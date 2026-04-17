#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Drop-in decode-time MoE slot planner for llama.cpp-style expert logging.
//
// This version fixes the real bug found in the small-model sanity check:
// - planning multiple fills in one round now uses a TEMPORARY slot state,
//   so one slot cannot be assigned to multiple experts in the same plan.
// - applying fills now removes any previous occupant from gid_to_slot_ even if
//   the planner did not explicitly record it as an eviction.
//
// It also adds a model-shape-aware configuration helper so the planner can be
// configured from the current model instead of assuming Gemma-style defaults.
//
// Intended wiring:
// 1. Construct SlotPlanner with config derived from the loaded model.
// 2. Call begin_decode_step(token_index) when ubatch.n_tokens == 1.
// 3. For each MoE layer callback on that token, call observe_layer(layer, values).
// 4. AFTER process_ubatch(...) returns, call finish_decode_step_and_get_plan().
// 5. Use the returned fills for async CPU->GPU copies for the NEXT token.
//
// Important: do not call finish_decode_step_and_get_plan() before the current
// token's callback data has been collected, or the planner will be working on
// stale state.

namespace llama_moe_plan {

struct Config {
    int n_layers = 30;
    int n_experts_per_layer = 128;
    int top_k = 8;

    // Global slot pool size across all layers.
    int n_global_slots = 1152;

    // How many fills to suggest for the NEXT token after observing current token.
    int max_fills_per_token = 4;

    // Rolling stats window.
    int hot_window = 32;

    // Layer stability window and threshold.
    int stability_window = 32;
    double stability_threshold = 0.30;

    // Do not evict if seen in one of the last N decode steps when avoidable.
    int protect_recent_steps = 2;

    // Prefer experts that make almost-complete layers even better.
    double score_completeness = 4.0;
    double score_hotness = 1.5;
    double score_recency = 1.0;
    double score_layer_stability = 3.0;

    // Mild bias for currently selected experts from current token.
    double score_current_presence = 0.5;

    // Small bonus to experts that have been resident for a while.
    double eviction_stickiness = 0.2;
};

inline Config make_model_shaped_config(
        int n_layers,
        int n_experts_per_layer,
        int top_k,
        int n_global_slots,
        int max_fills_per_token = 4) {
    Config cfg;
    cfg.n_layers = n_layers;
    cfg.n_experts_per_layer = n_experts_per_layer;
    cfg.top_k = top_k;
    cfg.n_global_slots = n_global_slots;
    cfg.max_fills_per_token = max_fills_per_token;
    return cfg;
}

struct FillDecision {
    int slot = -1;
    int layer = -1;
    int expert = -1;
    int gid = -1;

    int evict_gid = -1;     // -1 means empty slot used
    int evict_layer = -1;
    int evict_expert = -1;

    double score = 0.0;
    std::string reason;
};

struct LayerPlanSummary {
    int layer = -1;
    double stability = 0.0;
    bool stable_enough = false;

    int selected_count = 0;
    int resident_count = 0;
    int missing_count = 0;

    std::vector<int> selected_experts;
    std::vector<int> resident_experts;
    std::vector<int> missing_experts;
};

struct TokenPlan {
    int token_index = -1;
    int next_token_index = -1;

    std::vector<LayerPlanSummary> layer_summaries;
    std::vector<FillDecision> fills;

    int total_selected = 0;
    int total_resident = 0;
    int total_missing = 0;
    double resident_fraction = 0.0;

    std::string debug_string() const {
        std::ostringstream ss;
        ss << "token=" << token_index << " next=" << next_token_index
           << " selected=" << total_selected
           << " resident=" << total_resident
           << " missing=" << total_missing
           << " resident_fraction=" << resident_fraction;
        for (const auto & f : fills) {
            ss << "\n  fill slot=" << f.slot
               << " gid=" << f.gid
               << " (L" << f.layer << ":E" << f.expert << ")"
               << " evict=";
            if (f.evict_gid >= 0) {
                ss << f.evict_gid << " (L" << f.evict_layer << ":E" << f.evict_expert << ")";
            } else {
                ss << "none";
            }
            ss << " score=" << f.score << " reason=" << f.reason;
        }
        return ss.str();
    }
};

class SlotPlanner {
public:
    explicit SlotPlanner(Config cfg = {}) : cfg_(cfg) {
        reconfigure(cfg_);
    }

    void reconfigure(const Config & cfg) {
        cfg_ = cfg;
        slot_to_gid_.assign(std::max(0, cfg_.n_global_slots), -1);
        reset();
    }

    void reset() {
        token_steps_.clear();
        current_step_.clear();
        current_token_index_ = -1;
        seed_slots();
        gid_last_seen_.clear();
        gid_freq_.clear();
        recent_steps_per_gid_.clear();
        layer_jaccard_hist_.clear();
        layer_prev_selected_.clear();
    }

    void clear_slots() {
        gid_to_slot_.clear();
        std::fill(slot_to_gid_.begin(), slot_to_gid_.end(), -1);
    }

    const Config & config() const { return cfg_; }

    void begin_decode_step(int token_index) {
        current_token_index_ = token_index;
        current_step_.clear();
    }

    void observe_layer(int layer, const std::vector<int32_t> & selected_experts) {
        if (layer < 0 || layer >= cfg_.n_layers) {
            return;
        }

        std::vector<int> uniq;
        uniq.reserve(selected_experts.size());
        for (int32_t e : selected_experts) {
            if (e < 0 || e >= cfg_.n_experts_per_layer) {
                continue;
            }
            uniq.push_back((int) e);
        }
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        current_step_[layer] = std::move(uniq);
    }

    TokenPlan finish_decode_step_and_get_plan() {
        normalize_current_step_from_previous();

        TokenPlan plan;
        plan.token_index = current_token_index_;
        plan.next_token_index = current_token_index_ + 1;

        build_layer_summaries(plan);
        plan.fills = choose_fills(plan.layer_summaries);
        apply_fills(plan.fills);
        commit_current_step_histories();

        return plan;
    }

    int gid_to_slot(int gid) const {
        const auto it = gid_to_slot_.find(gid);
        return it == gid_to_slot_.end() ? -1 : it->second;
    }

    int make_gid(int layer, int expert) const {
        return layer * cfg_.n_experts_per_layer + expert;
    }

    std::pair<int, int> split_gid(int gid) const {
        return {gid / cfg_.n_experts_per_layer, gid % cfg_.n_experts_per_layer};
    }

private:
    struct Candidate {
        int gid = -1;
        int layer = -1;
        int expert = -1;
        double score = 0.0;
        std::string reason;
    };

    Config cfg_;
    int current_token_index_ = -1;

    std::map<int, std::vector<int>> current_step_;
    std::deque<std::map<int, std::vector<int>>> token_steps_;

    std::unordered_map<int, int> gid_to_slot_;
    std::vector<int> slot_to_gid_;

    std::unordered_map<int, int> gid_last_seen_;
    std::unordered_map<int, int> gid_freq_;
    std::unordered_map<int, std::deque<int>> recent_steps_per_gid_;
    std::unordered_map<int, std::deque<double>> layer_jaccard_hist_;
    std::unordered_map<int, std::vector<int>> layer_prev_selected_;

    void seed_slots() {
        gid_to_slot_.clear();
        std::fill(slot_to_gid_.begin(), slot_to_gid_.end(), -1);

        const int total_gids = cfg_.n_layers * cfg_.n_experts_per_layer;
        const int n_seed = std::min((int) slot_to_gid_.size(), total_gids);

        for (int gid = 0; gid < n_seed; ++gid) {
            slot_to_gid_[gid] = gid;
            gid_to_slot_[gid] = gid;
        }
    }


    static double jaccard_sorted(const std::vector<int> & a, const std::vector<int> & b) {
        size_t i = 0, j = 0;
        int inter = 0;
        int uni = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] == b[j]) {
                ++inter; ++uni; ++i; ++j;
            } else if (a[i] < b[j]) {
                ++uni; ++i;
            } else {
                ++uni; ++j;
            }
        }
        uni += (int) (a.size() - i + b.size() - j);
        return uni == 0 ? 1.0 : (double) inter / (double) uni;
    }

    void normalize_current_step_from_previous() {
        for (int l = 0; l < cfg_.n_layers; ++l) {
            if (current_step_.find(l) == current_step_.end()) {
                const auto it = layer_prev_selected_.find(l);
                current_step_[l] = it == layer_prev_selected_.end() ? std::vector<int>{} : it->second;
            }
        }
    }

    double layer_stability(int layer) const {
        const auto it = layer_jaccard_hist_.find(layer);
        if (it == layer_jaccard_hist_.end() || it->second.empty()) {
            return 0.0;
        }
        double s = 0.0;
        for (double x : it->second) s += x;
        return s / (double) it->second.size();
    }

    int gid_hotness(int gid) const {
        const auto it = gid_freq_.find(gid);
        return it == gid_freq_.end() ? 0 : it->second;
    }

    bool gid_seen_recently(int gid, int n_steps) const {
        const auto it = recent_steps_per_gid_.find(gid);
        if (it == recent_steps_per_gid_.end()) {
            return false;
        }
        for (int step : it->second) {
            if (current_token_index_ - step < n_steps) {
                return true;
            }
        }
        return false;
    }

    void build_layer_summaries(TokenPlan & plan) const {
        for (int l = 0; l < cfg_.n_layers; ++l) {
            LayerPlanSummary s;
            s.layer = l;
            s.stability = layer_stability(l);
            s.stable_enough = s.stability >= cfg_.stability_threshold;

            const auto it = current_step_.find(l);
            if (it != current_step_.end()) {
                s.selected_experts = it->second;
            }
            s.selected_count = (int) s.selected_experts.size();
            plan.total_selected += s.selected_count;

            for (int e : s.selected_experts) {
                const int gid = make_gid(l, e);
                if (gid_to_slot(gid) >= 0) {
                    s.resident_experts.push_back(e);
                    ++plan.total_resident;
                } else {
                    s.missing_experts.push_back(e);
                    ++plan.total_missing;
                }
            }
            s.resident_count = (int) s.resident_experts.size();
            s.missing_count = (int) s.missing_experts.size();
            plan.layer_summaries.push_back(std::move(s));
        }
        plan.resident_fraction = plan.total_selected == 0 ? 1.0 : (double) plan.total_resident / (double) plan.total_selected;
    }

    static int first_free_slot_in(const std::vector<int> & slot_to_gid) {
        for (int i = 0; i < (int) slot_to_gid.size(); ++i) {
            if (slot_to_gid[i] < 0) {
                return i;
            }
        }
        return -1;
    }

    int choose_victim_gid_with_state(
            const std::unordered_set<int> & reserved,
            const std::vector<int> & temp_slot_to_gid) const {
        int victim = -1;
        double best = std::numeric_limits<double>::infinity();

        for (int pass = 0; pass < 2; ++pass) {
            for (int slot = 0; slot < (int) temp_slot_to_gid.size(); ++slot) {
                const int gid = temp_slot_to_gid[slot];
                if (gid < 0) continue;
                if (reserved.find(gid) != reserved.end()) continue;

                const bool protected_recent = gid_seen_recently(gid, cfg_.protect_recent_steps);
                if (pass == 0 && protected_recent) continue;

                double score = 0.0;
                score += cfg_.eviction_stickiness * (double) gid_hotness(gid);
                const auto it_last = gid_last_seen_.find(gid);
                if (it_last != gid_last_seen_.end()) {
                    const int age = std::max(1, current_token_index_ - it_last->second);
                    score += cfg_.eviction_stickiness * (1.0 / (double) age);
                }

                if (score < best) {
                    best = score;
                    victim = gid;
                }
            }
            if (victim >= 0) {
                return victim;
            }
        }
        return -1;
    }

    std::vector<FillDecision> choose_fills(const std::vector<LayerPlanSummary> & layers) const {
        std::vector<Candidate> cands;
        cands.reserve(cfg_.n_layers * std::max(1, cfg_.top_k));

        for (const auto & ls : layers) {
            if (!ls.stable_enough) {
                continue;
            }
            if (ls.missing_count == 0) {
                continue;
            }

            const double completeness = (double) ls.resident_count / std::max(1, ls.selected_count);
            const int closeness = ls.resident_count;
            const double layer_base =
                cfg_.score_layer_stability * ls.stability +
                cfg_.score_completeness * (1.0 + completeness + 0.25 * closeness);

            for (int e : ls.missing_experts) {
                const int gid = make_gid(ls.layer, e);
                Candidate c;
                c.gid = gid;
                c.layer = ls.layer;
                c.expert = e;
                c.score = layer_base;
                c.score += cfg_.score_hotness * (double) gid_hotness(gid);

                const auto it_last = gid_last_seen_.find(gid);
                if (it_last != gid_last_seen_.end()) {
                    const int age = std::max(1, current_token_index_ - it_last->second);
                    c.score += cfg_.score_recency * (1.0 / (double) age);
                }
                c.score += cfg_.score_current_presence;

                std::ostringstream why;
                why << "stable_layer=" << ls.layer
                    << " stability=" << ls.stability
                    << " resident=" << ls.resident_count
                    << "/" << ls.selected_count
                    << " hot=" << gid_hotness(gid);
                c.reason = why.str();
                cands.push_back(std::move(c));
            }
        }

        std::sort(cands.begin(), cands.end(), [](const Candidate & a, const Candidate & b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.expert < b.expert;
        });

        std::vector<FillDecision> fills;
        fills.reserve(cfg_.max_fills_per_token);

        // IMPORTANT: plan against temporary cache state so we cannot assign the
        // same slot multiple times in one planning round.
        std::vector<int> temp_slot_to_gid = slot_to_gid_;
        std::unordered_map<int, int> temp_gid_to_slot = gid_to_slot_;
        std::unordered_set<int> reserved_gids;

        for (const auto & c : cands) {
            if ((int) fills.size() >= cfg_.max_fills_per_token) {
                break;
            }
            if (temp_gid_to_slot.find(c.gid) != temp_gid_to_slot.end()) {
                continue;
            }
            if (!reserved_gids.insert(c.gid).second) {
                continue;
            }

            FillDecision fd;
            fd.gid = c.gid;
            fd.layer = c.layer;
            fd.expert = c.expert;
            fd.score = c.score;
            fd.reason = c.reason;

            int slot = first_free_slot_in(temp_slot_to_gid);
            if (slot >= 0) {
                fd.slot = slot;
                temp_slot_to_gid[slot] = c.gid;
                temp_gid_to_slot[c.gid] = slot;
                fills.push_back(std::move(fd));
                continue;
            }

            const int victim_gid = choose_victim_gid_with_state(reserved_gids, temp_slot_to_gid);
            if (victim_gid < 0) {
                continue;
            }
            const auto it_slot = temp_gid_to_slot.find(victim_gid);
            if (it_slot == temp_gid_to_slot.end()) {
                continue;
            }
            slot = it_slot->second;
            fd.slot = slot;
            fd.evict_gid = victim_gid;
            auto [vl, ve] = split_gid(victim_gid);
            fd.evict_layer = vl;
            fd.evict_expert = ve;

            // Immediately update temporary state.
            temp_gid_to_slot.erase(victim_gid);
            temp_slot_to_gid[slot] = c.gid;
            temp_gid_to_slot[c.gid] = slot;
            reserved_gids.insert(victim_gid);
            fills.push_back(std::move(fd));
        }

        return fills;
    }

    void apply_fills(const std::vector<FillDecision> & fills) {
        for (const auto & f : fills) {
            if (f.slot < 0 || f.slot >= (int) slot_to_gid_.size()) {
                continue;
            }

            // Remove whatever currently occupies that slot, even if the plan did
            // not explicitly track it as an eviction. This keeps gid_to_slot_
            // and slot_to_gid_ consistent.
            const int old_gid = slot_to_gid_[f.slot];
            if (old_gid >= 0) {
                gid_to_slot_.erase(old_gid);
            }
            if (f.evict_gid >= 0) {
                gid_to_slot_.erase(f.evict_gid);
            }

            slot_to_gid_[f.slot] = f.gid;
            gid_to_slot_[f.gid] = f.slot;
        }
    }

    void commit_current_step_histories() {
        token_steps_.push_back(current_step_);
        while ((int) token_steps_.size() > cfg_.hot_window) {
            const auto & old = token_steps_.front();
            for (const auto & kv : old) {
                const int layer = kv.first;
                const auto & exps = kv.second;
                for (int e : exps) {
                    const int gid = make_gid(layer, e);
                    auto it = gid_freq_.find(gid);
                    if (it != gid_freq_.end()) {
                        if (--it->second <= 0) {
                            gid_freq_.erase(it);
                        }
                    }
                }
            }
            token_steps_.pop_front();
        }

        for (const auto & kv : current_step_) {
            const int layer = kv.first;
            const auto & exps = kv.second;
            for (int e : exps) {
                const int gid = make_gid(layer, e);
                gid_freq_[gid] += 1;
                gid_last_seen_[gid] = current_token_index_;
                auto & dq = recent_steps_per_gid_[gid];
                dq.push_back(current_token_index_);
                while ((int) dq.size() > cfg_.protect_recent_steps + 4) {
                    dq.pop_front();
                }
            }
        }

        for (int l = 0; l < cfg_.n_layers; ++l) {
            const auto it = current_step_.find(l);
            const std::vector<int> & curr = it == current_step_.end() ? empty_vec() : it->second;
            const auto pit = layer_prev_selected_.find(l);
            const std::vector<int> & prev = pit == layer_prev_selected_.end() ? empty_vec() : pit->second;
            const double jac = jaccard_sorted(prev, curr);
            auto & hist = layer_jaccard_hist_[l];
            hist.push_back(jac);
            while ((int) hist.size() > cfg_.stability_window) {
                hist.pop_front();
            }
            layer_prev_selected_[l] = curr;
        }
    }

    static const std::vector<int> & empty_vec() {
        static const std::vector<int> kEmpty;
        return kEmpty;
    }
};

} // namespace llama_moe_plan

#ifdef LLAMA_MOE_SLOT_PLANNER_EXAMPLE_MAIN
int main() {
    using namespace llama_moe_plan;
    Config cfg;

    SlotPlanner planner(
        cfg
        // make_model_shaped_config(
        // /*n_layers=*/22,
        // /*n_experts_per_layer=*/4,
        // /*top_k=*/2,
        // /*n_global_slots=*/16,
        // /*max_fills_per_token=*/4)
    );

    planner.begin_decode_step(0);
    for (int l = 0; l < 22; ++l) {
        planner.observe_layer(l, {0, 2});
    }
    auto p0 = planner.finish_decode_step_and_get_plan();
    std::printf("%s\n", p0.debug_string().c_str());

    planner.begin_decode_step(1);
    for (int l = 0; l < 22; ++l) {
        planner.observe_layer(l, {0, 3});
    }
    auto p1 = planner.finish_decode_step_and_get_plan();
    std::printf("%s\n", p1.debug_string().c_str());
    return 0;
}
#endif
