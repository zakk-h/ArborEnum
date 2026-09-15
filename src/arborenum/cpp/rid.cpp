#include "rid_divisive_core.cpp"
#include "rid_divisive_frontier.cpp"

class ArborEnumRID : public ArborEnum {
public:
    using ArborEnum::ArborEnum;

    std::vector<ExactImportanceInterval>
    get_exact_replacement_importance_intervals_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,
        const std::vector<std::vector<int>>& variable_columns_in = {},
        const std::vector<int>& bb_pred_eval = {},
        const std::vector<std::vector<int>>&
            matched_group_of_row_by_variable_eval = {},
        const std::vector<std::vector<int>>&
            matched_group_size_by_variable_eval = {},
        bool sum_samplewise_extrema = false
    ) const {
        if (!rid_divisive_interval_active || sum_samplewise_extrema) {
            return ArborEnum::get_exact_replacement_importance_intervals_packed_trie(
                X_row_major,
                y_eval,
                budget_override,
                variable_columns_in,
                bb_pred_eval,
                matched_group_of_row_by_variable_eval,
                matched_group_size_by_variable_eval,
                sum_samplewise_extrema
            );
        }

        if (!bb_pred_eval.empty()) {
            throw std::runtime_error(
                "Divisive model reliance does not support deferral."
            );
        }

        return rid_exact_divisive_importance_intervals(
            *this,
            X_row_major,
            y_eval,
            budget_override < 0 ? result->budget : budget_override,
            variable_columns_in,
            matched_group_of_row_by_variable_eval,
            matched_group_size_by_variable_eval
        );
    }
};

#define ArborEnum ArborEnumRID
#define compute_rid_subtractive_mr_bootstrap compute_rid_subtractive_mr_bootstrap_base
#include "rid_base.cpp"
#undef compute_rid_subtractive_mr_bootstrap
#undef ArborEnum

RIDResult compute_rid_subtractive_mr_bootstrap(
    const std::vector<std::vector<uint8_t>>& X_row_major,
    const std::vector<int>& y,
    int n_bootstraps,
    int n_scramble_evals,
    double lambda,
    int depth_budget,
    double rashomon_mult,
    int lookahead_k,
    uint64_t seed,
    bool memory_efficient,
    ArborEnum::KeyMode key_mode = ArborEnum::KeyMode::HASH64,
    bool trie_cache_enabled = true,
    const std::vector<std::vector<int>>& binning_map_vars = {},
    const std::vector<int>& continuous_starts = {},
    bool use_anytime_fit = false,
    double second_rashomon_mult = -1.0,
    double multiplier_step_size = 0.01,
    const std::vector<int>& proxy_threshold_features = {},
    const std::vector<int>& initial_active_threshold_features = {},
    int refinement_width = 1,
    int max_refinement_rounds = -1,
    bool use_multipass = true,
    bool rule_list_mode = false,
    int proxy_style = 0,
    bool majority_leaf_only = false,
    bool cache_cheap_subproblems = false,
    int greedy_split_mode = 1,
    int greedy_continuous_mode = 0,
    bool proxy_caching = true,
    int proxy_refinement_mode = 0,
    bool continuous_proxy_in_lickety = true,
    bool continuous_proxy_in_depthd_exact = true,
    bool continuous_proxy_in_greedy = true,
    double runtime_limit_seconds = -1.0,
    double memory_limit_mb = -1.0,
    bool use_deferral = false,
    double eta_defer = 0.0,
    const std::vector<int>& bb_pred = {},
    bool return_joint_samples = false,
    bool lossless = false,
    const std::vector<std::vector<std::vector<int>>>& matched_groups_by_variable = {},
    bool additive = false,
    int importance_interval_mode = 0,
    int subsample = -1,
    int root_budget = -1
) {
    const bool activate =
        lossless &&
        importance_interval_mode == 1 &&
        n_scramble_evals == rid_divisive_interval_sentinel;

    const bool previous = rid_divisive_interval_active;
    rid_divisive_interval_active = activate;

    try {
        RIDResult result = compute_rid_subtractive_mr_bootstrap_base(
            X_row_major,
            y,
            n_bootstraps,
            activate ? 1 : n_scramble_evals,
            lambda,
            depth_budget,
            rashomon_mult,
            lookahead_k,
            seed,
            memory_efficient,
            key_mode,
            trie_cache_enabled,
            binning_map_vars,
            continuous_starts,
            use_anytime_fit,
            second_rashomon_mult,
            multiplier_step_size,
            proxy_threshold_features,
            initial_active_threshold_features,
            refinement_width,
            max_refinement_rounds,
            use_multipass,
            rule_list_mode,
            proxy_style,
            majority_leaf_only,
            cache_cheap_subproblems,
            greedy_split_mode,
            greedy_continuous_mode,
            proxy_caching,
            proxy_refinement_mode,
            continuous_proxy_in_lickety,
            continuous_proxy_in_depthd_exact,
            continuous_proxy_in_greedy,
            runtime_limit_seconds,
            memory_limit_mb,
            use_deferral,
            eta_defer,
            bb_pred,
            return_joint_samples,
            lossless,
            matched_groups_by_variable,
            additive,
            importance_interval_mode,
            subsample,
            root_budget
        );
        rid_divisive_interval_active = previous;
        return result;
    } catch (...) {
        rid_divisive_interval_active = previous;
        throw;
    }
}
