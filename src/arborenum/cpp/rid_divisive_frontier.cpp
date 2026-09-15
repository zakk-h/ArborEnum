static RIDDivisiveFrontiers rid_collect_divisive_frontiers(
    const TreeTrieNode* node,
    int budget,
    const RIDDivisiveMask& original_mask,
    const RIDDivisiveMask& replacement_root_mask,
    const RIDDivisiveReplacementState& state,
    int variable,
    const std::vector<int>& internal_to_variable,
    const std::vector<RIDDivisiveMask>& x_bits,
    const std::vector<RIDDivisiveMask>& y_bits,
    int n,
    const std::vector<std::vector<int>>& matched_group_of_row_by_variable,
    const std::vector<std::vector<int>>& matched_group_size_by_variable
) {
    RIDDivisiveFrontiers out;
    if (!node || budget < 0) return out;

    constexpr int INF = std::numeric_limits<int>::max();
    if (node->min_objective == INF || node->min_objective > budget) {
        return out;
    }

    std::vector<RIDDivisivePoint> lower_candidates;
    std::vector<RIDDivisivePoint> upper_candidates;

    for (const auto& leaf : node->leaves) {
        if (leaf.loss > budget) continue;

        const int original_mistakes =
            rid_divisive_wrong_count(
                original_mask,
                leaf.prediction,
                y_bits
            );

        const double replacement_mistakes =
            rid_divisive_replacement_mistakes(
                state,
                variable,
                leaf.prediction,
                original_mistakes,
                n,
                y_bits,
                matched_group_of_row_by_variable,
                matched_group_size_by_variable
            );

        const RIDDivisivePoint point{
            leaf.loss,
            original_mistakes,
            replacement_mistakes
        };

        lower_candidates.push_back(point);
        upper_candidates.push_back(point);

        if (original_mistakes == 0) {
            out.min_zero_obj = std::min(out.min_zero_obj, leaf.loss);
        }
    }

    for (const auto& split : node->splits) {
        const TreeTrieNode* left_node = split.left.get();
        const TreeTrieNode* right_node = split.right.get();
        if (!left_node || !right_node) continue;

        const int min_left = left_node->min_objective;
        const int min_right = right_node->min_objective;
        if (min_left == INF || min_right == INF) continue;

        int left_budget = budget - min_right;
        int right_budget = budget - min_left;
        if (left_budget < 0 || right_budget < 0) continue;

        left_budget = std::min(left_budget, left_node->budget);
        right_budget = std::min(right_budget, right_node->budget);
        if (left_budget < min_left || right_budget < min_right) continue;

        if (
            split.feature < 0 ||
            split.feature >= (int)internal_to_variable.size()
        ) {
            throw std::runtime_error(
                "Divisive model reliance saw an invalid split feature."
            );
        }

        const int split_variable =
            internal_to_variable[(std::size_t)split.feature];
        if (split_variable < 0) {
            throw std::runtime_error(
                "Divisive model reliance saw an unmapped split feature."
            );
        }

        const auto& x_feature = x_bits[(std::size_t)split.feature];
        RIDDivisiveMask original_left =
            rid_divisive_and(original_mask, x_feature);
        RIDDivisiveMask original_right =
            rid_divisive_andnot(original_mask, x_feature);

        RIDDivisiveReplacementState left_state;
        RIDDivisiveReplacementState right_state;

        if (!state.used) {
            if (variable == split_variable) {
                left_state.used = true;
                right_state.used = true;
                left_state.target_rows = original_mask;
                right_state.target_rows = original_mask;
                left_state.replacement_values =
                    rid_divisive_and(replacement_root_mask, x_feature);
                right_state.replacement_values =
                    rid_divisive_andnot(replacement_root_mask, x_feature);
            }
        } else {
            left_state.used = true;
            right_state.used = true;

            if (variable == split_variable) {
                left_state.target_rows = state.target_rows;
                right_state.target_rows = state.target_rows;
                left_state.replacement_values =
                    rid_divisive_and(state.replacement_values, x_feature);
                right_state.replacement_values =
                    rid_divisive_andnot(state.replacement_values, x_feature);
            } else {
                left_state.target_rows =
                    rid_divisive_and(state.target_rows, x_feature);
                right_state.target_rows =
                    rid_divisive_andnot(state.target_rows, x_feature);
                left_state.replacement_values = state.replacement_values;
                right_state.replacement_values = state.replacement_values;
            }
        }

        auto left = rid_collect_divisive_frontiers(
            left_node,
            left_budget,
            original_left,
            replacement_root_mask,
            left_state,
            variable,
            internal_to_variable,
            x_bits,
            y_bits,
            n,
            matched_group_of_row_by_variable,
            matched_group_size_by_variable
        );

        auto right = rid_collect_divisive_frontiers(
            right_node,
            right_budget,
            original_right,
            replacement_root_mask,
            right_state,
            variable,
            internal_to_variable,
            x_bits,
            y_bits,
            n,
            matched_group_of_row_by_variable,
            matched_group_size_by_variable
        );

        if (
            left.min_zero_obj != INF &&
            right.min_zero_obj != INF &&
            left.min_zero_obj <= budget - right.min_zero_obj
        ) {
            out.min_zero_obj = std::min(
                out.min_zero_obj,
                left.min_zero_obj + right.min_zero_obj
            );
        }

        for (const auto& lp : left.lower) {
            for (const auto& rp : right.lower) {
                const int obj = lp.obj + rp.obj;
                if (obj > budget) continue;
                lower_candidates.push_back(
                    RIDDivisivePoint{
                        obj,
                        lp.original_mistakes + rp.original_mistakes,
                        lp.replacement_mistakes + rp.replacement_mistakes
                    }
                );
            }
        }

        for (const auto& lp : left.upper) {
            for (const auto& rp : right.upper) {
                const int obj = lp.obj + rp.obj;
                if (obj > budget) continue;
                upper_candidates.push_back(
                    RIDDivisivePoint{
                        obj,
                        lp.original_mistakes + rp.original_mistakes,
                        lp.replacement_mistakes + rp.replacement_mistakes
                    }
                );
            }
        }
    }

    out.lower = rid_divisive_prune(
        std::move(lower_candidates),
        true,
        n
    );
    out.upper = rid_divisive_prune(
        std::move(upper_candidates),
        false,
        n
    );

    return out;
}

static std::vector<std::pair<double, double>>
rid_exact_divisive_importance_intervals(
    const ArborEnum& model,
    const std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& y,
    int budget,
    const std::vector<std::vector<int>>& variable_columns,
    const std::vector<std::vector<int>>& matched_group_of_row_by_variable,
    const std::vector<std::vector<int>>& matched_group_size_by_variable
) {
    if (!model.result) return {};

    const int n = (int)X.size();
    const int d = n > 0 ? (int)X[0].size() : 0;
    const int n_words = (n + 63) / 64;
    const int number_of_variables = (int)variable_columns.size();

    if (n <= 0 || d <= 0 || number_of_variables <= 0) return {};

    std::vector<int> internal_to_variable((std::size_t)d, -1);
    for (int variable = 0; variable < number_of_variables; ++variable) {
        for (int column : variable_columns[(std::size_t)variable]) {
            if (column < 0 || column >= d) {
                throw std::runtime_error(
                    "Divisive model reliance received an invalid variable column."
                );
            }
            internal_to_variable[(std::size_t)column] = variable;
        }
    }

    std::vector<RIDDivisiveMask> x_bits(
        (std::size_t)d,
        rid_divisive_empty_mask(n_words)
    );

    for (int row = 0; row < n; ++row) {
        const std::size_t wi = (std::size_t)(row >> 6);
        const uint64_t bit = 1ULL << (row & 63);
        for (int column = 0; column < d; ++column) {
            if (X[(std::size_t)row][(std::size_t)column]) {
                x_bits[(std::size_t)column].w[wi] |= bit;
            }
        }
    }

    int number_of_classes = 0;
    for (int value : y) {
        number_of_classes = std::max(number_of_classes, value + 1);
    }

    std::vector<RIDDivisiveMask> y_bits(
        (std::size_t)number_of_classes,
        rid_divisive_empty_mask(n_words)
    );

    for (int row = 0; row < n; ++row) {
        const int cls = y[(std::size_t)row];
        if (cls < 0 || cls >= number_of_classes) {
            throw std::runtime_error(
                "Divisive model reliance received an invalid class label."
            );
        }
        y_bits[(std::size_t)cls].w[(std::size_t)(row >> 6)] |=
            1ULL << (row & 63);
    }

    RIDDivisiveMask root = rid_divisive_empty_mask(n_words);
    for (int row = 0; row < n; ++row) {
        root.w[(std::size_t)(row >> 6)] |= 1ULL << (row & 63);
    }

    if (!matched_group_of_row_by_variable.empty()) {
        if (
            (int)matched_group_of_row_by_variable.size() != number_of_variables ||
            (int)matched_group_size_by_variable.size() != number_of_variables
        ) {
            throw std::runtime_error(
                "Divisive model reliance received inconsistent matched groups."
            );
        }
    }

    std::vector<std::pair<double, double>> out(
        (std::size_t)number_of_variables,
        {0.0, 0.0}
    );

    for (int variable = 0; variable < number_of_variables; ++variable) {
        RIDDivisiveReplacementState root_state;
        auto frontiers = rid_collect_divisive_frontiers(
            model.result.get(),
            budget,
            root,
            root,
            root_state,
            variable,
            internal_to_variable,
            x_bits,
            y_bits,
            n,
            matched_group_of_row_by_variable,
            matched_group_size_by_variable
        );

        if (
            frontiers.min_zero_obj != std::numeric_limits<int>::max() &&
            frontiers.min_zero_obj <= budget
        ) {
            throw std::runtime_error(
                "Divisive model reliance is undefined because a feasible tree has zero original evaluation loss."
            );
        }

        if (frontiers.lower.empty() || frontiers.upper.empty()) {
            return {};
        }

        double lower = std::numeric_limits<double>::infinity();
        double upper = -std::numeric_limits<double>::infinity();

        for (const auto& point : frontiers.lower) {
            if (point.obj > budget) continue;
            lower = std::min(
                lower,
                point.replacement_mistakes /
                    (double)point.original_mistakes
            );
        }

        for (const auto& point : frontiers.upper) {
            if (point.obj > budget) continue;
            upper = std::max(
                upper,
                point.replacement_mistakes /
                    (double)point.original_mistakes
            );
        }

        if (!std::isfinite(lower) || !std::isfinite(upper)) return {};
        out[(std::size_t)variable] = {lower, upper};
    }

    return out;
}

