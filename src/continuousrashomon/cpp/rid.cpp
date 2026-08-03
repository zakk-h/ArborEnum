// implementation for the rashomon importance distribution - not one of our contributions but still integrated with the method.
#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cmath>

using std::cout;

struct RIDResult {
    std::vector<double> mean_sub_mr;
    std::vector<std::vector<double>> cdf_x;
    std::vector<std::vector<double>> cdf_p;
};

static inline uint64_t popcnt64_u(uint64_t x) {
#if defined(_MSC_VER)
    return (uint64_t)__popcnt64(x);
#else
    return (uint64_t)__builtin_popcountll(x);
#endif
}

// y_bits[c] has bit i = 1 iff y[i] == c
static inline std::vector<Packed> build_yc_packed(
    const std::vector<int>& y,
    int n_classes,
    int n_words,
    uint64_t tail_mask
) {
    std::vector<Packed> y_bits;
    y_bits.reserve((size_t)n_classes);
    for (int c = 0; c < n_classes; ++c) y_bits.emplace_back((size_t)n_words);

    for (int i = 0; i < (int)y.size(); ++i) {
        const int c = y[i];
        // (optional) assert 0 <= c < n_classes
        y_bits[(size_t)c].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
    }

    if (n_words > 0) {
        for (int c = 0; c < n_classes; ++c) {
            y_bits[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
        }
    }
    return y_bits;
}

static inline int count_correct_packed_multi(
    const PackedPredMulti& pred,
    const std::vector<Packed>& y_bits,
    int n_words,
    uint64_t tail_mask
) {
    const int C = (int)y_bits.size();
    uint64_t correct = 0;

    for (int c = 0; c < C; ++c) {
        const auto& pw = pred.by_class[(size_t)c].w;
        const auto& yw = y_bits[(size_t)c].w;

        for (int w = 0; w < n_words; ++w) {
            uint64_t bits = pw[(size_t)w] & yw[(size_t)w];
            if (w == n_words - 1) bits &= tail_mask;
            correct += popcnt64_u(bits);
        }
    }
    return (int)correct;
}



static inline void bootstrap_indices(int n, std::mt19937_64& rng, std::vector<int>& idx) {
    std::uniform_int_distribution<int> unif(0, n - 1);
    idx.resize(n);
    for (int i = 0; i < n; ++i) idx[i] = unif(rng);
}

static inline void make_bootstrap_dataset(
    const std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& y,
    const std::vector<int>& idx,
    std::vector<std::vector<uint8_t>>& Xb,
    std::vector<int>& yb
) {
    const int n = (int)idx.size();
    const int d = (int)X[0].size();
    Xb.assign(n, std::vector<uint8_t>(d));
    yb.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const int s = idx[i];
        Xb[i] = X[s];
        yb[i] = y[s];
    }
}

static inline void rowmajor_to_colmajor_bool(
    const std::vector<std::vector<uint8_t>>& X_row,
    std::vector<std::vector<bool>>& X_col
) {
    const int n = (int)X_row.size();
    const int d = (int)X_row[0].size();
    X_col.assign(d, std::vector<bool>(n, false));
    for (int i = 0; i < n; ++i) {
        const auto& row = X_row[i];
        for (int j = 0; j < d; ++j) {
            X_col[j][i] = (row[j] != 0);
        }
    }
}

static inline void make_permutation(int n, std::mt19937_64& rng, std::vector<int>& perm) {
    perm.resize(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
}

// scramble a single feature, but represented by multiple binary columns
static inline void scramble_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& cols,
    const std::vector<int>& perm,
    std::vector<std::vector<uint8_t>>& saved_cols
) {
    const int n = (int)X.size();
    saved_cols.assign(cols.size(), std::vector<uint8_t>(n));

    // save originals
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];
        for (int i = 0; i < n; ++i) saved_cols[ci][i] = X[i][col];
    }

    // apply same permutation to each column in the block
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];
        for (int i = 0; i < n; ++i) X[i][col] = saved_cols[ci][perm[i]];
    }
}

static inline void restore_block_inplace(
    std::vector<std::vector<uint8_t>>& X,
    const std::vector<int>& cols,
    const std::vector<std::vector<uint8_t>>& saved_cols
) {
    const int n = (int)X.size();
    for (size_t ci = 0; ci < cols.size(); ++ci) {
        const int col = cols[ci];
        for (int i = 0; i < n; ++i) X[i][col] = saved_cols[ci][i];
    }
}

static inline int count_correct(const std::vector<uint8_t>& preds, const std::vector<int>& y) {
    const int n = (int)y.size();
    int c = 0;
    for (int i = 0; i < n; ++i) c += (preds[i] == (uint8_t)y[i]);
    return c;
}

RIDResult compute_rid_subtractive_mr_bootstrap(
    const std::vector<std::vector<uint8_t>>& X_row_major,
    const std::vector<int>& y,
    int n_bootstraps,
    double lambda,
    int depth_budget,
    double rashomon_mult,
    int lookahead_k,
    uint64_t seed,
    bool memory_efficient,
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
    double memory_limit_mb = -1.0
) {
    const int n_full = (int)X_row_major.size();
    const int d = (int)X_row_major[0].size();

    const double resolved_second_rashomon_mult =
        second_rashomon_mult < 0.0
            ? rashomon_mult
            : second_rashomon_mult;


    // build var->cols mapping.
    // if no binning map is provided, assume no relationship between binary features
    std::vector<std::vector<int>> var_cols;
    if (!binning_map_vars.empty()) {
        var_cols = binning_map_vars;
    } else {
        var_cols.resize((size_t)d);
        for (int j = 0; j < d; ++j) var_cols[(size_t)j] = std::vector<int>{j};
    }

    const int V = (int)var_cols.size();

    std::mt19937_64 rng(seed);

    RIDResult out;
    out.mean_sub_mr.assign(V, 0.0);
    out.cdf_x.assign(V, {});
    out.cdf_p.assign(V, {});

    // for each feature j, we aggregate a weighted empirical distribution of delta_correct = correct_orig - correct_scrambled
    std::vector<std::unordered_map<double, double>> mass_by_delta(V); // maps feature, delta to mass

    for (int b = 0; b < n_bootstraps; ++b) {
        std::vector<int> idx;
        bootstrap_indices(n_full, rng, idx);

        std::vector<std::vector<uint8_t>> Xb;
        std::vector<int> yb;
        make_bootstrap_dataset(X_row_major, y, idx, Xb, yb);

        const int n = (int)Xb.size();
        const int n_words = (n + 63) / 64;
        const uint64_t tail_mask = (n % 64) ? ((1ULL << (n % 64)) - 1ULL) : ~0ULL;
        int y_max = 0;
        for (int i = 0; i < (int)yb.size(); ++i) y_max = std::max(y_max, yb[i]);
        const int n_classes = y_max + 1;

        const auto y_bits = build_yc_packed(yb, n_classes, n_words, tail_mask);

        // row-major -> col-major bool for training
        std::vector<std::vector<bool>> Xcol;
        rowmajor_to_colmajor_bool(Xb, Xcol);

        PRAXIS model;

        model.set_greedy_split_mode(greedy_split_mode);
        model.set_greedy_continuous_mode(greedy_continuous_mode);

        const bool any_proxy_is_restricted =
            !continuous_proxy_in_lickety ||
            !continuous_proxy_in_depthd_exact ||
            !continuous_proxy_in_greedy;

        if (
            !continuous_starts.empty() &&
            any_proxy_is_restricted &&
            proxy_threshold_features.empty()
        ) {
            throw std::runtime_error(
                "Continuous RID requires nonempty proxy_threshold_features "
                "when any proxy component is restricted."
            );
        }

        if (use_anytime_fit) {
            model.fit_anytime(
                Xcol,
                yb,
                lambda,
                static_cast<int8_t>(depth_budget),
                rashomon_mult,
                resolved_second_rashomon_mult,
                multiplier_step_size,
                static_cast<int8_t>(lookahead_k),
                use_multipass,
                rule_list_mode,
                proxy_style,
                majority_leaf_only,
                cache_cheap_subproblems,
                proxy_caching,
                proxy_threshold_features,
                initial_active_threshold_features,
                refinement_width,
                max_refinement_rounds,
                proxy_refinement_mode,
                continuous_proxy_in_lickety,
                continuous_proxy_in_depthd_exact,
                continuous_proxy_in_greedy,
                continuous_starts,
                runtime_limit_seconds,
                memory_limit_mb
            );
        } else {
            model.fit(
                Xcol,
                yb,
                lambda,
                depth_budget,
                rashomon_mult,
                lookahead_k,
                -1,                           // root_budget
                use_multipass,
                rule_list_mode,
                proxy_style,
                majority_leaf_only,
                cache_cheap_subproblems,
                proxy_caching,
                proxy_threshold_features,    // allowed_proxy_features
                !continuous_proxy_in_lickety,
                !continuous_proxy_in_depthd_exact,
                !continuous_proxy_in_greedy,
                true,                         // rashomon_mode
                continuous_starts,
                false

            );
        }

        const uint64_t T64 = model.result ? model.result->count_trees() : 0ULL;
        const int T = (int)T64;
        if (T == 0) continue;
        cout << "Finished RID bootstrap: " << (b + 1) << " / " << n_bootstraps << " with " << T << " trees\n";

        // pre-sample permutations for each feature (one scramble per feature per bootstrap)
        // std::vector<std::vector<int>> perms((size_t)V);
        // for (int v = 0; v < V; ++v) make_permutation(n, rng, perms[(size_t)v]);

        // reuse buffer for column/block scrambling
        std::vector<std::vector<uint8_t>> saved_cols;

        // hardcoded number of random scrambles per variable per bootstrap
        const int n_scrambles_per_var = 5;

        const int budget_override = (int)llround((1.0 + rashomon_mult) * (double)model.result->min_objective);
        if (memory_efficient) {
            // memory-efficient method:
            // use only trees within budget_override. if there are too many,
            // uniformly sample 1000 objective-sorted tree indices from [0, T_budget - 1].

            const uint64_t T_budget =
                model.result ? model.result->count_leq(budget_override) : 0ULL;

            if (T_budget == 0) continue;

            std::vector<uint64_t> tree_indices;

            if (T_budget <= 1000ULL) {
                tree_indices.reserve((size_t)T_budget);
                for (uint64_t t = 0; t < T_budget; ++t) {
                    tree_indices.push_back(t);
                }
            } else {
                tree_indices.reserve(1000);

                std::unordered_set<uint64_t> seen;
                seen.reserve(2048);

                std::uniform_int_distribution<uint64_t> unif_tree(0ULL, T_budget - 1ULL);

                while (tree_indices.size() < 1000) {
                    const uint64_t t = unif_tree(rng);
                    if (seen.insert(t).second) {
                        tree_indices.push_back(t);
                    }
                }

                std::sort(tree_indices.begin(), tree_indices.end());
            }

            const uint64_t Tvec = (uint64_t)tree_indices.size();

            std::vector<int> correct_orig;
            correct_orig.reserve((size_t)Tvec);

            for (uint64_t k = 0; k < Tvec; ++k) {
                const uint64_t t = tree_indices[(size_t)k];
                const auto preds_orig = model.get_predictions(t, Xb);
                correct_orig.push_back(count_correct(preds_orig, yb));
            }

            const double wt_tree = 1.0 / ((double)n_bootstraps * (double)Tvec);

            for (int v = 0; v < V; ++v) {
                const std::vector<int>& cols = var_cols[(size_t)v];

                std::vector<double> delta_sum_by_tree((size_t)Tvec, 0.0);

                for (int s = 0; s < n_scrambles_per_var; ++s) {
                    std::vector<int> perm;
                    make_permutation(n, rng, perm);

                    scramble_block_inplace(Xb, cols, perm, saved_cols);

                    for (uint64_t k = 0; k < Tvec; ++k) {
                        const uint64_t t = tree_indices[(size_t)k];

                        const auto preds_scr = model.get_predictions(t, Xb);
                        const int correct_scr = count_correct(preds_scr, yb);

                        delta_sum_by_tree[(size_t)k] +=
                            (double)(correct_orig[(size_t)k] - correct_scr);
                    }

                    restore_block_inplace(Xb, cols, saved_cols);
                }

                for (uint64_t k = 0; k < Tvec; ++k) {
                    const double avg_delta_correct =
                        delta_sum_by_tree[(size_t)k] / (double)n_scrambles_per_var;

                    out.mean_sub_mr[v] += wt_tree * (avg_delta_correct / (double)n);
                    mass_by_delta[v][avg_delta_correct] += wt_tree;
                }
            }

            continue;
        }

        auto orig = model.get_all_predictions_packed_trie(Xb, budget_override);
        const uint64_t Tvec = (uint64_t)orig.size();

        if (Tvec == 0) continue;

        // weight per tree per bootstrap
        const double wt_tree = 1.0 / ((double)n_bootstraps * (double)Tvec); // we may return more trees than we use (within new budget), so Tvec here


        std::vector<int> correct_orig((size_t)Tvec, 0);
        for (uint64_t t = 0; t < Tvec; ++t) {
            correct_orig[(size_t)t] = count_correct_packed_multi(orig[(size_t)t].pred, y_bits, n_words, tail_mask);
        }
    
        // Consider this optimization: convert the bootstrap to column major once, and scramble the columns in column major (which is probably slightly slower), and then replace get_all_predictions_packed_trie to take in column major instead of taking in row and converting to column.
        // I think precomputing column major once instead of f times is better, even if it is not ideal for the scrambling.
        
        for (int v = 0; v < V; ++v) {
            const std::vector<int>& cols = var_cols[(size_t)v];

            // delta_sum_by_tree[t] accumulates correct_orig[t] - correct_scrambled[t]
            // over multiple independent scrambles of the same variable.
            std::vector<double> delta_sum_by_tree((size_t)Tvec, 0.0);

            for (int s = 0; s < n_scrambles_per_var; ++s) {
                std::vector<int> perm;
                make_permutation(n, rng, perm);

                scramble_block_inplace(Xb, cols, perm, saved_cols);

                auto scr = model.get_all_predictions_packed_trie(Xb, budget_override);
                const uint64_t Tuse = Tvec;

                for (uint64_t t = 0; t < Tuse; ++t) {
                    const int correct_scr = count_correct_packed_multi(
                        scr[(size_t)t].pred,
                        y_bits,
                        n_words,
                        tail_mask
                    );

                    delta_sum_by_tree[(size_t)t] +=
                        (double)(correct_orig[(size_t)t] - correct_scr);
                }

                restore_block_inplace(Xb, cols, saved_cols);
            }

            for (uint64_t t = 0; t < Tvec; ++t) {
                const double avg_delta_correct =
                    delta_sum_by_tree[(size_t)t] / (double)n_scrambles_per_var;

                out.mean_sub_mr[v] += wt_tree * (avg_delta_correct / (double)n);
                mass_by_delta[v][avg_delta_correct] += wt_tree;
            }
        }

        
    }

    // build weighted CDF for each feature from the mass map
    for (int v = 0; v < V; ++v) {
        std::vector<std::pair<double, double>> items;
        items.reserve(mass_by_delta[v].size());
        for (const auto& kv : mass_by_delta[v]) items.push_back(kv);

        std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        out.cdf_x[v].reserve(items.size());
        out.cdf_p[v].reserve(items.size());

        double cum = 0.0;
        for (const auto& kv : items) {
            const double delta = kv.first;
            const double w = kv.second;
            cum += w;
            out.cdf_x[v].push_back((double)delta / (double)n_full);
            out.cdf_p[v].push_back(cum);
        }
    }


    return out;
}