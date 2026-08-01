// pybind for our continuous rashomon set algorithms.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/praxis_continuous.cpp"
#include "cpp/rid.cpp"

namespace py = pybind11;

static std::vector<std::vector<double>>
numpy_double_2d_to_row_major(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    const std::string& name
) {
    py::buffer_info info = X.request();

    if (info.ndim != 2) {
        throw std::runtime_error(name + " must be 2D.");
    }

    const int n = static_cast<int>(info.shape[0]);
    const int p = static_cast<int>(info.shape[1]);

    auto *ptr = static_cast<double*>(info.ptr);

    std::vector<std::vector<double>> out(
        n,
        std::vector<double>(p)
    );

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            out[i][j] = ptr[(std::size_t)i * (std::size_t)p + (std::size_t)j];
        }
    }

    return out;
}

static std::vector<std::vector<uint8_t>>
numpy_uint8_2d_to_row_major(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
    const std::string& name
) {
    py::buffer_info info = X.request();

    if (info.ndim != 2) {
        throw std::runtime_error(name + " must be 2D.");
    }

    const int n = static_cast<int>(info.shape[0]);
    const int p = static_cast<int>(info.shape[1]);

    auto *ptr = static_cast<uint8_t*>(info.ptr);

    std::vector<std::vector<uint8_t>> out(
        n,
        std::vector<uint8_t>(p)
    );

    for (int i = 0; i < n; ++i) {
        if (p > 0) {
            std::memcpy(
                out[(std::size_t)i].data(),
                ptr + (std::size_t)i * (std::size_t)p,
                (std::size_t)p * sizeof(uint8_t)
            );
        }
    }

    return out;
}

static std::vector<int>
numpy_int_1d_to_vector(
    py::array_t<int, py::array::c_style | py::array::forcecast> y,
    const std::string& name
) {
    py::buffer_info info = y.request();

    if (info.ndim != 1) {
        throw std::runtime_error(name + " must be 1D.");
    }

    const int n = static_cast<int>(info.shape[0]);
    auto *ptr = static_cast<int*>(info.ptr);

    return std::vector<int>(ptr, ptr + n);
}

static int
find_closest_full_binary_column(
    const std::vector<std::vector<uint8_t>>& X_full,
    const std::vector<uint8_t>& active_col
) {
    const int n = (int)X_full.size();
    if (n == 0) {
        throw std::runtime_error("X_full has zero rows.");
    }

    const int d = (int)X_full[0].size();
    if ((int)active_col.size() != n) {
        throw std::runtime_error("active_col length does not match X_full rows.");
    }

    int best_idx = -1;
    int best_dist = std::numeric_limits<int>::max();

    for (int f = 0; f < d; ++f) {
        int dist = 0;

        for (int i = 0; i < n; ++i) {
            const uint8_t a = active_col[(std::size_t)i] ? 1 : 0;
            const uint8_t b = X_full[(std::size_t)i][(std::size_t)f] ? 1 : 0;
            dist += (a != b);
        }

        if (dist < best_dist) {
            best_dist = dist;
            best_idx = f;

            if (best_dist == 0) break;
        }
    }

    if (best_idx < 0) {
        throw std::runtime_error("Failed to snap active feature to full binarized feature.");
    }

    return best_idx;
}

static void
sort_unique_ints(std::vector<int>& xs) {
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
}

static PRAXIS::KeyMode
parse_key_mode(const std::string& key_mode_str) {
    std::string s = key_mode_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    std::replace(s.begin(), s.end(), '-', '_');
    std::replace(s.begin(), s.end(), ' ', '_');

    if (
        s == "exact" ||
        s == "bitvector" ||
        s == "bit_vector"
    ) {
        return PRAXIS::KeyMode::EXACT;
    }

    if (
        s == "literal" ||
        s == "lits" ||
        s == "lits_exact" ||
        s == "itemset"
    ) {
        return PRAXIS::KeyMode::LITS_EXACT;
    }

    if (
        s == "128" ||
        s == "hash128" ||
        s == "hash_128" ||
        s == "fingerprint128" ||
        s == "fingerprint_128" ||
        s == "128bit" ||
        s == "128_bit" ||
        s == "hash128bit" ||
        s == "hash_128_bit"
    ) {
        return PRAXIS::KeyMode::HASH128;
    }

    return PRAXIS::KeyMode::HASH64;
}

static int
parse_greedy_continuous_mode(const std::string& mode_str) {
    if (
        mode_str == "numerical" ||
        mode_str == "numeric" ||
        mode_str == "sorted"
    ) {
        return 1;
    }

    if (
        mode_str == "binary" ||
        mode_str == "binarized" ||
        mode_str == "threshold" ||
        mode_str == "thresholds"
    ) {
        return 0;
    }

    throw std::runtime_error(
        "greedy_continuous_mode must be 'binary' or 'numerical'."
    );
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "PRAXIS C++ core bindings";

    py::class_<PRAXIS>(m, "PRAXIS")
        .def(py::init<>())

        .def(
            "fit",
            [](PRAXIS &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::array_t<int,     py::array::c_style | py::array::forcecast> y,
               double lambda_reg,
               int depth_budget,
               double rashomon_mult,
               double multiplicative_slack,
               std::string key_mode_str,
               bool trie_cache_enabled,
               int lookahead_k,
               int root_budget,
               bool use_multipass,
               bool rule_list_mode,
               int oracle_style,
               bool majority_leaf_only,
               bool cache_cheap_subproblems,
               int greedy_split_mode,
               std::string greedy_continuous_mode,
               bool proxy_caching,
               std::vector<int> allowed_proxy_features,
               bool restrict_proxy_in_lickety,
               bool restrict_proxy_in_depthd_exact,
               bool restrict_proxy_in_greedy,
               bool rashomon_mode,
               std::vector<int> continuous_starts,
               bool stronger_rollout
            ) {

                py::buffer_info xinfo = X.request();
                py::buffer_info yinfo = y.request();

                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                int n_samples  = static_cast<int>(xinfo.shape[0]);
                int n_features = static_cast<int>(xinfo.shape[1]);

                auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);
                auto *y_ptr = static_cast<int*>(yinfo.ptr);

                std::vector<std::vector<bool>> X_col_major(
                    n_features,
                    std::vector<bool>(n_samples)
                );

                for (int f = 0; f < n_features; ++f) {
                    for (int i = 0; i < n_samples; ++i) {
                        uint8_t v = x_ptr[i * n_features + f];
                        X_col_major[f][i] = (v != 0);
                    }
                }

                std::vector<int> y_vec(y_ptr, y_ptr + n_samples);

                // PRAXIS::KeyMode km;
                // if (key_mode_str == "exact" || key_mode_str == "bitvector") {
                //     km = PRAXIS::KeyMode::EXACT;
                // } else if (key_mode_str == "literal" || key_mode_str == "lits" || key_mode_str == "lits_exact" || key_mode_str == "itemset") {
                //     km = PRAXIS::KeyMode::LITS_EXACT;
                // } else {
                //     km = PRAXIS::KeyMode::HASH64;
                // }
                PRAXIS::KeyMode km = parse_key_mode(key_mode_str);


                self.set_key_mode(km);
                self.set_trie_cache_enabled(trie_cache_enabled);
                self.set_multiplicative_slack(multiplicative_slack);
                self.set_use_multipass(use_multipass);
                self.set_rule_list_mode(rule_list_mode);
                self.set_cache_cheap_subproblems(cache_cheap_subproblems);
                self.set_greedy_split_mode(greedy_split_mode);
                self.set_greedy_continuous_mode(parse_greedy_continuous_mode(greedy_continuous_mode));
                self.set_majority_leaf_only(majority_leaf_only);
                self.set_proxy_caching_enabled(proxy_caching);
                self.set_stronger_rollout(stronger_rollout);

                self.fit(
                    X_col_major,
                    y_vec,
                    lambda_reg,
                    depth_budget,
                    rashomon_mult,
                    lookahead_k,
                    root_budget,
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
                    majority_leaf_only,
                    cache_cheap_subproblems,
                    proxy_caching,
                    allowed_proxy_features,
                    restrict_proxy_in_lickety,
                    restrict_proxy_in_depthd_exact,
                    restrict_proxy_in_greedy,
                    rashomon_mode,
                    continuous_starts,
                    stronger_rollout
                );
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("rashomon_mult") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = false,
            py::arg("lookahead_k") = 1,
            py::arg("root_budget") = -1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("greedy_continuous_mode") = "binary",
            py::arg("proxy_caching") = true,
            py::arg("allowed_proxy_features") = std::vector<int>{},
            py::arg("restrict_proxy_in_lickety") = false,
            py::arg("restrict_proxy_in_depthd_exact") = false,
            py::arg("restrict_proxy_in_greedy") = false,
            py::arg("rashomon_mode") = true,
            py::arg("continuous_starts") = std::vector<int>{},
            py::arg("stronger_rollout") = false
        )

        .def(
            "fit_then_extend",
            [](PRAXIS &self,
               py::array_t<
                   uint8_t,
                   py::array::c_style | py::array::forcecast
               > X,
               py::array_t<
                   int,
                   py::array::c_style | py::array::forcecast
               > y,
               double lambda_reg,
               int depth_budget,
               double first_rashomon_mult,
               double second_rashomon_mult,
               double multiplier_step_size,
               double multiplicative_slack,
               std::string key_mode_str,
               bool trie_cache_enabled,
               int lookahead_k,
               bool use_multipass,
               bool rule_list_mode,
               int oracle_style,
               bool majority_leaf_only,
               bool cache_cheap_subproblems,
               int greedy_split_mode,
               std::string greedy_continuous_mode,
               bool proxy_caching,
               std::vector<int> allowed_proxy_features,
               bool restrict_proxy_in_lickety,
               bool restrict_proxy_in_depthd_exact,
               bool restrict_proxy_in_greedy,
               std::vector<int> continuous_starts,
               bool stronger_rollout,
               double runtime_limit_seconds,
               double memory_limit_mb
            ) {
                py::buffer_info xinfo = X.request();
                py::buffer_info yinfo = y.request();

                if (xinfo.ndim != 2) {
                    throw std::runtime_error(
                        "X must be 2D (n_samples x n_features)."
                    );
                }

                if (yinfo.ndim != 1) {
                    throw std::runtime_error(
                        "y must be 1D."
                    );
                }

                const int n_samples =
                    static_cast<int>(xinfo.shape[0]);

                const int n_features =
                    static_cast<int>(xinfo.shape[1]);

                if (
                    static_cast<int>(yinfo.shape[0]) !=
                    n_samples
                ) {
                    throw std::runtime_error(
                        "y length must match X rows."
                    );
                }

                auto *x_ptr =
                    static_cast<uint8_t*>(xinfo.ptr);

                auto *y_ptr =
                    static_cast<int*>(yinfo.ptr);

                std::vector<std::vector<bool>>
                    X_col_major(
                        static_cast<std::size_t>(
                            n_features
                        ),
                        std::vector<bool>(
                            static_cast<std::size_t>(
                                n_samples
                            )
                        )
                    );

                for (int f = 0; f < n_features; ++f) {
                    for (int i = 0; i < n_samples; ++i) {
                        const uint8_t v =
                            x_ptr[
                                static_cast<std::size_t>(i) *
                                static_cast<std::size_t>(
                                    n_features
                                ) +
                                static_cast<std::size_t>(f)
                            ];

                        X_col_major[
                            static_cast<std::size_t>(f)
                        ][
                            static_cast<std::size_t>(i)
                        ] = (v != 0);
                    }
                }

                std::vector<int> y_vec(
                    y_ptr,
                    y_ptr + n_samples
                );

                self.set_key_mode(
                    parse_key_mode(key_mode_str)
                );

                self.set_trie_cache_enabled(
                    trie_cache_enabled
                );

                self.set_multiplicative_slack(
                    multiplicative_slack
                );

                self.set_use_multipass(
                    use_multipass
                );

                self.set_rule_list_mode(
                    rule_list_mode
                );

                self.set_cache_cheap_subproblems(
                    cache_cheap_subproblems
                );

                self.set_greedy_split_mode(
                    greedy_split_mode
                );

                self.set_greedy_continuous_mode(
                    parse_greedy_continuous_mode(
                        greedy_continuous_mode
                    )
                );

                self.set_majority_leaf_only(
                    majority_leaf_only
                );

                self.set_proxy_caching_enabled(
                    proxy_caching
                );

                self.set_stronger_rollout(
                    stronger_rollout
                );

                self.fit_then_extend(
                    X_col_major,
                    y_vec,
                    lambda_reg,
                    static_cast<int8_t>(depth_budget),
                    first_rashomon_mult,
                    second_rashomon_mult,
                    multiplier_step_size,
                    static_cast<int8_t>(lookahead_k),
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
                    majority_leaf_only,
                    cache_cheap_subproblems,
                    proxy_caching,
                    allowed_proxy_features,
                    restrict_proxy_in_lickety,
                    restrict_proxy_in_depthd_exact,
                    restrict_proxy_in_greedy,
                    continuous_starts,
                    stronger_rollout,
                    runtime_limit_seconds,
                    memory_limit_mb
                );
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("first_rashomon_mult") = 0.01,
            py::arg("second_rashomon_mult") = 0.03,
            py::arg("multiplier_step_size") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = true,
            py::arg("lookahead_k") = 1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("greedy_continuous_mode") =
                "binary",
            py::arg("proxy_caching") = true,
            py::arg("allowed_proxy_features") =
                std::vector<int>{},
            py::arg("restrict_proxy_in_lickety") =
                false,
            py::arg("restrict_proxy_in_depthd_exact") =
                false,
            py::arg("restrict_proxy_in_greedy") =
                false,
            py::arg("continuous_starts") =
                std::vector<int>{},
            py::arg("stronger_rollout") = false,
            py::arg("runtime_limit_seconds") = -1.0,
            py::arg("memory_limit_mb") = -1.0
        )

        .def(
            "fit_anytime",
            [](PRAXIS &self,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
            py::array_t<int,     py::array::c_style | py::array::forcecast> y,
            double lambda_reg,
            int depth_budget,
            double rashomon_mult,
            double second_rashomon_mult,
            double multiplier_step_size,
            double multiplicative_slack,
            std::string key_mode_str,
            bool trie_cache_enabled,
            int lookahead_k,
            bool use_multipass,
            bool rule_list_mode,
            int oracle_style,
            bool majority_leaf_only,
            bool cache_cheap_subproblems,
            int greedy_split_mode,
            std::string greedy_continuous_mode,
            bool proxy_caching,
            std::vector<int> proxy_threshold_features,
            std::vector<int> initial_active_threshold_features,
            int refinement_width,
            int max_refinement_rounds,
            int proxy_refinement_mode,
            bool continuous_proxy_in_lickety,
            bool continuous_proxy_in_depthd_exact,
            bool continuous_proxy_in_greedy,
            std::vector<int> continuous_starts,
            double runtime_limit_seconds,
            double memory_limit_mb
            ) {
                py::buffer_info xinfo = X.request();
                py::buffer_info yinfo = y.request();

                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                if (yinfo.ndim != 1) {
                    throw std::runtime_error("y must be 1D.");
                }

                int n_samples  = static_cast<int>(xinfo.shape[0]);
                int n_features = static_cast<int>(xinfo.shape[1]);

                if (static_cast<int>(yinfo.shape[0]) != n_samples) {
                    throw std::runtime_error("y length must match X rows.");
                }

                auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);
                auto *y_ptr = static_cast<int*>(yinfo.ptr);

                std::vector<std::vector<bool>> X_col_major(
                    n_features,
                    std::vector<bool>(n_samples)
                );

                for (int f = 0; f < n_features; ++f) {
                    for (int i = 0; i < n_samples; ++i) {
                        uint8_t v = x_ptr[
                            (std::size_t)i * (std::size_t)n_features +
                            (std::size_t)f
                        ];
                        X_col_major[(std::size_t)f][(std::size_t)i] = (v != 0);
                    }
                }

                std::vector<int> y_vec(y_ptr, y_ptr + n_samples);

                PRAXIS::KeyMode km = parse_key_mode(key_mode_str);

                self.set_key_mode(km);
                self.set_trie_cache_enabled(trie_cache_enabled);
                self.set_multiplicative_slack(multiplicative_slack);
                self.set_use_multipass(use_multipass);
                self.set_rule_list_mode(rule_list_mode);
                self.set_cache_cheap_subproblems(cache_cheap_subproblems);
                self.set_greedy_split_mode(greedy_split_mode);
                self.set_greedy_continuous_mode(
                    parse_greedy_continuous_mode(greedy_continuous_mode)
                );
                self.set_majority_leaf_only(majority_leaf_only);
                self.set_proxy_caching_enabled(proxy_caching);

                self.fit_anytime(
                    X_col_major,
                    y_vec,
                    lambda_reg,
                    static_cast<int8_t>(depth_budget),
                    rashomon_mult,
                    second_rashomon_mult,
                    multiplier_step_size,
                    static_cast<int8_t>(lookahead_k),
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
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
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("rashomon_mult") = 0.01,
            py::arg("second_rashomon_mult") = 0.01,
            py::arg("multiplier_step_size") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = true,
            py::arg("lookahead_k") = 1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("greedy_continuous_mode") = "binary",
            py::arg("proxy_caching") = true,
            py::arg("proxy_threshold_features") = std::vector<int>{},
            py::arg("initial_active_threshold_features") = std::vector<int>{},
            py::arg("refinement_width") = 1,
            py::arg("max_refinement_rounds") = -1,
            py::arg("proxy_refinement_mode") = 0,
            py::arg("continuous_proxy_in_lickety") = false,
            py::arg("continuous_proxy_in_depthd_exact") = false,
            py::arg("continuous_proxy_in_greedy") = false,
            py::arg("continuous_starts") = std::vector<int>{},
            py::arg("runtime_limit_seconds") = -1.0,
            py::arg("memory_limit_mb") = -1.0
        )

        .def(
            "prepare_continuous_data",
            [](PRAXIS &self,
            py::array_t<double, py::array::c_style | py::array::forcecast> X_num,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X_bin,
            py::array_t<int, py::array::c_style | py::array::forcecast> y,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
                X_initial_active,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
                X_proxy_active,
            int max_number_thresholds_per_feature
            ) {
                auto X_num_vec =
                    numpy_double_2d_to_row_major(X_num, "X_num");

                auto X_bin_vec =
                    numpy_uint8_2d_to_row_major(X_bin, "X_bin");

                auto y_vec =
                    numpy_int_1d_to_vector(y, "y");

                auto X_initial_active_vec =
                    numpy_uint8_2d_to_row_major(
                        X_initial_active,
                        "X_initial_active"
                    );

                auto X_proxy_active_vec =
                    numpy_uint8_2d_to_row_major(
                        X_proxy_active,
                        "X_proxy_active"
                    );

                self.prepare_continuous_data(
                    X_num_vec,
                    X_bin_vec,
                    y_vec,
                    X_initial_active_vec,
                    X_proxy_active_vec,
                    max_number_thresholds_per_feature
                );
            },
            py::arg("X_num"),
            py::arg("X_bin"),
            py::arg("y"),
            py::arg("X_initial_active"),
            py::arg("X_proxy_active"),
            py::arg("max_number_thresholds_per_feature") = -1
        )

        .def(
            "fit_prepared_anytime",
            [](PRAXIS &self,
            double lambda_reg,
            int depth_budget,
            double rashomon_mult,
            double second_rashomon_mult,
            double multiplier_step_size,
            double multiplicative_slack,
            std::string key_mode_str,
            bool trie_cache_enabled,
            int lookahead_k,
            bool use_multipass,
            bool rule_list_mode,
            int oracle_style,
            bool majority_leaf_only,
            bool cache_cheap_subproblems,
            int greedy_split_mode,
            std::string greedy_continuous_mode,
            bool proxy_caching,
            std::vector<int> proxy_threshold_features,
            std::vector<int> initial_active_threshold_features,
            int refinement_width,
            int max_refinement_rounds,
            int proxy_refinement_mode,
            bool continuous_proxy_in_lickety,
            bool continuous_proxy_in_depthd_exact,
            bool continuous_proxy_in_greedy,
            double runtime_limit_seconds,
            double memory_limit_mb
            ) {
                PRAXIS::KeyMode km = parse_key_mode(key_mode_str);

                self.set_key_mode(km);
                self.set_trie_cache_enabled(trie_cache_enabled);
                self.set_multiplicative_slack(multiplicative_slack);
                self.set_use_multipass(use_multipass);
                self.set_rule_list_mode(rule_list_mode);
                self.set_cache_cheap_subproblems(cache_cheap_subproblems);
                self.set_greedy_split_mode(greedy_split_mode);
                self.set_greedy_continuous_mode(
                    parse_greedy_continuous_mode(greedy_continuous_mode)
                );
                self.set_majority_leaf_only(majority_leaf_only);
                self.set_proxy_caching_enabled(proxy_caching);

                self.fit_prepared_anytime(
                    lambda_reg,
                    static_cast<int8_t>(depth_budget),
                    rashomon_mult,
                    second_rashomon_mult,
                    multiplier_step_size,
                    static_cast<int8_t>(lookahead_k),
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
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
                    runtime_limit_seconds,
                    memory_limit_mb
                );
            },
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("rashomon_mult") = 0.01,
            py::arg("second_rashomon_mult") = 0.01,
            py::arg("multiplier_step_size") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = true,
            py::arg("lookahead_k") = 1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("greedy_continuous_mode") = "binary",
            py::arg("proxy_caching") = true,
            py::arg("proxy_threshold_features") = std::vector<int>{},
            py::arg("initial_active_threshold_features") = std::vector<int>{},
            py::arg("refinement_width") = 1,
            py::arg("max_refinement_rounds") = -1,
            py::arg("proxy_refinement_mode") = 0,
            py::arg("continuous_proxy_in_lickety") = false,
            py::arg("continuous_proxy_in_depthd_exact") = false,
            py::arg("continuous_proxy_in_greedy") = false,
            py::arg("runtime_limit_seconds") = -1.0,
            py::arg("memory_limit_mb") = -1.0
        )

        .def(
            "fit_prepared",
            [](PRAXIS &self,
               double lambda_reg,
               int depth_budget,
               double rashomon_mult,
               double multiplicative_slack,
               std::string key_mode_str,
               bool trie_cache_enabled,
               int lookahead_k,
               int root_budget,
               bool use_multipass,
               bool rule_list_mode,
               int oracle_style,
               bool majority_leaf_only,
               bool cache_cheap_subproblems,
               int greedy_split_mode,
               std::string greedy_continuous_mode,
               bool proxy_caching,
               bool restrict_proxy_in_lickety,
               bool restrict_proxy_in_depthd_exact,
               bool restrict_proxy_in_greedy,
               bool rashomon_mode,
               bool stronger_rollout
            ) {
                PRAXIS::KeyMode km = parse_key_mode(key_mode_str);

                self.set_key_mode(km);
                self.set_trie_cache_enabled(trie_cache_enabled);
                self.set_multiplicative_slack(multiplicative_slack);
                self.set_use_multipass(use_multipass);
                self.set_rule_list_mode(rule_list_mode);
                self.set_cache_cheap_subproblems(cache_cheap_subproblems);
                self.set_greedy_split_mode(greedy_split_mode);
                self.set_greedy_continuous_mode(parse_greedy_continuous_mode(greedy_continuous_mode));
                self.set_majority_leaf_only(majority_leaf_only);
                self.set_proxy_caching_enabled(proxy_caching);
                self.set_stronger_rollout(stronger_rollout);

                self.fit_prepared(
                    lambda_reg,
                    static_cast<int8_t>(depth_budget),
                    rashomon_mult,
                    static_cast<int8_t>(lookahead_k),
                    root_budget,
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
                    majority_leaf_only,
                    cache_cheap_subproblems,
                    proxy_caching,
                    restrict_proxy_in_lickety,
                    restrict_proxy_in_depthd_exact,
                    restrict_proxy_in_greedy,
                    rashomon_mode,
                    stronger_rollout
                );
            },
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("rashomon_mult") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = false,
            py::arg("lookahead_k") = 1,
            py::arg("root_budget") = -1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("greedy_continuous_mode") = "binary",
            py::arg("proxy_caching") = true,
            py::arg("restrict_proxy_in_lickety") = false,
            py::arg("restrict_proxy_in_depthd_exact") = false,
            py::arg("restrict_proxy_in_greedy") = false,
            py::arg("rashomon_mode") = true,
            py::arg("stronger_rollout") = false
        )

        .def(
            "fit_prepared_then_extend",
            [](PRAXIS &self,
               double lambda_reg,
               int depth_budget,
               double first_rashomon_mult,
               double second_rashomon_mult,
               double multiplier_step_size,
               double multiplicative_slack,
               std::string key_mode_str,
               bool trie_cache_enabled,
               int lookahead_k,
               bool use_multipass,
               bool rule_list_mode,
               int oracle_style,
               bool majority_leaf_only,
               bool cache_cheap_subproblems,
               int greedy_split_mode,
               std::string greedy_continuous_mode,
               bool proxy_caching,
               bool restrict_proxy_in_lickety,
               bool restrict_proxy_in_depthd_exact,
               bool restrict_proxy_in_greedy,
               bool stronger_rollout,
               double runtime_limit_seconds,
               double memory_limit_mb
            ) {
                self.set_key_mode(
                    parse_key_mode(key_mode_str)
                );

                self.set_trie_cache_enabled(
                    trie_cache_enabled
                );

                self.set_multiplicative_slack(
                    multiplicative_slack
                );

                self.set_use_multipass(
                    use_multipass
                );

                self.set_rule_list_mode(
                    rule_list_mode
                );

                self.set_cache_cheap_subproblems(
                    cache_cheap_subproblems
                );

                self.set_greedy_split_mode(
                    greedy_split_mode
                );

                self.set_greedy_continuous_mode(
                    parse_greedy_continuous_mode(
                        greedy_continuous_mode
                    )
                );

                self.set_majority_leaf_only(
                    majority_leaf_only
                );

                self.set_proxy_caching_enabled(
                    proxy_caching
                );

                self.set_stronger_rollout(
                    stronger_rollout
                );

                self.fit_prepared_then_extend(
                    lambda_reg,
                    static_cast<int8_t>(depth_budget),
                    first_rashomon_mult,
                    second_rashomon_mult,
                    multiplier_step_size,
                    static_cast<int8_t>(lookahead_k),
                    use_multipass,
                    rule_list_mode,
                    oracle_style,
                    majority_leaf_only,
                    cache_cheap_subproblems,
                    proxy_caching,
                    restrict_proxy_in_lickety,
                    restrict_proxy_in_depthd_exact,
                    restrict_proxy_in_greedy,
                    stronger_rollout,
                    runtime_limit_seconds,
                    memory_limit_mb
                );
            },
            py::arg("lambda_reg") = 0.01,
            py::arg("depth_budget") = 5,
            py::arg("first_rashomon_mult") = 0.01,
            py::arg("second_rashomon_mult") = 0.03,
            py::arg("multiplier_step_size") = 0.01,
            py::arg("multiplicative_slack") = 0.0,
            py::arg("key_mode") = "hash",
            py::arg("trie_cache_enabled") = true,
            py::arg("lookahead_k") = 1,
            py::arg("use_multipass") = true,
            py::arg("rule_list_mode") = false,
            py::arg("oracle_style") = 0,
            py::arg("majority_leaf_only") = false,
            py::arg("cache_cheap_subproblems") = false,
            py::arg("greedy_split_mode") = 1,
            py::arg("greedy_continuous_mode") =
                "binary",
            py::arg("proxy_caching") = true,
            py::arg("restrict_proxy_in_lickety") =
                false,
            py::arg("restrict_proxy_in_depthd_exact") =
                false,
            py::arg("restrict_proxy_in_greedy") =
                false,
            py::arg("stronger_rollout") = false,
            py::arg("runtime_limit_seconds") = -1.0,
            py::arg("memory_limit_mb") = -1.0
        )

        .def("count_trees",
             [](PRAXIS &self) {
                 return self.result ? self.result->count_trees() : 0ULL;
             })

        .def("count_distinct_or_nodes",
             [](const PRAXIS &self) {
                 return self.count_distinct_or_nodes();
             })

        .def("count_graph_features",
             [](const PRAXIS &self) {
                 return self.count_graph_features();
             })

        .def("get_min_objective",
             [](PRAXIS &self) {
                 return self.result
                        ? self.result->min_objective
                        : std::numeric_limits<int>::max();
             })

        .def("get_root_histogram",
             [](PRAXIS &self) {
                 if (!self.result) {
                     return std::vector<std::pair<int, std::uint64_t>>{};
                 }
                 self.result->ensure_hist_built();
                 const auto &hist = self.result->hist;
                 std::vector<std::pair<int, std::uint64_t>> out;
                 out.reserve(hist.size());
                 for (const auto &e : hist) {
                     out.emplace_back(e.obj, e.cnt);
                 }
                 return out;
             })

        .def(
            "get_predictions",
            [](const PRAXIS &self,
               std::uint64_t tree_index,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X) {
                py::buffer_info xinfo = X.request();
                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                int n_samples  = static_cast<int>(xinfo.shape[0]);
                int n_features = static_cast<int>(xinfo.shape[1]);
                auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);

                std::vector<std::vector<uint8_t>> X_row_major(
                    n_samples, std::vector<uint8_t>(n_features));
                for (int i = 0; i < n_samples; ++i) {
                    for (int f = 0; f < n_features; ++f) {
                        X_row_major[i][f] = x_ptr[i * n_features + f];
                    }
                }

                auto preds = self.get_predictions(tree_index, X_row_major);

                py::array_t<uint8_t> out(n_samples);
                auto out_info = out.request();
                auto *out_ptr = static_cast<uint8_t*>(out_info.ptr);
                std::memcpy(
                    out_ptr, preds.data(),
                    static_cast<std::size_t>(n_samples) * sizeof(uint8_t));
                return out;
            },
            py::arg("tree_index"),
            py::arg("X")
        )

        .def(
            "set_greedy_continuous_mode",
            [](PRAXIS& self, std::string mode) {
                self.set_greedy_continuous_mode(
                    parse_greedy_continuous_mode(mode)
                );
            },
            py::arg("mode")
        )

        .def(
            "get_greedy_continuous_mode",
            [](const PRAXIS& self) {
                return self.get_greedy_continuous_mode();
            }
        )

        .def(
            "get_all_predictions",
            [](const PRAXIS &self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               bool stack) {
                py::buffer_info xinfo = X.request();
                if (xinfo.ndim != 2) {
                    throw std::runtime_error("X must be 2D (n_samples x n_features)");
                }

                int n_samples  = static_cast<int>(xinfo.shape[0]);
                int n_features = static_cast<int>(xinfo.shape[1]);
                auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);

                std::vector<std::vector<uint8_t>> X_row_major(
                    n_samples, std::vector<uint8_t>(n_features));
                for (int i = 0; i < n_samples; ++i) {
                    for (int f = 0; f < n_features; ++f) {
                        X_row_major[i][f] = x_ptr[i * n_features + f];
                    }
                }

                auto all_preds = self.get_all_predictions(X_row_major);
                std::uint64_t total = all_preds.size();

                if (!stack) {
                    py::list lst;
                    for (std::uint64_t t = 0; t < total; ++t) {
                        py::array_t<uint8_t> arr(n_samples);
                        auto info = arr.request();
                        auto *ptr = static_cast<uint8_t*>(info.ptr);
                        std::memcpy(
                            ptr,
                            all_preds[t].data(),
                            static_cast<std::size_t>(n_samples) * sizeof(uint8_t));
                        lst.append(arr);
                    }
                    return py::object(lst);
                } else {
                    py::array_t<uint8_t> out(
                        {static_cast<py::ssize_t>(total),
                         static_cast<py::ssize_t>(n_samples)});
                    auto out_info = out.request();
                    auto *out_ptr = static_cast<uint8_t*>(out_info.ptr);
                    for (std::uint64_t t = 0; t < total; ++t) {
                        std::memcpy(
                            out_ptr + t * n_samples,
                            all_preds[t].data(),
                            static_cast<std::size_t>(n_samples) * sizeof(uint8_t));
                    }
                    return py::object(out);
                }
            },
            py::arg("X"),
            py::arg("stack") = false
        )

        .def(
            "get_tree_objective",
            [](const PRAXIS &self, std::uint64_t tree_index) {
                auto obj_pair = self.get_ith_tree_objective(tree_index);
                // obj_pair.first  = unnormalized objective (int)
                // obj_pair.second = normalized objective (double)
                return py::make_tuple(obj_pair.first, obj_pair.second);
            },
            py::arg("tree_index")
        )

        .def(
            "get_tree_paths",
            [](const PRAXIS &self, std::uint64_t tree_index) {
                auto result = self.get_tree_paths(tree_index);
                const auto &paths = result.first;
                const auto &preds = result.second;

                py::list py_paths;
                for (const auto &p : paths) {
                    py::list py_path;
                    for (int v : p) {
                        py_path.append(v);
                    }
                    py_paths.append(py_path);
                }

                py::array_t<int> py_preds(preds.size());
                auto info = py_preds.request();
                auto *ptr = static_cast<int*>(info.ptr);
                for (std::size_t i = 0; i < preds.size(); ++i) {
                    ptr[i] = preds[i];
                }

                return py::make_tuple(py_paths, py_preds);
            },
            py::arg("tree_index")
        )

                .def(
            "root_lickety_objective_lookahead1",
            [](PRAXIS &self, int depth_budget) {
                return self.root_lickety_objective_lookahead1(depth_budget);
            },
            py::arg("depth_budget")
        )

        .def(
            "reachable_prediction_mask_for_training_sample",
            [](const PRAXIS &self, int sample_idx) {
                return self.reachable_prediction_mask_for_training_sample(sample_idx);
            },
            py::arg("sample_idx")
        )

        .def(
            "training_sample_has_multiple_reachable_predictions",
            [](const PRAXIS &self, int sample_idx) {
                return self.training_sample_has_multiple_reachable_predictions(sample_idx);
            },
            py::arg("sample_idx")
        )

        .def(
            "training_samples_with_multiple_reachable_predictions",
            [](const PRAXIS &self) {
                return self.training_samples_with_multiple_reachable_predictions();
            }
        );

    m.def(
        "rid_subtractive_model_reliance",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
        py::array_t<int,     py::array::c_style | py::array::forcecast> y,
        int n_boot,
        double lambda_reg,
        int depth_budget,
        double rashomon_mult,
        int lookahead_k,
        std::uint64_t seed,
        bool memory_efficient,
        py::object binning_map_obj ) {
            py::buffer_info xinfo = X.request();
            py::buffer_info yinfo = y.request();

            if (xinfo.ndim != 2) throw std::runtime_error("X must be 2D");
            if (yinfo.ndim != 1) throw std::runtime_error("y must be 1D");

            int n_samples  = (int)xinfo.shape[0];
            int n_features = (int)xinfo.shape[1];
            if ((int)yinfo.shape[0] != n_samples) throw std::runtime_error("y must match X rows");

            auto *x_ptr = static_cast<uint8_t*>(xinfo.ptr);
            auto *y_ptr = static_cast<int*>(yinfo.ptr);

            // build row-major X
            std::vector<std::vector<uint8_t>> X_row_major(n_samples, std::vector<uint8_t>(n_features));
            for (int i = 0; i < n_samples; ++i) {
                std::memcpy(X_row_major[i].data(),
                            x_ptr + (std::size_t)i * (std::size_t)n_features,
                            (std::size_t)n_features * sizeof(uint8_t));
            }

            std::vector<int> y_vec(y_ptr, y_ptr + n_samples);   
            
            // binning map stuff
            int d = n_features;
            std::vector<std::vector<int>> groups;
            if (binning_map_obj.is_none()) {
                groups.resize(d);
                for (int j = 0; j < d; ++j) groups[j] = {j};
            } else {
                py::dict bm = binning_map_obj.cast<py::dict>();

                std::vector<int> keys;
                for (auto item : bm) keys.push_back(py::cast<int>(item.first));
                std::sort(keys.begin(), keys.end());

                groups.reserve(keys.size());
                for (int k : keys) {
                    py::list lst = bm[py::int_(k)].cast<py::list>();
                    std::vector<int> cols;
                    for (auto h : lst) cols.push_back(py::cast<int>(h));
                    groups.push_back(std::move(cols));
                }
            }


            RIDResult r = compute_rid_subtractive_mr_bootstrap(
                X_row_major,
                y_vec,
                n_boot,
                lambda_reg,
                depth_budget,
                rashomon_mult,
                lookahead_k,
                seed,
                memory_efficient,
                groups
            );

            py::dict out;
            out["mean_sub_mr"] = r.mean_sub_mr; // vector<double>
            out["cdf_x"] = r.cdf_x; // vector<vector<double>>
            out["cdf_p"] = r.cdf_p; // vector<vector<double>>
            return out;
        },
        py::arg("X"),
        py::arg("y"),
        py::arg("n_boot") = 10,
        py::arg("lambda_reg") = 0.01,
        py::arg("depth_budget") = 5,
        py::arg("rashomon_mult") = 0.05,
        py::arg("lookahead_k") = 1,
        py::arg("seed") = 0,
        py::arg("memory_efficient") = false,
        py::arg("binning_map") = py::none()
    );

    m.def(
        "rid_subtractive_model_reliance_continuous",
        [](
            py::array_t<
                double,
                py::array::c_style | py::array::forcecast
            > X_num,
            py::array_t<
                uint8_t,
                py::array::c_style | py::array::forcecast
            > X_bin,
            py::array_t<
                int,
                py::array::c_style | py::array::forcecast
            > y,
            py::array_t<
                uint8_t,
                py::array::c_style | py::array::forcecast
            > X_proxy_active,
            py::array_t<
                uint8_t,
                py::array::c_style | py::array::forcecast
            > X_initial_active,
            int n_boot,
            double lambda_reg,
            int depth_budget,
            double rashomon_mult,
            double second_rashomon_mult,
            double multiplier_step_size,
            int lookahead_k,
            std::uint64_t seed,
            bool memory_efficient,
            bool use_anytime_fit,
            int refinement_width,
            int max_refinement_rounds,
            int proxy_refinement_mode,
            bool continuous_proxy_in_lickety,
            bool continuous_proxy_in_depthd_exact,
            bool continuous_proxy_in_greedy,
            bool use_multipass,
            bool rule_list_mode,
            int proxy_style,
            bool majority_leaf_only,
            bool cache_cheap_subproblems,
            bool proxy_caching,
            double runtime_limit_seconds,
            double memory_limit_mb
        ) {
            py::buffer_info num_info = X_num.request();
            py::buffer_info bin_info = X_bin.request();
            py::buffer_info yinfo = y.request();

            py::buffer_info proxy_active_info =
                X_proxy_active.request();

            py::buffer_info initial_active_info =
                X_initial_active.request();

            if (num_info.ndim != 2) {
                throw std::runtime_error("X_num must be 2D.");
            }

            if (bin_info.ndim != 2) {
                throw std::runtime_error("X_bin must be 2D.");
            }

            if (yinfo.ndim != 1) {
                throw std::runtime_error("y must be 1D.");
            }

            if (proxy_active_info.ndim != 2) {
                throw std::runtime_error(
                    "X_proxy_active must be 2D."
                );
            }

            if (initial_active_info.ndim != 2) {
                throw std::runtime_error(
                    "X_initial_active must be 2D."
                );
            }

            const int n_num =
                static_cast<int>(num_info.shape[0]);

            const int p_num =
                static_cast<int>(num_info.shape[1]);

            const int n_bin =
                static_cast<int>(bin_info.shape[0]);

            const int p_bin =
                static_cast<int>(bin_info.shape[1]);

            const int n_proxy_active =
                static_cast<int>(proxy_active_info.shape[0]);

            const int p_proxy_active =
                static_cast<int>(proxy_active_info.shape[1]);

            const int n_initial_active =
                static_cast<int>(initial_active_info.shape[0]);

            const int p_initial_active =
                static_cast<int>(initial_active_info.shape[1]);

            const int n_y =
                static_cast<int>(yinfo.shape[0]);

            if (n_num != n_bin) {
                throw std::runtime_error(
                    "X_num and X_bin must have the same number "
                    "of rows."
                );
            }

            if (n_proxy_active != n_num) {
                throw std::runtime_error(
                    "X_proxy_active and X_num must have the "
                    "same number of rows."
                );
            }

            if (n_initial_active != n_num) {
                throw std::runtime_error(
                    "X_initial_active and X_num must have the "
                    "same number of rows."
                );
            }

            if (n_y != n_num) {
                throw std::runtime_error(
                    "y length must match X_num/X_bin rows."
                );
            }

            const int n = n_num;

            auto* num_ptr =
                static_cast<double*>(num_info.ptr);

            auto* bin_ptr =
                static_cast<uint8_t*>(bin_info.ptr);

            auto* y_ptr =
                static_cast<int*>(yinfo.ptr);

            auto* proxy_active_ptr =
                static_cast<uint8_t*>(proxy_active_info.ptr);

            auto* initial_active_ptr =
                static_cast<uint8_t*>(initial_active_info.ptr);

            std::vector<int> y_vec(
                y_ptr,
                y_ptr + n
            );

            std::vector<std::vector<uint8_t>> X_full(
                static_cast<std::size_t>(n),
                std::vector<uint8_t>{}
            );

            for (int i = 0; i < n; ++i) {
                X_full[static_cast<std::size_t>(i)].reserve(
                    static_cast<std::size_t>(
                        p_bin + p_num * 8
                    )
                );
            }

            std::vector<std::vector<int>> groups;
            groups.reserve(
                static_cast<std::size_t>(
                    p_bin + p_num
                )
            );

            std::vector<int> continuous_starts;

            // ordinary binary features
            for (int j = 0; j < p_bin; ++j) {
                const int col_idx =
                    static_cast<int>(X_full[0].size());

                groups.push_back(
                    std::vector<int>{col_idx}
                );

                for (int i = 0; i < n; ++i) {
                    const uint8_t value =
                        bin_ptr[
                            static_cast<std::size_t>(i) *
                            static_cast<std::size_t>(p_bin) +
                            static_cast<std::size_t>(j)
                        ];

                    X_full[
                        static_cast<std::size_t>(i)
                    ].push_back(value ? 1 : 0);
                }
            }

            // fully threshold-binarize numerical features
            for (int j = 0; j < p_num; ++j) {
                std::vector<double> vals;
                vals.reserve(
                    static_cast<std::size_t>(n)
                );

                for (int i = 0; i < n; ++i) {
                    vals.push_back(
                        num_ptr[
                            static_cast<std::size_t>(i) *
                            static_cast<std::size_t>(p_num) +
                            static_cast<std::size_t>(j)
                        ]
                    );
                }

                std::sort(
                    vals.begin(),
                    vals.end()
                );

                vals.erase(
                    std::unique(
                        vals.begin(),
                        vals.end()
                    ),
                    vals.end()
                );

                // constant numerical feature has no threshold
                if (vals.size() <= 1) {
                    continue;
                }

                const int group_start =
                    static_cast<int>(X_full[0].size());

                continuous_starts.push_back(group_start);

                std::vector<int> this_group;
                this_group.reserve(vals.size() - 1);

                // threshold columns are x <= unique value,
                // excluding the maximum unique value.
                for (
                    std::size_t q = 0;
                    q + 1 < vals.size();
                    ++q
                ) {
                    const double threshold = vals[q];

                    const int col_idx =
                        static_cast<int>(X_full[0].size());

                    this_group.push_back(col_idx);

                    for (int i = 0; i < n; ++i) {
                        const double xij =
                            num_ptr[
                                static_cast<std::size_t>(i) *
                                static_cast<std::size_t>(p_num) +
                                static_cast<std::size_t>(j)
                            ];

                        X_full[
                            static_cast<std::size_t>(i)
                        ].push_back(
                            xij <= threshold ? 1 : 0
                        );
                    }
                }

                groups.push_back(
                    std::move(this_group)
                );
            }

            if (
                X_full.empty() ||
                X_full[0].empty()
            ) {
                throw std::runtime_error(
                    "Continuous RID produced zero binary "
                    "features."
                );
            }

            // snap proxy columns separately.
            std::vector<int> proxy_threshold_features;
            proxy_threshold_features.reserve(
                static_cast<std::size_t>(
                    p_proxy_active
                )
            );

            for (int a = 0; a < p_proxy_active; ++a) {
                std::vector<uint8_t> active_col(
                    static_cast<std::size_t>(n),
                    0
                );

                for (int i = 0; i < n; ++i) {
                    active_col[
                        static_cast<std::size_t>(i)
                    ] =
                        proxy_active_ptr[
                            static_cast<std::size_t>(i) *
                            static_cast<std::size_t>(
                                p_proxy_active
                            ) +
                            static_cast<std::size_t>(a)
                        ] ? 1 : 0;
                }

                proxy_threshold_features.push_back(
                    find_closest_full_binary_column(
                        X_full,
                        active_col
                    )
                );
            }

            sort_unique_ints(
                proxy_threshold_features
            );

            // snap initial enumeration columns
            std::vector<int>
                initial_active_threshold_features;

            initial_active_threshold_features.reserve(
                static_cast<std::size_t>(
                    p_initial_active
                )
            );

            for (int a = 0; a < p_initial_active; ++a) {
                std::vector<uint8_t> active_col(
                    static_cast<std::size_t>(n),
                    0
                );

                for (int i = 0; i < n; ++i) {
                    active_col[
                        static_cast<std::size_t>(i)
                    ] =
                        initial_active_ptr[
                            static_cast<std::size_t>(i) *
                            static_cast<std::size_t>(
                                p_initial_active
                            ) +
                            static_cast<std::size_t>(a)
                        ] ? 1 : 0;
                }

                initial_active_threshold_features.push_back(
                    find_closest_full_binary_column(
                        X_full,
                        active_col
                    )
                );
            }

            sort_unique_ints(
                initial_active_threshold_features
            );

            const bool any_proxy_is_restricted =
                !continuous_proxy_in_lickety ||
                !continuous_proxy_in_depthd_exact ||
                !continuous_proxy_in_greedy;

            if (
                use_anytime_fit &&
                !continuous_starts.empty() &&
                any_proxy_is_restricted &&
                proxy_threshold_features.empty()
            ) {
                throw std::runtime_error(
                    "Anytime continuous RID requires nonempty X_proxy_active "
                    "when any proxy is restricted."
                );
            }

            RIDResult r =
                compute_rid_subtractive_mr_bootstrap(
                    X_full,
                    y_vec,
                    n_boot,
                    lambda_reg,
                    depth_budget,
                    rashomon_mult,
                    lookahead_k,
                    seed,
                    memory_efficient,
                    groups,
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
                    proxy_caching,
                    proxy_refinement_mode,
                    continuous_proxy_in_lickety,
                    continuous_proxy_in_depthd_exact,
                    continuous_proxy_in_greedy,
                    runtime_limit_seconds,
                    memory_limit_mb
                );

            py::dict out;
            out["mean_sub_mr"] = r.mean_sub_mr;
            out["cdf_x"] = r.cdf_x;
            out["cdf_p"] = r.cdf_p;

            out["proxy_threshold_features"] =
                proxy_threshold_features;

            out["initial_active_threshold_features"] =
                initial_active_threshold_features;

            out["continuous_starts"] =
                continuous_starts;

            out["binning_map"] =
                groups;

            return out;
        },
        py::arg("X_num"),
        py::arg("X_bin"),
        py::arg("y"),
        py::arg("X_proxy_active"),
        py::arg("X_initial_active"),
        py::arg("n_boot") = 10,
        py::arg("lambda_reg") = 0.01,
        py::arg("depth_budget") = 5,
        py::arg("rashomon_mult") = 0.05,
        py::arg("second_rashomon_mult") = -1.0,
        py::arg("multiplier_step_size") = 0.01,
        py::arg("lookahead_k") = 1,
        py::arg("seed") = 0,
        py::arg("memory_efficient") = false,
        py::arg("use_anytime_fit") = false,
        py::arg("refinement_width") = 1,
        py::arg("max_refinement_rounds") = -1,
        py::arg("proxy_refinement_mode") = 0,
        py::arg("continuous_proxy_in_lickety") = true,
        py::arg("continuous_proxy_in_depthd_exact") = true,
        py::arg("continuous_proxy_in_greedy") = true,
        py::arg("use_multipass") = true,
        py::arg("rule_list_mode") = false,
        py::arg("proxy_style") = 0,
        py::arg("majority_leaf_only") = false,
        py::arg("cache_cheap_subproblems") = false,
        py::arg("proxy_caching") = true,
        py::arg("runtime_limit_seconds") = -1.0,
        py::arg("memory_limit_mb") = -1.0
    );


}