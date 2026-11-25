    struct ProxySplitIndex {
        Objective objective;
        int feature;
    };
    std::unordered_map<KLA, ProxySplitIndex, KLA::Hash> proxy_split_indices_;
    uint64_t sparsification_index_hits_ = 0, sparsification_leaf_shortcuts_ = 0;
    std::vector<double> sparsification_probability_;
    uint64_t sparsification_seed_ = 0;
    bool sparsification_enabled_ = false;
    bool sparsification_index_enabled_ = false;
    std::unordered_map<K2, int, K2::Hash> sparsification_splits_;
    uint64_t sparsification_scan_count_ = 0;
    double sparsification_scan_seconds_ = 0;

    void clear_sparsification_state_() {
        if (!sparsification_splits_.empty()) sparsification_splits_.clear();
        if (!proxy_split_indices_.empty()) proxy_split_indices_.clear();
        sparsification_index_hits_ = 0;
        sparsification_leaf_shortcuts_ = 0;
        sparsification_scan_count_ = 0;
        sparsification_scan_seconds_ = 0;
    }

    void remember_proxy_split_(uint64_t key, int8_t depth, Objective objective, int feature, int tag) {
        const int remaining = static_cast<int>(depth) - proxy_delta_;
        if (!sparsification_index_enabled_ || remaining <= 0 ||
            remaining > static_cast<int>(sparsification_probability_.size()) ||
            sparsification_probability_[remaining - 1] <= 0 || feature < 0) return;
        const KLA state{key, depth, tag};
        const auto found = proxy_split_indices_.find(state);
        if (found == proxy_split_indices_.end()) proxy_split_indices_.emplace(state, ProxySplitIndex{objective, feature});
        else if (objective < found->second.objective ||
            (objective == found->second.objective && feature < found->second.feature))
            found->second = {objective, feature};
    }

    int proxy_split_index_tag_(int8_t depth, int8_t lookahead) const {
        const auto exact_tag = [&]() {
            if (depth == 1) {
                if (use_restricted_depthd_exact_proxy_())
                    return should_route_continuous_lickety_depth1_to_binary_greedy_() ? -1 : 0;
                if (should_add_continuous_greedy_split_to_binary_lickety_()) return -2;
            }
            return use_kla_cache() ? static_cast<int>(depth) - 1 : 0;
        };
        const auto greedy_tag = [&]() {
            const bool exact_stump = depth == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2);
            if (restrict_proxy_in_greedy) return exact_stump ? exact_tag() : -1;
            if (greedy_continuous_mode == GreedyContinuousMode::BINARY) {
                if (should_add_continuous_greedy_split_to_binary_lickety_()) return -2;
                if (exact_stump) return exact_tag();
            }
            return -1;
        };
        if (lookahead_init == 0) return greedy_tag();
        const int k = std::min<int>(lookahead, depth - 1);
        if (proxy_style == 4) return k <= 0 || k == depth - 1 ? exact_tag() : -3 - k;
        if (!use_restricted_lickety_proxy_() && depth == 1 &&
            !should_route_continuous_lickety_depth1_to_binary_greedy_() && use_restricted_depthd_exact_proxy_())
            return greedy_tag();
        return use_kla_cache() ? k : 0;
    }

    bool sparsify_subproblem_(const Packed& mask, int8_t depth) const {
        const int remaining = static_cast<int>(depth) - proxy_delta_;
        if (remaining <= 0 || remaining > static_cast<int>(sparsification_probability_.size())) return false;
        const double probability = sparsification_probability_[remaining - 1];
        if (probability <= 0) return false;
        if (probability >= 1) return true;
        const uint64_t hash = mix64(hash_mask64(mask.w.data(), n_words, tail_mask) ^
            mix64(sparsification_seed_) ^ mix64(static_cast<uint64_t>(remaining)));
        return static_cast<double>(hash >> 11) * 0x1.0p-53 < probability;
    }

    int cached_sparsification_split_(const Packed& mask, int8_t depth, const PathKey& pk) const {
        if (lookahead_init < 0 || depth <= 0) return -1;
        const auto cached_objective = [&](const Packed& child, const PathKey& child_pk) {
            if (depth == 1) return leaf_objective(child);
            const uint64_t key = key_of_subproblem(child, child_pk);
            Objective best = std::numeric_limits<Objective>::max();
            if (auto it = greedy_cache.find(K2{key, static_cast<int8_t>(depth - 1)}); it != greedy_cache.end())
                best = std::min(best, it->second);
            if (use_kla_cache()) {
                for (int k = 0; k < depth - 1; ++k) {
                    if (auto it = lickety_cache_kla.find(KLA{key, static_cast<int8_t>(depth - 1), static_cast<int8_t>(k)}); it != lickety_cache_kla.end())
                        best = std::min(best, it->second);
                }
            } else if (auto it = lickety_cache_k2.find(K2{key, static_cast<int8_t>(depth - 1)}); it != lickety_cache_k2.end())
                best = std::min(best, it->second);
            return best;
        };
        Objective best = leaf_objective(mask);
        int feature = -1;
        Packed left(n_words), right(n_words);
        for (int f = 0; f < n_features; ++f) {
            if ((f & 255) == 0 && resource_limit_reached_()) break;
            split_threshold_bits_(mask, f, left, right);
            if (!left.any() || !right.any()) continue;
            const PathKey* pk_left = &empty_pk();
            const PathKey* pk_right = &empty_pk();
            PathKey local_left, local_right;
            make_child_pks_if_needed_(f, pk, pk_left, pk_right, local_left, local_right);
            const Objective l = cached_objective(left, *pk_left);
            const Objective r = cached_objective(right, *pk_right);
            if (l == std::numeric_limits<Objective>::max() || r == std::numeric_limits<Objective>::max()) continue;
            if (l + r < best) { best = l + r; feature = f; }
        }
        return feature;
    }

    std::shared_ptr<TreeTrieNode> sparsified_split_(std::shared_ptr<TreeTrieNode> node,
        const Packed& mask, int8_t depth, Objective budget, const PathKey& pk,
        const ContinuousPath& cpath, int8_t k_here, const std::vector<int>* active_features) {
        const K2 key{hash_mask64(mask.w.data(), n_words, tail_mask), depth};
        auto found = sparsification_splits_.find(key);
        int feature;
        if (found == sparsification_splits_.end()) {
            const Objective objective = proxy_completion_objective_(mask, depth, k_here, pk, cpath);
            if (sparsification_index_enabled_ && objective == leaf_objective(mask)) {
                feature = -1;
                ++sparsification_leaf_shortcuts_;
            } else {
                const auto index = sparsification_index_enabled_
                    ? proxy_split_indices_.find(KLA{key_of_subproblem(mask, pk), depth, proxy_split_index_tag_(depth, k_here)})
                    : proxy_split_indices_.end();
                if (index != proxy_split_indices_.end() && index->second.objective == objective) {
                    feature = index->second.feature;
                    ++sparsification_index_hits_;
                } else {
                    const auto start = std::chrono::steady_clock::now();
                    feature = cached_sparsification_split_(mask, depth, pk);
                    sparsification_scan_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    ++sparsification_scan_count_;
                }
            }
            sparsification_splits_.emplace(key, feature);
        } else feature = found->second;
        if (feature >= 0 && !resource_limit_reached_()) {
            Packed left(n_words), right(n_words);
            split_threshold_bits_(mask, feature, left, right);
            if (left.any() && right.any()) {
                const PathKey* pk_left = &empty_pk();
                const PathKey* pk_right = &empty_pk();
                PathKey local_left, local_right;
                make_child_pks_if_needed_(feature, pk, pk_left, pk_right, local_left, local_right);
                ContinuousPath current = materialize_continuous_path_(cpath);
                const ContinuousPath* path_left = &current;
                const ContinuousPath* path_right = &current;
                ContinuousPath local_path_left, local_path_right;
                make_child_continuous_paths_if_needed_(feature, current, path_left, path_right, local_path_left, local_path_right);
                const Objective l = proxy_completion_objective_(left, depth - 1, k_here, *pk_left, *path_left);
                const Objective r = proxy_completion_objective_(right, depth - 1, k_here, *pk_right, *path_right);
                const bool feasible = rule_list_mode
                    ? !(l > objective_remainder(budget, gamma) && r > objective_remainder(budget, gamma))
                    : l + r <= budget;
                if (feasible) {
                    auto existing = std::find_if(node->splits.begin(), node->splits.end(),
                        [feature](const SplitNode& split) { return split.feature == feature; });
                    auto children = solve_siblings_extend(existing == node->splits.end() ? nullptr : existing->left,
                        existing == node->splits.end() ? nullptr : existing->right, l, r, left, right, budget, depth,
                        *pk_left, *pk_right, *path_left, *path_right, active_features);
                    if (children_have_feasible_pair_(children.first, children.second, budget)) {
                        if (existing == node->splits.end()) node->add_split(feature, children.first, children.second);
                        else {
                            existing->left = children.first;
                            existing->right = children.second;
                            node->min_objective = std::min(node->min_objective,
                                children.first->min_objective + children.second->min_objective);
                        }
                    }
                }
            }
        }
        node->hist.clear();
        node->hist_built = false;
        for (auto& split : node->splits) split.num_valid_trees = 0;
        if (!node_has_solution_(node)) return nullptr;
        if (trie_cache_enabled) trie_cache[K2{key_of_subproblem(mask, pk), depth}] = node;
        return node;
    }

public:
    void set_sparsification(const std::vector<double>& probability, uint64_t seed, bool cache_proxy_splits = true) {
        for (double value : probability)
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("Sparsification probabilities must be finite and between 0 and 1.");
        sparsification_probability_ = probability;
        sparsification_seed_ = seed;
        sparsification_enabled_ = std::any_of(probability.begin(), probability.end(), [](double p) { return p > 0; });
        sparsification_index_enabled_ = sparsification_enabled_ && cache_proxy_splits;
        clear_sparsification_state_();
        if (!sparsification_index_enabled_ && proxy_split_indices_.bucket_count() > 1)
            decltype(proxy_split_indices_)().swap(proxy_split_indices_);
    }

    std::map<std::string, double> get_sparsification_stats() const {
        return {{"enabled", sparsification_enabled_ ? 1.0 : 0.0},
            {"selected_subproblems", static_cast<double>(sparsification_splits_.size())},
            {"split_scans", static_cast<double>(sparsification_scan_count_)},
            {"split_index_enabled", sparsification_index_enabled_ ? 1.0 : 0.0},
            {"split_index_hits", static_cast<double>(sparsification_index_hits_)},
            {"split_index_entries", static_cast<double>(proxy_split_indices_.size())},
            {"leaf_shortcuts", static_cast<double>(sparsification_leaf_shortcuts_)},
            {"split_scan_seconds", sparsification_scan_seconds_},
            {"proxy_refinement_skipped", sparsification_enabled_ ? 1.0 : 0.0}};
    }

private:
