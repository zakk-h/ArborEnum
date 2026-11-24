    std::unordered_map<K2, std::vector<Objective>, K2::Hash> proxy_trace_values_;

    void remember_proxy_trace_values_() {
        auto remember = [&](const K2& key, Objective objective) {
            if (key.depth > proxy_delta_) return;
            auto& values = proxy_trace_values_[key];
            if (std::find(values.begin(), values.end(), objective) == values.end()) values.push_back(objective);
        };
        for (const auto& [key, objective] : greedy_cache) remember(key, objective);
        for (const auto& [key, objective] : lickety_cache_k2) remember(key, objective);
        for (const auto& [key, objective] : lickety_cache_kla) remember(K2{key.k, key.depth}, objective);
    }

    static bool proxy_objectives_match_(Objective left, Objective right) {
        if constexpr (regression_objective) {
            return std::abs(left - right) <= 64 * std::numeric_limits<double>::epsilon()
                * std::max({1.0, std::abs(static_cast<double>(left)), std::abs(static_cast<double>(right))});
        } else return left == right;
    }

    std::vector<Objective> proxy_cached_values_(const Packed& mask, int8_t depth, const PathKey& pk) const {
        std::vector<Objective> values{leaf_objective(mask)};
        if (depth > 0) {
            const uint64_t key = key_of_subproblem(mask, pk);
            if (auto it = greedy_cache.find(K2{key, depth}); it != greedy_cache.end()) values.push_back(it->second);
            if (auto it = lickety_cache_k2.find(K2{key, depth}); it != lickety_cache_k2.end()) values.push_back(it->second);
            for (int k = 0; k < depth; ++k) {
                if (auto it = lickety_cache_kla.find(KLA{key, depth, k}); it != lickety_cache_kla.end()) values.push_back(it->second);
            }
            if (auto it = proxy_trace_values_.find(K2{key, depth}); it != proxy_trace_values_.end())
                values.insert(values.end(), it->second.begin(), it->second.end());
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    std::shared_ptr<TreeTrieNode> cached_proxy_graph_(const Packed& mask, int8_t depth, const PathKey& pk, Objective expected) const {
        const auto leaf = best_leaf_action(mask);
        if (proxy_objectives_match_(leaf.loss, expected)) {
            auto node = std::make_shared<TreeTrieNode>();
            node->budget = leaf.loss;
            node->add_leaf(leaf.prediction, leaf.loss);
            return node;
        }
        if (depth <= 0) return nullptr;
        Packed left(n_words), right(n_words);
        for (int feature = 0; feature < n_features; ++feature) {
            split_threshold_bits_(mask, feature, left, right);
            if (!left.any() || !right.any()) continue;
            const PathKey* pk_left = &empty_pk();
            const PathKey* pk_right = &empty_pk();
            PathKey local_left, local_right;
            make_child_pks_if_needed_(feature, pk, pk_left, pk_right, local_left, local_right);
            const auto left_values = proxy_cached_values_(left, depth - 1, *pk_left);
            const auto right_values = proxy_cached_values_(right, depth - 1, *pk_right);
            std::size_t i = 0, j = right_values.size();
            while (i < left_values.size() && j > 0) {
                const Objective sum = left_values[i] + right_values[j - 1];
                if (proxy_objectives_match_(sum, expected)) {
                    auto left_node = cached_proxy_graph_(left, depth - 1, *pk_left, left_values[i]);
                    auto right_node = cached_proxy_graph_(right, depth - 1, *pk_right, right_values[j - 1]);
                    if (left_node && right_node) {
                        auto node = std::make_shared<TreeTrieNode>();
                        node->budget = left_node->min_objective + right_node->min_objective;
                        node->add_split(feature, left_node, right_node);
                        return node;
                    }
                    ++i;
                } else if (sum < expected) ++i;
                else --j;
            }
        }
        return nullptr;
    }

    std::shared_ptr<TreeTrieNode> proxy_frontier_(std::shared_ptr<TreeTrieNode> node, const Packed& mask,
        int8_t depth, Objective budget, const PathKey& pk, const ContinuousPath& cpath) {
        const Objective objective = proxy_objective_for_gap_(mask, depth, pk, cpath);
        if (objective > budget) return node_has_solution_(node) ? node : nullptr;
        if (!node) node = std::make_shared<TreeTrieNode>();
        node->budget = budget;
        node->leaves.clear();
        node->splits.clear();
        node->hist.clear();
        node->hist_built = false;
        node->min_objective = std::numeric_limits<Objective>::max();
        node->add_leaf(best_leaf_action(mask).prediction, objective);
        node->proxy_pending = true;
        if (trie_cache_enabled) trie_cache[K2{key_of_subproblem(mask, pk), depth}] = node;
        return node;
    }

    void expand_proxy_frontiers_dfs_(const std::shared_ptr<TreeTrieNode>& node, const Packed& mask,
        int8_t depth, const PathKey& pk, std::unordered_set<const TreeTrieNode*>& visited) {
        if (!node || !visited.insert(node.get()).second) return;
        if (depth == proxy_delta_) {
            if (node->proxy_pending) {
                auto tree = cached_proxy_graph_(mask, depth, pk, node->min_objective);
                if (!tree) throw std::runtime_error("Could not reconstruct the cached proxy completion.");
                node->leaves = std::move(tree->leaves);
                node->splits = std::move(tree->splits);
                node->min_objective = tree->min_objective;
                node->proxy_pending = false;
            }
        } else {
            Packed left(n_words), right(n_words);
            for (auto& split : node->splits) {
                split_threshold_bits_(mask, split.feature, left, right);
                const PathKey* pk_left = &empty_pk();
                const PathKey* pk_right = &empty_pk();
                PathKey local_left, local_right;
                make_child_pks_if_needed_(split.feature, pk, pk_left, pk_right, local_left, local_right);
                expand_proxy_frontiers_dfs_(split.left, left, depth - 1, *pk_left, visited);
                expand_proxy_frontiers_dfs_(split.right, right, depth - 1, *pk_right, visited);
                split.num_valid_trees = 0;
            }
            node->min_objective = std::numeric_limits<Objective>::max();
            for (const auto& leaf : node->leaves) node->min_objective = std::min(node->min_objective, leaf.loss);
            for (const auto& split : node->splits) node->min_objective = std::min(node->min_objective, split.left->min_objective + split.right->min_objective);
        }
        node->hist.clear();
        node->hist_built = false;
    }

    void expand_proxy_frontiers_(const std::shared_ptr<TreeTrieNode>& graph, const Packed& mask, int8_t depth) {
        std::unordered_set<const TreeTrieNode*> visited;
        expand_proxy_frontiers_dfs_(graph, mask, depth, empty_pk(), visited);
    }
