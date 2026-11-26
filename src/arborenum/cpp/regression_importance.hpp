    Packed regression_importance_training_mask_;

    Packed regression_importance_train_mask_() const {
        return regression_importance_training_mask_.w.empty() ? root_mask_() : regression_importance_training_mask_;
    }

    std::shared_ptr<TreeTrieNode> regression_importance_graph_(const PredNode* tree, const Packed& mask) const {
        if (!tree) throw std::runtime_error("The fitted tree is empty.");
        auto graph = std::make_shared<TreeTrieNode>();
        if (tree->feature < 0) {
#if defined(ARBORENUM_LOGISTIC)
            long double loss = 0;
            for (int word = 0; word < n_words; ++word) {
                uint64_t bits = mask.w[word];
                while (bits) {
                    const int row = word * 64 + std::countr_zero(bits);
                    loss += logistic_loss_(row, tree->prediction);
                    bits &= bits - 1;
                }
            }
            graph->add_leaf(tree->prediction, gamma + static_cast<Objective>(loss));
#else
            const auto moments = regression_moments_(mask);
            if (tree->prediction == DEFER_PREDICTION)
                graph->add_leaf(DEFER_PREDICTION, gamma + defer_penalty_from_count_(moments.count) + count_bb_wrong(mask));
            else graph->add_leaf(static_cast<Prediction>(moments.mean), gamma + static_cast<Objective>(moments.sse));
#endif
        } else {
            Packed left(n_words), right(n_words);
            split_threshold_bits_(mask, tree->feature, left, right);
            graph->add_split(tree->feature, regression_importance_graph_(tree->left.get(), left),
                regression_importance_graph_(tree->right.get(), right));
        }
        graph->budget = graph->min_objective;
        return graph;
    }

    std::vector<ExactImportanceInterval> regression_continuous_importance_(
        const ExactReplacementIntervalEvalSetup_& setup, const TreeTrieNode* graph) const {
        using Real = long double;
        constexpr Real inf = std::numeric_limits<Real>::infinity();
        constexpr int none = -1;
        const Real budget = setup.budget;
        if (!std::isfinite(budget) || budget < 0) {
            throw std::invalid_argument("Regression importance requires a finite nonnegative budget.");
        }
        const int p = static_cast<int>(setup.variable_columns.size());
        std::vector<ExactImportanceInterval> out(p, {0.0, 0.0});
        std::vector<uint8_t> used(p, 0);
        std::unordered_set<const TreeTrieNode*> visited;
        std::vector<const TreeTrieNode*> nodes{graph};
        while (!nodes.empty()) {
            const auto* node = nodes.back();
            nodes.pop_back();
            if (!node || !visited.insert(node).second) continue;
            for (const auto& s : node->splits) {
                used[setup.internal_to_variable.at(s.feature)] = 1;
                nodes.push_back(s.left.get());
                nodes.push_back(s.right.get());
            }
        }
        const Objective impossible = std::numeric_limits<Objective>::max();
        std::unordered_map<const TreeTrieNode*, Objective> minimum;
        const auto min_structure = [&](auto&& self, const TreeTrieNode* node) -> Objective {
            if (!node) return impossible;
            if (const auto it = minimum.find(node); it != minimum.end()) return it->second;
            const Objective cap = std::min(node->budget, setup.budget);
            Objective best = impossible;
            for (const auto& leaf : node->leaves)
                if (leaf.loss <= cap) best = std::min(best, leaf.loss);
            for (const auto& split : node->splits) {
                const Objective l = self(self, split.left.get()), r = self(self, split.right.get());
                if (l == impossible || r == impossible) continue;
                const Objective q = l + r;
                if (q <= cap) best = std::min(best, q);
            }
            minimum.emplace(node, best);
            return best;
        };
        if (min_structure(min_structure, graph) == impossible) return {};
        const auto& ctx = setup.ctx;
        const Real inv_n = 1.0L / ctx.n_eval;
        struct Key {
            const TreeTrieNode* node;
            std::vector<uint64_t> bits;
            bool operator==(const Key& rhs) const {
                return node == rhs.node && bits == rhs.bits;
            }
        };
        struct KeyHash {
            std::size_t operator()(const Key& key) const {
                uint64_t h = mix64(static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(key.node)));
                for (uint64_t w : key.bits) h = mix64(h ^ w);
                return static_cast<std::size_t>(h);
            }
        };
        struct Leaf {
            Real q, a, b, c, radius;
        };
        struct Alt {
            int leaf = -1, left = -1, right = -1;
        };
        struct State {
            Objective cap, minq;
            std::vector<Alt> alts;
        };
        struct Value {
            Real f = -std::numeric_limits<Real>::infinity(), p = 0, q = 0;
            int alt = -1;
        };
        struct Occ {
            int state, alt = -1, left = -1, right = -1;
        };
        struct Branch {
            std::vector<Occ> tree;
            Real ub, eta;
        };
        const auto branch_less = [](const Branch& a, const Branch& b) { return a.ub < b.ub; };
        for (int variable = 0; variable < p; ++variable) {
            if (!used[variable]) continue;
            std::vector<State> states;
            std::vector<Leaf> leaves;
            std::unordered_map<Key, int, KeyHash> ids;
            std::unordered_map<const TreeTrieNode*, bool> contains;
            std::unordered_map<const TreeTrieNode*, int> zero_states;
            const auto has_variable = [&](auto&& self, const TreeTrieNode* node) -> bool {
                if (!node) return false;
                if (const auto it = contains.find(node); it != contains.end()) return it->second;
                bool found = false;
                for (const auto& split : node->splits) {
                    if (setup.internal_to_variable[split.feature] == variable ||
                        self(self, split.left.get()) || self(self, split.right.get())) {
                        found = true;
                        break;
                    }
                }
                contains.emplace(node, found);
                return found;
            };
            const auto build = [&](auto&& self, const TreeTrieNode* node, const Packed& train,
                                   const Packed& recipient, const Packed& donor) -> int {
                if (!node || minimum.at(node) == impossible) return none;
                if (!recipient.any() || !donor.any() ||
                    (donor.count() == ctx.n_eval && !has_variable(has_variable, node))) {
                    if (const auto it = zero_states.find(node); it != zero_states.end()) return it->second;
                    const int leaf = static_cast<int>(leaves.size());
                    leaves.push_back({minimum.at(node), 0, 0, 0, 0});
                    const int id = static_cast<int>(states.size());
                    states.push_back({std::min(node->budget, setup.budget), minimum.at(node), {{leaf, none, none}}});
                    zero_states.emplace(node, id);
                    return id;
                }
                Key key{node, {}};
                key.bits.reserve(train.w.size() + recipient.w.size() + donor.w.size());
                key.bits.insert(key.bits.end(), train.w.begin(), train.w.end());
                key.bits.insert(key.bits.end(), recipient.w.begin(), recipient.w.end());
                key.bits.insert(key.bits.end(), donor.w.begin(), donor.w.end());
                if (const auto it = ids.find(key); it != ids.end()) return it->second;
                const Objective cap = std::min(node->budget, setup.budget);
                State state{cap, std::numeric_limits<Objective>::max(), {}};
                const int m = train.count();
                for (const auto& leaf : node->leaves) {
                    if (leaf.loss > cap) continue;
                    const bool deferred = leaf.prediction == DEFER_PREDICTION;
                    if (!deferred && !m) throw std::runtime_error("Empty regression leaf in importance graph.");
                    const int donor_count = donor.count();
                    const Real w = donor_count * inv_n;
                    Real a = 0, b = 0, c = 0;
                    if (donor_count != ctx.n_eval) {
                        for (int word = 0; word < ctx.n_words; ++word) {
                            uint64_t bits = recipient.w[word];
                            while (bits) {
                                const int bit = std::countr_zero(bits);
                                const int row = word * 64 + bit;
                                const Real weight = w - static_cast<Real>((donor.w[word] >> bit) & 1ULL);
                                const Real prediction = deferred ? setup.y_bits.bb_predictions.at(row) : leaf.prediction;
                                const Real residual = static_cast<Real>(setup.y_bits.targets[row]) - prediction;
                                c += weight * residual * residual;
                                if (!deferred) {
                                    a += weight;
                                    b -= 2 * weight * residual;
                                }
                                bits &= bits - 1;
                            }
                        }
                    }
                    const Real radius = deferred ? 0 : std::sqrt(std::max(0.0L, budget - leaf.loss));
                    const int id = static_cast<int>(leaves.size());
                    leaves.push_back({leaf.loss, deferred ? 0 : a * inv_n / m,
                        deferred ? 0 : b * inv_n / std::sqrt(static_cast<Real>(m)), c * inv_n, radius});
                    state.alts.push_back({id, none, none});
                    state.minq = std::min(state.minq, leaf.loss);
                }
                for (const auto& split : node->splits) {
                    if (!split.left || !split.right ||
                        minimum.at(split.left.get()) == impossible || minimum.at(split.right.get()) == impossible ||
                        minimum.at(split.left.get()) + minimum.at(split.right.get()) > cap) continue;
                    const int f = split.feature;
                    Packed tl(n_words), tr(n_words);
                    and_bits(train, X_bits[f], tl);
                    andnot_bits(train, X_bits[f], tr);
                    Packed el(ctx.n_words), er(ctx.n_words);
                    const bool is_variable = setup.internal_to_variable[f] == variable;
                    const Packed& source = is_variable ? donor : recipient;
                    and_bits_eval(source, ctx.X_bits_eval[f], el, ctx.n_words, ctx.tail_mask);
                    andnot_bits_eval(source, ctx.X_bits_eval[f], er, ctx.n_words, ctx.tail_mask);
                    const int l = is_variable ? self(self, split.left.get(), tl, recipient, el)
                                              : self(self, split.left.get(), tl, el, donor);
                    const int r = is_variable ? self(self, split.right.get(), tr, recipient, er)
                                              : self(self, split.right.get(), tr, er, donor);
                    if (l == none || r == none) continue;
                    const Objective q = states[l].minq + states[r].minq;
                    if (q > cap) continue;
                    state.alts.push_back({none, l, r});
                    state.minq = std::min(state.minq, q);
                }
                if (state.alts.empty()) return none;
                const int id = static_cast<int>(states.size());
                states.push_back(std::move(state));
                ids.emplace(std::move(key), id);
                return id;
            };
            const int root = build(build, graph, regression_importance_train_mask_(), setup.root_mask, setup.root_mask);
            if (root == none) return {};
            ids.clear();
            ids.rehash(0);
            contains.clear();
            contains.rehash(0);
            zero_states.clear();
            zero_states.rehash(0);
            const auto solve = [&](Real sign) -> Real {
                const auto leaf_value = [&](int id, Real eta) {
                    const auto& l = leaves[id];
                    const Real a = sign * l.a, b = std::abs(l.b);
                    Real z = l.radius;
                    if (eta > a) z = std::min(z, b / (2 * (eta - a)));
                    else if (eta == a && b == 0) z = 0;
                    const Real q = l.q + z * z;
                    const Real p = sign * l.c + z * (a * z + b);
                    return Value{p - eta * q, p, q, none};
                };
                const auto fixed_optimum = [&](const std::vector<int>& ls, Objective base) -> Real {
                    const Real R = budget - static_cast<Real>(base);
                    if (R < 0) return -inf;
                    Real value = 0, lo = 0, norm_b = 0;
                    int linear = none;
                    for (int id : ls) {
                        const auto& l = leaves[id];
                        value += sign * l.c;
                        if (!l.radius) continue;
                        const Real a = sign * l.a;
                        if (a > lo) { lo = a; linear = none; }
                        if (a == lo && l.b == 0) linear = id;
                        norm_b = std::hypot(norm_b, l.b);
                    }
                    if (!R) return value;
                    const auto energy = [&](Real eta) {
                        Real e = 0;
                        for (int id : ls) {
                            const auto& l = leaves[id];
                            if (!l.radius || l.b == 0) continue;
                            const Real d = eta - sign * l.a;
                            if (d <= 0) return inf;
                            const Real z = std::abs(l.b) / (2 * d);
                            e += z * z;
                            if (e > R) return e;
                        }
                        return e;
                    };
                    Real eta = lo;
                    if (energy(lo) > R) {
                        Real hi = lo + norm_b / (2 * std::sqrt(R));
                        if (!(hi > lo)) hi = std::nextafter(lo, inf);
                        for (int k = 0; energy(hi) > R && k < 128; ++k) {
                            hi = lo + 2 * (hi - lo);
                        }
                        for (int k = 0; k < 100; ++k) {
                            const Real mid = lo + (hi - lo) / 2;
                            if (mid == lo || mid == hi) break;
                            if (energy(mid) > R) lo = mid;
                            else hi = mid;
                        }
                        eta = hi;
                    }
                    Real spent = 0;
                    for (int id : ls) {
                        const auto& l = leaves[id];
                        if (!l.radius || l.b == 0) continue;
                        Real z = std::abs(l.b) / (2 * (eta - sign * l.a));
                        z = std::min(z, std::sqrt(std::max(0.0L, R - spent)));
                        spent += z * z;
                        value += z * (sign * l.a * z + std::abs(l.b));
                    }
                    if (linear != none && eta == sign * leaves[linear].a && eta > 0)
                        value += eta * std::max(0.0L, R - spent);
                    return value;
                };
                Real initial_eta = 0;
                for (const auto& l : leaves) {
                    initial_eta = std::max(initial_eta, std::abs(l.a));
                    if (l.radius > 0) initial_eta = std::max(initial_eta, std::abs(l.b) / (2 * l.radius));
                }
                if (!(initial_eta > 0) || !std::isfinite(initial_eta)) initial_eta = 1;
                struct DualRow {
                    Real eta = -1;
                    std::vector<Value> values;
                    std::vector<uint64_t> stamps;
                    uint64_t epoch = 0, age = 0;
                };
                const std::size_t row_bytes = states.size() * (sizeof(Value) + sizeof(uint64_t));
                const std::size_t row_count = std::max<std::size_t>(2,
                    std::min<std::size_t>(8, (64ULL << 20) / std::max<std::size_t>(1, row_bytes)));
                std::vector<DualRow> dual_rows(row_count);
                Value* dp = nullptr;
                uint64_t* stamp = nullptr;
                uint64_t epoch = 0, clock = 0;
                const auto select_row = [&](Real eta, Real parent_eta, Real best_eta) {
                    DualRow* row = nullptr;
                    for (auto& cached : dual_rows) {
                        if (cached.eta == eta) { row = &cached; break; }
                    }
                    if (!row) {
                        for (auto& cached : dual_rows) {
                            if (cached.eta != 0 && cached.eta != parent_eta && cached.eta != best_eta &&
                                (!row || cached.age < row->age)) row = &cached;
                        }
                        if (!row) {
                            for (auto& cached : dual_rows) {
                                if (cached.eta != 0 && (!row || cached.age < row->age)) row = &cached;
                            }
                        }
                        row->eta = eta;
                        ++row->epoch;
                        if (row->values.empty()) {
                            row->values.resize(states.size());
                            row->stamps.resize(states.size(), 0);
                        }
                    }
                    row->age = ++clock;
                    dp = row->values.data();
                    stamp = row->stamps.data();
                    epoch = row->epoch;
                };
                Real incumbent = -inf;
                std::vector<Branch> queue;
                const auto combine = [](const Value& l, const Value& r, int alt) {
                    return Value{l.f + r.f, l.p + r.p, l.q + r.q, alt};
                };
                const auto optimize = [&](auto&& self, int id, Real eta) -> Value {
                    if (stamp[id] == epoch) return dp[id];
                    Value best;
                    const auto& st = states[id];
                    for (int k = 0; k < static_cast<int>(st.alts.size()); ++k) {
                        const auto& a = st.alts[k];
                        Value v = a.leaf != none ? leaf_value(a.leaf, eta)
                            : combine(self(self, a.left, eta), self(self, a.right, eta), k);
                        v.alt = k;
                        if (v.f > best.f || (v.f == best.f && v.q < best.q)) best = v;
                    }
                    stamp[id] = epoch;
                    return dp[id] = best;
                };
                const auto bound = [&](auto&& self, const Branch& br, int i, Real eta) -> Value {
                    const auto& occ = br.tree[i];
                    if (occ.alt == none) return optimize(optimize, occ.state, eta);
                    const auto& a = states[occ.state].alts[occ.alt];
                    if (a.leaf != none) return leaf_value(a.leaf, eta);
                    return combine(self(self, br, occ.left, eta), self(self, br, occ.right, eta), occ.alt);
                };
                const auto min_cost = [&](auto&& self, const Branch& br, int i) -> Objective {
                    const auto& occ = br.tree[i];
                    const auto& st = states[occ.state];
                    if (occ.alt == none) return st.minq;
                    const auto& a = st.alts[occ.alt];
                    if (a.leaf != none) return static_cast<Objective>(leaves[a.leaf].q);
                    const Objective l = self(self, br, occ.left), r = self(self, br, occ.right);
                    if (l == std::numeric_limits<Objective>::max() || r == std::numeric_limits<Objective>::max())
                        return std::numeric_limits<Objective>::max();
                    const Objective q = l + r;
                    return q <= st.cap ? q : std::numeric_limits<Objective>::max();
                };
                struct CandidateHash {
                    std::size_t operator()(const std::vector<int>& ls) const {
                        uint64_t h = 0;
                        for (int id : ls) h = mix64(h ^ static_cast<uint64_t>(id));
                        return static_cast<std::size_t>(h);
                    }
                };
                std::unordered_set<std::vector<int>, CandidateHash> evaluated;
                std::vector<int> candidate;
                const auto collect = [&](auto&& self, const Branch& br, int id, int occurrence, bool cheapest) -> Objective {
                    const auto& st = states[id];
                    int k = none;
                    const bool fixed = occurrence != none && br.tree[occurrence].alt != none;
                    if (fixed) k = br.tree[occurrence].alt;
                    else if (cheapest) {
                        for (int j = 0; j < static_cast<int>(st.alts.size()); ++j) {
                            const auto& a = st.alts[j];
                            const Objective q = a.leaf != none ? static_cast<Objective>(leaves[a.leaf].q)
                                : states[a.left].minq + states[a.right].minq;
                            if (q == st.minq) { k = j; break; }
                        }
                    } else k = dp[id].alt;
                    if (k == none) return std::numeric_limits<Objective>::max();
                    const auto& a = st.alts[k];
                    if (a.leaf != none) {
                        candidate.push_back(a.leaf);
                        return static_cast<Objective>(leaves[a.leaf].q);
                    }
                    const Objective l = self(self, br, a.left, fixed ? br.tree[occurrence].left : none, cheapest);
                    const Objective r = self(self, br, a.right, fixed ? br.tree[occurrence].right : none, cheapest);
                    if (l == std::numeric_limits<Objective>::max() || r == std::numeric_limits<Objective>::max())
                        return std::numeric_limits<Objective>::max();
                    const Objective q = l + r;
                    return q <= st.cap ? q : std::numeric_limits<Objective>::max();
                };
                const auto update = [&](const Branch& br, bool cheapest) {
                    candidate.clear();
                    const Objective q = collect(collect, br, root, 0, cheapest);
                    if (q != std::numeric_limits<Objective>::max() && q <= budget) {
                        if (evaluated.size() >= 8192) evaluated.clear();
                        if (evaluated.insert(candidate).second)
                            incumbent = std::max(incumbent, fixed_optimum(candidate, q));
                    }
                };
                const auto tolerance = [&]() { return 1e-10L * std::max(1.0L, std::abs(incumbent)); };
                const auto prunable = [&](Real ub) {
                    return std::isfinite(incumbent) && ub <= incumbent + tolerance();
                };
                const auto evaluate = [&](Branch br) {
                    const Objective q = min_cost(min_cost, br, 0);
                    if (q == std::numeric_limits<Objective>::max() || q > budget) return;
                    update(br, true);
                    bool complete = true;
                    for (const auto& occ : br.tree) if (occ.alt == none) { complete = false; break; }
                    if (complete || prunable(br.ub)) return;
                    const Real parent_eta = br.eta;
                    const auto trial = [&](Real eta) {
                        select_row(eta, parent_eta, br.eta);
                        const auto v = bound(bound, br, 0, eta);
                        const Real ub = v.f + eta * budget;
                        const Real roundoff = 512 * std::numeric_limits<Objective>::epsilon() *
                            (1 + std::abs(v.f) + std::abs(eta * budget) + std::abs(v.p));
                        if (ub + roundoff < br.ub) { br.ub = ub + roundoff; br.eta = eta; }
                        update(br, false);
                        return v.q;
                    };
                    Real lo = 0, hi = 0;
                    if (trial(0) > budget && !prunable(br.ub)) {
                        hi = parent_eta > 0 ? parent_eta : initial_eta;
                        Real qhi = trial(hi);
                        for (int k = 0; qhi > budget && k < 48 && !prunable(br.ub); ++k) {
                            lo = hi;
                            hi *= 2;
                            qhi = trial(hi);
                        }
                        if (qhi <= budget) {
                            for (int k = 0; k < 24 && !prunable(br.ub); ++k) {
                                const Real mid = lo + (hi - lo) / 2;
                                if (mid == lo || mid == hi) break;
                                if (trial(mid) > budget) lo = mid;
                                else hi = mid;
                            }
                        }
                    }
                    if (!prunable(br.ub)) {
                        queue.push_back(std::move(br));
                        std::push_heap(queue.begin(), queue.end(), branch_less);
                    }
                };
                evaluate(Branch{{Occ{root}}, inf, 0});
                while (!queue.empty()) {
                    std::pop_heap(queue.begin(), queue.end(), branch_less);
                    Branch br = std::move(queue.back());
                    queue.pop_back();
                    if (prunable(br.ub)) break;
                    int i = none;
                    for (int k = 0; k < static_cast<int>(br.tree.size()); ++k) {
                        if (br.tree[k].alt == none) { i = k; break; }
                    }
                    if (i == none) continue;
                    const auto& st = states[br.tree[i].state];
                    for (int k = 0; k < static_cast<int>(st.alts.size()); ++k) {
                        Branch child = br;
                        child.tree[i].alt = k;
                        const auto& a = st.alts[k];
                        if (a.leaf == none) {
                            child.tree[i].left = static_cast<int>(child.tree.size());
                            child.tree.push_back(Occ{a.left});
                            child.tree[i].right = static_cast<int>(child.tree.size());
                            child.tree.push_back(Occ{a.right});
                        }
                        evaluate(std::move(child));
                    }
                }
                return incumbent;
            };
            const Real lower = -solve(-1), upper = solve(1);
            if (!std::isfinite(lower) || !std::isfinite(upper)) return {};
            out[variable] = {static_cast<double>(lower), static_cast<double>(upper)};
        }
        return out;
    }
