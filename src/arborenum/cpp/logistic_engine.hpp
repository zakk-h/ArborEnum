    struct LogisticLeaf {
        int count = 0;
        int positives = 0;
        double correction = 0;
        double loss = 0;
        double lower_loss = 0;
        double residual = 0;
    };

#if defined(ARBORENUM_LOGISTIC)
    std::vector<double> logistic_offsets_;
    std::vector<double> logistic_evaluation_offsets_;
    std::vector<double> logistic_offset_exponentials_;
    mutable std::array<std::vector<double>, 4> logistic_leaf_exponentials_;
    std::vector<double> logistic_minimum_;
    std::vector<double> logistic_variation_;
    std::vector<uint64_t> logistic_positive_words_;
    std::vector<double> logistic_offset_values_;
    std::vector<int> logistic_offset_groups_;
    mutable std::vector<int> logistic_group_counts_, logistic_group_positives_, logistic_active_groups_;
    double logistic_clip_ = 30;
    double logistic_tolerance_ = 1e-10;
    int logistic_iterations_ = 50;
    bool logistic_constant_offset_ = true;
    double logistic_offset_ = 0;
    double logistic_r0_ = 0, logistic_r1_ = 0;
    double logistic_low0_ = 0, logistic_low1_ = 0;
    double logistic_high0_ = 0, logistic_high1_ = 0;
    std::vector<double> logistic_logs_, logistic_entropy_terms_;
    mutable double logistic_warm_start_ = 0;
    mutable std::unordered_map<uint64_t, LogisticLeaf> logistic_leaves_;
    mutable std::vector<double> logistic_gradients_, logistic_hessians_;
    mutable std::vector<int> logistic_active_rows_;
    mutable std::vector<long double> logistic_bucket_gradients_, logistic_bucket_hessians_;
    mutable std::vector<int> logistic_bucket_counts_;
    mutable std::vector<std::pair<int, int>> logistic_sparse_rows_;

    static double logistic_softplus_(double value) {
        return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
    }

    static double logistic_sigmoid_(double value) {
        const double e = std::exp(-std::abs(value));
        return value >= 0 ? 1 / (1 + e) : e / (1 + e);
    }

    double logistic_loss_(int row, double correction) const {
        return logistic_softplus_((1 - 2 * y_train[row]) *
            ((logistic_constant_offset_ ? logistic_offset_ : logistic_offsets_[row]) + correction));
    }

    void clear_logistic_leaf_cache_() {
        logistic_leaves_.clear();
        logistic_warm_start_ = 0;
    }

    void clear_logistic_state_() {
        clear_logistic_leaf_cache_();
        logistic_evaluation_offsets_.clear();
        logistic_warm_start_ = 0;
        for (auto& values : logistic_leaf_exponentials_) values.clear();
        logistic_gradients_.clear();
        logistic_hessians_.clear();
        logistic_minimum_.clear();
        logistic_variation_.clear();
        logistic_positive_words_.clear();
        logistic_offset_exponentials_.clear();
        logistic_offset_values_.clear();
        logistic_offset_groups_.clear();
        logistic_group_counts_.clear();
        logistic_group_positives_.clear();
        logistic_active_groups_.clear();
    }

public:
    void set_logistic_evaluation_offsets(const std::vector<double>& offsets) {
        for (double value : offsets) if (!std::isfinite(value))
            throw std::invalid_argument("Evaluation base logits must be finite.");
        logistic_evaluation_offsets_ = offsets;
    }

    void set_logistic_options(const std::vector<double>& offsets, double clip = 30,
            int iterations = 50, double tolerance = 1e-10) {
        if (!std::isfinite(clip) || clip <= 0 || iterations < 1 ||
                !std::isfinite(tolerance) || tolerance <= 0)
            throw std::invalid_argument("Invalid logistic leaf optimization parameters.");
        for (double offset : offsets) if (!std::isfinite(offset))
            throw std::invalid_argument("Base logits must be finite.");
        logistic_offsets_ = offsets;
        logistic_clip_ = clip;
        logistic_iterations_ = iterations;
        logistic_tolerance_ = tolerance;
        clear_fit_state_();
    }

private:
    void initialize_logistic_() {
        const int n = static_cast<int>(y_train.size());
        if (!logistic_offsets_.empty() && static_cast<int>(logistic_offsets_.size()) != n)
            throw std::invalid_argument("base_logits must have one value per training observation.");
        logistic_constant_offset_ = logistic_offsets_.empty() || std::all_of(
            logistic_offsets_.begin(), logistic_offsets_.end(), [&](double z) { return z == logistic_offsets_[0]; });
        logistic_offset_ = logistic_offsets_.empty() ? 0 : logistic_offsets_[0];
        logistic_low0_ = logistic_r0_ = logistic_softplus_(logistic_offset_ - logistic_clip_);
        logistic_high1_ = logistic_r1_ = logistic_softplus_(-logistic_offset_ - logistic_clip_);
        logistic_low1_ = logistic_softplus_(-logistic_offset_ + logistic_clip_);
        logistic_high0_ = logistic_softplus_(logistic_offset_ + logistic_clip_);
        logistic_logs_.resize(n + 1);
        logistic_entropy_terms_.assign(n + 1, 0);
        for (int count = 1; count <= n; ++count) {
            logistic_logs_[count] = std::log(static_cast<double>(count));
            logistic_entropy_terms_[count] = count * logistic_logs_[count];
        }
        logistic_offset_exponentials_.clear();
        logistic_offset_values_.clear();
        logistic_offset_groups_.clear();
        logistic_group_counts_.clear();
        logistic_group_positives_.clear();
        logistic_active_groups_.clear();
        if (!logistic_constant_offset_) {
            logistic_offset_values_ = logistic_offsets_;
            std::sort(logistic_offset_values_.begin(), logistic_offset_values_.end());
            logistic_offset_values_.erase(std::unique(logistic_offset_values_.begin(),
                logistic_offset_values_.end()), logistic_offset_values_.end());
            if (logistic_offset_values_.size() <= static_cast<std::size_t>(std::max(1, n / 4))) {
                logistic_offset_groups_.resize(n);
                for (int row = 0; row < n; ++row)
                    logistic_offset_groups_[row] = static_cast<int>(std::lower_bound(logistic_offset_values_.begin(),
                        logistic_offset_values_.end(), logistic_offsets_[row]) - logistic_offset_values_.begin());
                logistic_group_counts_.assign(logistic_offset_values_.size(), 0);
                logistic_group_positives_.assign(logistic_offset_values_.size(), 0);
            } else logistic_offset_values_.clear();
        }
        if (!logistic_constant_offset_ && logistic_offset_groups_.empty() && logistic_clip_ <= 100 &&
                std::all_of(logistic_offsets_.begin(), logistic_offsets_.end(), [](double z) { return std::abs(z) <= 600; })) {
            logistic_offset_exponentials_.resize(n);
            for (int row = 0; row < n; ++row)
                logistic_offset_exponentials_[row] = std::exp(-std::abs(logistic_offsets_[row]));
        }
        logistic_minimum_.resize(n);
        logistic_variation_.resize(n);
        logistic_positive_words_.assign((n + 63) / 64, 0);
        for (int row = 0; row < n; ++row) {
            if (y_train[row] != 0 && y_train[row] != 1)
                throw std::invalid_argument("Logistic targets must be zero or one.");
            if (y_train[row] == 1) logistic_positive_words_[row >> 6] |= 1ULL << (row & 63);
            const double value = (1 - 2 * y_train[row]) *
                (logistic_constant_offset_ ? logistic_offset_ : logistic_offsets_[row]);
            logistic_minimum_[row] = logistic_softplus_(value - logistic_clip_);
            const double variation = value >= logistic_clip_
                ? 2 * logistic_clip_ + logistic_softplus_(-value - logistic_clip_) - logistic_softplus_(-value + logistic_clip_)
                : logistic_softplus_(value + logistic_clip_) - logistic_minimum_[row];
            logistic_variation_[row] = std::nextafter(std::max(0.0, variation),
                std::numeric_limits<double>::infinity());
        }
        logistic_leaves_.clear();
    }

    double logistic_roundoff_(double value, int count) const {
        return 128 * std::numeric_limits<double>::epsilon() *
            (1 + std::abs(value) + count * logistic_clip_);
    }

    LogisticLeaf logistic_counts_leaf_(int count, int positives) const {
        LogisticLeaf result;
        result.count = count;
        result.positives = positives;
        if (!count) return result;
        const double unconstrained = positives == 0 ? -std::numeric_limits<double>::infinity()
            : positives == count ? std::numeric_limits<double>::infinity()
            : logistic_logs_[positives] - logistic_logs_[count - positives] - logistic_offset_;
        result.correction = std::clamp(unconstrained, -logistic_clip_, logistic_clip_);
        if (unconstrained <= -logistic_clip_) result.loss = (count - positives) * logistic_low0_ + positives * logistic_low1_;
        else if (unconstrained >= logistic_clip_) result.loss = (count - positives) * logistic_high0_ + positives * logistic_high1_;
        else if (std::min(positives, count - positives) * 64LL < count)
            result.loss = (count - positives) * std::log1p(static_cast<double>(positives) / (count - positives))
                + positives * std::log1p(static_cast<double>(count - positives) / positives);
        else result.loss = std::max(0.0, logistic_entropy_terms_[count] - logistic_entropy_terms_[positives] - logistic_entropy_terms_[count - positives]);
        const double residual = (count - positives) * logistic_r0_ + positives * logistic_r1_;
        result.residual = std::max(0.0, residual - logistic_roundoff_(residual, count));
        result.lower_loss = std::max(0.0, result.loss - logistic_roundoff_(result.loss, count));
        return result;
    }

    LogisticLeaf logistic_leaf_(const Packed& mask) const {
        const uint64_t key = regression_mask_key_(mask);
        if (auto it = logistic_leaves_.find(key); it != logistic_leaves_.end()) return it->second;
        LogisticLeaf result;
        const bool factored = !logistic_offset_exponentials_.empty();
        if (factored) for (auto& values : logistic_leaf_exponentials_) values.clear();
        for (int group : logistic_active_groups_) {
            logistic_group_counts_[group] = 0;
            logistic_group_positives_[group] = 0;
        }
        logistic_active_groups_.clear();
        long double residual = 0;
        for (int word = 0; word < n_words; ++word) {
            uint64_t bits = mask.w[word];
            result.count += popcnt64(bits);
            result.positives += popcnt64(bits & logistic_positive_words_[word]);
            if (!logistic_constant_offset_) while (bits) {
                const int row = word * 64 + std::countr_zero(bits);
                residual += logistic_minimum_[row];
                if (factored) logistic_leaf_exponentials_[(logistic_offsets_[row] >= 0 ? 0 : 2) +
                    (y_train[row] == 1)].push_back(logistic_offset_exponentials_[row]);
                if (!logistic_offset_groups_.empty()) {
                    const int group = logistic_offset_groups_[row];
                    if (!logistic_group_counts_[group]++) logistic_active_groups_.push_back(group);
                    logistic_group_positives_[group] += y_train[row] == 1;
                }
                bits &= bits - 1;
            }
        }
        if (logistic_constant_offset_) result = logistic_counts_leaf_(result.count, result.positives);
        else if (result.count) {
            result.residual = std::max(0.0, static_cast<double>(residual) -
                logistic_roundoff_(static_cast<double>(residual), result.count));
            if (!result.positives || result.positives == result.count) {
                result.correction = result.positives ? logistic_clip_ : -logistic_clip_;
                result.loss = static_cast<double>(residual);
                result.lower_loss = result.residual;
                logistic_leaves_.emplace(key, result);
                return result;
            }
            const auto derivatives = [&](double correction) {
                long double gradient = 0, hessian = 0;
                if (!logistic_offset_groups_.empty()) {
                    for (int group : logistic_active_groups_) {
                        const double logit = logistic_offset_values_[group] + correction;
                        const double e = std::exp(-std::abs(logit));
                        const int count = logistic_group_counts_[group];
                        const int positives = logistic_group_positives_[group];
                        const double inverse = 1 / (1 + e);
                        gradient += logit >= 0 ? ((count - positives) - positives * e) * inverse
                                              : ((count - positives) * e - positives) * inverse;
                        hessian += count * e * inverse * inverse;
                    }
                    return std::pair<double, double>{static_cast<double>(gradient), static_cast<double>(hessian)};
                }
                if (factored) {
                    const double positive_factor = std::exp(correction), negative_factor = std::exp(-correction);
                    const auto accumulate = [&]<bool Positive, bool Label>(const std::vector<double>& values) {
                        const double factor = Positive ? negative_factor : positive_factor;
                        for (std::size_t begin = 0; begin < values.size(); begin += 32) {
                            double block_gradient = 0, block_hessian = 0;
                            const std::size_t end = std::min(begin + 32, values.size());
                            for (std::size_t index = begin; index < end; ++index) {
                                const double q = values[index] * factor;
                                const double inverse = 1 / (1 + q);
                                if constexpr (Label) block_gradient -= Positive ? q * inverse : inverse;
                                else block_gradient += Positive ? inverse : q * inverse;
                                block_hessian += q * inverse * inverse;
                            }
                            gradient += block_gradient;
                            hessian += block_hessian;
                        }
                    };
                    accumulate.template operator()<true, false>(logistic_leaf_exponentials_[0]);
                    accumulate.template operator()<true, true>(logistic_leaf_exponentials_[1]);
                    accumulate.template operator()<false, false>(logistic_leaf_exponentials_[2]);
                    accumulate.template operator()<false, true>(logistic_leaf_exponentials_[3]);
                    return std::pair<double, double>{static_cast<double>(gradient), static_cast<double>(hessian)};
                }
                const double positive_factor = factored ? std::exp(correction) : 0;
                const double negative_factor = factored ? std::exp(-correction) : 0;
                for (int word = 0; word < n_words; ++word) {
                    uint64_t bits = mask.w[word];
                    while (bits) {
                        const int row = word * 64 + std::countr_zero(bits);
                        const double offset = logistic_offsets_[row];
                        const bool positive = factored ? offset >= 0 : offset + correction >= 0;
                        const double e = factored ? logistic_offset_exponentials_[row] *
                            (positive ? negative_factor : positive_factor) : std::exp(-std::abs(offset + correction));
                        const double inverse = 1 / (1 + e);
                        gradient += y_train[row] == 0 ? (positive ? inverse : e * inverse)
                                                     : -(positive ? e * inverse : inverse);
                        hessian += e * inverse * inverse;
                        bits &= bits - 1;
                    }
                }
                return std::pair<double, double>{static_cast<double>(gradient), static_cast<double>(hessian)};
            };
            double gradient = 0;
            if (result.positives == 0) result.correction = -logistic_clip_;
            else if (result.positives == result.count) result.correction = logistic_clip_;
            else if (derivatives(-logistic_clip_).first >= 0) result.correction = -logistic_clip_;
            else if (derivatives(logistic_clip_).first <= 0) result.correction = logistic_clip_;
            else {
                double low = -logistic_clip_, high = logistic_clip_;
                double correction = std::clamp(logistic_warm_start_, low, high);
                for (int iteration = 0; iteration < logistic_iterations_; ++iteration) {
                    const auto [g, h] = derivatives(correction);
                    if (std::abs(g) <= logistic_tolerance_ * std::max(1, result.count)) break;
                    if (g > 0) high = correction;
                    else low = correction;
                    const double step = h > 0 ? correction - g / h : std::numeric_limits<double>::infinity();
                    const double next = std::isfinite(step) && step > low && step < high ? step : low + (high - low) / 2;
                    if (next == correction) break;
                    correction = next;
                }
                result.correction = correction;
            }
            gradient = derivatives(result.correction).first;
            long double loss = 0;
            if (!logistic_offset_groups_.empty()) {
                for (int group : logistic_active_groups_) {
                    const double logit = logistic_offset_values_[group] + result.correction;
                    const int count = logistic_group_counts_[group];
                    const int positives = logistic_group_positives_[group];
                    loss += (count - positives) * logistic_softplus_(logit) + positives * logistic_softplus_(-logit);
                }
            } else {
                const bool factored = !logistic_offset_exponentials_.empty();
                const double positive_factor = factored ? std::exp(result.correction) : 0;
                const double negative_factor = factored ? std::exp(-result.correction) : 0;
                for (int word = 0; word < n_words; ++word) {
                    uint64_t bits = mask.w[word];
                    while (bits) {
                        const int row = word * 64 + std::countr_zero(bits);
                        if (factored) {
                            const double offset = logistic_offsets_[row];
                            const double q = logistic_offset_exponentials_[row] *
                                (offset >= 0 ? negative_factor : positive_factor);
                            const double signed_logit = (1 - 2 * y_train[row]) * (offset + result.correction);
                            loss += std::max(signed_logit, 0.0) + std::log1p(q <= 1 ? q : 1 / q);
                        } else loss += logistic_loss_(row, result.correction);
                        bits &= bits - 1;
                    }
                }
            }
            result.loss = static_cast<double>(loss);
            const double tangent = gradient >= 0 ? gradient * (-logistic_clip_ - result.correction)
                                                 : gradient * (logistic_clip_ - result.correction);
            result.lower_loss = std::max(0.0, result.loss + tangent - logistic_roundoff_(result.loss, result.count));
            logistic_warm_start_ = result.correction;
        }
        logistic_leaves_.emplace(key, result);
        return result;
    }

    double logistic_budget_(double bound, int count) const {
        const double relative = 16 * std::numeric_limits<double>::epsilon() *
            (std::max(0, static_cast<int>(trained_depth_budget)) + 2) * (1 + std::log1p(count));
        return std::nextafter(bound + relative * std::abs(bound), std::numeric_limits<double>::infinity());
    }

    Objective logistic_lower_bound_(const Packed& mask, int depth) const {
        const auto leaf = logistic_leaf_(mask);
        if (!leaf.count) return 0;
        if (depth <= 0) return leaf.lower_loss + gamma;
        return std::min(leaf.lower_loss + gamma, leaf.residual + 2 * gamma);
    }

    bool logistic_leaf_optimal_(const Packed& mask, Objective leaf) const {
        const auto value = logistic_leaf_(mask);
        return !value.count || !value.positives || value.positives == value.count || leaf <= value.residual + 2 * gamma;
    }

    long double logistic_movement_(const Packed& mask, int source, int destination, Objective cutoff) const {
        if (cutoff <= 0) return 0;
        if (source == destination) return 0;
        long double error = 0;
        if (auto* scope = regression_scope_(mask, source);
                scope && destination >= scope->first && destination < scope->end) {
            auto& prefix = scope->logistic_prefix;
            if (prefix.empty()) {
                const int group = continuous_group_pos_for_threshold_(source);
                const auto& positions = regression_bucket_positions_(group);
                prefix.assign(scope->end - scope->first + 2, 0);
                for (int word = 0; word < n_words; ++word) {
                    uint64_t bits = mask.w[word];
                    while (bits) {
                        const int row = word * 64 + std::countr_zero(bits);
                        prefix[positions[row] + 1] += logistic_variation_[row];
                        bits &= bits - 1;
                    }
                }
                for (std::size_t i = 1; i < prefix.size(); ++i) prefix[i] += prefix[i - 1];
            }
            const int low = std::min(source, destination) - scope->first;
            const int high = std::max(source, destination) - scope->first;
            error = prefix[high + 1] - prefix[low + 1];
        } else {
            for (int word = 0; word < n_words; ++word) {
                uint64_t bits = mask.w[word] & (X_bits[source].w[word] ^ X_bits[destination].w[word]);
                while (bits) {
                    error += logistic_variation_[word * 64 + std::countr_zero(bits)];
                    bits &= bits - 1;
                    if (error >= cutoff) return error;
                }
            }
        }
        return error + logistic_roundoff_(static_cast<double>(error), n_samples);
    }

    GreedyObjFirstSplit logistic_stump_(const Packed& mask, const std::vector<int>& allowed, bool continuous) const {
        const auto parent = logistic_leaf_(mask);
        GreedyObjFirstSplit best{parent.count ? parent.loss + gamma : 0, -1};
        if (parent.count < 2 || logistic_leaf_optimal_(mask, best.obj)) return best;
        Packed left(n_words), right(n_words);
        const auto evaluate = [&](int feature, int count = -1, int positives = 0) {
            if (!allowed.empty() && !std::binary_search(allowed.begin(), allowed.end(), feature)) return;
            Objective objective;
            if (logistic_constant_offset_) {
                if (count < 0) {
                    count = positives = 0;
                    for (int word = 0; word < n_words; ++word) {
                        const uint64_t bits = mask.w[word] & X_bits[feature].w[word];
                        count += popcnt64(bits);
                        positives += popcnt64(bits & logistic_positive_words_[word]);
                    }
                }
                if (!count || count == parent.count) return;
                objective = logistic_counts_leaf_(count, positives).loss +
                    logistic_counts_leaf_(parent.count - count, parent.positives - positives).loss + 2 * gamma;
            } else {
                split_threshold_bits_(mask, feature, left, right);
                if (!left.any() || !right.any()) return;
                objective = leaf_objective(left) + leaf_objective(right);
            }
            if (objective < best.obj) best = {objective, feature};
        };
        const int binary_end = continuous ? first_continuous_feature_() : n_features;
        if (allowed.empty()) for (int feature = 0; feature < binary_end; ++feature) evaluate(feature);
        else for (int feature : allowed) if (feature < binary_end) evaluate(feature);
        if (continuous) for (int group = 0; group < static_cast<int>(continuous_starts.size()); ++group) {
            const int first = continuous_starts[group], end = continuous_group_end_(group);
            if (!logistic_constant_offset_) {
                for (int feature = first; feature < end; ++feature) evaluate(feature);
                continue;
            }
            const auto& positions = regression_bucket_positions_(group);
            logistic_bucket_counts_.assign(end - first + 1, 0);
            regression_stump_counts_.assign(end - first + 1, 0);
            for (int word = 0; word < n_words; ++word) {
                uint64_t bits = mask.w[word];
                while (bits) {
                    const int row = word * 64 + std::countr_zero(bits);
                    ++logistic_bucket_counts_[positions[row]];
                    regression_stump_counts_[positions[row]] += y_train[row] == 1;
                    bits &= bits - 1;
                }
            }
            int count = 0, positives = 0;
            for (int feature = first; feature < end; ++feature) {
                count += logistic_bucket_counts_[feature - first];
                positives += regression_stump_counts_[feature - first];
                if (logistic_bucket_counts_[feature - first] || !allowed.empty()) evaluate(feature, count, positives);
            }
        }
        return best;
    }

    std::pair<int, double> logistic_taylor_split_(const Packed& mask, bool restricted) const {
        const auto parent = logistic_leaf_(mask);
        if (parent.count < 2) return {-1, 0};
        const auto& allowed = proxy_features_for_(ProxyLoopKind::Greedy);
        const auto permitted = [&](int feature) {
            return !restricted || allowed.empty() || std::binary_search(allowed.begin(), allowed.end(), feature);
        };
        logistic_gradients_.resize(n_samples);
        logistic_hessians_.resize(n_samples);
        long double total_gradient = 0, total_hessian = 0;
        auto& rows = logistic_active_rows_;
        rows.clear();
        rows.reserve(parent.count);
        const bool factored = !logistic_offset_exponentials_.empty();
        const double positive_factor = factored ? std::exp(parent.correction) : 0;
        const double negative_factor = factored ? std::exp(-parent.correction) : 0;
        const double common_logit = logistic_offset_ + parent.correction;
        const double common_negative_gradient = logistic_sigmoid_(common_logit);
        const double common_positive_gradient = -logistic_sigmoid_(-common_logit);
        const double common_e = std::exp(-std::abs(common_logit));
        const double common_hessian = common_e / ((1 + common_e) * (1 + common_e));
        for (int word = 0; word < n_words; ++word) {
            uint64_t bits = mask.w[word];
            while (bits) {
                const int row = word * 64 + std::countr_zero(bits);
                rows.push_back(row);
                const double logit = (logistic_constant_offset_ ? logistic_offset_ : logistic_offsets_[row]) + parent.correction;
                const bool positive = factored ? logistic_offsets_[row] >= 0 : logit >= 0;
                const double e = logistic_constant_offset_ ? common_e : factored
                    ? logistic_offset_exponentials_[row] * (positive ? negative_factor : positive_factor)
                    : std::exp(-std::abs(logit));
                const double inverse = 1 / (1 + e);
                const double g = logistic_constant_offset_
                    ? (y_train[row] == 0 ? common_negative_gradient : common_positive_gradient)
                    : y_train[row] == 0 ? (positive ? inverse : e * inverse)
                                       : -(positive ? e * inverse : inverse);
                const double h = logistic_constant_offset_ ? common_hessian : e * inverse * inverse;
                logistic_gradients_[row] = g;
                logistic_hessians_[row] = h;
                total_gradient += g;
                total_hessian += h;
                bits &= bits - 1;
            }
        }
        const auto quadratic = [&](long double gradient, long double hessian) {
            const long double low = -logistic_clip_ - parent.correction;
            const long double high = logistic_clip_ - parent.correction;
            const long double delta = hessian > 0 ? std::clamp(-gradient / hessian, low, high)
                : gradient > 0 ? low : gradient < 0 ? high : 0;
            return gradient * delta + hessian * delta * delta / 2;
        };
        int best_feature = -1;
        long double best_score = std::numeric_limits<long double>::infinity();
        const auto consider = [&](int feature, int count, long double gradient, long double hessian) {
            if (!count || count == parent.count || !permitted(feature)) return;
            const long double score = quadratic(gradient, hessian) +
                quadratic(total_gradient - gradient, std::max(0.0L, total_hessian - hessian));
            if (score < best_score || (score == best_score && (best_feature < 0 || feature < best_feature))) {
                best_score = score;
                best_feature = feature;
            }
        };
        const int binary_end = first_continuous_feature_();
        for (int feature = 0; feature < binary_end; ++feature) {
            if (!permitted(feature)) continue;
            int count = 0;
            long double gradient = 0, hessian = 0;
            for (int word = 0; word < n_words; ++word) {
                uint64_t bits = mask.w[word] & X_bits[feature].w[word];
                count += popcnt64(bits);
                while (bits) {
                    const int row = word * 64 + std::countr_zero(bits);
                    gradient += logistic_gradients_[row];
                    hessian += logistic_hessians_[row];
                    bits &= bits - 1;
                }
            }
            consider(feature, count, gradient, hessian);
        }
        for (int group = 0; group < static_cast<int>(continuous_starts.size()); ++group) {
            const int first = continuous_starts[group], end = continuous_group_end_(group);
            const auto& positions = regression_bucket_positions_(group);
            int count = 0;
            long double gradient = 0, hessian = 0;
            if (end - first > 4LL * parent.count) {
                logistic_sparse_rows_.clear();
                for (int row : rows) if (positions[row] < end - first)
                    logistic_sparse_rows_.emplace_back(first + positions[row], row);
                std::sort(logistic_sparse_rows_.begin(), logistic_sparse_rows_.end());
                std::size_t index = 0;
                while (index < logistic_sparse_rows_.size()) {
                    int feature = logistic_sparse_rows_[index].first;
                    do {
                        const int row = logistic_sparse_rows_[index++].second;
                        ++count;
                        gradient += logistic_gradients_[row];
                        hessian += logistic_hessians_[row];
                    } while (index < logistic_sparse_rows_.size() && logistic_sparse_rows_[index].first == feature);
                    if (restricted && !allowed.empty()) {
                        const auto it = std::lower_bound(allowed.begin(), allowed.end(), feature);
                        const int next = index < logistic_sparse_rows_.size() ? logistic_sparse_rows_[index].first : end;
                        if (it == allowed.end() || *it >= next) continue;
                        feature = *it;
                    }
                    consider(feature, count, gradient, hessian);
                }
            } else {
                logistic_bucket_counts_.assign(end - first + 1, 0);
                logistic_bucket_gradients_.assign(end - first + 1, 0);
                logistic_bucket_hessians_.assign(end - first + 1, 0);
                for (int row : rows) {
                    const int bucket = positions[row];
                    ++logistic_bucket_counts_[bucket];
                    logistic_bucket_gradients_[bucket] += logistic_gradients_[row];
                    logistic_bucket_hessians_[bucket] += logistic_hessians_[row];
                }
                for (int feature = first; feature < end; ++feature) {
                    count += logistic_bucket_counts_[feature - first];
                    gradient += logistic_bucket_gradients_[feature - first];
                    hessian += logistic_bucket_hessians_[feature - first];
                    if (logistic_bucket_counts_[feature - first] || (restricted && !allowed.empty()))
                        consider(feature, count, gradient, hessian);
                }
            }
        }
        return {best_feature, static_cast<double>(best_score)};
    }

    GreedyObjFirstSplit logistic_greedy_(const Packed& mask, int8_t depth, const PathKey& pk, bool restricted) {
        const auto leaf = logistic_leaf_(mask);
        GreedyObjFirstSplit result{leaf.count ? leaf.loss + gamma : 0, -1};
        if (depth <= 0 || resource_limit_reached_()) return result;
        const uint64_t key = key_of_subproblem(mask, pk);
        const K2 cache_key{key, depth};
        if (leaf.count < 2 || logistic_leaf_optimal_(mask, result.obj)) {
            if (proxy_caching_enabled) {
                greedy_cache.insert_or_assign(cache_key, result.obj);
                greedy_first_split_cache.insert_or_assign(cache_key, result);
            }
            if (sparsification_index_enabled_) remember_proxy_split_(key, depth, result.obj, -1, -1);
            return result;
        }
        if (proxy_caching_enabled) {
            const auto objective = greedy_cache.find(cache_key);
            const auto split = greedy_first_split_cache.find(cache_key);
            if (objective != greedy_cache.end() &&
                    (split == greedy_first_split_cache.end() || objective->second < split->second.obj))
                return {objective->second, -1};
            if (split != greedy_first_split_cache.end()) return split->second;
        }
        const auto [feature, score] = logistic_taylor_split_(mask, restricted);
        if (feature >= 0) {
            Packed left(n_words), right(n_words);
            split_threshold_bits_(mask, feature, left, right);
            PathKey pk_left, pk_right;
            const PathKey* left_key = &empty_pk();
            const PathKey* right_key = &empty_pk();
            make_child_pks_if_needed_(feature, pk, left_key, right_key, pk_left, pk_right);
            const Objective objective = logistic_greedy_(left, depth - 1, *left_key, restricted).obj +
                logistic_greedy_(right, depth - 1, *right_key, restricted).obj;
            if (objective < result.obj) result = {objective, feature};
        }
        if (proxy_caching_enabled) {
            greedy_cache.insert_or_assign(cache_key, result.obj);
            greedy_first_split_cache.insert_or_assign(cache_key, result);
        }
        if (sparsification_index_enabled_) remember_proxy_split_(key, depth, result.obj, result.first_feat, -1);
        return result;
    }
#else
public:
    void set_logistic_options(const std::vector<double>&, double = 30, int = 50, double = 1e-10) {
        throw std::invalid_argument("Use the logistic objective module.");
    }
private:
    void initialize_logistic_() {}
    void clear_logistic_state_() {}
    void clear_logistic_leaf_cache_() {}
    double logistic_loss_(int, double) const { return 0; }
    LogisticLeaf logistic_leaf_(const Packed&) const { return {}; }
    double logistic_budget_(double bound, int) const { return bound; }
    Objective logistic_lower_bound_(const Packed&, int) const { return 0; }
    bool logistic_leaf_optimal_(const Packed&, Objective) const { return false; }
    long double logistic_movement_(const Packed&, int, int, Objective) const { return 0; }
    GreedyObjFirstSplit logistic_stump_(const Packed&, const std::vector<int>&, bool) const { return {}; }
    GreedyObjFirstSplit logistic_greedy_(const Packed&, int8_t, const PathKey&, bool) { return {}; }
#endif
