#include <tuple>
#include <type_traits>
#include <utility>

static thread_local bool rid_divisive_interval_active = false;
static constexpr int rid_divisive_interval_sentinel = -2147483000;

struct RIDDivisiveMask {
    std::vector<uint64_t> w;
};

struct RIDDivisiveReplacementState {
    RIDDivisiveMask target_rows;
    RIDDivisiveMask replacement_values;
    bool used = false;
};

struct RIDDivisivePoint {
    int obj = 0;
    int original_mistakes = 0;
    double replacement_mistakes = 0.0;
};

struct RIDDivisiveFrontiers {
    std::vector<RIDDivisivePoint> lower;
    std::vector<RIDDivisivePoint> upper;
    int min_zero_obj = std::numeric_limits<int>::max();
};

static inline RIDDivisiveMask rid_divisive_empty_mask(int n_words) {
    RIDDivisiveMask out;
    out.w.assign((std::size_t)n_words, 0ULL);
    return out;
}

static inline int rid_divisive_count(const RIDDivisiveMask& mask) {
    int total = 0;
    for (uint64_t word : mask.w) {
#if defined(_MSC_VER)
        total += (int)__popcnt64(word);
#else
        total += __builtin_popcountll(word);
#endif
    }
    return total;
}

static inline int rid_divisive_wrong_count(
    const RIDDivisiveMask& mask,
    int prediction,
    const std::vector<RIDDivisiveMask>& y_bits
) {
    if (prediction < 0 || prediction >= (int)y_bits.size()) {
        throw std::runtime_error(
            "Divisive model reliance saw an invalid leaf prediction."
        );
    }

    int total = 0;
    const auto& correct = y_bits[(std::size_t)prediction].w;

    for (std::size_t i = 0; i < mask.w.size(); ++i) {
        const uint64_t wrong = mask.w[i] & ~correct[i];
#if defined(_MSC_VER)
        total += (int)__popcnt64(wrong);
#else
        total += __builtin_popcountll(wrong);
#endif
    }

    return total;
}

static inline RIDDivisiveMask rid_divisive_and(
    const RIDDivisiveMask& a,
    const RIDDivisiveMask& b
) {
    RIDDivisiveMask out = rid_divisive_empty_mask((int)a.w.size());
    for (std::size_t i = 0; i < a.w.size(); ++i) {
        out.w[i] = a.w[i] & b.w[i];
    }
    return out;
}

static inline RIDDivisiveMask rid_divisive_andnot(
    const RIDDivisiveMask& a,
    const RIDDivisiveMask& b
) {
    RIDDivisiveMask out = rid_divisive_empty_mask((int)a.w.size());
    for (std::size_t i = 0; i < a.w.size(); ++i) {
        out.w[i] = a.w[i] & ~b.w[i];
    }
    return out;
}

static inline double rid_divisive_replacement_mistakes(
    const RIDDivisiveReplacementState& state,
    int variable,
    int prediction,
    int original_mistakes,
    int n,
    const std::vector<RIDDivisiveMask>& y_bits,
    const std::vector<std::vector<int>>& matched_group_of_row_by_variable,
    const std::vector<std::vector<int>>& matched_group_size_by_variable
) {
    if (!state.used) {
        return (double)original_mistakes;
    }

    const int wrong_target =
        rid_divisive_wrong_count(state.target_rows, prediction, y_bits);

    if (matched_group_of_row_by_variable.empty()) {
        return
            (double)wrong_target *
            (double)rid_divisive_count(state.replacement_values) /
            (double)n;
    }

    const auto& group_of_row =
        matched_group_of_row_by_variable[(std::size_t)variable];
    const auto& group_sizes =
        matched_group_size_by_variable[(std::size_t)variable];

    std::vector<int> wrong_counts(group_sizes.size(), 0);
    std::vector<int> replacement_counts(group_sizes.size(), 0);

    for (int row = 0; row < n; ++row) {
        const std::size_t wi = (std::size_t)(row >> 6);
        const uint64_t bit = 1ULL << (row & 63);
        const int group = group_of_row[(std::size_t)row];

        if (state.replacement_values.w[wi] & bit) {
            ++replacement_counts[(std::size_t)group];
        }

        if (
            (state.target_rows.w[wi] & bit) &&
            !(y_bits[(std::size_t)prediction].w[wi] & bit)
        ) {
            ++wrong_counts[(std::size_t)group];
        }
    }

    double total = 0.0;

    for (std::size_t group = 0; group < group_sizes.size(); ++group) {
        const int size = group_sizes[group];
        if (size <= 0) continue;
        total +=
            (double)wrong_counts[group] *
            (double)replacement_counts[group] /
            (double)size;
    }

    return total;
}

static std::vector<RIDDivisivePoint> rid_divisive_prune(
    std::vector<RIDDivisivePoint> points,
    bool minimize,
    int n
) {
    if (points.empty()) return points;

    if (minimize) {
        std::sort(
            points.begin(),
            points.end(),
            [](const RIDDivisivePoint& a, const RIDDivisivePoint& b) {
                if (a.obj != b.obj) return a.obj < b.obj;
                if (a.original_mistakes != b.original_mistakes) {
                    return a.original_mistakes > b.original_mistakes;
                }
                return a.replacement_mistakes < b.replacement_mistakes;
            }
        );
    } else {
        std::sort(
            points.begin(),
            points.end(),
            [](const RIDDivisivePoint& a, const RIDDivisivePoint& b) {
                if (a.obj != b.obj) return a.obj < b.obj;
                if (a.original_mistakes != b.original_mistakes) {
                    return a.original_mistakes < b.original_mistakes;
                }
                return a.replacement_mistakes > b.replacement_mistakes;
            }
        );
    }

    const int m = n + 2;
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> bit(
        (std::size_t)m + 1,
        minimize ? inf : -inf
    );

    auto query = [&](int idx) {
        double best = minimize ? inf : -inf;
        for (int i = idx; i > 0; i -= i & -i) {
            if (minimize) {
                best = std::min(best, bit[(std::size_t)i]);
            } else {
                best = std::max(best, bit[(std::size_t)i]);
            }
        }
        return best;
    };

    auto update = [&](int idx, double value) {
        for (int i = idx; i <= m; i += i & -i) {
            if (minimize) {
                bit[(std::size_t)i] =
                    std::min(bit[(std::size_t)i], value);
            } else {
                bit[(std::size_t)i] =
                    std::max(bit[(std::size_t)i], value);
            }
        }
    };

    std::vector<RIDDivisivePoint> out;
    out.reserve(points.size());

    for (const auto& point : points) {
        const int b = point.original_mistakes;
        if (b < 0 || b > n) {
            throw std::runtime_error(
                "Divisive model reliance saw an invalid original loss."
            );
        }

        const int idx = minimize ? n - b + 1 : b + 1;
        const double best = query(idx);
        const bool dominated = minimize
            ? best <= point.replacement_mistakes
            : best >= point.replacement_mistakes;

        if (dominated) continue;

        out.push_back(point);
        update(idx, point.replacement_mistakes);
    }

    return out;
}
