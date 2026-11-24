#pragma once
#include <list>

struct RegressionMoments {
    int count = 0;
    long double mean = 0;
    long double sse = 0;

    void add(double value) {
        const long double delta = value - mean;
        mean += delta / ++count;
        sse += delta * (value - mean);
    }
};

class RegressionKMeans {
    std::vector<int64_t> weights_;
    std::vector<long double> sums_, squares_, previous_, current_;
    std::shared_ptr<const std::vector<long double>> reciprocals_;

    static std::shared_ptr<const std::vector<long double>> reciprocals(int64_t count) {
        static thread_local std::shared_ptr<const std::vector<long double>> cached;
        const std::size_t size = static_cast<std::size_t>(std::min<int64_t>(count, 65536)) + 1;
        if (!cached || cached->size() < size) {
            auto values = std::make_shared<std::vector<long double>>(size);
            for (std::size_t i = 1; i < size; ++i) (*values)[i] = 1.0L / i;
            cached = std::move(values);
        }
        return cached;
    }
    std::vector<int> minima_, columns_, reduced_;
    std::vector<double> losses_;
    double unavoidable_ = 0;
    int optimum_ = 0;
    int m_ = 0;

    long double cost(int first, int last) const {
        const int64_t w = weights_[last] - weights_[first];
        const long double s = sums_[last] - sums_[first];
        const long double correction = static_cast<uint64_t>(w) < reciprocals_->size()
            ? s * s * (*reciprocals_)[w] : s * s / w;
        return std::max(0.0L, squares_[last] - squares_[first] - correction);
    }

    long double value(int row, int col) const {
        return col < row ? previous_[col] + cost(col, row)
                         : std::numeric_limits<long double>::infinity();
    }

    void search(int first, int stride, int count, const int* columns, int column_count, int offset) {
        if (!count) return;
        int* reduced = reduced_.data() + offset;
        int size = 0;
        for (int j = 0; j < column_count; ++j) {
            const int col = columns[j];
            while (size && value(first + stride * (size - 1), col)
                    < value(first + stride * (size - 1), reduced[size - 1])) --size;
            if (size < count) reduced[size++] = col;
        }
        search(first + stride, stride * 2, count / 2, reduced, size, offset + size);
        int start = 0;
        for (int i = 0; i < count; i += 2) {
            const int row = first + stride * i;
            const int end = i + 1 < count ? minima_[row + stride] : reduced[size - 1];
            long double best = std::numeric_limits<long double>::infinity();
            int arg = reduced[start];
            int next = start;
            for (int j = start; j < size && reduced[j] <= end; ++j) {
                const long double candidate = value(row, reduced[j]);
                if (candidate < best) {
                    best = candidate;
                    arg = reduced[j];
                    next = j;
                }
            }
            minima_[row] = arg;
            current_[row] = best;
            start = next;
        }
    }

public:
    void reset(std::vector<std::pair<double, int>> points, double unavoidable = 0) {
        unavoidable_ = unavoidable;
        optimum_ = 0;
        losses_.assign(1, 0);
        std::sort(points.begin(), points.end());
        std::size_t unique = 0;
        for (const auto& point : points) {
            if (unique && points[unique - 1].first == point.first) points[unique - 1].second += point.second;
            else points[unique++] = point;
        }
        points.resize(unique);
        const int m = m_ = static_cast<int>(points.size());
        if (!m) return;
        weights_.assign(m + 1, 0);
        sums_.assign(m + 1, 0);
        squares_.assign(m + 1, 0);
        const long double shift = points[m / 2].first;
        for (int j = 0; j < m; ++j) {
            const long double v = points[j].first - shift;
            const long double w = points[j].second;
            weights_[j + 1] = weights_[j] + w;
            sums_[j + 1] = sums_[j] + w * v;
            squares_[j + 1] = squares_[j] + w * v * v;
        }
        reciprocals_ = reciprocals(weights_[m]);
        previous_.assign(m + 1, std::numeric_limits<long double>::infinity());
        current_.resize(m + 1);
        minima_.resize(m + 1);
        columns_.resize(m);
        std::iota(columns_.begin(), columns_.end(), 0);
        reduced_.resize(2 * m + 4);
        previous_[0] = 0;
        for (int j = 1; j <= m; ++j) previous_[j] = cost(0, j);
        losses_.push_back(static_cast<double>(previous_[m]));
    }

    bool extend(int cap, double penalty, const std::function<bool()>& interrupted = {}) {
        cap = std::min(cap, m_);
        if (optimum_) return true;
        const int m = m_;
        for (int k = static_cast<int>(losses_.size()); k <= cap; ++k) {
            if (interrupted && interrupted()) return false;
            search(k, 1, m - k + 1, columns_.data() + k - 1, m - k + 1, 0);
            losses_.push_back(static_cast<double>(current_[m]));
            if (penalty > 0 && previous_[m] - current_[m] <= penalty) {
                optimum_ = k - 1;
                return true;
            }
            previous_.swap(current_);
        }
        return true;
    }

    int size() const { return m_; }
    int optimum() const { return optimum_; }
    double unavoidable() const { return unavoidable_; }
    const std::vector<double>& losses() const { return losses_; }
    std::size_t memory_bytes() const {
        return sizeof(*this) + weights_.capacity() * sizeof(int64_t)
            + (sums_.capacity() + squares_.capacity()
            + previous_.capacity() + current_.capacity()) * sizeof(long double)
            + (minima_.capacity() + columns_.capacity() + reduced_.capacity()) * sizeof(int)
            + losses_.capacity() * sizeof(double);
    }

    double solve(std::vector<std::pair<double, int>> points, int depth,
                 double penalty, double unavoidable = 0,
                 const std::function<bool()>& interrupted = {}) {
        if (points.empty()) return 0;
        reset(std::move(points), unavoidable);
        const int cap = depth >= 31 ? m_ : std::min(m_, 1 << std::max(0, depth));
        if (!penalty && cap == m_) return unavoidable_;
        if (!extend(cap, penalty, interrupted)) return unavoidable + penalty;
        const int chosen = optimum_ ? std::min(cap, optimum_) : cap;
        return unavoidable_ + losses_[chosen] + penalty * chosen;
    }
};

class RegressionKMeansBounds {
    struct Result {
        std::vector<double> losses;
        double unavoidable = 0;
        int count = 0;
        int optimum = 0;
    };
    using Work = std::list<std::pair<uint64_t, RegressionKMeans>>;
    std::unordered_map<uint64_t, Result> results_;
    Work work_;
    std::unordered_map<uint64_t, Work::iterator> working_;
    std::size_t bytes_ = 0;
    double penalty_ = -1;

public:
    RegressionKMeansBounds() = default;
    RegressionKMeansBounds(const RegressionKMeansBounds&) {}
    RegressionKMeansBounds& operator=(const RegressionKMeansBounds&) { clear(); return *this; }

    void clear() {
        results_.clear();
        working_.clear();
        work_.clear();
        bytes_ = 0;
        penalty_ = -1;
    }

    template<class Build>
    double get(uint64_t key, int depth, double penalty, Build&& build,
               const std::function<bool()>& interrupted = {}) {
        if (penalty_ != penalty) {
            clear();
            penalty_ = penalty;
        }
        auto& result = results_[key];
        int cap = depth >= 31 ? std::numeric_limits<int>::max() : 1 << std::max(0, depth);
        if (result.count) {
            cap = std::min(cap, result.count);
            if (!penalty && cap == result.count) return result.unavoidable;
            const int chosen = result.optimum ? std::min(cap, result.optimum) : cap;
            if (chosen < static_cast<int>(result.losses.size())) {
                return result.unavoidable + result.losses[chosen] + penalty * chosen;
            }
        }
        auto found = working_.find(key);
        if (found == working_.end()) {
            std::vector<std::pair<double, int>> points;
            double unavoidable = 0;
            build(points, unavoidable);
            if (points.empty()) return 0;
            work_.emplace_front(key, RegressionKMeans{});
            work_.front().second.reset(std::move(points), unavoidable);
            found = working_.emplace(key, work_.begin()).first;
        } else {
            bytes_ -= found->second->second.memory_bytes();
            work_.splice(work_.begin(), work_, found->second);
        }
        auto& solver = found->second->second;
        cap = std::min(cap, solver.size());
        const bool saturated = !penalty && cap == solver.size();
        const bool complete = saturated || solver.extend(cap, penalty, interrupted);
        bytes_ += solver.memory_bytes();
        if (complete) {
            result.losses = solver.losses();
            result.unavoidable = solver.unavoidable();
            result.count = solver.size();
            result.optimum = solver.optimum();
        }
        const int chosen = result.optimum ? std::min(cap, result.optimum) : cap;
        const double bound = saturated ? solver.unavoidable()
            : complete ? result.unavoidable + result.losses[chosen] + penalty * chosen
                       : solver.unavoidable() + penalty;
        while (bytes_ > 64ULL * 1024 * 1024 && work_.size() > 1) {
            bytes_ -= work_.back().second.memory_bytes();
            working_.erase(work_.back().first);
            work_.pop_back();
        }
        return bound;
    }
};
