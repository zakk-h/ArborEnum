#pragma once
#include <list>
#include <optional>

class RegressionFeatureExtrema {
    std::vector<int> counts_;
    std::vector<long double> sums_, squares_;
    long double shift_ = 0;
    std::vector<double> prefix_min_, prefix_max_, suffix_min_, suffix_max_;

public:
    void build(const std::vector<std::pair<int, double>>& points, int thresholds) {
        const double inf = std::numeric_limits<double>::infinity();
        shift_ = points.empty() ? 0 : points.front().second;
        counts_.assign(thresholds + 2, 0);
        sums_.assign(thresholds + 2, 0);
        squares_.assign(thresholds + 2, 0);
        prefix_min_.assign(thresholds + 3, inf);
        prefix_max_.assign(thresholds + 3, -inf);
        for (const auto& [position, value] : points) {
            const long double centered = static_cast<long double>(value) - shift_;
            ++counts_[position + 1];
            sums_[position + 1] += centered;
            squares_[position + 1] += centered * centered;
            prefix_min_[position + 1] = std::min(prefix_min_[position + 1], value);
            prefix_max_[position + 1] = std::max(prefix_max_[position + 1], value);
        }
        suffix_min_ = prefix_min_;
        suffix_max_ = prefix_max_;
        for (int i = 1; i <= thresholds + 1; ++i) {
            counts_[i] += counts_[i - 1];
            sums_[i] += sums_[i - 1];
            squares_[i] += squares_[i - 1];
            prefix_min_[i] = std::min(prefix_min_[i], prefix_min_[i - 1]);
            prefix_max_[i] = std::max(prefix_max_[i], prefix_max_[i - 1]);
        }
        for (int i = thresholds + 1; i > 0; --i) {
            suffix_min_[i] = std::min(suffix_min_[i], suffix_min_[i + 1]);
            suffix_max_[i] = std::max(suffix_max_[i], suffix_max_[i + 1]);
        }
    }

    std::optional<long double> cutoff_error(int source, int destination, bool source_range, long double cutoff) const {
        if (source == destination) return 0;
        const auto [minimum, maximum] = donor_range(source, destination, source_range);
        if (!std::isfinite(minimum)) return std::numeric_limits<long double>::infinity();
        const int first = std::min(source, destination) + 1;
        const int last = std::max(source, destination) + 1;
        const int count = counts_[last] - counts_[first];
        if (!count) return 0;
        const long double sum = sums_[last] - sums_[first];
        const long double square = squares_[last] - squares_[first];
        const long double low = static_cast<long double>(minimum) - shift_;
        const long double high = static_cast<long double>(maximum) - shift_;
        const long double a = square - 2 * low * sum + count * low * low;
        const long double b = square - 2 * high * sum + count * high * high;
        const long double scale = squares_.back() + std::abs(2 * low * sum)
            + std::abs(2 * high * sum) + count * (low * low + high * high) + 1;
        const long double margin = 64 * std::numeric_limits<long double>::epsilon()
            * counts_.back() * scale;
        const long double lower = std::max(a, b) - margin;
        if (lower >= cutoff) return lower;
        const long double upper = std::max(0.0L, a + b) + margin;
        if (upper < cutoff) return upper;
        return std::nullopt;
    }

    std::pair<double, double> donor_range(int source, int destination, bool source_range = false) const {
        const int index = source_range ? source : destination;
        return destination > source ? std::pair{suffix_min_[index + 2], suffix_max_[index + 2]}
                                    : std::pair{prefix_min_[index + 1], prefix_max_[index + 1]};
    }
};

class RegressionThresholdIndex {
    struct Node {
        int left = 0;
        int right = 0;
        int count = 0;
        long double sum = 0;
        long double square = 0;
    };
    std::vector<Node> nodes_{Node{}};
    std::vector<int> roots_;
    std::vector<double> values_;
    long double shift_ = 0;

    int insert(int old, int low, int high, int position, double value) {
        const int index = static_cast<int>(nodes_.size());
        nodes_.push_back(nodes_[old]);
        ++nodes_[index].count;
        const long double centered = value - shift_;
        nodes_[index].sum += centered;
        nodes_[index].square += centered * centered;
        if (low < high) {
            const int mid = low + (high - low) / 2;
            if (position <= mid) {
                const int child = insert(nodes_[old].left, low, mid, position, value);
                nodes_[index].left = child;
            } else {
                const int child = insert(nodes_[old].right, mid + 1, high, position, value);
                nodes_[index].right = child;
            }
        }
        return index;
    }

    Node difference(int upper, int lower) const {
        Node out;
        out.count = nodes_[upper].count - nodes_[lower].count;
        out.sum = nodes_[upper].sum - nodes_[lower].sum;
        out.square = nodes_[upper].square - nodes_[lower].square;
        return out;
    }

    int extreme(int upper, int lower, bool maximum) const {
        int low = 0;
        int high = static_cast<int>(values_.size()) - 1;
        while (low < high) {
            const int mid = low + (high - low) / 2;
            const int first = maximum ? nodes_[upper].right : nodes_[upper].left;
            const int second = maximum ? nodes_[lower].right : nodes_[lower].left;
            const bool chosen = nodes_[first].count > nodes_[second].count;
            const bool right = chosen ? maximum : !maximum;
            upper = right ? nodes_[upper].right : nodes_[upper].left;
            lower = right ? nodes_[lower].right : nodes_[lower].left;
            if (right) low = mid + 1;
            else high = mid;
        }
        return low;
    }

    Node prefix(int upper, int lower, long double midpoint) const {
        Node out;
        int low = 0;
        int high = static_cast<int>(values_.size()) - 1;
        while (low < high) {
            const int mid = low + (high - low) / 2;
            if (static_cast<long double>(values_[mid]) - shift_ <= midpoint) {
                const Node left = difference(nodes_[upper].left, nodes_[lower].left);
                out.count += left.count;
                out.sum += left.sum;
                upper = nodes_[upper].right;
                lower = nodes_[lower].right;
                low = mid + 1;
            } else {
                upper = nodes_[upper].left;
                lower = nodes_[lower].left;
                high = mid;
            }
        }
        if (static_cast<long double>(values_[low]) - shift_ <= midpoint) {
            const Node leaf = difference(upper, lower);
            out.count += leaf.count;
            out.sum += leaf.sum;
        }
        return out;
    }

public:
    std::size_t memory_bytes() const {
        return sizeof(*this) + nodes_.capacity() * sizeof(Node)
            + roots_.capacity() * sizeof(int) + values_.capacity() * sizeof(double);
    }

    void build(std::vector<std::pair<int, double>> points, int thresholds) {
        values_.reserve(points.size());
        for (const auto& point : points) values_.push_back(point.second);
        std::sort(values_.begin(), values_.end());
        values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
        roots_.assign(thresholds + 2, 0);
        if (values_.empty()) return;
        shift_ = values_[values_.size() / 2];
        nodes_.reserve(1 + points.size() * (2 + std::bit_width(values_.size())));
        std::sort(points.begin(), points.end());
        std::size_t next = 0;
        for (int bucket = 0; bucket <= thresholds; ++bucket) {
            int root = roots_[bucket];
            while (next < points.size() && points[next].first == bucket) {
                const double value = points[next++].second;
                const int rank = static_cast<int>(std::lower_bound(values_.begin(), values_.end(), value) - values_.begin());
                root = insert(root, 0, static_cast<int>(values_.size()) - 1, rank, value);
            }
            roots_[bucket + 1] = root;
        }
    }

    long double error(int source, int destination, bool source_range = false) const {
        if (source == destination || values_.empty()) return 0;
        const int from = roots_[source + 1];
        const int to = roots_[destination + 1];
        const int donor = source_range ? from : to;
        const int donor_upper = destination > source ? roots_.back() : donor;
        const int donor_lower = destination > source ? donor : 0;
        if (nodes_[donor_upper].count == nodes_[donor_lower].count) return std::numeric_limits<long double>::infinity();
        const long double low = static_cast<long double>(values_[extreme(donor_upper, donor_lower, false)]) - shift_;
        const long double high = static_cast<long double>(values_[extreme(donor_upper, donor_lower, true)]) - shift_;
        const int moved_upper = destination > source ? to : from;
        const int moved_lower = destination > source ? from : to;
        const Node all = difference(moved_upper, moved_lower);
        const Node lower = prefix(moved_upper, moved_lower, (low + high) / 2);
        return std::max(0.0L, all.square + high * high * lower.count
            + low * low * (all.count - lower.count)
            - 2 * (high * lower.sum + low * (all.sum - lower.sum)));
    }
};

template<class Key, class Hash>
class RegressionThresholdCache {
    using Entries = std::list<std::pair<Key, RegressionThresholdIndex>>;
    Entries entries_;
    std::unordered_map<Key, typename Entries::iterator, Hash> lookup_;
    std::size_t bytes_ = 0;
    std::size_t limit_;
    std::size_t peak_bytes_ = 0;
    uint64_t hits_ = 0, misses_ = 0, evictions_ = 0;

public:
    explicit RegressionThresholdCache(std::size_t limit = 128ULL * 1024 * 1024) : limit_(limit) {}
    RegressionThresholdCache(const RegressionThresholdCache& other) : limit_(other.limit_) {}
    RegressionThresholdCache& operator=(const RegressionThresholdCache& other) {
        if (this != &other) {
            clear();
            limit_ = other.limit_;
        }
        return *this;
    }
    RegressionThresholdCache(RegressionThresholdCache&&) = default;
    RegressionThresholdCache& operator=(RegressionThresholdCache&&) = default;

    template<class Build>
    const RegressionThresholdIndex& get(const Key& key, Build&& build) {
        if (auto found = lookup_.find(key); found != lookup_.end()) {
            ++hits_;
            entries_.splice(entries_.begin(), entries_, found->second);
            return found->second->second;
        }
        ++misses_;
        RegressionThresholdIndex index;
        build(index);
        bytes_ += index.memory_bytes();
        peak_bytes_ = std::max(peak_bytes_, bytes_);
        entries_.emplace_front(key, std::move(index));
        lookup_.emplace(key, entries_.begin());
        while (bytes_ > limit_ && entries_.size() > 1) {
            bytes_ -= entries_.back().second.memory_bytes();
            lookup_.erase(entries_.back().first);
            entries_.pop_back();
            ++evictions_;
        }
        return entries_.front().second;
    }

    void set_limit(std::size_t limit) { clear(); limit_ = limit; }
    std::size_t limit() const { return limit_; }
    std::size_t bytes() const { return bytes_; }
    std::size_t peak_bytes() const { return peak_bytes_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t evictions() const { return evictions_; }

    void clear() {
        lookup_.clear();
        entries_.clear();
        bytes_ = 0;
        peak_bytes_ = 0;
        hits_ = misses_ = evictions_ = 0;
    }
};
