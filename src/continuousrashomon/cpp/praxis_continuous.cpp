#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <cstdint>
#include <deque>
#include <map>

using namespace std;

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int popcnt64(uint64_t x) {
      return static_cast<int>(__popcnt64(x));
  }
#else
  static inline int popcnt64(uint64_t x) {
      return __builtin_popcountll(x);
  }
#endif

#if defined(__AVX512F__)
  #include <immintrin.h>
  #define PRAXIS_USE_AVX512 1
#else
  #define PRAXIS_USE_AVX512 0
#endif

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
  #define PRAXIS_USE_AVX512_POPCNT 1
#else
  #define PRAXIS_USE_AVX512_POPCNT 0
#endif

static inline int popcount_words(const uint64_t* a, int n_words) {
#if PRAXIS_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i pc = _mm512_popcnt_epi64(va);
        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        total += (uint64_t)popcnt64(a[i]);
    }

    return (int)total;
#else
    int total = 0;
    for (int i = 0; i < n_words; ++i) {
        total += popcnt64(a[i]);
    }
    return total;
#endif
}

static inline int popcount_and_words(
    const uint64_t* a,
    const uint64_t* b,
    int n_words
) {
#if PRAXIS_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vc = _mm512_and_si512(va, vb);
        __m512i pc = _mm512_popcnt_epi64(vc);
        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        total += (uint64_t)popcnt64(a[i] & b[i]);
    }

    return (int)total;
#else
    int total = 0;
    for (int i = 0; i < n_words; ++i) {
        total += popcnt64(a[i] & b[i]);
    }
    return total;
#endif
}

static inline int popcount_xor_and_words(
    const uint64_t* mask,
    const uint64_t* a,
    const uint64_t* b,
    int n_words
) {
#if PRAXIS_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i vm = _mm512_loadu_si512((const void*)(mask + i));
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));

        __m512i diff = _mm512_xor_si512(va, vb);
        __m512i active_diff = _mm512_and_si512(vm, diff);
        __m512i pc = _mm512_popcnt_epi64(active_diff);

        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        total += (uint64_t)popcnt64(mask[i] & (a[i] ^ b[i]));
    }

    return (int)total;
#else
    int total = 0;
    for (int i = 0; i < n_words; ++i) {
        total += popcnt64(mask[i] & (a[i] ^ b[i]));
    }
    return total;
#endif
}

static inline void and_words(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    int n_words,
    uint64_t tail_mask
) {
    if (n_words <= 0) return;
#if PRAXIS_USE_AVX512
    int i = 0;
    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vc = _mm512_and_si512(va, vb);
        _mm512_storeu_si512((void*)(out + i), vc);
    }
    for (; i < n_words; ++i) out[i] = a[i] & b[i];
#else
    for (int i = 0; i < n_words; ++i) out[i] = a[i] & b[i];
#endif
    out[n_words - 1] &= tail_mask;
}

static inline void andnot_words(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    int n_words,
    uint64_t tail_mask
) {
    if (n_words <= 0) return;
#if PRAXIS_USE_AVX512
    int i = 0;
    for (; i + 8 <= n_words; i += 8) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));
        __m512i vc = _mm512_andnot_si512(vb, va); // ~b & a
        _mm512_storeu_si512((void*)(out + i), vc);
    }
    for (; i < n_words; ++i) out[i] = a[i] & ~b[i];
#else
    for (int i = 0; i < n_words; ++i) out[i] = a[i] & ~b[i];
#endif
    out[n_words - 1] &= tail_mask;
}

static inline int popcount_and_make_split_words(
    const uint64_t* mask,
    const uint64_t* split,
    uint64_t* left,
    uint64_t* right,
    int n_words,
    uint64_t tail_mask
) {
    if (n_words <= 0) return 0;
#if PRAXIS_USE_AVX512_POPCNT
    int i = 0;
    __m512i acc = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i vm = _mm512_loadu_si512((const void*)(mask + i));
        __m512i vs = _mm512_loadu_si512((const void*)(split + i));

        __m512i vl = _mm512_and_si512(vm, vs);
        __m512i vr = _mm512_andnot_si512(vs, vm); // ~split & mask

        _mm512_storeu_si512((void*)(left + i), vl);
        _mm512_storeu_si512((void*)(right + i), vr);

        __m512i pc = _mm512_popcnt_epi64(vl);
        acc = _mm512_add_epi64(acc, pc);
    }

    alignas(64) uint64_t tmp[8];
    _mm512_store_si512((void*)tmp, acc);

    uint64_t total =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] +
        tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < n_words; ++i) {
        const uint64_t l = mask[i] & split[i];
        const uint64_t r = mask[i] & ~split[i];
        left[i] = l;
        right[i] = r;
        total += (uint64_t)popcnt64(l);
    }

    left[n_words - 1] &= tail_mask;
    right[n_words - 1] &= tail_mask;

    return (int)total;
#else
    int total = 0;
    for (int i = 0; i < n_words; ++i) {
        const uint64_t l = mask[i] & split[i];
        const uint64_t r = mask[i] & ~split[i];
        left[i] = l;
        right[i] = r;
        total += popcnt64(l);
    }
    left[n_words - 1] &= tail_mask;
    right[n_words - 1] &= tail_mask;
    return total;
#endif
}

static inline bool any_words(const uint64_t* a, int n_words) {
    if (n_words <= 0) return false;

#if PRAXIS_USE_AVX512
    int i = 0;
    __m512i accum = _mm512_setzero_si512();

    for (; i + 8 <= n_words; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(a + i));
        accum = _mm512_or_si512(accum, v);
    }

    if (_mm512_test_epi64_mask(accum, accum) != 0) {
        return true;
    }

    for (; i < n_words; ++i) {
        if (a[i]) return true;
    }

    return false;
#else
    for (int i = 0; i < n_words; ++i) {
        if (a[i]) return true;
    }
    return false;
#endif
}

using Lit = uint32_t; // 32-bit literal = 2*feat + sign
using PathKey = std::vector<Lit>;

struct ContinuousPathEntry {
    int threshold_index = -1; // actual binarized threshold-column index
    bool went_true = false; // true means left branch: x <= threshold
};

using ContinuousPath = std::vector<ContinuousPathEntry>;

static inline const ContinuousPath& empty_continuous_path() {
    static const ContinuousPath p;
    return p;
}

struct Packed {
    vector<uint64_t> w; // words (64-bit each)
    Packed() = default;
    explicit Packed(size_t nwords) : w(nwords, 0ULL) {} // allocates a vector of nwords many 64-bit words, with all bits off

    inline void clear() {
        if (!w.empty()) {
            std::memset(w.data(), 0, w.size() * sizeof(uint64_t));
        }
    }

    inline bool any() const {
        return any_words(w.data(), (int)w.size());
    }

    inline int count() const {
        return popcount_words(w.data(), (int)w.size());
    }

};

// scramble a 64-bit value
static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

// hash the array of words into a 64-bit hash value: many-to-one in theory, but with our expected amount of pruning, something like 54k total keys for a reasonably sized rashomon set computation, which yields something like 10^-11 probability of having a collision somewhere.
// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
static inline uint64_t hash_mask64(const uint64_t* w, int n_words, uint64_t tail_mask) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < n_words; ++i) {
        uint64_t x = w[i];
        if (i == n_words - 1) x &= tail_mask;
        uint64_t m = mix64(x);
        h ^= m + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// 2*feat + sign
static inline Lit enc_lit(int feat, int sign01) {
    if (feat < 0) {
        throw std::runtime_error("enc_lit got negative feature index.");
    }

    const uint64_t lit =
        (static_cast<uint64_t>(feat) << 1) |
        static_cast<uint64_t>(sign01 & 1);

    if (lit > static_cast<uint64_t>(std::numeric_limits<Lit>::max())) {
        throw std::runtime_error("feature index too large for Lit encoding.");
    }

    return static_cast<Lit>(lit);
}

static inline const PathKey& empty_pk() {
    static const PathKey k;
    return k;
}

// insert literal into PathKey, maintaining sorted canonical order
static inline void pk_insert_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.insert(it, lit);
}

// remove literal from PathKey (must exist)
static inline void pk_erase_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.erase(it);
}

struct PackedPredMulti {
    std::vector<Packed> by_class; // size = num_classes, each is n_words over eval rows
};

// bucket by objective
struct ObjBucketMulti {
    int obj;
    std::vector<PackedPredMulti> preds; // one entry per tree at this objective
};

struct PredPackWithObj {
    int obj;
    PackedPredMulti pred;
};

// used for exact, non-probabilistic keyks at the expense of more memory. we intern the exact bytes of a mask/bitvector and assign a small integer ID.
// first unique mask id 0, second unique mask id 1 and so on.
class MaskIdTable {
public:
    uint32_t intern(const Packed& mask, int n_words, uint64_t tail_mask) {
        const size_t bytes = (size_t)n_words * sizeof(uint64_t); // constant across the dataset, how many words needed * 64 bit length
        string key;
        key.resize(bytes);
        // uint64_t* out = reinterpret_cast<uint64_t*>(&key[0]); // pointer to the start of key
        for (int i = 0; i < n_words; ++i) {
            uint64_t x = mask.w[i];
            if (i == n_words - 1) x &= tail_mask; // the last word may have padding bits, tail_mask zeroes out the unused bits.
            // out[i] = x; // the byte representation of mask.w - we need to convert to use as a key in the unordered map
            std::memcpy(&key[i * sizeof(uint64_t)], &x, sizeof(uint64_t));
        }
        auto it = table.find(key); // have we seen this bitmask before?
        if (it != table.end()) return it->second; // return the previously assigned id if it points to the entry, meaning we have it already
        uint32_t id = (uint32_t)pool_size++; // use the value, then increment it
        table.emplace(std::move(key), id); // store without copying
        return id;
    }

    size_t size() const { return pool_size; }

private:
    unordered_map<string, uint32_t> table;
    size_t pool_size = 0;
};

class LitIdTable {
public:
    uint32_t intern(const std::vector<Lit>& lits) {
        const size_t bytes = lits.size() * sizeof(Lit);
        std::string key;
        key.resize(bytes);
        if (bytes) std::memcpy(&key[0], lits.data(), bytes);

        auto it = table.find(key);
        if (it != table.end()) return it->second;
        uint32_t id = (uint32_t)pool_size++;
        table.emplace(std::move(key), id);
        return id;
    }


    size_t size() const { return pool_size; }

private:
    std::unordered_map<std::string, uint32_t> table;
    size_t pool_size = 0;
};


// two structures to define the key type used in hash maps
// K2: for greedy and lickety cache (subproblem, depth)
// K3: tries (subproblem, depth, budget)
// define equality with operator== and how to hash the keys for unordered_map

struct K2 {
    uint64_t k; // hash or interned-id
    int depth;
    bool operator==(const K2& o) const { return k == o.k && depth == o.depth; } // element-wise equality
    struct Hash { // custom hash
        size_t operator()(const K2& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct K3 {
    uint64_t k; // hash or interned-id
    int depth;
    int budget;
    bool operator==(const K3& o) const { return k == o.k && depth == o.depth && budget == o.budget; }
    struct Hash {
        size_t operator()(const K3& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t b = (size_t)x.budget;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= b + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct KLA {
    uint64_t k;
    int depth;
    int la; // lookahead used for this call
    bool operator==(const KLA& o) const { return k == o.k && depth == o.depth && la == o.la; }
    struct Hash {
        size_t operator()(const KLA& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t a = (size_t)x.la;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= a + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct HistEntry {
    int obj;
    uint64_t cnt;
};
static inline bool hist_less(const HistEntry& a, const HistEntry& b){ return a.obj < b.obj; } // helper for sorting

struct TreeTrieNode; // fwd

struct LeafNode {
    int prediction; // 0/1 (kept for completeness)
    int loss;       // gamma + miscls
};

struct SplitNode {
    int feature = -1;
    shared_ptr<TreeTrieNode> left;
    shared_ptr<TreeTrieNode> right;
    uint64_t num_valid_trees = 0; // trees contributed by this split under parent's budget
};

struct TreeTrieNode {
    int budget = 0;
    int min_objective = numeric_limits<int>::max();
    vector<LeafNode> leaves; // (prediction,loss) for as many [<=2 in binary classification] if within budget
    vector<SplitNode> splits; // stores splitnodes which have the feature they split on and left and right trienodes
    vector<HistEntry> hist; // sorted ascending by obj; counts aggregated (obj, count) are elements
    bool hist_built = false; // wait until the end to build the histograms because we don't know if they'll be used in the final trie because of multipass

    uint64_t count_trees() const {
        ensure_hist_built();
        uint64_t s = 0;
        for (const auto& e : hist) s += e.cnt;
        return s;
    }

    uint64_t count_leq(int objective) const {
        ensure_hist_built();
        if (hist.empty()) return 0ULL;
        uint64_t total = 0ULL;
        for (const auto& e : hist) {
            if (e.obj > objective) break; // hist is sorted ascending by obj
            total += e.cnt;
        }
        return total;
    }

    void add_hist(int obj, uint64_t add_cnt = 1) {
        auto it = lower_bound(hist.begin(), hist.end(), HistEntry{obj,0}, hist_less); // find the first poisition in hist where obj could be inserted without breaking sort order
        if (it != hist.end() && it->obj == obj) it->cnt += add_cnt; // if it already exists, just increment
        else hist.insert(it, HistEntry{obj, add_cnt}); // otherwise, add
        if (obj < min_objective) min_objective = obj; // keep min_objective fresh
    }
    
    void add_leaf(int prediction, int loss) {
        leaves.push_back(LeafNode{prediction, loss});
        if (loss < min_objective) min_objective = loss;
    }
    
    void add_leaf_and_build(int prediction, int loss) { // assumes you call within budget
        leaves.push_back(LeafNode{prediction, loss});
        add_hist(loss, 1);
    }

    void add_split(int feat,
               const shared_ptr<TreeTrieNode>& L,
               const shared_ptr<TreeTrieNode>& R) {
        SplitNode s;
        s.feature = feat;
        s.left  = L;
        s.right = R;
        s.num_valid_trees = 0; // will be filled in post-processing

        if (L && R) {
            int min_sum = (L->min_objective == numeric_limits<int>::max() ||
                        R->min_objective == numeric_limits<int>::max())
                        ? numeric_limits<int>::max()
                        : (L->min_objective + R->min_objective);
            if (min_sum < min_objective) min_objective = min_sum;
        }
        splits.push_back(std::move(s));
    }

    void add_split_and_build(int feat,
                   const shared_ptr<TreeTrieNode>& L,
                   const shared_ptr<TreeTrieNode>& R) {
        SplitNode s;
        s.feature = feat;
        s.left  = L;
        s.right = R;

        int min_sum = L->min_objective + R->min_objective;
        if (min_sum < min_objective)
            min_objective = min_sum;

        if ((L && !L->hist.empty()) && (R && !R->hist.empty())) {
            unordered_map<int, uint64_t> sum_counts; // make a temporary map to map obj -> count until we know how they distribute in full to then transfer to the vector-based histogram
            sum_counts.reserve(L->hist.size() * 2); // 2x is a good starting estimate

            uint64_t valid = 0;
            vector<int> R_objs; R_objs.reserve(R->hist.size()); // split (obj, cnt) into two parallel ararys, just for R due to the binary search needs in the future
            vector<uint64_t> R_cnts; R_cnts.reserve(R->hist.size());
            for (auto &e : R->hist) { R_objs.push_back(e.obj); R_cnts.push_back(e.cnt); }

            for (const auto& le : L->hist) {
                if (le.obj > budget) break; // should never happen by invariant but if we do lossy caching/more heuristics
                int rem = budget - le.obj; // R cannot exceed
                auto it_end = upper_bound(R_objs.begin(), R_objs.end(), rem); // find the first index strictly greater than rim
                int idx_end = (int)distance(R_objs.begin(), it_end); // gets the index of it_end (it_end is an iterator)
                for (int j = 0; j < idx_end; ++j) { // go until the last index that doesn't exceed rem
                    int tot = le.obj + R_objs[j];
                    uint64_t addc = le.cnt * R_cnts[j];
                    sum_counts[tot] += addc;
                    valid += addc;
                }
            }
            // we've updated our temporary sum_counts map, now we must merge it into the existing histogram
            if (!sum_counts.empty()) {
                vector<HistEntry> tmp; tmp.reserve(sum_counts.size());
                for (auto &kv : sum_counts) tmp.push_back(HistEntry{kv.first, kv.second}); // back the (obj, count) format and sorting
                sort(tmp.begin(), tmp.end(), hist_less);

                // now, we have to aggregate this into the histogram for all splits at that node
                vector<HistEntry> merged; merged.reserve(hist.size() + tmp.size());
                // simply merge two sorted lists into a new list and swap it in
                size_t i=0, j=0;
                while (i < hist.size() && j < tmp.size()) {
                    if (hist[i].obj < tmp[j].obj) merged.push_back(hist[i++]);
                    else if (tmp[j].obj < hist[i].obj) merged.push_back(tmp[j++]);
                    else { merged.push_back(HistEntry{hist[i].obj, hist[i].cnt + tmp[j].cnt}); ++i; ++j; }
                }
                while (i < hist.size()) merged.push_back(hist[i++]);
                while (j < tmp.size()) merged.push_back(tmp[j++]);
                hist.swap(merged);
            }
            s.num_valid_trees = valid;
        }

        splits.push_back(std::move(s)); // adding this split information to the trienode
    }

    // post-process the trie to build per-node histograms using the existing helpers.
    // assumes leaves/splits/min_objective/budget are already set by construct_trie.
    static void build_histograms_post(TreeTrieNode* node) {
        if (node->hist_built) return;

        // ensure children are processed first (post-order)
        for (auto &s : node->splits) {
            if (s.left)  build_histograms_post(s.left.get());
            if (s.right) build_histograms_post(s.right.get());
        }

        // rebuild this node's histogram from scratch
        std::vector<SplitNode> saved = std::move(node->splits);
        node->splits.clear();
        node->hist.clear();
        node->hist_built = false; // (will set true at end)

        // add leaf contributions
        for (const auto &leaf : node->leaves) {
            node->add_hist(leaf.loss, 1); // could call add_leaf_and_build but that is overkill here
        }

        // re-add splits, letting add_split_and_build do the heavy lifting:
        // merges L/R histograms into node->hist
        // computes s.num_valid_trees
        // refreshes min_objective though that isn't needed
        for (auto &s : saved) {
            node->add_split_and_build(s.feature, s.left, s.right);
        }

        node->hist_built = true;
    }

    void ensure_hist_built() const {
        if (!hist_built) {
            TreeTrieNode::build_histograms_post(const_cast<TreeTrieNode*>(this));
        }
    }

};

struct PredNode {
    int feature;  // -1 for leaf
    int prediction; // only meaningful if feature == -1
    shared_ptr<PredNode> left;
    shared_ptr<PredNode> right;
};

// for joint rashomon set prediction / rid
// struct ObjBucket {
//     int obj;
//     std::vector<Packed> preds; // each is a prediction bitvector for one tree at this obj. predictions for all trees with an objective.
// };

struct EvalCtx {
    int n_eval = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL;
    std::vector<Packed> X_bits_eval; // everything needed for evaluation dataset
};


class PRAXIS {
public:
    enum class KeyMode { HASH64, EXACT, LITS_EXACT };

    enum class GreedyContinuousMode {
        BINARY = 0,
        NUMERICAL = 1
    };

private:
    int n_samples = 0;
    int n_features = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL; // to clear high bits in last word
    int gamma = 0;
    int8_t trained_depth_budget = -1; 

    int best_objective = 0;
    int obj_bound = 0;

    double multiplicative_slack = 0.0;

    vector<Packed> X_bits; // vector of Packed, each Packed is a feature column. packed is a sequence of 64-bit words where each bit corresponds to the row value for the column
    // Packed Ypos; // each bit of a word is the label for the row
    int num_classes = 0;
    std::vector<Packed> Y_bits; // vector is size size num_classes; Y_bits[c] has 1s where y==c
    std::vector<int> continuous_starts;

    bool has_prepared_data = false;

    std::vector<std::vector<bool>> prepared_X_col_major;
    std::vector<int> prepared_y;
    std::vector<int> prepared_continuous_starts;
    std::vector<int> prepared_allowed_proxy_features;

    KeyMode key_mode = KeyMode::HASH64; // will change later in fit
    GreedyContinuousMode greedy_continuous_mode = GreedyContinuousMode::BINARY;
    bool trie_cache_enabled = false;
    bool proxy_caching_enabled = true;
    mutable MaskIdTable mask_ids; // used only if in exact mode
    mutable LitIdTable lit_ids; // for itemset mode

    int lookahead_init = 1; // will be changed later
    bool use_multipass = true; // sim
    bool rule_list_mode = false;
    bool majority_leaf_only = false;
    bool cache_cheap_subproblems = false;
    bool evaluated_use_min_objectives = false;
    int greedy_split_mode = 1;
    // int num_proxy_features = -1; // <=0 means use all feature. positive for feature selection
    
    // empty means unrestricted / all features.
    // non empty means these exact feature indices are allowed when the relevant boolean is on.
    std::vector<int> allowed_proxy_features;
    bool restrict_proxy_in_lickety = false;
    bool restrict_proxy_in_depthd_exact = false;
    bool restrict_proxy_in_greedy = false;
    
    int proxy_style = 0; // 0=constant k (current), 1=cyclic (recursively applying split), 2=cyclic-consistent, 3=split without postprocessing, 4=split with postprocessing
    std::vector<int> k_at_depth; // size depth_budget (only used for style 2)

    unordered_map<K2, int, K2::Hash> greedy_cache;
    unordered_map<K2,  int, K2::Hash>  lickety_cache_k2; // used when lookahead_init <= 1
    unordered_map<KLA, int, KLA::Hash>  lickety_cache_kla; // used when lookahead_init > 1

    unordered_map<K3, shared_ptr<TreeTrieNode>, K3::Hash> trie_cache; // if trie_cache_enabled is on

    std::vector<std::vector<double>> numerical_X_cols_for_greedy;
    std::vector<std::vector<int>> numerical_global_sorted_idx;
    std::vector<std::vector<double>> numerical_unique_values_for_greedy;
    std::vector<int> y_train;

    // bitwise and of the bitvectors represented as lists of words. makes a sparser list of words
    inline void and_bits(const Packed& a, const Packed& b, Packed& out) const {
        and_words(a.w.data(), b.w.data(), out.w.data(), n_words, tail_mask);
    }

    inline void andnot_bits(const Packed& a, const Packed& b, Packed& out) const {
        andnot_words(a.w.data(), b.w.data(), out.w.data(), n_words, tail_mask);
    }

    inline int popcount_and(const Packed& a, const Packed& b) const {
        return popcount_and_words(a.w.data(), b.w.data(), n_words);
    }

    inline uint64_t key_of_mask(const Packed& mask) const {
        if (key_mode == KeyMode::LITS_EXACT) {
            throw std::runtime_error("key_of_mask called in LITS_EXACT mode; use key_of_state(mask, pk)");
        }
        return key_of_state(mask, empty_pk());
    }



    inline uint64_t key_of_state(const Packed& mask, const PathKey& pk) const {
        switch (key_mode) {
            case KeyMode::HASH64:
                return hash_mask64(mask.w.data(), n_words, tail_mask);
            case KeyMode::EXACT:
                return (uint64_t)mask_ids.intern(mask, n_words, tail_mask); // cast 32->64
            case KeyMode::LITS_EXACT:
                return (uint64_t)lit_ids.intern(pk);
        }
        return 0;
    }

    inline uint64_t key_of_subproblem(const Packed& mask, const PathKey& pk) const {
        if (key_mode == KeyMode::LITS_EXACT) {
            return key_of_state(mask, pk); // interns pk
        } else {
            return key_of_mask(mask); 
        }
    }

    // inline int proxy_feat_count_() const {
    //     return (num_proxy_features > 0) ? std::min(num_proxy_features, n_features) : n_features;
    // }

    enum class ProxyLoopKind {
        Lickety,
        DepthDExact,
        Greedy
    };

    inline bool should_restrict_proxy_features_(ProxyLoopKind kind) const {
        if (allowed_proxy_features.empty()) return false;

        switch (kind) {
            case ProxyLoopKind::Lickety:
                return restrict_proxy_in_lickety;
            case ProxyLoopKind::DepthDExact:
                return restrict_proxy_in_depthd_exact;
            case ProxyLoopKind::Greedy:
                return restrict_proxy_in_greedy;
        }
        return false;
    }

    inline const std::vector<int>& proxy_features_for_(ProxyLoopKind kind) const {
        static const std::vector<int> empty;
        if (should_restrict_proxy_features_(kind)) return allowed_proxy_features;
        return empty; // empty means use normal 0..n_features-1 loop
    }

    inline bool use_restricted_greedy_proxy_() const {
        return !allowed_proxy_features.empty() && restrict_proxy_in_greedy;
    }

    inline bool use_restricted_depthd_exact_proxy_() const {
        return !allowed_proxy_features.empty() && restrict_proxy_in_depthd_exact;
    }

    inline bool use_restricted_lickety_proxy_() const {
        return !allowed_proxy_features.empty() && restrict_proxy_in_lickety;
    }

    // inline int greedy_proxy_objective_(
    //     const Packed& mask,
    //     int8_t depth,
    //     const PathKey& pk
    // ) {
    //     if (use_restricted_greedy_proxy_()) {
    //         return train_greedy(mask, depth, pk);
    //     }
    //     return train_greedy_continuous(mask, depth, pk);
    // }

    int greedy_proxy_objective_(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        // if greedy is feature-restricted, always use the existing binary feature loop.
        if (restrict_proxy_in_greedy) {
            return train_greedy(mask, depth_budget, pk);
        }

        // unrestricted greedy: choose how continuous features are evaluated.
        if (greedy_continuous_mode == GreedyContinuousMode::BINARY) {
            // scan fully binarized threshold columns by continuous group.
            return train_greedy_continuous(mask, depth_budget, pk, cpath);
        }

        // use raw numerical sorted lists for continuous features.
        return greedy_numerical_entry_point(mask, depth_budget, pk);
    }

    inline int depthd_exact_proxy_objective_(
        const Packed& mask,
        int8_t depth,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (use_restricted_depthd_exact_proxy_()) {
            return depthd_exact_solver_cached(mask, depth, pk);
        }
 
        return depthd_exact_solver_cached_continuous(mask, depth, pk, cpath);
    }

    inline int lickety_proxy_objective_(
        const Packed& mask,
        int8_t depth,
        int8_t k,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (use_restricted_lickety_proxy_()) {
            if (proxy_style == 4) {
                return split_algorithm(mask, depth, k, pk);
            }
            return generalized_lickety_split(mask, depth, k, pk);
        }

        if (proxy_style == 4) {
            return split_algorithm(mask, depth, k, pk);
        }
        return generalized_lickety_split_continuous(mask, depth, k, pk, cpath);
    }

    inline bool use_kla_cache() const { return lookahead_init > 1; }

    void reserve_caches_mid_() {
        // default max_load_factor.

        greedy_cache.reserve(4'000'000);

        if (use_kla_cache()) {
            lickety_cache_kla.reserve(2'000'000);
        } else {
            lickety_cache_k2.reserve(2'000'000);
        }

        if (trie_cache_enabled) {
            trie_cache.reserve(512);
        }
    }
        
    inline int count_total(const Packed& mask) const { return mask.count(); } // number of active samples
    // inline int count_pos(const Packed& mask) const { return popcount_and(mask, Ypos); } // number of active samples that are positive

    inline void count_per_class(const Packed& mask, std::vector<int>& counts) const {
        counts.assign((size_t)num_classes, 0);
        for (int c = 0; c < num_classes; ++c) {
            counts[(size_t)c] = popcount_and(mask, Y_bits[(size_t)c]);
        }
    }

    inline int count_class(const Packed& mask, int c) const {
        return popcount_and(mask, Y_bits[(size_t)c]);
    }


    static inline double entropy(double p) {
        const double eps = 1e-12;
        p = max(eps, min(1.0 - eps, p));
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    }

    static inline double entropy_multiclass(const std::vector<int>& cnts, int n) {
        if (n <= 0) return 0.0;
        const double invn = 1.0 / (double)n;
        double H = 0.0;
        for (int c = 0; c < (int)cnts.size(); ++c) {
            const int k = cnts[(size_t)c];
            if (k <= 0) continue;
            const double p = (double)k * invn;
            H -= p * log2(p);
        }
        return H;
    }


    // count the distinct subproblems/bitvectors/literals/fingerprints, not considering depth or greedy/lickety
    size_t count_distinct_subproblems_union() const {
        std::unordered_set<uint64_t> seen;

        const size_t lsz = use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size();
        const size_t approx = greedy_cache.size() + lsz;
        seen.reserve(approx * 2 + 16);

        // greedy: key is (k, depth). we ignore depth by inserting only k.
        for (const auto& kv : greedy_cache) {
            seen.insert(kv.first.k);
        }

        // lickety: either (k, depth) or (k, depth, la). ignore depth/la by inserting only k.
        if (use_kla_cache()) {
            for (const auto& kv : lickety_cache_kla) {
                seen.insert(kv.first.k);
            }
        } else {
            for (const auto& kv : lickety_cache_k2) {
                seen.insert(kv.first.k);
            }
        }

        return seen.size();
    }

    

public:
    shared_ptr<TreeTrieNode> result;

    void set_key_mode(KeyMode m) { key_mode = m; }
    void set_trie_cache_enabled(bool on) { trie_cache_enabled = on; }
    void set_multiplicative_slack(double s) { multiplicative_slack = s; }
    void set_use_multipass(bool on) { use_multipass = on; }
    void set_rule_list_mode(bool on) { rule_list_mode = on; }
    void set_majority_leaf_only(bool on) { majority_leaf_only = on; }
    void set_cache_cheap_subproblems(bool on) { cache_cheap_subproblems = on; }
    void set_greedy_split_mode(int m) { greedy_split_mode = m; }
    void set_proxy_caching_enabled(bool on) { proxy_caching_enabled = on; }
    void set_evaluated_use_min_objectives(bool on) { evaluated_use_min_objectives = on; }

    void set_allowed_proxy_features(const std::vector<int>& feats) {
        allowed_proxy_features.clear();
        allowed_proxy_features.reserve(feats.size());

        for (int f : feats) {
            if (f < 0 || f >= n_features) {
                throw std::runtime_error("allowed_proxy_features contains out-of-range feature index.");
            }
            allowed_proxy_features.push_back(f);
        }

        std::sort(allowed_proxy_features.begin(), allowed_proxy_features.end());
        allowed_proxy_features.erase(
            std::unique(allowed_proxy_features.begin(), allowed_proxy_features.end()),
            allowed_proxy_features.end()
        );
    }

    void set_proxy_feature_restrictions(
        bool use_in_lickety,
        bool use_in_depthd_exact,
        bool use_in_greedy
    ) {
        restrict_proxy_in_lickety = use_in_lickety;
        restrict_proxy_in_depthd_exact = use_in_depthd_exact;
        restrict_proxy_in_greedy = use_in_greedy;
    }

    void set_greedy_continuous_mode(int mode) {
        if (mode == 0) {
            greedy_continuous_mode = GreedyContinuousMode::BINARY;
        } else if (mode == 1) {
            greedy_continuous_mode = GreedyContinuousMode::NUMERICAL;
        } else {
            throw std::runtime_error(
                "set_greedy_continuous_mode expects 0 for BINARY or 1 for NUMERICAL."
            );
        }

        greedy_cache.clear();
    }

    int get_greedy_continuous_mode() const {
        return greedy_continuous_mode == GreedyContinuousMode::BINARY ? 0 : 1;
    }

    void prepare_continuous_data(
        const std::vector<std::vector<double>>& X_num_row_major,
        const std::vector<std::vector<uint8_t>>& X_bin_row_major,
        const std::vector<int>& y,
        const std::vector<std::vector<uint8_t>>& X_active_row_major
    ) {
        validate_rectangular_matrix_(X_num_row_major, "X_num_row_major");
        validate_rectangular_matrix_(X_bin_row_major, "X_bin_row_major");
        validate_rectangular_matrix_(X_active_row_major, "X_active_row_major");

        const int n_num = (int)X_num_row_major.size();
        const int n_bin = (int)X_bin_row_major.size();
        const int n_active = (int)X_active_row_major.size();

        int n = -1;

        if (n_num > 0) n = n_num;
        if (n_bin > 0) {
            if (n < 0) n = n_bin;
            else if (n_bin != n) throw std::runtime_error("X_bin_row_major row count does not match X_num_row_major.");
        }
        if (n_active > 0) {
            if (n < 0) n = n_active;
            else if (n_active != n) throw std::runtime_error("X_active_row_major row count does not match X_num_row_major.");
        }

        if (n < 0) {
            throw std::runtime_error("At least one of X_num_row_major, X_bin_row_major, or X_active_row_major must be nonempty.");
        }

        if ((int)y.size() != n) {
            throw std::runtime_error("y length does not match number of rows.");
        }

        const int p_num = (n_num > 0) ? (int)X_num_row_major[0].size() : 0;
        const int p_bin = (n_bin > 0) ? (int)X_bin_row_major[0].size() : 0;
        const int p_active = (n_active > 0) ? (int)X_active_row_major[0].size() : 0;

        std::vector<std::vector<bool>> X_cols;
        std::vector<int> cont_starts;
        std::vector<std::vector<double>> numerical_cols_for_greedy;
        std::vector<std::vector<int>> numerical_global_sorted_idx_local;
        std::vector<std::vector<double>> numerical_unique_values_for_greedy_local;

        // first append already-binary columns.
        // these are ordinary binary features, not continuous threshold groups.
        X_cols.reserve((std::size_t)(p_bin + p_num * 8));

        for (int f = 0; f < p_bin; ++f) {
            std::vector<bool> col((std::size_t)n, false);

            for (int i = 0; i < n; ++i) {
                const uint8_t v = X_bin_row_major[(std::size_t)i][(std::size_t)f];
                col[(std::size_t)i] = (v != 0);
            }

            X_cols.push_back(std::move(col));
        }

        // fully threshold-binarize each numeric column.
        // each numeric feature becomes one contiguous continuous threshold group.
        for (int f = 0; f < p_num; ++f) {
            std::vector<double> vals = sorted_unique_values_(X_num_row_major, f);

            // constant numeric feature contributes no threshold columns.
            if (vals.size() <= 1) continue;

            const int group_start = (int)X_cols.size();
            cont_starts.push_back(group_start);

            // vals[q] corresponds to threshold column group_start + q,
            // for q = 0, ..., vals.size() - 2.
            // vals.back() has no threshold column because x <= max is trivial.
            numerical_unique_values_for_greedy_local.push_back(vals);

            // store the raw numerical column corresponding to this continuous group.
            std::vector<double> col_num((std::size_t)n);
            for (int i = 0; i < n; ++i) {
                col_num[(std::size_t)i] =
                    X_num_row_major[(std::size_t)i][(std::size_t)f];
            }

            // store globally sorted row indices for this numerical feature.
            std::vector<int> order((std::size_t)n);
            for (int i = 0; i < n; ++i) order[(std::size_t)i] = i;

            std::stable_sort(
                order.begin(),
                order.end(),
                [&](int a, int b) {
                    const double xa = col_num[(std::size_t)a];
                    const double xb = col_num[(std::size_t)b];
                    if (xa < xb) return true;
                    if (xb < xa) return false;
                    return a < b;
                }
            );

            numerical_cols_for_greedy.push_back(std::move(col_num));
            numerical_global_sorted_idx_local.push_back(std::move(order));

            // use every unique value except the last as a <= threshold.
            for (std::size_t t = 0; t + 1 < vals.size(); ++t) {
                const double thr = vals[t];

                std::vector<bool> col((std::size_t)n, false);
                for (int i = 0; i < n; ++i) {
                    col[(std::size_t)i] =
                        X_num_row_major[(std::size_t)i][(std::size_t)f] <= thr;
                }

                X_cols.push_back(std::move(col));
            }
        }

        if (X_cols.empty()) {
            throw std::runtime_error("prepare_continuous_data produced zero binary features.");
        }

        // map active binary columns to closest columns in the full binarized X.
        std::vector<int> active_features;
        active_features.reserve((std::size_t)p_active);

        for (int a = 0; a < p_active; ++a) {
            std::vector<uint8_t> active_col((std::size_t)n, 0);

            for (int i = 0; i < n; ++i) {
                active_col[(std::size_t)i] =
                    X_active_row_major[(std::size_t)i][(std::size_t)a] ? 1 : 0;
            }

            int best_idx = -1;
            int best_dist = std::numeric_limits<int>::max();

            for (int f = 0; f < (int)X_cols.size(); ++f) {
                const int d = hamming_distance_binary_column_(
                    X_cols[(std::size_t)f],
                    active_col
                );

                if (d < best_dist) {
                    best_dist = d;
                    best_idx = f;

                    // Perfect match. Cannot improve.
                    if (best_dist == 0) break;
                }
            }

            if (best_idx < 0) {
                throw std::runtime_error("Failed to map active binary feature to full binarized feature.");
            }

            active_features.push_back(best_idx);
        }

        sort_unique_ints_inplace_(active_features);

        prepared_X_col_major = std::move(X_cols);
        prepared_y = y;
        prepared_continuous_starts = std::move(cont_starts);
        prepared_allowed_proxy_features = std::move(active_features);

        numerical_X_cols_for_greedy = std::move(numerical_cols_for_greedy);
        numerical_global_sorted_idx = std::move(numerical_global_sorted_idx_local);
        numerical_unique_values_for_greedy =
            std::move(numerical_unique_values_for_greedy_local);

        if (numerical_X_cols_for_greedy.size() != prepared_continuous_starts.size() ||
            numerical_global_sorted_idx.size() != prepared_continuous_starts.size() ||
            numerical_unique_values_for_greedy.size() != prepared_continuous_starts.size()) {
            throw std::logic_error(
                "Numerical greedy arrays do not align with prepared_continuous_starts."
            );
        }

        has_prepared_data = true;
    }

    void fit_prepared(
        double lambda,
        int8_t depth_budget,
        double rashomon_mult,
        int8_t lookahead_k,
        int root_budget,
        bool use_multipass_flag,
        bool rule_list_mode_flag,
        int proxy_style_in,
        bool majority_leaf_only_flag,
        bool cache_cheap_subproblems_flag,
        bool proxy_caching_flag,
        bool restrict_proxy_in_lickety_in,
        bool restrict_proxy_in_depthd_exact_in,
        bool restrict_proxy_in_greedy_in,
        bool rashomon_mode
    ) {
        if (!has_prepared_data) {
            throw std::runtime_error("No prepared data. Call prepare_continuous_data(...) before fit_prepared(...).");
        }

        fit(
            prepared_X_col_major,
            prepared_y,
            lambda,
            depth_budget,
            rashomon_mult,
            lookahead_k,
            root_budget,
            use_multipass_flag,
            rule_list_mode_flag,
            proxy_style_in,
            majority_leaf_only_flag,
            cache_cheap_subproblems_flag,
            proxy_caching_flag,
            prepared_allowed_proxy_features,
            restrict_proxy_in_lickety_in,
            restrict_proxy_in_depthd_exact_in,
            restrict_proxy_in_greedy_in,
            rashomon_mode,
            prepared_continuous_starts
        );
    }

    void fit(const std::vector<std::vector<bool>>& X_col_major,
             const std::vector<int>& y,
             double lambda,
             int8_t depth_budget,
             double rashomon_mult,
             int8_t lookahead_k,
             int root_budget,
             bool use_multipass_flag,
             bool rule_list_mode_flag,
             int proxy_style_in,
             bool majority_leaf_only_flag,
             bool cache_cheap_subproblems_flag,
             bool proxy_caching_flag,
             const std::vector<int>& allowed_proxy_features_in,
             bool restrict_proxy_in_lickety_in,
             bool restrict_proxy_in_depthd_exact_in,
             bool restrict_proxy_in_greedy_in,
             bool rashomon_mode,
             const std::vector<int>& continuous_starts_in
            ) {

        clear_fit_state_();

        if (X_col_major.empty()) {
            throw std::runtime_error("X_col_major is empty.");
        }
        if (X_col_major[0].empty()) {
            throw std::runtime_error("X_col_major has zero samples.");
        }

        n_features = (int)X_col_major.size();
        n_samples  = (int)X_col_major[0].size();
        continuous_starts = continuous_starts_in;
        y_train = y;
       
        if (greedy_continuous_mode == GreedyContinuousMode::NUMERICAL) {
            if (numerical_X_cols_for_greedy.size() != continuous_starts.size() ||
                numerical_global_sorted_idx.size() != continuous_starts.size() ||
                numerical_unique_values_for_greedy.size() != continuous_starts.size()) {
                throw std::runtime_error(
                    "Numerical greedy arrays must align one-to-one with continuous_starts."
                );
            }
        }
        n_words = (n_samples + 63) / 64; // 64 -> 1, 65 -> 2
        tail_mask = (n_samples % 64) ? ((1ULL << (n_samples % 64)) - 1ULL) : ~0ULL; // if multiple of 64, all 1s. otherwise, n_samples % 64 1s followed by 0s.
        gamma = (int)llround(lambda * (double)n_samples);
        trained_depth_budget = depth_budget;
        lookahead_init = lookahead_k;
        use_multipass = use_multipass_flag;
        rule_list_mode = rule_list_mode_flag;
        majority_leaf_only = majority_leaf_only_flag;
        cache_cheap_subproblems = cache_cheap_subproblems_flag;
        proxy_style = proxy_style_in;
        proxy_caching_enabled = proxy_caching_flag;
        if (!rashomon_mode) { // force proxy caching on in single-tree mode - required for this codebase
            proxy_caching_enabled = true;
            cache_cheap_subproblems = true;
        } 

        reserve_caches_mid_();

        restrict_proxy_in_lickety = restrict_proxy_in_lickety_in;
        restrict_proxy_in_depthd_exact = restrict_proxy_in_depthd_exact_in;
        restrict_proxy_in_greedy = restrict_proxy_in_greedy_in;

        allowed_proxy_features.clear();
        allowed_proxy_features.reserve(allowed_proxy_features_in.size());

        for (int f : allowed_proxy_features_in) {
            if (f < 0 || f >= n_features) {
                throw std::runtime_error("allowed_proxy_features contains out-of-range feature index.");
            }
            allowed_proxy_features.push_back(f);
        }

        std::sort(allowed_proxy_features.begin(), allowed_proxy_features.end());
        allowed_proxy_features.erase(
            std::unique(allowed_proxy_features.begin(), allowed_proxy_features.end()),
            allowed_proxy_features.end()
        );

        // if (num_proxy_features_in <= 0) num_proxy_features = n_features;
        // else num_proxy_features = std::min(num_proxy_features_in, n_features);

        X_bits.assign(n_features, Packed(n_words)); // length n_features with entries of Packed, initialized to all 0s bits, we will set below.
        for (int f = 0; f < n_features; ++f) {
            auto &col = X_bits[f].w; // reference to the array of 64-bit words for that feature
            for (int i = 0; i < n_samples; ++i) {
                if (X_col_major[f][i]) col[i>>6] |= (1ULL << (i & 63)); // i>>6 integer division by 64 to answer what word are we in. then i & 63 = i % 64 to get index within word. (1ULL << (i & 63)) creates a 64-bit mask with exactly one bit = 1 at the position you need, then do bitwise or with ol[i >> 6] to set the position to 1 if it is not already set.
                // if feature f is true for sample i, set the bit corresponding to row i to true in the packed column by doing bitwise or with the current column and a 64 bit word with exactly one 1.
            }
            col[n_words-1] &= tail_mask; // bitwise and to 0 out invalid
        }

        // so the format is now a contiguous array of 64 bit words. if our remainder is 5 (and we are in word x), then our bit is the 5th least significant one in the word.
        // so increasing row index increases the bit position within a word.
        // last word is padded with 0 (at the front, most significant bits)

        // Ypos = Packed(n_words);
        // for (int i = 0; i < n_samples; ++i) {
        //     if (y[i]) Ypos.w[i>>6] |= (1ULL << (i & 63));
        // }
        // Ypos.w[n_words-1] &= tail_mask;
        int y_max = 0;
        for (int i = 0; i < n_samples; ++i) y_max = std::max(y_max, y[i]);
        num_classes = y_max + 1;

        Y_bits.assign((size_t)num_classes, Packed(n_words));
        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c].clear();
        }

        for (int i = 0; i < n_samples; ++i) {
            const int yi = y[i];
            Y_bits[(size_t)yi].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }
        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
        }

        Packed root(n_words);
        for (int i = 0; i < n_words-1; ++i) root.w[i] = ~0ULL; // not 0, so all 1.
        root.w[n_words-1] = tail_mask; // enforce 0s for out of scope

        const PathKey& root_pk = empty_pk();
        const ContinuousPath& root_cpath = empty_continuous_path();

        // SINGLE TREE SUPPORT - always use all features
        if (!rashomon_mode) {
            int single_obj = 0;
            if (lookahead_init <= 0) {
                single_obj = train_greedy_continuous(root, depth_budget, root_pk, root_cpath);
            } else {
                if (proxy_style == 4) {
                    single_obj = split_algorithm(root, depth_budget, lookahead_init, root_pk);
                } else {
                    single_obj = generalized_lickety_split_continuous(root, depth_budget, lookahead_init, root_pk, root_cpath);
                }
            }
            
            std::cout << "Single-tree objective: " << single_obj
                    << " (" << (double)single_obj / (double)n_samples << ")\n";
            return; // done; do NOT build trie or do rashomon things
        }

        if (root_budget >= 0) {
            // user-specified bound: skip reference solution
            obj_bound = root_budget;
            std::cout << "Objective bound (user-set): " << obj_bound << "\n";

        } else {
            if (lookahead_init <= 0) { // set based on greedy even if our proxy is a leaf
                best_objective = greedy_proxy_objective_(root, depth_budget, root_pk, root_cpath);
            } else {
                if (proxy_style == 4) {
                    best_objective = split_algorithm(root, depth_budget, lookahead_init, root_pk);
                } else {
                    best_objective = lickety_proxy_objective_(root, depth_budget, lookahead_init, root_pk, root_cpath);
                }
            }
            cout << "Best objective: " << best_objective
                << " (" << (double)best_objective / (double)n_samples << ")\n";

            obj_bound = (int)llround(best_objective * (1.0 + rashomon_mult) * (1.0 + multiplicative_slack));

            cout << "Objective bound: " << obj_bound << "\n";
        }

        if (proxy_style == 2) { // we define to be depth_budget+1 size but depth 0 doesn't actually matter
            k_at_depth.assign(depth_budget + 1, 1);
            int K = lookahead_init;
            int kk = K;
            for (int d = depth_budget; d >= 0; --d) {
                k_at_depth[d] = std::min(d, kk);
                kk = (kk > 1) ? (kk - 1) : K; // increment down then wrap
            }
        } else {
            k_at_depth.clear();
        }


        result = construct_trie(root, depth_budget, obj_bound, root_pk, root_cpath);

        // cout << "Found " << result->count_trees() << " trees\n"; // we'll let the user compute this query if they want it because it is somewhat expensive
        cout << "Minimum objective: " << result->min_objective << "\n";
        cout << "Cache sizes - Greedy: " << greedy_cache.size()
            << ", Lickety: " << (use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size())
            << ", Trie: " << trie_cache.size();
        // cout << ", Distinct subproblems (greedy U lickety): " << count_distinct_subproblems_union();
        // if (key_mode == KeyMode::EXACT) {
        //     cout << ", Unique masks: " << mask_ids.size();
        // }
        // if (key_mode == KeyMode::LITS_EXACT) {
        //     cout << ", Unique literal subproblems: " << lit_ids.size();
        // }
        cout << ", Trie cache: " << (trie_cache_enabled ? "ON" : "OFF");
        cout << "\n";
    }

    // predict using the i-th tree in the Rashomon set: X_row_major: binary [n_samples][n_features]
    std::vector<uint8_t> get_predictions(uint64_t tree_index, const std::vector<std::vector<uint8_t>>& X_row_major) const {
        const std::size_t n_samples = X_row_major.size();
        if (n_samples == 0) return {};

        const std::size_t n_features = X_row_major[0].size();
        const int8_t depth_budget = trained_depth_budget;

        if ((int)n_features != this->n_features) {
            throw std::runtime_error("Prediction X has different number of features than training.");
        }

        std::shared_ptr<PredNode> tree;

        // single tree mode
        if (!result) {
            if (tree_index != 0) {
                throw std::runtime_error("Single-tree mode only supports tree_index == 0.");
            }
            if (depth_budget < 0) {
                throw std::runtime_error("trained_depth_budget not set. Call fit() first.");
            }

            Packed root((size_t)n_words);
            for (int i = 0; i < n_words - 1; ++i) root.w[(size_t)i] = ~0ULL;
            root.w[(size_t)(n_words - 1)] = tail_mask;

            const PathKey& root_pk = empty_pk();
            tree = build_best_tree_from_caches(root, depth_budget, root_pk);
        } 
        // standard rashomon mode
        else {
            tree = get_ith_tree(tree_index);
        }

        std::vector<uint8_t> out(n_samples, 0);

        std::vector<int> idx(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i) {
            idx[i] = static_cast<int>(i);
        }

        predict_tree_recursive(tree.get(), X_row_major, out, idx);
        return out;
    }


    // get predictions from all trees in the rashomon set, as a vector of prediction vectors (one per tree).
    std::vector<std::vector<uint8_t>> get_all_predictions(
        const std::vector<std::vector<uint8_t>>& X_row_major
    ) const {
        uint64_t total = result ? result->count_trees() : 0ULL;
        std::vector<std::vector<uint8_t>> all;
        all.reserve(static_cast<std::size_t>(total));
        for (uint64_t i = 0; i < total; ++i) {
            all.push_back(get_predictions(i, X_row_major));
        }
        return all;
    }

    std::pair<std::vector<std::vector<int>>, std::vector<int>>
    get_tree_paths(std::uint64_t tree_index) const {
        // single tree mode
        if (!result) {
            if (tree_index != 0) {
                throw std::out_of_range("Single-tree mode only supports tree_index == 0.");
            }

            // root mask - all samples active
            Packed root(n_words);
            for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
            root.w[n_words - 1] = tail_mask;

            const PathKey& root_pk = empty_pk();
            const int8_t depth_budget = trained_depth_budget;

            auto tree = build_best_tree_from_caches(root, depth_budget, root_pk);

            std::vector<std::vector<int>> paths;
            std::vector<int> preds;
            std::vector<int> current;
            collect_paths(tree.get(), current, paths, preds);
            return {paths, preds};
        }

        // standard rashomon mode
        auto tree = get_ith_tree(tree_index);
        std::vector<std::vector<int>> paths;
        std::vector<int> preds;
        std::vector<int> current;

        collect_paths(tree.get(), current, paths, preds);
        return {paths, preds};
    }


    // for individual decision tree algorithm
    std::pair<std::vector<std::vector<int>>, std::vector<int>>
        get_tree_paths_from_tree(const std::shared_ptr<PredNode>& tree) const {
            if (!tree) {
                throw std::runtime_error("Null tree passed to get_tree_paths_from_tree.");
            }

            std::vector<std::vector<int>> paths;
            std::vector<int> preds;
            std::vector<int> current;

            collect_paths(tree.get(), current, paths, preds);
            return {paths, preds};
        }


    // return (unnormalized_objective, normalized_objective) for the ith tree
    std::pair<int, double> get_ith_tree_objective(std::uint64_t i) const {
        // if (!result) {
        //     throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        // }
        if (!result) {
            if (i != 0) throw std::out_of_range("Single-tree mode only supports i==0.");
            if (!proxy_caching_enabled) {
                throw std::runtime_error("Single-tree objective requires proxy_caching_enabled, or recompute objective.");
            }
            if (trained_depth_budget < 0) {
                throw std::runtime_error("trained_depth_budget not set. Call fit() first.");
            }

            // root mask
            Packed root((size_t)n_words);
            for (int w = 0; w < n_words - 1; ++w) root.w[(size_t)w] = ~0ULL;
            root.w[(size_t)(n_words - 1)] = tail_mask;

            const PathKey& root_pk = empty_pk();
            const int8_t d = trained_depth_budget;

            const uint64_t km = key_of_subproblem(root, root_pk);

            int best = std::numeric_limits<int>::max(); // it suffices to return the minimum among the root caches

            // greedy
            if (auto itg = greedy_cache.find(K2{km, d}); itg != greedy_cache.end())
                best = std::min(best, itg->second);

            // lickety
            if (use_kla_cache()) {
                for (int kk = 0; kk <= (int)(d - 1); ++kk) {
                    auto it = lickety_cache_kla.find(KLA{km, d, kk});
                    if (it != lickety_cache_kla.end()) best = std::min(best, it->second);
                }
            } else {
                if (auto it = lickety_cache_k2.find(K2{km, d}); it != lickety_cache_k2.end())
                    best = std::min(best, it->second);
            }

            if (best == std::numeric_limits<int>::max()) {
                throw std::runtime_error("Root objective not found in caches (greedy/lickety).");
            }

            double normalized = (double)best / (double)n_samples;
            return {best, normalized};
        }

        // count_trees will ensure that the histograms are built at the root and every child node (by building them if they are not yet built)
        std::uint64_t total = result->count_trees();
        if (i >= total) {
            throw std::out_of_range("Tree index out of range in get_ith_tree_objective");
        }

        std::uint64_t cum = 0;
        int target_obj = -1;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                break;
            }
            cum += e.cnt;
        }

        if (target_obj < 0) {
            throw std::runtime_error("Failed to locate objective bucket in get_ith_tree_objective");
        }

        double normalized = (n_samples > 0)
            ? static_cast<double>(target_obj) / static_cast<double>(n_samples)
            : 0.0;

        return {target_obj, normalized};
    }

    // root LicketySPLIT objective with lookahead=1 so the user can compare frontier cuts to the reference solution
    int root_lickety_objective_lookahead1(int depth_budget) {
        if (n_samples == 0) {
            throw std::runtime_error("Model not fitted.");
        }

        Packed root(n_words);
        for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
        root.w[n_words - 1] = tail_mask;

        PathKey root_pk;
        return generalized_lickety_split_continuous(root, depth_budget, /*k=*/1, root_pk);
    }

private:
    template <typename T>
    static void validate_rectangular_matrix_(
        const std::vector<std::vector<T>>& X,
        const std::string& name
    ) {
        if (X.empty()) return;

        const std::size_t p = X[0].size();
        for (std::size_t i = 1; i < X.size(); ++i) {
            if (X[i].size() != p) {
                throw std::runtime_error(name + " must be rectangular.");
            }
        }
    }

    static std::vector<double> sorted_unique_values_(
        const std::vector<std::vector<double>>& X_num_row_major,
        int col
    ) {
        std::vector<double> vals;
        vals.reserve(X_num_row_major.size());

        for (std::size_t i = 0; i < X_num_row_major.size(); ++i) {
            const double v = X_num_row_major[i][(std::size_t)col];

            if (std::isnan(v)) {
                throw std::runtime_error("X_num contains NaN.");
            }

            vals.push_back(v);
        }

        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        return vals;
    }

    static int hamming_distance_binary_column_(
        const std::vector<bool>& a,
        const std::vector<uint8_t>& b
    ) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Hamming distance column sizes do not match.");
        }

        int d = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const bool bv = (b[i] != 0);
            if (a[i] != bv) ++d;
        }
        return d;
    }

    static void sort_unique_ints_inplace_(std::vector<int>& xs) {
        std::sort(xs.begin(), xs.end());
        xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    }

    void clear_fit_state_() {
        result.reset();

        X_bits.clear();
        Y_bits.clear();
        continuous_starts.clear();

        greedy_cache.clear();
        lickety_cache_k2.clear();
        lickety_cache_kla.clear();
        trie_cache.clear();

        mask_ids = MaskIdTable();
        lit_ids = LitIdTable();

        n_samples = 0;
        n_features = 0;
        n_words = 0;
        tail_mask = ~0ULL;
        gamma = 0;
        trained_depth_budget = -1;
        best_objective = 0;
        obj_bound = 0;
        num_classes = 0;
        k_at_depth.clear();
    }

    inline bool mask_has_row_(const Packed& mask, int row) const {
        const int w = row >> 6;
        const int b = row & 63;
        return ((mask.w[(std::size_t)w] >> b) & 1ULL) != 0ULL;
    }

    struct NumericalGreedyState {
        // one sorted row list per numerical continuous group.
        // sorted_idx_by_num[g] is sorted by numerical_X_cols_for_greedy[g].
        std::vector<std::vector<int>> sorted_idx_by_num;
    };

    shared_ptr<TreeTrieNode> construct_trie(const Packed& mask, int8_t depth, int budget, const PathKey& pk, const ContinuousPath& cpath = empty_continuous_path()) {
        const uint64_t k = key_of_subproblem(mask, pk);
        K3 key{k, depth, budget};

        if (trie_cache_enabled) {
            if (auto it = trie_cache.find(key); it != trie_cache.end()) {
                return it->second; // exact lookup for simplicity
            }
        }

        auto node = make_shared<TreeTrieNode>(); // wraps in shared pointer so memory management is automatic
        node->budget = budget;

        int n_sub = 0;

        if (num_classes == 2) {
            int pos = 0;
            count_total_pos_binary(mask, n_sub, pos);

            if (!majority_leaf_only) {
                // predict 0: mistakes are positives
                const int cost0 = gamma + pos;

                // predict 1: mistakes are negatives
                const int cost1 = gamma + (n_sub - pos);

                if (cost0 <= budget) node->add_leaf(0, cost0);
                if (cost1 <= budget) node->add_leaf(1, cost1);
            } else {
                // choose majority class; tie breaks toward class 1
                const int neg = n_sub - pos;
                const int best_c = (pos >= neg) ? 1 : 0;
                const int mis = std::min(pos, neg);
                const int best_cost = gamma + mis;

                if (best_cost <= budget) node->add_leaf(best_c, best_cost);
            }
        } else {
            n_sub = count_total(mask);

            std::vector<int> cnts;
            count_per_class(mask, cnts);

            if (!majority_leaf_only) {
                for (int c = 0; c < num_classes; ++c) {
                    const int mis = n_sub - cnts[(size_t)c];
                    const int cost = gamma + mis;
                    if (cost <= budget) node->add_leaf(c, cost);
                }
            } else {
                // choose argmax class
                int best_c = 0;
                int best_cnt = cnts[0];
                for (int c = 1; c < num_classes; ++c) {
                    const int v = cnts[(size_t)c];
                    if (v > best_cnt || (v == best_cnt && c > best_c)) {
                        best_cnt = v;
                        best_c = c;
                    }
                }

                const int mis = n_sub - best_cnt;
                const int best_cost = gamma + mis;
                if (best_cost <= budget) node->add_leaf(best_c, best_cost);
            }
        }

        if (depth == 0 || budget < 2 * gamma) {
            if (trie_cache_enabled) trie_cache.emplace(key, node);
            return node;
        }

        Packed L(n_words), R(n_words);

        const int8_t k_here = (proxy_style == 2 && depth >= 0 && depth < (int)k_at_depth.size())
            ? k_at_depth[depth-1]
            : lookahead_init;

        const int first_continuous_feature = continuous_starts.empty() ? n_features : continuous_starts[0];

        for (int f = 0; f < first_continuous_feature; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);

            if (!L.any() || !R.any()) continue;

            // build child pks (canonical sorted)
            // pk refs default to EMPTY
            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            // only build PKs in LITS_EXACT
            PathKey pkL_local, pkR_local;
            if (key_mode == KeyMode::LITS_EXACT) {
                pkL_local = pk;
                pkR_local = pk;
                pk_insert_sorted(pkL_local, enc_lit(f, 1));
                pk_insert_sorted(pkR_local, enc_lit(f, 0));
                pkLp = &pkL_local;
                pkRp = &pkR_local;
            }

            int lossL, lossR;

            // to evaluate whether lossL+lossR is within budget (for non rule list mode), we can first handle an early pruning case
            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
            } else if (lookahead_init == 0) {
                lossL = greedy_proxy_objective_(L, depth - 1, *pkLp, cpath);
            } else {
                if (proxy_style == 4) {
                    lossL = split_algorithm(L, depth - 1, k_here, *pkLp);
                } else {
                    lossL = lickety_proxy_objective_(L, depth - 1, k_here, *pkLp, cpath);
                }
            }

            // either L or R would work here, could take larger, but very cheap to just choose one
            if (!rule_list_mode) {
                if (lossL + gamma > budget) continue;
            }

            // now compute R if we need it for more information
            if (lookahead_init < 0) {
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossR = greedy_proxy_objective_(R, depth - 1, *pkRp, cpath);
            } else {
                if (proxy_style == 4) {
                    lossR = split_algorithm(R, depth - 1, k_here, *pkRp);
                } else {
                    lossR = lickety_proxy_objective_(R, depth - 1, k_here, *pkRp, cpath);
                }
            }

            // standard pruning logic in paper
            if (!rule_list_mode) {
                if (lossL + lossR > budget) continue; // approximation decision tree rashomon set
            } else {
                if (lossL > budget - gamma && lossR > budget - gamma) continue; // exact rule list rashomon set
            }

            
            // LR has two entries: first and second.
            std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>> LR; 
            if (rule_list_mode) {
                LR = solve_siblings(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp, cpath, cpath);
            } else if (use_multipass) {
                LR = solve_siblings(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp, cpath, cpath);
            } else {
                LR = symmetric_single_pass(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp, cpath, cpath);
            }

            // the left and right TreeTrieNode (OR nodes) to be added to the AND/OR graph being built
            if (!LR.first || !LR.second) continue; // safeguard, especially needed if we allow non-injective keys
            
            node->add_split(f, LR.first, LR.second); // add split with left and right subtries
        }

        // continuous groups, already fully binarized
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start_idx = continuous_starts[(size_t)cont_pos];

            const int raw_end_idx = (cont_pos + 1 < (int)continuous_starts.size())
                ? continuous_starts[(size_t)(cont_pos + 1)]
                : n_features;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start_idx,
                    raw_end_idx,
                    cpath
                );

            if (start_idx >= end_idx) {
                continue;
            }

            enumerate_continuous_feature_for_trie(
                node,
                mask,
                depth,
                budget,
                pk,
                cpath,
                k_here,
                start_idx,
                end_idx,
                L,
                R
            );
        }

        if (trie_cache_enabled) trie_cache.emplace(key, node);
        return node;
    }

    inline int bit_distance_between_thresholds_(
        const Packed& mask,
        int feat_a,
        int feat_b
    ) const {
        // number of active samples whose side changes between threshold columns.
        int s = 0;

        const Packed& A = X_bits[(size_t)feat_a];
        const Packed& B = X_bits[(size_t)feat_b];

        return popcount_xor_and_words(
            mask.w.data(),
            A.w.data(),
            B.w.data(),
            n_words
        );
    }

    inline bool bit_distance_at_least_(
        const Packed& mask,
        int feat_a,
        int feat_b,
        int delta
    ) const {
        // need at least 0 moved samples is always true.
        if (delta <= 0) return true;

        // same threshold has distance 0.
        if (feat_a == feat_b) return false;

        int s = 0;

        const Packed& A = X_bits[(size_t)feat_a];
        const Packed& B = X_bits[(size_t)feat_b];

        for (int t = 0; t < n_words; ++t) {
            const uint64_t diff =
                mask.w[(size_t)t] &
                (A.w[(size_t)t] ^ B.w[(size_t)t]);

            s += popcnt64(diff);

            // early exit: tightening only needs to know whether
            // we have moved at least delta active samples.
            if (s >= delta) return true;
        }

        return false;
    }

    // partitioning subproblem into left and right using feat split
    inline void split_threshold_bits_(
        const Packed& mask,
        int feat,
        Packed& L,
        Packed& R
    ) const {
        const Packed& Xf = X_bits[(size_t)feat];

        for (int t = 0; t < n_words; ++t) {
            const uint64_t mw = mask.w[(size_t)t];
            const uint64_t lw = mw & Xf.w[(size_t)t];

            L.w[(size_t)t] = lw;
            R.w[(size_t)t] = mw & ~Xf.w[(size_t)t];
        }

        L.w[(size_t)(n_words - 1)] &= tail_mask;
        R.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    // map is red black tree, so both of these are log
    // returns the largest evaluated threshold index strictly less than i
    inline int predecessor_eval_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int i
    ) const {
        auto it = evaluated.lower_bound(i);
        if (it == evaluated.begin()) return -1;
        --it;
        return it->first;
    }

    // smallest strictly larger
    inline int successor_eval_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int j
    ) const {
        auto it = evaluated.upper_bound(j);
        if (it == evaluated.end()) return -1;
        return it->first;
    }

    inline int eval_left_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int feat
    ) const {
        return evaluated.at(feat).first;
    }

    inline int eval_right_(
        const std::map<int, std::pair<int,int>>& evaluated,
        int feat
    ) const {
        return evaluated.at(feat).second;
    }

    // if at budget, that is fine
    inline bool violates_rashomon_bound_(int z, int budget) const {
        return z > budget;
    }

    // need to improve by gap to be within budget
    inline int rashomon_gap_(int z, int budget) const {
        return z - budget;
    }

    // int tighten_lower_bound_bitvector_(
    //     const Packed& mask,
    //     int end_idx,
    //     int u,
    //     int i,
    //     int delta
    // ) const {
    //     // return min k >= i such that distance(u,k) >= delta.
    //     // search interval is [i, end_idx).
    //     // return end_idx if no such k exists.
    //     // k is the point that we would start looking at after if we know i is really bad - delta bad. we look to the right of k, k+1 usually.

    //     if (i >= end_idx) return end_idx;

    //     int lo = i;
    //     int hi = end_idx - 1;
    //     int ans = end_idx;

    //     // we can binary search because distance increases. in practice, we may find linearly scanning is better because we can escape really early. evaluate.
    //     while (lo <= hi) {
    //         const int mid = lo + ((hi - lo) >> 1);

    //         const bool moved_enough = bit_distance_at_least_(mask, u, mid, delta);

    //         if (moved_enough) {
    //             ans = mid;
    //             hi = mid - 1;
    //         } else {
    //             lo = mid + 1;
    //         }
    //     }

    //     return ans;
    // }

    // int tighten_upper_bound_bitvector_(
    //     const Packed& mask,
    //     int start_idx,
    //     int v,
    //     int j,
    //     int delta
    // ) const {
    //     // return max k <= j such that distance(k,v) >= delta.
    //     // search interval is [start_idx, j].
    //     // return start_idx - 1 if no such k exists.

    //     if (j < start_idx) return start_idx - 1;

    //     int lo = start_idx;
    //     int hi = j;
    //     int ans = start_idx - 1;

    //     while (lo <= hi) {
    //         const int mid = lo + ((hi - lo) >> 1);

    //         const bool moved_enough = bit_distance_at_least_(mask, mid, v, delta);

    //         if (moved_enough) {
    //             ans = mid;
    //             lo = mid + 1;
    //         } else {
    //             hi = mid - 1;
    //         }
    //     }

    //     return ans;
    // }

    // int tighten_lower_bound_bitvector_(
    //     const Packed& mask,
    //     int end_idx,
    //     int u,
    //     int i,
    //     int delta
    // ) const {
    //     if (delta <= 0) return i;
    //     if (i >= end_idx) return end_idx;

    //     for (int k = i; k < end_idx; ++k) {
    //         if (bit_distance_at_least_(mask, u, k, delta)) {
    //             return k;
    //         }
    //     }

    //     return end_idx;
    // }

    // int tighten_upper_bound_bitvector_(
    //     const Packed& mask,
    //     int start_idx,
    //     int v,
    //     int j,
    //     int delta
    // ) const {
    //     if (delta <= 0) return j;
    //     if (j < start_idx) return start_idx - 1;

    //     for (int k = j; k >= start_idx; --k) {
    //         if (bit_distance_at_least_(mask, k, v, delta)) {
    //             return k;
    //         }
    //     }

    //     return start_idx - 1;
    // }

    int tighten_lower_bound_bitvector_(
        const Packed& mask,
        int end_idx,
        int u,
        int i,
        int delta
    ) const {
        // return min k >= i such that distance(u,k) >= delta.
        // search interval is [i, end_idx).
        // return end_idx if no such k exists.

        if (delta <= 0) return i;
        if (i >= end_idx) return end_idx;

        // Local probe: nearby thresholds often satisfy the movement condition,
        // and bit_distance_at_least_ can early-exit.
        const int probe_end = std::min(end_idx, i + 2);
        for (int k = i; k < probe_end; ++k) {
            if (bit_distance_at_least_(mask, u, k, delta)) {
                return k;
            }
        }

        // If neither i nor i+1 worked, binary search the remaining suffix.
        int lo = probe_end;
        int hi = end_idx - 1;
        int ans = end_idx;

        while (lo <= hi) {
            const int mid = lo + ((hi - lo) >> 1);

            if (bit_distance_at_least_(mask, u, mid, delta)) {
                ans = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }

        return ans;
    }

    int tighten_upper_bound_bitvector_(
        const Packed& mask,
        int start_idx,
        int v,
        int j,
        int delta
    ) const {
        // return max k <= j such that distance(k,v) >= delta.
        // search interval is [start_idx, j].
        // return start_idx - 1 if no such k exists.

        if (delta <= 0) return j;
        if (j < start_idx) return start_idx - 1;

        // Local probe: nearby thresholds often satisfy the movement condition.
        const int probe_start = std::max(start_idx, j - 1);
        for (int k = j; k >= probe_start; --k) {
            if (bit_distance_at_least_(mask, k, v, delta)) {
                return k;
            }
        }

        // If neither j nor j-1 worked, binary search the remaining prefix.
        int lo = start_idx;
        int hi = probe_start - 1;
        int ans = start_idx - 1;

        while (lo <= hi) {
            const int mid = lo + ((hi - lo) >> 1);

            if (bit_distance_at_least_(mask, mid, v, delta)) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        return ans;
    }

    // start_idx and end_idx are the fixed threshold boundaries of the continuous feature.
    // i and j are the current threshold indices for the interval being pruned.
    std::pair<int,int> shrink_interval_with_bound_bitvector_(
        const std::map<int, std::pair<int,int>>& evaluated,
        const Packed& mask,
        int start_idx,
        int end_idx,
        int i,
        int j,
        int M_L,
        int M_R,
        int budget
    ) const {
        // we will leverage the closest thing we evaluated before, and the closest thing we evaluated above.
        const int u = predecessor_eval_(evaluated, i);
        const int v = successor_eval_(evaluated, j);

        // if the closest evaluated threshold to the left contributes a left side
        // and the closest evaluated threshold to the right contributes a right side
        // whose sum already violates the budget, prune the whole interval, because we haven't even accounted for the points in the middle.
        if (u >= 0 && v >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_v = eval_right_(evaluated, v);

            if (violates_rashomon_bound_(L_u + R_v, budget)) {
                return {1, 0};
            }
        }

        // if predecessor itself violates, move the lower endpoint rightward
        // until enough active samples have changed sides.
        if (u >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_u = eval_right_(evaluated, u);
            const int P_u = L_u + R_u;

            if (violates_rashomon_bound_(P_u, budget)) {
                const int delta = rashomon_gap_(P_u, budget);
                // given a failed predecessor threshold u, and a current interval starting at i, 
                // find the first threshold at or after i that is at least delta samples away from u
                const int new_i = tighten_lower_bound_bitvector_(
                    mask,
                    end_idx,
                    u,
                    i,
                    delta
                );

                i = std::max(i, new_i);
            }
        }

        // if successor itself violates, move the upper endpoint leftward
        // until enough active samples have changed sides.
        if (v >= 0) {
            const int L_v = eval_left_(evaluated, v);
            const int R_v = eval_right_(evaluated, v);
            const int P_v = L_v + R_v;

            if (violates_rashomon_bound_(P_v, budget)) {
                const int delta = rashomon_gap_(P_v, budget);

                const int new_j = tighten_upper_bound_bitvector_(
                    mask,
                    start_idx,
                    v,
                    j,
                    delta
                );

                j = std::min(j, new_j);
            }
        }

        i = std::max(i, M_L);
        j = std::min(j, M_R);

        return {i, j};
    }

    inline int threshold_distance_bitvector_(
        const Packed& mask,
        int a,
        int b
    ) const {
        if (a == b) return 0;

        return popcount_xor_and_words(
            mask.w.data(),
            X_bits[(size_t)a].w.data(),
            X_bits[(size_t)b].w.data(),
            n_words
        );
    }

    void enumerate_continuous_feature_for_trie(
        shared_ptr<TreeTrieNode>& node,
        const Packed& mask,
        int8_t depth,
        int budget,
        const PathKey& pk,
        const ContinuousPath& cpath,
        int8_t k_here,
        int start_idx,
        int end_idx,
        Packed& L,
        Packed& R
    ){
        // continuous threshold group is [start_idx, end_idx).
        // threshold columns are already binarized and monotone ordered.
        if (start_idx >= end_idx) return;

        // evaluated[feat] = {lossL, lossR}
        //
        // feat is the absolute threshold-column feature index.
        // this scratch table includes both feasible and failed evaluated thresholds.
        // successful splits are stored persistently by node->add_split(feat, ...).
        // std::map<int, std::pair<int,int>> evaluated;

        std::deque<std::pair<int,int>> Q;
        Q.push_back({start_idx, end_idx - 1});

        int M_L = start_idx;
        int M_R = end_idx - 1;

        while (!Q.empty()) {
            auto [i, j] = Q.front();
            Q.pop_front();

            // auto shrunk = shrink_interval_with_bound_bitvector_(
            //     evaluated,
            //     mask,
            //     start_idx,
            //     end_idx,
            //     i,
            //     j,
            //     M_L,
            //     M_R,
            //     budget
            // );

            i = std::max(i, M_L);
            j = std::min(j, M_R);

            if (i > j) continue;

            const int feat = i + ((j - i) >> 1); // midpoint of thresholds, recall we aren't deduplicating

            split_threshold_bits_(mask, feat, L, R);

            const bool left_empty = !L.any();
            const bool right_empty = !R.any();

            if (left_empty || right_empty) {
                if (left_empty && !right_empty) {
                    // threshold too low: all thresholds <= feat also have empty left.
                    // only higher thresholds can become valid.
                    if (feat + 1 <= j) {
                        Q.push_back({feat + 1, j});
                    }
                } else if (!left_empty && right_empty) {
                    // threshold too high: all thresholds >= feat also have empty right.
                    // only lower thresholds can become valid.
                    if (i <= feat - 1) {
                        Q.push_back({i, feat - 1});
                    }
                }
                // if both are empty, mask itself is empty, which should not happen here.
                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int lossL;
            int lossR;

            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
            } else if (lookahead_init == 0) {
                lossL = greedy_proxy_objective_(L, depth - 1, *pkLp, *cpathLp);
            } else {
                if (proxy_style == 4) {
                    lossL = split_algorithm(L, depth - 1, k_here, *pkLp);
                } else {
                    lossL = lickety_proxy_objective_(L, depth - 1, k_here, *pkLp, *cpathLp);
                }
            }

            if (lookahead_init < 0) {
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossR = greedy_proxy_objective_(R, depth - 1, *pkRp, *cpathRp);
            } else {
                if (proxy_style == 4) {
                    lossR = split_algorithm(R, depth - 1, k_here, *pkRp);
                } else {
                    lossR = lickety_proxy_objective_(R, depth - 1, k_here, *pkRp, *cpathRp);
                }
            }

            // evaluated[feat] = {lossL, lossR};

            const int total_proxy = lossL + lossR;

            bool feasible;
            if (!rule_list_mode) {
                feasible = (total_proxy <= budget);
            } else {
                feasible = !(lossL > budget - gamma && lossR > budget - gamma);
            }

            if (feasible) {
                std::pair<
                    std::shared_ptr<TreeTrieNode>,
                    std::shared_ptr<TreeTrieNode>
                > LR;

                if (rule_list_mode) {
                    LR = solve_siblings(
                        lossL,
                        lossR,
                        L,
                        R,
                        budget,
                        depth,
                        *pkLp,
                        *pkRp,
                        *cpathLp,
                        *cpathRp
                    );
                } else if (use_multipass) {
                    LR = solve_siblings(
                        lossL,
                        lossR,
                        L,
                        R,
                        budget,
                        depth,
                        *pkLp,
                        *pkRp,
                        *cpathLp,
                        *cpathRp
                    );
                } else {
                    LR = symmetric_single_pass(
                        lossL,
                        lossR,
                        L,
                        R,
                        budget,
                        depth,
                        *pkLp,
                        *pkRp,
                        *cpathLp,
                        *cpathRp
                    );
                }

                if (LR.first && LR.second) {
                    // persistent successful split storage.
                    // feat is the actual threshold-column index.
                    node->add_split(feat, LR.first, LR.second);

                    // if (evaluated_use_min_objectives) {
                    //     evaluated[feat] = {
                    //         LR.first->min_objective,
                    //         LR.second->min_objective
                    //     };
                    // } else {
                    //     evaluated[feat] = {lossL, lossR};
                    // }
                } 

                if (i <= feat - 1) {
                    Q.push_back({i, feat - 1});
                }

                if (feat + 1 <= j) {
                    Q.push_back({feat + 1, j});
                }

                continue; // we enumerate nearby thresholds because they are high quality too
            }
            /*
            if (feasible) {
                // given a threshold split_feat and guessed child objective values guessL, guessR, 
                // try to build the left/right Rashomon subtries under this split, 
                // add the split to the parent node if both children exist, 
                // and report the true child minimum objectives
                auto solve_and_add_from_guesses = [&](
                    int split_feat,
                    int guessL,
                    int guessR,
                    int& out_minL,
                    int& out_minR
                ) -> bool {
                    split_threshold_bits_(mask, split_feat, L, R);

                    if (!L.any() || !R.any()) {
                        return false;
                    }

                    const PathKey* local_pkLp = &empty_pk();
                    const PathKey* local_pkRp = &empty_pk();

                    PathKey local_pkL;
                    PathKey local_pkR;

                    make_child_pks_if_needed_(
                        split_feat,
                        pk,
                        local_pkLp,
                        local_pkRp,
                        local_pkL,
                        local_pkR
                    );

                    const ContinuousPath* local_cpathLp = &cpath;
                    const ContinuousPath* local_cpathRp = &cpath;

                    ContinuousPath local_cpathL;
                    ContinuousPath local_cpathR;

                    make_child_continuous_paths_if_needed_(
                        split_feat,
                        cpath,
                        local_cpathLp,
                        local_cpathRp,
                        local_cpathL,
                        local_cpathR
                    );

                    std::pair<
                        std::shared_ptr<TreeTrieNode>,
                        std::shared_ptr<TreeTrieNode>
                    > LR;

                    if (rule_list_mode) {
                        LR = solve_siblings(
                            guessL,
                            guessR,
                            L,
                            R,
                            budget,
                            depth,
                            *local_pkLp,
                            *local_pkRp,
                            *local_cpathLp,
                            *local_cpathRp
                        );
                    } else if (use_multipass) {
                        LR = solve_siblings(
                            guessL,
                            guessR,
                            L,
                            R,
                            budget,
                            depth,
                            *local_pkLp,
                            *local_pkRp,
                            *local_cpathLp,
                            *local_cpathRp
                        );
                    } else {
                        LR = symmetric_single_pass(
                            guessL,
                            guessR,
                            L,
                            R,
                            budget,
                            depth,
                            *local_pkLp,
                            *local_pkRp,
                            *local_cpathLp,
                            *local_cpathRp
                        );
                    }

                    if (!LR.first || !LR.second) {
                        // evaluated[split_feat] = {guessL, guessR};
                        return false;
                    }

                    node->add_split(split_feat, LR.first, LR.second);

                    out_minL = LR.first->min_objective;
                    out_minR = LR.second->min_objective;

                    // store the actual subgraph minima
                    // evaluated[split_feat] = {out_minL, out_minR};

                    return true;
                };

                int base_minL = std::numeric_limits<int>::max();
                int base_minR = std::numeric_limits<int>::max();

                const bool base_ok = solve_and_add_from_guesses(
                    feat,
                    lossL,
                    lossR,
                    base_minL,
                    base_minR
                );
                
                // this should never happen, would be a collision on the fingerprints if so
                if (!base_ok) {
                    // evaluated[feat] = {lossL, lossR};

                    if (i <= feat - 1) {
                        Q.push_back({i, feat - 1});
                    }

                    if (feat + 1 <= j) {
                        Q.push_back({feat + 1, j});
                    }

                    continue;
                }

                // in rule-list mode, keep the old behavior because the feasibility condition
                // is not simply minL + minR <= budget.
                if (rule_list_mode) {
                    if (i <= feat - 1) {
                        Q.push_back({i, feat - 1});
                    }

                    if (feat + 1 <= j) {
                        Q.push_back({feat + 1, j});
                    }

                    continue;
                }

                // walk left from the feasible threshold.
                // moving left means:
                // left child loses x points  -> do not decrease its min objective
                // right child gains x points -> add x to the right guess
                {
                    int anchor_feat = feat;
                    int anchor_minL = base_minL;
                    int anchor_minR = base_minR;
                    int delta = budget - (anchor_minL + anchor_minR);

                    int t = feat - 1; // start of walking left, feat is midpoint

                    for (; t >= i; --t) {
                        if (delta < 0) break; // if delta = 0, consecutive columns may be the same when acting on this subproblem, we don't want to skip them

                        const int x = threshold_distance_bitvector_(
                            mask,
                            t,
                            anchor_feat
                        );

                        if (x > delta) {
                            break;
                        }

                        const int guessL = anchor_minL;
                        const int guessR = anchor_minR + x;

                        int new_minL = std::numeric_limits<int>::max(); // the method will fill these in 
                        int new_minR = std::numeric_limits<int>::max();

                        const bool ok = solve_and_add_from_guesses(
                            t,
                            guessL,
                            guessR,
                            new_minL,
                            new_minR
                        );

                        if (!ok) {
                            break;
                        }

                        anchor_feat = t; // new reference location
                        anchor_minL = new_minL; // reference location best objectives
                        anchor_minR = new_minR;
                        delta = budget - (anchor_minL + anchor_minR); // new delta at this reference
                    }

                    // continue normal interval search on the part not covered by the local feasible walk
                    if (i <= t) {
                        Q.push_back({i, t});
                    }
                }

                // walk right from the feasible threshold.
                // moving right means:
                // left child gains x points  -> add x to the left guess
                // right child loses x points -> do not decrease its min objective
                {
                    int anchor_feat = feat;
                    int anchor_minL = base_minL;
                    int anchor_minR = base_minR;
                    int delta = budget - (anchor_minL + anchor_minR);

                    int t = feat + 1;

                    for (; t <= j; ++t) {
                        if (delta < 0) break;

                        const int x = threshold_distance_bitvector_(
                            mask,
                            anchor_feat,
                            t
                        );

                        if (x > delta) {
                            break;
                        }

                        const int guessL = anchor_minL + x;
                        const int guessR = anchor_minR;

                        int new_minL = std::numeric_limits<int>::max();
                        int new_minR = std::numeric_limits<int>::max();

                        const bool ok = solve_and_add_from_guesses(
                            t,
                            guessL,
                            guessR,
                            new_minL,
                            new_minR
                        );

                        if (!ok) {
                            break;
                        }

                        anchor_feat = t;
                        anchor_minL = new_minL;
                        anchor_minR = new_minR;
                        delta = budget - (anchor_minL + anchor_minR);
                    }

                    // continue normal interval search on the part
                    // not covered by the local feasible walk.
                    if (t <= j) {
                        Q.push_back({t, j});
                    }
                }

                continue; // we just finished everything we do if a threshold is within budget, go to the next iteration and get a new threshold
            }
            */


            // evaluated[feat] = {lossL, lossR};

            // failed split pruning.
            // if left is pure/zero-error leaf objective, farther-left thresholds
            // cannot improve the left side enough, so move M_L right.
            if (lossL == gamma) {
                M_L = std::max(M_L, feat + 1);
            }

            // if right is pure/zero-error leaf objective, farther-right thresholds
            // cannot improve the right side enough, so move M_R left.
            if (lossR == gamma) {
                M_R = std::min(M_R, feat - 1);
            }

            const int delta = total_proxy - budget;

            if (delta <= 0) {
                // this is only possible in rule-list mode, where infeasible means
                // both sides individually exceed budget - gamma, not necessarily
                // that lossL + lossR > budget.
                if (i <= feat - 1) {
                    Q.push_back({i, feat - 1});
                }

                if (feat + 1 <= j) {
                    Q.push_back({feat + 1, j});
                }

                continue;
            }

            const int a = tighten_upper_bound_bitvector_(
                mask,
                start_idx,
                feat,
                feat - 1,
                delta
            );

            const int b = tighten_lower_bound_bitvector_(
                mask,
                end_idx,
                feat,
                feat + 1,
                delta
            );

            if (i <= a) {
                Q.push_back({i, a});
            }

            if (b <= j) {
                Q.push_back({b, j});
            }
        }
    }

    // returns left and right treetrienode. the left and right mask are constants, even as you recurse on construct_trie
    pair<shared_ptr<TreeTrieNode>, shared_ptr<TreeTrieNode>>
    solve_siblings(int loss_l, int loss_r,
               const Packed& Lmask, const Packed& Rmask,
               int budget, int8_t depth,
               const PathKey& pkL, const PathKey& pkR,
               const ContinuousPath& cpathL, const ContinuousPath& cpathR) {
        int left_budget  = budget - loss_r;
        shared_ptr<TreeTrieNode> left_node =
            (left_budget >= 0) ? construct_trie(Lmask, depth - 1, left_budget, pkL, cpathL)
                               : nullptr; // handles some potential issues with non-injective keys
        int min_left = (left_node ? left_node->min_objective : numeric_limits<int>::max());

        int right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
        shared_ptr<TreeTrieNode> right_node =
            (right_budget >= 0) ? construct_trie(Rmask, depth - 1, right_budget, pkR, cpathR)
                                : nullptr;
        int min_right = (right_node ? right_node->min_objective : numeric_limits<int>::max());

        while (true) {
            bool improved = false;

            int new_left_budget = (min_right == numeric_limits<int>::max()) ? -1 : (budget - min_right);
            if (new_left_budget > left_budget) {
                left_budget = new_left_budget;
                if (left_budget >= 0) {
                    left_node = construct_trie(Lmask, depth - 1, left_budget, pkL, cpathL);
                    int new_min_left = left_node->min_objective;
                    if (new_min_left < min_left) min_left = new_min_left;
                }
            }

            int new_right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
            if (new_right_budget > right_budget) {
                right_budget = new_right_budget;
                if (right_budget >= 0) {
                    right_node = construct_trie(Rmask, depth - 1, right_budget, pkR, cpathR);
                    int new_min_right = right_node->min_objective;
                    if (new_min_right < min_right) { min_right = new_min_right; improved = true; }
                }
            }

            if (!improved) break;
        }

        return {left_node, right_node};
    }

    // this is solely for ableation study purposes - if practical we would subtract minobjective from the other side
    std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>>
    symmetric_single_pass(int loss_l, int loss_r,
                 const Packed& Lmask, const Packed& Rmask,
                 int budget, int8_t depth,
                 const PathKey& pkL, const PathKey& pkR,
                 const ContinuousPath& cpathL, const ContinuousPath& cpathR) {
        int left_budget  = budget - loss_r;
        int right_budget = budget - loss_l;

        std::shared_ptr<TreeTrieNode> left_node  = nullptr;
        std::shared_ptr<TreeTrieNode> right_node = nullptr;

        if (left_budget >= 0) { // robustness incase we change pruning
            left_node = construct_trie(Lmask, depth - 1, left_budget, pkL, cpathL);
        }
        if (right_budget >= 0) {
            right_node = construct_trie(Rmask, depth - 1, right_budget, pkR, cpathR);
        }

        return {left_node, right_node};
    
    }

    inline void count_total_pos_binary(const Packed& mask, int& n, int& pos) const {
        const Packed& Ypos = Y_bits[(size_t)1];
        count_total_and_pos_words(mask.w.data(), Ypos.w.data(), n_words, n, pos);
    }

    static inline void count_total_and_pos_words(
        const uint64_t* mask,
        const uint64_t* ypos,
        int n_words,
        int& n,
        int& pos
    ) {
    #if PRAXIS_USE_AVX512_POPCNT
        int i = 0;
        __m512i acc_n = _mm512_setzero_si512();
        __m512i acc_p = _mm512_setzero_si512();

        for (; i + 8 <= n_words; i += 8) {
            __m512i vm = _mm512_loadu_si512((const void*)(mask + i));
            __m512i vy = _mm512_loadu_si512((const void*)(ypos + i));
            __m512i vp = _mm512_and_si512(vm, vy);

            acc_n = _mm512_add_epi64(acc_n, _mm512_popcnt_epi64(vm));
            acc_p = _mm512_add_epi64(acc_p, _mm512_popcnt_epi64(vp));
        }

        alignas(64) uint64_t tmp_n[8];
        alignas(64) uint64_t tmp_p[8];

        _mm512_store_si512((void*)tmp_n, acc_n);
        _mm512_store_si512((void*)tmp_p, acc_p);

        uint64_t tn =
            tmp_n[0] + tmp_n[1] + tmp_n[2] + tmp_n[3] +
            tmp_n[4] + tmp_n[5] + tmp_n[6] + tmp_n[7];

        uint64_t tp =
            tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] +
            tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];

        for (; i < n_words; ++i) {
            const uint64_t mw = mask[i];
            tn += (uint64_t)popcnt64(mw);
            tp += (uint64_t)popcnt64(mw & ypos[i]);
        }

        n = (int)tn;
        pos = (int)tp;
    #else
        int tn = 0;
        int tp = 0;
        for (int i = 0; i < n_words; ++i) {
            const uint64_t mw = mask[i];
            tn += popcnt64(mw);
            tp += popcnt64(mw & ypos[i]);
        }
        n = tn;
        pos = tp;
    #endif
    }

    int leaf_objective(const Packed& mask) const {
        if (num_classes == 2) {
            int n_sub, pos;
            count_total_pos_binary(mask, n_sub, pos);

            if (n_sub == 0) return 0;

            return gamma + std::min(pos, n_sub - pos);
        }

        const int n_sub = count_total(mask);
        if (n_sub == 0) return 0;

        int best_cnt = 0;
        for (int c = 0; c < num_classes; ++c) {
            best_cnt = std::max(best_cnt, count_class(mask, c));
        }
        const int mis = n_sub - best_cnt;
        return gamma + mis;
    }

    inline void make_child_pks_if_needed_(
        int feat,
        const PathKey& pk,
        const PathKey*& pkLp,
        const PathKey*& pkRp,
        PathKey& pkL_local,
        PathKey& pkR_local
    ) const {
        pkLp = &empty_pk();
        pkRp = &empty_pk();
        if (key_mode == KeyMode::LITS_EXACT) {
            pkL_local = pk;
            pkR_local = pk;
            pk_insert_sorted(pkL_local, enc_lit(feat, 1));
            pk_insert_sorted(pkR_local, enc_lit(feat, 0));
            pkLp = &pkL_local;
            pkRp = &pkR_local;
        }
    }

    inline bool is_continuous_threshold_feature_(int feat) const {
        return feat >= first_continuous_feature_() && feat < n_features;
    }

    inline std::pair<int,int> tighten_continuous_interval_from_path_(
        int start_idx,
        int end_idx,
        const ContinuousPath& cpath
    ) const {
        int lo = start_idx;
        int hi = end_idx; // [lo, hi), by that i mean we have the first threshold of the feature, and the first threshold of the next feature

        for (const auto& e : cpath) {
            const int t = e.threshold_index;

            if (t < start_idx || t >= end_idx) {
                continue; // decision belongs to another continuous feature group
            }

            if (e.went_true) {
                // we are in the x <= threshold branch.
                // for nested <= thresholds, all thresholds >= t are now all-true
                // on this subproblem, so they cannot produce a valid split.
                hi = std::min(hi, t);
            } else {
                // we are in the x > threshold branch.
                // all thresholds <= t are now all-false on this subproblem,
                // so they cannot produce a valid split.
                lo = std::max(lo, t + 1);
            }
        }

        return {lo, hi};
    }

    inline void make_child_continuous_paths_if_needed_(
        int feat,
        const ContinuousPath& cpath,
        const ContinuousPath*& cpathLp,
        const ContinuousPath*& cpathRp,
        ContinuousPath& cpathL_local,
        ContinuousPath& cpathR_local
    ) const {
        cpathLp = &cpath;
        cpathRp = &cpath;

        if (!is_continuous_threshold_feature_(feat)) {
            return;
        }

        cpathL_local = cpath;
        cpathR_local = cpath;

        cpathL_local.push_back(ContinuousPathEntry{feat, true});
        cpathR_local.push_back(ContinuousPathEntry{feat, false});

        cpathLp = &cpathL_local;
        cpathRp = &cpathR_local;
    }

    inline int leaf_objective_binary_from_counts(int n_sub, int pos) const {
        if (n_sub == 0) return 0;
        return gamma + std::min(pos, n_sub - pos);
    }

    // continuous land
    enum class ContinuousEvalMode {
        Lickety,
        Exact
    };

    struct ContinuousBestSplitResult {
        int best_sum = std::numeric_limits<int>::max();
        int best_feat = -1; // actual threshold-column feature index
    };

    inline int first_continuous_feature_() const {
        return continuous_starts.empty() ? n_features : continuous_starts[0];
    }

    inline int continuous_group_end_(int cont_pos) const {
        return (cont_pos + 1 < (int)continuous_starts.size())
            ? continuous_starts[(size_t)(cont_pos + 1)]
            : n_features;
    }

    // best-search mode: equality is not useful, because we only need strict improvement.
    inline bool violates_best_bound_(int z, int best) const {
        return z >= best;
    }

    // if z == best, we still need at least one active sample to move before
    // a neighboring threshold can possibly become strictly better.
    inline int best_gap_(int z, int best) const {
        return std::max(1, z - best);
    }

    std::pair<int,int> shrink_interval_with_best_bound_bitvector_(
        const std::map<int, std::pair<int,int>>& evaluated,
        const Packed& mask,
        int start_idx,
        int end_idx,
        int i,
        int j,
        int M_L,
        int M_R,
        int best
    ) const {
        const int u = predecessor_eval_(evaluated, i);
        const int v = successor_eval_(evaluated, j);

        // if predecessor-left plus successor-right already cannot strictly beat best,
        // then every threshold in the middle is hopeless.
        if (u >= 0 && v >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_v = eval_right_(evaluated, v);

            if (violates_best_bound_(L_u + R_v, best)) {
                return {1, 0};
            }
        }

        // if predecessor total cannot beat best, move lower endpoint right until
        // enough active samples have changed sides to make strict improvement possible.
        if (u >= 0) {
            const int L_u = eval_left_(evaluated, u);
            const int R_u = eval_right_(evaluated, u);
            const int P_u = L_u + R_u;

            if (violates_best_bound_(P_u, best)) {
                const int delta = best_gap_(P_u, best);

                const int new_i = tighten_lower_bound_bitvector_(
                    mask,
                    end_idx,
                    u,
                    i,
                    delta
                );

                i = std::max(i, new_i);
            }
        }

        // if successor total cannot beat best, move upper endpoint left until
        // enough active samples have changed sides to make strict improvement possible.
        if (v >= 0) {
            const int L_v = eval_left_(evaluated, v);
            const int R_v = eval_right_(evaluated, v);
            const int P_v = L_v + R_v;

            if (violates_best_bound_(P_v, best)) {
                const int delta = best_gap_(P_v, best);

                const int new_j = tighten_upper_bound_bitvector_(
                    mask,
                    start_idx,
                    v,
                    j,
                    delta
                );

                j = std::min(j, new_j);
            }
        }

        i = std::max(i, M_L);
        j = std::min(j, M_R);

        return {i, j};
    }

    int eval_with_lookahead_continuous(
        const Packed& mask,
        int8_t depth_budget,
        int8_t k,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        if (depth_budget == 1) {
            return depthd_exact_proxy_objective_(mask, 1, pk, cpath);
        }

        if (k <= 0) {
            return greedy_proxy_objective_(mask, depth_budget, pk, cpath);
        }

        return generalized_lickety_split_continuous(mask, depth_budget, k, pk, cpath);
    }

    ContinuousBestSplitResult search_continuous_feature_for_best_split_continuous_(
        const Packed& mask,
        int8_t depth_budget,
        int8_t child_k,
        const PathKey& pk,
        const ContinuousPath& cpath,
        int start_idx,
        int end_idx,
        int current_best,
        ContinuousEvalMode eval_mode
    ) {
        // unlike rashomon, if we are just current best at best, can prune. can also update current best
        ContinuousBestSplitResult out;
        out.best_sum = current_best;
        out.best_feat = -1;

        auto tightened = tighten_continuous_interval_from_path_(start_idx, end_idx, cpath);

        start_idx = tightened.first;
        end_idx = tightened.second;

        if (start_idx >= end_idx) return out;
        
        std::map<int, std::pair<int,int>> evaluated;

        std::deque<std::pair<int,int>> Q;
        Q.push_back({start_idx, end_idx - 1});

        int M_L = start_idx;
        int M_R = end_idx - 1;

        Packed L(n_words), R(n_words);

        while (!Q.empty()) {
            auto [i, j] = Q.front();
            Q.pop_front();

            auto shrunk = shrink_interval_with_best_bound_bitvector_(
                evaluated,
                mask,
                start_idx,
                end_idx,
                i,
                j,
                M_L,
                M_R,
                out.best_sum
            );

            i = shrunk.first;
            j = shrunk.second;

            if (i > j) continue;

            const int feat = i + ((j - i) >> 1);

            split_threshold_bits_(mask, feat, L, R);

            const bool left_empty = !L.any();
            const bool right_empty = !R.any();

            if (left_empty || right_empty) {
                if (left_empty && !right_empty) {
                    if (feat + 1 <= j) {
                        Q.push_back({feat + 1, j});
                    }
                } else if (!left_empty && right_empty) {
                    if (i <= feat - 1) {
                        Q.push_back({i, feat - 1});
                    }
                }
                continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int lossL;
            int lossR;

            if (eval_mode == ContinuousEvalMode::Exact) {
                if (depth_budget == 1) { // note that this is saying depth_budget-1 recursive call would give leaf objective, which is correct, it isn't that depth budget 1 is leaf.
                    lossL = leaf_objective(L);
                    lossR = leaf_objective(R);
                } else {
                    lossL = depthd_exact_solver_cached_continuous(
                        L,
                        depth_budget - 1,
                        *pkLp,
                        *cpathLp
                    );

                    lossR = depthd_exact_solver_cached_continuous(
                        R,
                        depth_budget - 1,
                        *pkRp,
                        *cpathRp
                    );
                }
            } else {
                lossL = eval_with_lookahead_continuous(
                    L,
                    depth_budget - 1,
                    child_k,
                    *pkLp,
                    *cpathLp
                );

                lossR = eval_with_lookahead_continuous(
                    R,
                    depth_budget - 1,
                    child_k,
                    *pkRp,
                    *cpathRp
                );
            }

            const int sum = lossL + lossR;

            evaluated[feat] = {lossL, lossR};

            if (sum < out.best_sum) {
                out.best_sum = sum;
                out.best_feat = feat;
            }

            // one-sided pure-side cutoffs.
            // if the left side is already at minimum possible leaf penalty,
            // farther-left thresholds cannot improve the left side.
            if (lossL == gamma) {
                M_L = std::max(M_L, feat + 1);
            }

            // if the right side is already at minimum possible leaf penalty,
            // farther-right thresholds cannot improve the right side.
            if (lossR == gamma) {
                M_R = std::min(M_R, feat - 1);
            }

            // we are bad, we can prune our neighbors
            if (sum >= out.best_sum) {
                const int delta = best_gap_(sum, out.best_sum); // if we are equal, we must move at least one sample

                const int a = tighten_upper_bound_bitvector_(
                    mask,
                    start_idx,
                    feat,
                    feat - 1,
                    delta
                );

                const int b = tighten_lower_bound_bitvector_(
                    mask,
                    end_idx,
                    feat,
                    feat + 1,
                    delta
                );

                if (i <= a) {
                    Q.push_back({i, a});
                }

                if (b <= j) {
                    Q.push_back({b, j});
                }
            } // else { // should never happen
            //     if (i <= feat - 1) {
            //         Q.push_back({i, feat - 1});
            //     }

            //     if (feat + 1 <= j) {
            //         Q.push_back({feat + 1, j});
            //     }
            // }
        }

        return out;
    }

    int generalized_lickety_split_continuous(
        const Packed& mask,
        int8_t depth_budget,
        int8_t k,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget == 0) {
            return leaf_objective(mask);
        }

        if (depth_budget == 1) {
            return depthd_exact_proxy_objective_(mask, 1, pk, cpath);
        }

        if (k > depth_budget - 1) {
            k = depth_budget - 1;
        }

        if (k == depth_budget - 1) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk, cpath);
        }

        uint64_t kmask = 0;
        K2  key2{0, depth_budget};
        KLA keyla{0, depth_budget, k};

        const bool use_kla = use_kla_cache();

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            keyla.k = kmask;

            if (use_kla) {
                if (auto it = lickety_cache_kla.find(keyla); it != lickety_cache_kla.end()) {
                    return it->second;
                }
            } else {
                if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end()) {
                    return it->second;
                }
            }
        }

        const int leaf_loss = leaf_objective(mask);

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                if (use_kla) {
                    lickety_cache_kla.emplace(keyla, leaf_loss);
                } else {
                    lickety_cache_k2.emplace(key2, leaf_loss);
                }
            }

            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum = leaf_loss;

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);

        const int8_t child_k = k - 1;

        // const int F = proxy_feat_count_();
        // const int first_cont = std::min(first_continuous_feature_(), F);
        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        // binary / ordinary feature columns only.
        for (int f = 0; f < first_cont; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);

            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                f,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const int left_loss = eval_with_lookahead_continuous(
                L,
                depth_budget - 1,
                child_k,
                *pkLp,
                cpath
            );

            const int right_loss = eval_with_lookahead_continuous(
                R,
                depth_budget - 1,
                child_k,
                *pkRp,
                cpath
            );

            const int sum = left_loss + right_loss;

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
            }
        }

        // continuous groups.
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    cpath
                );

            if (start_idx >= end_idx) continue;

            ContinuousBestSplitResult cres =
                search_continuous_feature_for_best_split_continuous_(
                    mask,
                    depth_budget,
                    child_k,
                    pk,
                    cpath,
                    start_idx,
                    end_idx,
                    best_sum,
                    ContinuousEvalMode::Lickety
                );

            if (cres.best_feat >= 0 && cres.best_sum < best_sum) {
                best_sum = cres.best_sum;
                best_feat = cres.best_feat;
            }
        }

        int ans = leaf_loss;

        int8_t k_recurse;

        if (proxy_style == 0) {
            // style 0: constant k.
            k_recurse = k;
        } else if (proxy_style == 3) {
            // SPLIT without postprocessing: tree is fully determined by the chosen split.
            ans = std::min(ans, best_sum);

            if (proxy_caching_enabled) {
                if (use_kla) {
                    lickety_cache_kla.emplace(keyla, ans);
                } else {
                    lickety_cache_k2.emplace(key2, ans);
                }
            }

            return ans;
        } else {
            // styles 1/2: recursively cycle k, k-1, ..., 1, k.
            k_recurse = (child_k == 0) ? lookahead_init : child_k;
        }

        if (best_feat >= 0) { // now partition for the best threshold
            split_threshold_bits_(mask, best_feat, bestL, bestR);

            const int8_t next_depth = depth_budget - 1;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                best_feat,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                best_feat,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            const int left_loss = generalized_lickety_split_continuous(
                bestL,
                next_depth,
                k_recurse,
                *pkLp,
                *cpathLp
            );

            const int right_loss = generalized_lickety_split_continuous(
                bestR,
                next_depth,
                k_recurse,
                *pkRp,
                *cpathRp
            );

            ans = std::min(ans, left_loss + right_loss);
            ans = std::min(ans, best_sum);
        }

        if (proxy_caching_enabled) {
            if (use_kla) {
                lickety_cache_kla.emplace(keyla, ans);
            } else {
                lickety_cache_k2.emplace(key2, ans);
            }
        }

        return ans;
    }

    int special_depth2_fixed_root_split_sum_bitvector_(
        const Packed& Lroot,
        const Packed& Rroot,
        const PathKey& pkL,
        const PathKey& pkR,
        int incumbent
    ) {
        constexpr int8_t DEPTH = 1;
        constexpr int8_t KTAG  = 0;

        uint64_t kL = 0;
        uint64_t kR = 0;

        bool have_cached_L = false;
        bool have_cached_R = false;

        int bestL = 0;
        int bestR = 0;

        if (proxy_caching_enabled) {
            kL = key_of_subproblem(Lroot, pkL);
            kR = key_of_subproblem(Rroot, pkR);

            have_cached_L = try_get_lickety_cached_(kL, DEPTH, KTAG, bestL);
            have_cached_R = try_get_lickety_cached_(kR, DEPTH, KTAG, bestR);

            if (have_cached_L && have_cached_R) {
                return bestL + bestR;
            }
        }

        const int leafL = leaf_objective(Lroot);
        const int leafR = leaf_objective(Rroot);

        if (!have_cached_L) bestL = leafL;
        if (!have_cached_R) bestR = leafR;

        int nLroot = 0;
        int nRroot = 0;

        int posLroot = 0;
        int posRroot = 0;

        std::vector<int> cntLroot;
        std::vector<int> cntRroot;

        if (num_classes == 2) {
            count_total_pos_binary(Lroot, nLroot, posLroot);
            count_total_pos_binary(Rroot, nRroot, posRroot);
        } else {
            nLroot = count_total(Lroot);
            nRroot = count_total(Rroot);

            count_per_class(Lroot, cntLroot);
            count_per_class(Rroot, cntRroot);
        }

        const bool scanL =
            !have_cached_L &&
            nLroot > 1 &&
            leafL > 2 * gamma;

        const bool scanR =
            !have_cached_R &&
            nRroot > 1 &&
            leafR > 2 * gamma;

        if (!scanL && !scanR) {
            if (proxy_caching_enabled) {
                if (!have_cached_L) {
                    cache_lickety_if_true_(
                        kL,
                        DEPTH,
                        KTAG,
                        bestL,
                        /*allow_cache=*/cache_cheap_subproblems || leafL > 2 * gamma
                    );
                }

                if (!have_cached_R) {
                    cache_lickety_if_true_(
                        kR,
                        DEPTH,
                        KTAG,
                        bestR,
                        /*allow_cache=*/cache_cheap_subproblems || leafR > 2 * gamma
                    );
                }
            }

            return bestL + bestR;
        }

        const int first_cont = first_continuous_feature_();

        Packed LL(n_words), LR(n_words);
        Packed RL(n_words), RR(n_words);

        // ordinary non-continuous binary features.
        for (int f2 = 0; f2 < first_cont; ++f2) {
            if (scanL) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(Lroot, X_bits[(std::size_t)f2], LL, LR, left_n);

                    if (left_n != 0 && left_n != nLroot) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                } else {
                    and_bits(Lroot, X_bits[(std::size_t)f2], LL);
                    andnot_bits(Lroot, X_bits[(std::size_t)f2], LR);

                    if (LL.any() && LR.any()) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                }
            }

            if (scanR) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(Rroot, X_bits[(std::size_t)f2], RL, RR, left_n);

                    if (left_n != 0 && left_n != nRroot) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                } else {
                    and_bits(Rroot, X_bits[(std::size_t)f2], RL);
                    andnot_bits(Rroot, X_bits[(std::size_t)f2], RR);

                    if (RL.any() && RR.any()) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                }
            }

            if (bestL + bestR <= 2 * gamma) {
                goto done_scanning;
            }
        }

        // continuous groups represented as contiguous monotone binary columns.
        // for a group: f = start, start+1, ..., end-1
        // columns are nested: X_bits[start] subset X_bits[start+1] subset ...
        // so for each threshold we update counts using only the newly-added delta = X_bits[f] \ X_bits[f - 1]
        // for each second continuous feature, scan the ordered threshold
        // block once and maintain the four child counters.

        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int start_idx = continuous_starts[(std::size_t)cont_pos];
            const int end_idx = continuous_group_end_(cont_pos);

            if (start_idx >= end_idx) {
                continue;
            }

            if (num_classes == 2) {
                const Packed& Ypos = Y_bits[(std::size_t)1];

                int L_second_left_n = 0;
                int L_second_left_pos = 0;

                int R_second_left_n = 0;
                int R_second_left_pos = 0;

                int prev_L_second_left_n = -1;
                int prev_L_second_left_pos = -1;
                int prev_R_second_left_n = -1;
                int prev_R_second_left_pos = -1;

                for (int f2 = start_idx; f2 < end_idx; ++f2) {
                    const Packed& cur = X_bits[(std::size_t)f2];

                    const Packed* prev =
                        (f2 == start_idx)
                            ? nullptr
                            : &X_bits[(std::size_t)(f2 - 1)];

                    int dL_n = 0;
                    int dL_pos = 0;

                    int dR_n = 0;
                    int dR_pos = 0;

                    for (int w = 0; w < n_words; ++w) {
                        const uint64_t curw = cur.w[(std::size_t)w];
                        const uint64_t prevw =
                            prev ? prev->w[(std::size_t)w] : 0ULL;

                        const uint64_t delta = curw & ~prevw;

                        const uint64_t dL =
                            delta & Lroot.w[(std::size_t)w];

                        const uint64_t dR =
                            delta & Rroot.w[(std::size_t)w];

                        dL_n += popcnt64(dL);
                        dR_n += popcnt64(dR);

                        dL_pos += popcnt64(dL & Ypos.w[(std::size_t)w]);
                        dR_pos += popcnt64(dR & Ypos.w[(std::size_t)w]);
                    }

                    L_second_left_n += dL_n;
                    L_second_left_pos += dL_pos;

                    R_second_left_n += dR_n;
                    R_second_left_pos += dR_pos;

                    if (scanL) {
                        const bool same_as_prev_L =
                            L_second_left_n == prev_L_second_left_n &&
                            L_second_left_pos == prev_L_second_left_pos;

                        prev_L_second_left_n = L_second_left_n;
                        prev_L_second_left_pos = L_second_left_pos;

                        if (!same_as_prev_L &&
                            L_second_left_n != 0 &&
                            L_second_left_n != nLroot) {
                            const int L_second_right_n =
                                nLroot - L_second_left_n;

                            const int L_second_right_pos =
                                posLroot - L_second_left_pos;

                            const int candL =
                                leaf_objective_binary_from_counts(
                                    L_second_left_n,
                                    L_second_left_pos
                                )
                                +
                                leaf_objective_binary_from_counts(
                                    L_second_right_n,
                                    L_second_right_pos
                                );

                            if (candL < bestL) {
                                bestL = candL;
                            }
                        }
                    }

                    if (scanR) {
                        const bool same_as_prev_R =
                            R_second_left_n == prev_R_second_left_n &&
                            R_second_left_pos == prev_R_second_left_pos;

                        prev_R_second_left_n = R_second_left_n;
                        prev_R_second_left_pos = R_second_left_pos;

                        if (!same_as_prev_R &&
                            R_second_left_n != 0 &&
                            R_second_left_n != nRroot) {
                            const int R_second_right_n =
                                nRroot - R_second_left_n;

                            const int R_second_right_pos =
                                posRroot - R_second_left_pos;

                            const int candR =
                                leaf_objective_binary_from_counts(
                                    R_second_left_n,
                                    R_second_left_pos
                                )
                                +
                                leaf_objective_binary_from_counts(
                                    R_second_right_n,
                                    R_second_right_pos
                                );

                            if (candR < bestR) {
                                bestR = candR;
                            }
                        }
                    }

                    if (bestL + bestR <= 2 * gamma) {
                        goto done_scanning;
                    }
                }
            } else {
                std::vector<int> L_second_left_cnt((std::size_t)num_classes, 0);
                std::vector<int> R_second_left_cnt((std::size_t)num_classes, 0);

                std::vector<int> prev_L_second_left_cnt((std::size_t)num_classes, -1);
                std::vector<int> prev_R_second_left_cnt((std::size_t)num_classes, -1);

                std::vector<int> L_second_right_cnt((std::size_t)num_classes, 0);
                std::vector<int> R_second_right_cnt((std::size_t)num_classes, 0);

                for (int f2 = start_idx; f2 < end_idx; ++f2) {
                    const Packed& cur = X_bits[(std::size_t)f2];

                    const Packed* prev =
                        (f2 == start_idx)
                            ? nullptr
                            : &X_bits[(std::size_t)(f2 - 1)];

                    for (int w = 0; w < n_words; ++w) {
                        const uint64_t curw = cur.w[(std::size_t)w];
                        const uint64_t prevw =
                            prev ? prev->w[(std::size_t)w] : 0ULL;

                        const uint64_t delta = curw & ~prevw;

                        const uint64_t dL =
                            delta & Lroot.w[(std::size_t)w];

                        const uint64_t dR =
                            delta & Rroot.w[(std::size_t)w];

                        if (dL) {
                            for (int c = 0; c < num_classes; ++c) {
                                L_second_left_cnt[(std::size_t)c] +=
                                    popcnt64(
                                        dL & Y_bits[(std::size_t)c].w[(std::size_t)w]
                                    );
                            }
                        }

                        if (dR) {
                            for (int c = 0; c < num_classes; ++c) {
                                R_second_left_cnt[(std::size_t)c] +=
                                    popcnt64(
                                        dR & Y_bits[(std::size_t)c].w[(std::size_t)w]
                                    );
                            }
                        }
                    }

                    if (scanL) {
                        const bool same_as_prev_L =
                            L_second_left_cnt == prev_L_second_left_cnt;

                        prev_L_second_left_cnt = L_second_left_cnt;

                        int L_second_left_n = 0;
                        for (int c = 0; c < num_classes; ++c) {
                            L_second_left_n +=
                                L_second_left_cnt[(std::size_t)c];
                        }

                        if (!same_as_prev_L &&
                            L_second_left_n != 0 &&
                            L_second_left_n != nLroot) {
                            for (int c = 0; c < num_classes; ++c) {
                                L_second_right_cnt[(std::size_t)c] =
                                    cntLroot[(std::size_t)c] -
                                    L_second_left_cnt[(std::size_t)c];
                            }

                            const int candL =
                                leaf_objective_multiclass_from_counts_(
                                    L_second_left_cnt
                                )
                                +
                                leaf_objective_multiclass_from_counts_(
                                    L_second_right_cnt
                                );

                            if (candL < bestL) {
                                bestL = candL;
                            }
                        }
                    }

                    if (scanR) {
                        const bool same_as_prev_R =
                            R_second_left_cnt == prev_R_second_left_cnt;

                        prev_R_second_left_cnt = R_second_left_cnt;

                        int R_second_left_n = 0;
                        for (int c = 0; c < num_classes; ++c) {
                            R_second_left_n +=
                                R_second_left_cnt[(std::size_t)c];
                        }

                        if (!same_as_prev_R &&
                            R_second_left_n != 0 &&
                            R_second_left_n != nRroot) {
                            for (int c = 0; c < num_classes; ++c) {
                                R_second_right_cnt[(std::size_t)c] =
                                    cntRroot[(std::size_t)c] -
                                    R_second_left_cnt[(std::size_t)c];
                            }

                            const int candR =
                                leaf_objective_multiclass_from_counts_(
                                    R_second_left_cnt
                                )
                                +
                                leaf_objective_multiclass_from_counts_(
                                    R_second_right_cnt
                                );

                            if (candR < bestR) {
                                bestR = candR;
                            }
                        }
                    }

                    if (bestL + bestR <= 2 * gamma) {
                        goto done_scanning;
                    }
                }
            }
        }

    done_scanning:

        if (proxy_caching_enabled) {
            if (!have_cached_L) {
                cache_lickety_if_true_(
                    kL,
                    DEPTH,
                    KTAG,
                    bestL,
                    /*allow_cache=*/true
                );
            }

            if (!have_cached_R) {
                cache_lickety_if_true_(
                    kR,
                    DEPTH,
                    KTAG,
                    bestR,
                    /*allow_cache=*/true
                );
            }
        }

        (void)incumbent;
        return bestL + bestR;
    }
    
    int special_depth2_exact_bitvector_no_cache_(
        const Packed& mask,
        int leaf_loss,
        int n_sub,
        const PathKey& pk
    ) {
        constexpr int8_t DEPTH = 2;
        constexpr int8_t KTAG  = 1;

        uint64_t kmask = 0;
        int cached = 0;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);

            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) {
                return cached;
            }
        }

        int best_sum = leaf_loss;

        const int first_cont = first_continuous_feature_();

        Packed L(n_words), R(n_words);

        // ordinary non-continuous binary root features.
        for (int f1 = 0; f1 < first_cont; ++f1) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[(std::size_t)f1], L, R, left_n);

                if (left_n == 0 || left_n == n_sub) {
                    continue;
                }
            } else {
                and_bits(mask, X_bits[(std::size_t)f1], L);
                andnot_bits(mask, X_bits[(std::size_t)f1], R);

                if (!L.any() || !R.any()) {
                    continue;
                }
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                f1,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const int sum = special_depth2_fixed_root_split_sum_bitvector_(
                L,
                R,
                *pkLp,
                *pkRp,
                best_sum
            );

            if (sum < best_sum) {
                best_sum = sum;

                if (best_sum <= 2 * gamma) {
                    goto done_depth2_scan;
                }
            }
        }

        // continuous root features represented as contiguous monotone threshold blocks.
        // we update the root-left mask incrementally by delta = mask & (X_bits[f] \ X_bits[f - 1])
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int start_idx = continuous_starts[(std::size_t)cont_pos];
            const int end_idx = continuous_group_end_(cont_pos);

            if (start_idx >= end_idx) {
                continue;
            }

            L.clear();

            if (num_classes == 2) {
                const Packed& Ypos = Y_bits[(std::size_t)1];

                int left_n = 0;
                int left_pos = 0;

                int prev_left_n = -1;
                int prev_left_pos = -1;

                for (int f1 = start_idx; f1 < end_idx; ++f1) {
                    const Packed& cur = X_bits[(std::size_t)f1];

                    const Packed* prev =
                        (f1 == start_idx)
                            ? nullptr
                            : &X_bits[(std::size_t)(f1 - 1)];

                    int delta_n = 0;
                    int delta_pos = 0;

                    for (int w = 0; w < n_words; ++w) {
                        const uint64_t curw = cur.w[(std::size_t)w];
                        const uint64_t prevw =
                            prev ? prev->w[(std::size_t)w] : 0ULL;

                        const uint64_t delta =
                            mask.w[(std::size_t)w] &
                            curw &
                            ~prevw;

                        L.w[(std::size_t)w] |= delta;

                        delta_n += popcnt64(delta);
                        delta_pos += popcnt64(
                            delta & Ypos.w[(std::size_t)w]
                        );
                    }

                    L.w[(std::size_t)(n_words - 1)] &= tail_mask;

                    left_n += delta_n;
                    left_pos += delta_pos;

                    const bool same_as_prev =
                        left_n == prev_left_n &&
                        left_pos == prev_left_pos;

                    prev_left_n = left_n;
                    prev_left_pos = left_pos;

                    if (same_as_prev || left_n == 0 || left_n == n_sub) {
                        continue;
                    }

                    andnot_bits(mask, L, R);

                    if (!R.any()) {
                        break;
                    }

                    const PathKey* pkLp = &empty_pk();
                    const PathKey* pkRp = &empty_pk();

                    PathKey pkL_local;
                    PathKey pkR_local;

                    make_child_pks_if_needed_(
                        f1,
                        pk,
                        pkLp,
                        pkRp,
                        pkL_local,
                        pkR_local
                    );

                    const int sum = special_depth2_fixed_root_split_sum_bitvector_(
                        L,
                        R,
                        *pkLp,
                        *pkRp,
                        best_sum
                    );

                    if (sum < best_sum) {
                        best_sum = sum;

                        if (best_sum <= 2 * gamma) {
                            goto done_depth2_scan;
                        }
                    }
                }
            } else {
                std::vector<int> left_counts((std::size_t)num_classes, 0);
                std::vector<int> prev_left_counts((std::size_t)num_classes, -1);

                int left_n = 0;

                for (int f1 = start_idx; f1 < end_idx; ++f1) {
                    const Packed& cur = X_bits[(std::size_t)f1];

                    const Packed* prev =
                        (f1 == start_idx)
                            ? nullptr
                            : &X_bits[(std::size_t)(f1 - 1)];

                    int delta_n = 0;

                    for (int w = 0; w < n_words; ++w) {
                        const uint64_t curw = cur.w[(std::size_t)w];
                        const uint64_t prevw =
                            prev ? prev->w[(std::size_t)w] : 0ULL;

                        const uint64_t delta =
                            mask.w[(std::size_t)w] &
                            curw &
                            ~prevw;

                        L.w[(std::size_t)w] |= delta;

                        delta_n += popcnt64(delta);

                        if (delta) {
                            for (int c = 0; c < num_classes; ++c) {
                                left_counts[(std::size_t)c] +=
                                    popcnt64(
                                        delta &
                                        Y_bits[(std::size_t)c].w[(std::size_t)w]
                                    );
                            }
                        }
                    }

                    L.w[(std::size_t)(n_words - 1)] &= tail_mask;

                    left_n += delta_n;

                    const bool same_as_prev =
                        left_counts == prev_left_counts;

                    prev_left_counts = left_counts;

                    if (same_as_prev || left_n == 0 || left_n == n_sub) {
                        continue;
                    }

                    andnot_bits(mask, L, R);

                    if (!R.any()) {
                        break;
                    }

                    const PathKey* pkLp = &empty_pk();
                    const PathKey* pkRp = &empty_pk();

                    PathKey pkL_local;
                    PathKey pkR_local;

                    make_child_pks_if_needed_(
                        f1,
                        pk,
                        pkLp,
                        pkRp,
                        pkL_local,
                        pkR_local
                    );

                    const int sum = special_depth2_fixed_root_split_sum_bitvector_(
                        L,
                        R,
                        *pkLp,
                        *pkRp,
                        best_sum
                    );

                    if (sum < best_sum) {
                        best_sum = sum;

                        if (best_sum <= 2 * gamma) {
                            goto done_depth2_scan;
                        }
                    }
                }
            }
        }

    done_depth2_scan:

        if (proxy_caching_enabled) {
            cache_lickety_if_true_(
                kmask,
                DEPTH,
                KTAG,
                best_sum,
                /*allow_cache=*/true
            );
        }

        return best_sum;
    }
    
    NumericalGreedyState make_numerical_state_for_mask_(
        const Packed& mask
    ) const {
        if (numerical_X_cols_for_greedy.size() != continuous_starts.size() ||
            numerical_global_sorted_idx.size() != continuous_starts.size() ||
            numerical_unique_values_for_greedy.size() != continuous_starts.size()) {
            throw std::logic_error(
                "Numerical greedy representation is not aligned with continuous_starts."
            );
        }

        NumericalGreedyState state;
        state.sorted_idx_by_num.resize(numerical_global_sorted_idx.size());

        for (std::size_t g = 0; g < numerical_global_sorted_idx.size(); ++g) {
            const auto& global_order = numerical_global_sorted_idx[g];
            auto& active_order = state.sorted_idx_by_num[g];

            active_order.clear();
            active_order.reserve(global_order.size());

            for (int row : global_order) {
                if (mask_has_row_(mask, row)) {
                    active_order.push_back(row);
                }
            }
        }

        return state;
    }

    int special_depth2_fixed_root_split_sum_numerical_(
        const Packed& Lroot,
        const Packed& Rroot,
        const PathKey& pkL,
        const PathKey& pkR,
        const NumericalGreedyState& state,
        int incumbent
    ) {
        constexpr int8_t DEPTH = 1;
        constexpr int8_t KTAG  = 0;

        uint64_t kL = 0;
        uint64_t kR = 0;

        bool have_cached_L = false;
        bool have_cached_R = false;

        int bestL = 0;
        int bestR = 0;

        if (proxy_caching_enabled) {
            kL = key_of_subproblem(Lroot, pkL);
            kR = key_of_subproblem(Rroot, pkR);

            have_cached_L = try_get_lickety_cached_(kL, DEPTH, KTAG, bestL);
            have_cached_R = try_get_lickety_cached_(kR, DEPTH, KTAG, bestR);

            if (have_cached_L && have_cached_R) {
                return bestL + bestR;
            }
        }

        const int leafL = leaf_objective(Lroot);
        const int leafR = leaf_objective(Rroot);

        if (!have_cached_L) bestL = leafL;
        if (!have_cached_R) bestR = leafR;

        int nLroot = 0;
        int nRroot = 0;
        int posLroot = 0;
        int posRroot = 0;

        std::vector<int> countsLroot;
        std::vector<int> countsRroot;

        if (num_classes == 2) {
            count_total_pos_binary(Lroot, nLroot, posLroot);
            count_total_pos_binary(Rroot, nRroot, posRroot);
        } else {
            nLroot = count_total(Lroot);
            nRroot = count_total(Rroot);

            count_per_class(Lroot, countsLroot);
            count_per_class(Rroot, countsRroot);
        }

        const bool scanL =
            !have_cached_L &&
            nLroot > 1 &&
            leafL > 2 * gamma;

        const bool scanR =
            !have_cached_R &&
            nRroot > 1 &&
            leafR > 2 * gamma;

        if (!scanL && !scanR) {
            if (proxy_caching_enabled) {
                if (!have_cached_L) {
                    cache_lickety_if_true_(
                        kL,
                        DEPTH,
                        KTAG,
                        bestL,
                        /*allow_cache=*/cache_cheap_subproblems || leafL > 2 * gamma
                    );
                }

                if (!have_cached_R) {
                    cache_lickety_if_true_(
                        kR,
                        DEPTH,
                        KTAG,
                        bestR,
                        /*allow_cache=*/cache_cheap_subproblems || leafR > 2 * gamma
                    );
                }
            }

            return bestL + bestR;
        }

        const int first_cont = first_continuous_feature_();
        const int G = (int)continuous_starts.size();

        if ((int)state.sorted_idx_by_num.size() != G) {
            throw std::logic_error(
                "Numerical depth-2 parent state is not aligned with continuous_starts."
            );
        }

        if ((int)numerical_X_cols_for_greedy.size() != G ||
            (int)numerical_unique_values_for_greedy.size() != G) {
            throw std::logic_error(
                "Numerical greedy arrays are not aligned with continuous_starts."
            );
        }

        Packed LL(n_words), LR(n_words);
        Packed RL(n_words), RR(n_words);

        // ordinary binary second-level features.
        for (int f2 = 0; f2 < first_cont; ++f2) {
            if (scanL) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(
                        Lroot,
                        X_bits[(std::size_t)f2],
                        LL,
                        LR,
                        left_n
                    );

                    if (left_n != 0 && left_n != nLroot) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                } else {
                    and_bits(Lroot, X_bits[(std::size_t)f2], LL);
                    andnot_bits(Lroot, X_bits[(std::size_t)f2], LR);

                    if (LL.any() && LR.any()) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                }
            }

            if (scanR) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(
                        Rroot,
                        X_bits[(std::size_t)f2],
                        RL,
                        RR,
                        left_n
                    );

                    if (left_n != 0 && left_n != nRroot) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                } else {
                    and_bits(Rroot, X_bits[(std::size_t)f2], RL);
                    andnot_bits(Rroot, X_bits[(std::size_t)f2], RR);

                    if (RL.any() && RR.any()) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                }
            }

            if (bestL + bestR <= 2 * gamma) {
                goto done_numerical_fixed_root;
            }
        }

        // numerical continuous second-level features.
        // For each second numerical feature g, scan the parent sorted rows once
        // and maintain four counters:
        //   Lroot + second-left, Lroot + second-right,
        //   Rroot + second-left, Rroot + second-right.
        for (int g = 0; g < G; ++g) {
            const auto& sorted_rows = state.sorted_idx_by_num[(std::size_t)g];

            if (sorted_rows.size() <= 1) {
                continue;
            }

            const std::vector<double>& x =
                numerical_X_cols_for_greedy[(std::size_t)g];

            if (num_classes == 2) {
                int L_second_left_n = 0;
                int L_second_left_pos = 0;

                int R_second_left_n = 0;
                int R_second_left_pos = 0;

                std::size_t i = 0;

                while (i < sorted_rows.size()) {
                    const double value = x[(std::size_t)sorted_rows[i]];

                    // move the entire tie block to second-left.
                    std::size_t j = i;
                    while (j < sorted_rows.size() &&
                           x[(std::size_t)sorted_rows[j]] == value) {
                        const int row = sorted_rows[j];

                        if (mask_has_row_(Lroot, row)) {
                            ++L_second_left_n;
                            if (y_train[(std::size_t)row] == 1) {
                                ++L_second_left_pos;
                            }
                        } else if (mask_has_row_(Rroot, row)) {
                            ++R_second_left_n;
                            if (y_train[(std::size_t)row] == 1) {
                                ++R_second_left_pos;
                            }
                        }

                        ++j;
                    }

                    // no valid threshold after the maximum parent value.
                    if (j >= sorted_rows.size()) {
                        break;
                    }

                    if (scanL &&
                        L_second_left_n != 0 &&
                        L_second_left_n != nLroot) {
                        const int L_second_right_n =
                            nLroot - L_second_left_n;

                        const int L_second_right_pos =
                            posLroot - L_second_left_pos;

                        const int candL =
                            leaf_objective_binary_from_counts(
                                L_second_left_n,
                                L_second_left_pos
                            )
                            +
                            leaf_objective_binary_from_counts(
                                L_second_right_n,
                                L_second_right_pos
                            );

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }

                    if (scanR &&
                        R_second_left_n != 0 &&
                        R_second_left_n != nRroot) {
                        const int R_second_right_n =
                            nRroot - R_second_left_n;

                        const int R_second_right_pos =
                            posRroot - R_second_left_pos;

                        const int candR =
                            leaf_objective_binary_from_counts(
                                R_second_left_n,
                                R_second_left_pos
                            )
                            +
                            leaf_objective_binary_from_counts(
                                R_second_right_n,
                                R_second_right_pos
                            );

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }

                    if (bestL + bestR <= 2 * gamma) {
                        goto done_numerical_fixed_root;
                    }

                    i = j;
                }
            } else {
                std::vector<int> L_second_left_counts((std::size_t)num_classes, 0);
                std::vector<int> R_second_left_counts((std::size_t)num_classes, 0);

                std::vector<int> L_second_right_counts((std::size_t)num_classes, 0);
                std::vector<int> R_second_right_counts((std::size_t)num_classes, 0);

                int L_second_left_n = 0;
                int R_second_left_n = 0;

                std::size_t i = 0;

                while (i < sorted_rows.size()) {
                    const double value = x[(std::size_t)sorted_rows[i]];

                    // move the entire tie block to second-left.
                    std::size_t j = i;
                    while (j < sorted_rows.size() &&
                           x[(std::size_t)sorted_rows[j]] == value) {
                        const int row = sorted_rows[j];
                        const int c = y_train[(std::size_t)row];

                        if (c < 0 || c >= num_classes) {
                            throw std::logic_error(
                                "Class label out of range in numerical depth-2 solver."
                            );
                        }

                        if (mask_has_row_(Lroot, row)) {
                            ++L_second_left_n;
                            ++L_second_left_counts[(std::size_t)c];
                        } else if (mask_has_row_(Rroot, row)) {
                            ++R_second_left_n;
                            ++R_second_left_counts[(std::size_t)c];
                        }

                        ++j;
                    }

                    // no valid threshold after the maximum parent value.
                    if (j >= sorted_rows.size()) {
                        break;
                    }

                    if (scanL &&
                        L_second_left_n != 0 &&
                        L_second_left_n != nLroot) {
                        for (int c = 0; c < num_classes; ++c) {
                            L_second_right_counts[(std::size_t)c] =
                                countsLroot[(std::size_t)c] -
                                L_second_left_counts[(std::size_t)c];
                        }

                        const int candL =
                            leaf_objective_multiclass_from_counts_(
                                L_second_left_counts
                            )
                            +
                            leaf_objective_multiclass_from_counts_(
                                L_second_right_counts
                            );

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }

                    if (scanR &&
                        R_second_left_n != 0 &&
                        R_second_left_n != nRroot) {
                        for (int c = 0; c < num_classes; ++c) {
                            R_second_right_counts[(std::size_t)c] =
                                countsRroot[(std::size_t)c] -
                                R_second_left_counts[(std::size_t)c];
                        }

                        const int candR =
                            leaf_objective_multiclass_from_counts_(
                                R_second_left_counts
                            )
                            +
                            leaf_objective_multiclass_from_counts_(
                                R_second_right_counts
                            );

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }

                    if (bestL + bestR <= 2 * gamma) {
                        goto done_numerical_fixed_root;
                    }

                    i = j;
                }
            }
        }

    done_numerical_fixed_root:

        if (proxy_caching_enabled) {
            if (!have_cached_L) {
                cache_lickety_if_true_(
                    kL,
                    DEPTH,
                    KTAG,
                    bestL,
                    /*allow_cache=*/true
                );
            }

            if (!have_cached_R) {
                cache_lickety_if_true_(
                    kR,
                    DEPTH,
                    KTAG,
                    bestR,
                    /*allow_cache=*/true
                );
            }
        }

        (void)incumbent;
        return bestL + bestR;
    }

    int special_depth2_exact_numerical_no_cache_(
        const Packed& mask,
        int leaf_loss,
        int n_sub,
        const NumericalGreedyState& state,
        const PathKey& pk
    ) {
        constexpr int8_t DEPTH = 2;
        constexpr int8_t KTAG  = 1;

        uint64_t kmask = 0;
        int cached = 0;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);

            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) {
                return cached;
            }
        }

        int best_sum = leaf_loss;

        const int first_cont = first_continuous_feature_();
        const int G = (int)continuous_starts.size();

        if ((int)state.sorted_idx_by_num.size() != G) {
            throw std::logic_error(
                "Numerical depth-2 state is not aligned with continuous_starts."
            );
        }

        if ((int)numerical_X_cols_for_greedy.size() != G ||
            (int)numerical_unique_values_for_greedy.size() != G) {
            throw std::logic_error(
                "Numerical greedy arrays are not aligned with continuous_starts."
            );
        }

        Packed L(n_words), R(n_words);

        // root split on ordinary binary features.
        // We do not partition numerical state anymore. The fixed-root solver
        // scans the parent sorted lists once per second feature.
        for (int f1 = 0; f1 < first_cont; ++f1) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[(std::size_t)f1], L, R, left_n);

                if (left_n == 0 || left_n == n_sub) {
                    continue;
                }
            } else {
                and_bits(mask, X_bits[(std::size_t)f1], L);
                andnot_bits(mask, X_bits[(std::size_t)f1], R);

                if (!L.any() || !R.any()) {
                    continue;
                }
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                f1,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const int sum = special_depth2_fixed_root_split_sum_numerical_(
                L,
                R,
                *pkLp,
                *pkRp,
                state,
                best_sum
            );

            if (sum < best_sum) {
                best_sum = sum;

                if (best_sum <= 2 * gamma) {
                    goto done_numerical_depth2;
                }
            }
        }

        // root split on numerical continuous features.
        // Scan root feature sorted rows once, building L incrementally.
        // For each valid root threshold, call the fixed-root solver using
        // the original parent state.
        for (int g = 0; g < G; ++g) {
            const auto& sorted_rows = state.sorted_idx_by_num[(std::size_t)g];

            if (sorted_rows.size() <= 1) {
                continue;
            }

            const std::vector<double>& x =
                numerical_X_cols_for_greedy[(std::size_t)g];

            const int start_feat = continuous_starts[(std::size_t)g];
            const int end_feat = continuous_group_end_(g);

            if (start_feat >= end_feat) {
                continue;
            }

            const std::vector<double>& vals =
                numerical_unique_values_for_greedy[(std::size_t)g];

            if (vals.size() <= 1) {
                continue;
            }

            const int expected_end_feat = start_feat + (int)vals.size() - 1;
            if (expected_end_feat != end_feat) {
                throw std::logic_error(
                    "continuous_starts/end does not match numerical_unique_values_for_greedy."
                );
            }

            L.clear();

            int nL = 0;
            std::size_t i = 0;

            while (i < sorted_rows.size()) {
                const double value = x[(std::size_t)sorted_rows[i]];

                // move the whole tie block for this root feature to root-left.
                std::size_t j = i;
                while (j < sorted_rows.size() &&
                       x[(std::size_t)sorted_rows[j]] == value) {
                    const int row = sorted_rows[j];

                    L.w[(std::size_t)(row >> 6)] |=
                        (1ULL << (row & 63));

                    ++nL;
                    ++j;
                }

                L.w[(std::size_t)(n_words - 1)] &= tail_mask;

                // no valid root threshold after the maximum value.
                if (j >= sorted_rows.size()) {
                    break;
                }

                if (nL == 0 || nL == n_sub) {
                    i = j;
                    continue;
                }

                // map value to the existing threshold column only as a consistency check.
                auto it = std::lower_bound(vals.begin(), vals.end(), value);

                if (it == vals.end() || *it != value) {
                    throw std::logic_error(
                        "Active numerical value not found in global unique values."
                    );
                }

                const int global_idx =
                    (int)std::distance(vals.begin(), it);

                if (global_idx + 1 >= (int)vals.size()) {
                    break;
                }

                const int feat = start_feat + global_idx;

                if (feat < start_feat || feat >= end_feat) {
                    throw std::logic_error(
                        "Mapped numerical threshold feature index is out of group range."
                    );
                }

                andnot_bits(mask, L, R);

                if (!R.any()) {
                    break;
                }

                const PathKey* pkLp = &empty_pk();
                const PathKey* pkRp = &empty_pk();

                PathKey pkL_local;
                PathKey pkR_local;

                make_child_pks_if_needed_(
                    feat,
                    pk,
                    pkLp,
                    pkRp,
                    pkL_local,
                    pkR_local
                );

                const int sum = special_depth2_fixed_root_split_sum_numerical_(
                    L,
                    R,
                    *pkLp,
                    *pkRp,
                    state,
                    best_sum
                );

                if (sum < best_sum) {
                    best_sum = sum;

                    if (best_sum <= 2 * gamma) {
                        goto done_numerical_depth2;
                    }
                }

                i = j;
            }
        }

    done_numerical_depth2:

        if (proxy_caching_enabled) {
            cache_lickety_if_true_(
                kmask,
                DEPTH,
                KTAG,
                best_sum,
                /*allow_cache=*/true
            );
        }

        return best_sum;
    }
    
    int special_depth2_exact_numerical_no_cache_(
        const Packed& mask,
        int leaf_loss,
        int n_sub,
        const PathKey& pk
    ) {
        NumericalGreedyState state = make_numerical_state_for_mask_(mask);

        return special_depth2_exact_numerical_no_cache_(
            mask,
            leaf_loss,
            n_sub,
            state,
            pk
        );
    }

    int depthd_exact_solver_cached_continuous(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        const int8_t DEPTH = depth_budget;
        const int8_t KTAG = depth_budget - 1;

        uint64_t kmask = 0;
        int cached = 0;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);

            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) {
                return cached;
            }
        }

        // if (depth_budget == 2) {
        //     int n_sub = 0;
        //     int leaf_loss = 0;

        //     if (num_classes == 2) {
        //         int pos = 0;
        //         count_total_pos_binary(mask, n_sub, pos);
        //         leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        //     } else {
        //         n_sub = count_total(mask);
        //         leaf_loss = leaf_objective(mask);
        //     }

        //     return special_depth2_exact_bitvector_no_cache_(
        //         mask,
        //         leaf_loss,
        //         n_sub,
        //         pk
        //     );
        // }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(
                    kmask,
                    DEPTH,
                    KTAG,
                    leaf_loss,
                    /*allow_cache=*/cache_cheap_subproblems
                );
            }

            return leaf_loss;
        }

        // if (depth_budget == 2) {
        //     if (greedy_continuous_mode == GreedyContinuousMode::NUMERICAL) {
        //         return special_depth2_exact_numerical_no_cache_(
        //             mask,
        //             leaf_loss,
        //             n_sub,
        //             pk
        //         );
        //     }

        //     return special_depth2_exact_bitvector_no_cache_(
        //         mask,
        //         leaf_loss,
        //         n_sub,
        //         pk
        //     );
        // }

        int best_sum = leaf_loss;

        Packed L(n_words), R(n_words);

        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        // binary / ordinary features.
        for (int f = 0; f < first_cont; ++f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);

                if (left_n == 0 || left_n == n_sub) {
                    continue;
                }
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);

                if (!L.any() || !R.any()) {
                    continue;
                }
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            PathKey pkL_local;
            PathKey pkR_local;

            make_child_pks_if_needed_(
                f,
                pk,
                pkLp,
                pkRp,
                pkL_local,
                pkR_local
            );

            const ContinuousPath* cpathLp = &cpath;
            const ContinuousPath* cpathRp = &cpath;

            ContinuousPath cpathL_local;
            ContinuousPath cpathR_local;

            make_child_continuous_paths_if_needed_(
                f,
                cpath,
                cpathLp,
                cpathRp,
                cpathL_local,
                cpathR_local
            );

            int left_best;
            int right_best;

            if (depth_budget == 1) {
                left_best = leaf_objective(L);
                right_best = leaf_objective(R);
            } else {
                left_best = depthd_exact_solver_cached_continuous(
                    L,
                    depth_budget - 1,
                    *pkLp,
                    *cpathLp
                );

                right_best = depthd_exact_solver_cached_continuous(
                    R,
                    depth_budget - 1,
                    *pkRp,
                    *cpathRp
                );
            }

            const int sum = left_best + right_best;

            if (sum < best_sum) {
                best_sum = sum;
            }
        }

        // continuous groups
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    cpath
                );

            if (start_idx >= end_idx) continue;

            ContinuousBestSplitResult cres =
                search_continuous_feature_for_best_split_continuous_(
                    mask,
                    depth_budget,
                    /*child_k=*/depth_budget - 2,
                    pk,
                    cpath,
                    start_idx,
                    end_idx,
                    best_sum,
                    ContinuousEvalMode::Exact
                );

            if (cres.best_feat >= 0 && cres.best_sum < best_sum) {
                best_sum = cres.best_sum;
            }
        }

        const int ans = best_sum;

        if (proxy_caching_enabled) {
            cache_lickety_if_true_(
                kmask,
                depth_budget,
                KTAG,
                ans,
                /*allow_cache=*/true
            );
        }

        return ans;
    }

    struct GainSplitResult {
        double score = -std::numeric_limits<double>::infinity();
        int feat = -1;
    };

    inline double split_score_from_counts_binary_(
        int n,
        int pos,
        int nL,
        int posL
    ) const {
        const int nR = n - nL;
        if (n <= 0 || nL <= 0 || nR <= 0) {
            return -std::numeric_limits<double>::infinity();
        }

        const int posR = pos - posL;

        const double left_H  = entropy((double)posL / (double)nL);
        const double right_H = entropy((double)posR / (double)nR);

        // parent entropy is constant, so maximize negative weighted child entropy.
        return -((double)nL / (double)n) * left_H
            -((double)nR / (double)n) * right_H;
    }

    inline double split_score_from_counts_multiclass_(
        const std::vector<int>& parent_counts,
        const std::vector<int>& left_counts,
        int n,
        int nL
    ) const {
        const int nR = n - nL;
        if (n <= 0 || nL <= 0 || nR <= 0) {
            return -std::numeric_limits<double>::infinity();
        }

        std::vector<int> right_counts((size_t)num_classes, 0);
        for (int c = 0; c < num_classes; ++c) {
            right_counts[(size_t)c] =
                parent_counts[(size_t)c] - left_counts[(size_t)c];
        }

        const double left_H  = entropy_multiclass(left_counts, nL);
        const double right_H = entropy_multiclass(right_counts, nR);

        // parent entropy is constant, so maximize negative weighted child entropy.
        return -((double)nL / (double)n) * left_H
            -((double)nR / (double)n) * right_H;
    }

    GainSplitResult best_binary_score_split_(
        const Packed& mask,
        int first_feat,
        int end_feat,
        int n_total,
        int pos_total
    ) const {
        GainSplitResult best;

        if (first_feat >= end_feat) return best;

        const Packed& Ypos = Y_bits[(size_t)1];

        for (int f = first_feat; f < end_feat; ++f) {
            const Packed& Xf = X_bits[(size_t)f];

            int nL = 0;
            int posL = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] & Xf.w[(size_t)w];

                nL += popcnt64(left_bits);
                posL += popcnt64(left_bits & Ypos.w[(size_t)w]);
            }

            if (nL == 0 || nL == n_total) continue;

            const double score = split_score_from_counts_binary_(
                n_total,
                pos_total,
                nL,
                posL
            );

            if (score > best.score) {
                best.score = score;
                best.feat = f;
            }
        }

        return best;
    }

    GainSplitResult best_continuous_score_split_(
        const Packed& mask,
        int start_idx,
        int end_idx,
        int n_total,
        int pos_total
    ) const {
        GainSplitResult best;

        if (start_idx >= end_idx) return best;

        const Packed& Ypos = Y_bits[(size_t)1];

        int prev_nL = -1;
        int prev_posL = -1;

        for (int feat = start_idx; feat < end_idx; ++feat) {
            const Packed& Xf = X_bits[(size_t)feat];

            int nL = 0;
            int posL = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] & Xf.w[(size_t)w];

                nL += popcnt64(left_bits);
                posL += popcnt64(left_bits & Ypos.w[(size_t)w]);
            }

            // same active split as previous threshold in this subproblem.
            if (nL == prev_nL && posL == prev_posL) {
                continue;
            }

            prev_nL = nL;
            prev_posL = posL;

            if (nL == 0 || nL == n_total) continue;

            const double score = split_score_from_counts_binary_(
                n_total,
                pos_total,
                nL,
                posL
            );

            if (score > best.score) {
                best.score = score;
                best.feat = feat;
            }
        }

        return best;
    }

    GainSplitResult best_binary_score_split_multiclass_(
        const Packed& mask,
        int first_feat,
        int end_feat,
        int n_total,
        const std::vector<int>& parent_counts
    ) const {
        GainSplitResult best;

        if (first_feat >= end_feat) return best;

        std::vector<int> left_counts((size_t)num_classes, 0);

        for (int f = first_feat; f < end_feat; ++f) {
            std::fill(left_counts.begin(), left_counts.end(), 0);

            const Packed& Xf = X_bits[(size_t)f];

            int nL = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] & Xf.w[(size_t)w];

                nL += popcnt64(left_bits);

                for (int c = 0; c < num_classes; ++c) {
                    left_counts[(size_t)c] += popcnt64(
                        left_bits & Y_bits[(size_t)c].w[(size_t)w]
                    );
                }
            }

            if (nL == 0 || nL == n_total) continue;

            const double score = split_score_from_counts_multiclass_(
                parent_counts,
                left_counts,
                n_total,
                nL
            );

            if (score > best.score) {
                best.score = score;
                best.feat = f;
            }
        }

        return best;
    }

    GainSplitResult best_continuous_score_split_multiclass_(
        const Packed& mask,
        int start_idx,
        int end_idx,
        int n_total,
        const std::vector<int>& parent_counts
    ) const {
        GainSplitResult best;

        if (start_idx >= end_idx) return best;

        std::vector<int> left_counts((size_t)num_classes, 0);
        std::vector<int> prev_left_counts((size_t)num_classes, -1);

        int prev_nL = -1;

        for (int feat = start_idx; feat < end_idx; ++feat) {
            std::fill(left_counts.begin(), left_counts.end(), 0);

            const Packed& Xf = X_bits[(size_t)feat];

            int nL = 0;

            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] & Xf.w[(size_t)w];

                nL += popcnt64(left_bits);

                for (int c = 0; c < num_classes; ++c) {
                    left_counts[(size_t)c] += popcnt64(
                        left_bits & Y_bits[(size_t)c].w[(size_t)w]
                    );
                }
            }

            if (nL == prev_nL && left_counts == prev_left_counts) {
                continue;
            }

            prev_nL = nL;
            prev_left_counts = left_counts;

            if (nL == 0 || nL == n_total) continue;

            const double score = split_score_from_counts_multiclass_(
                parent_counts,
                left_counts,
                n_total,
                nL
            );

            if (score > best.score) {
                best.score = score;
                best.feat = feat;
            }
        }

        return best;
    }

    // int greedy_numerical_entry_point(
    //     const Packed& mask,
    //     int8_t depth_budget,
    //     const PathKey& pk
    // ) {

    //     if (numerical_X_cols_for_greedy.size() != continuous_starts.size()) {
    //         throw std::logic_error(
    //             "Numerical greedy representation is not aligned with continuous_starts."
    //         );
    //     }

    //     NumericalGreedyState state;
    //     state.sorted_idx_by_num.resize(numerical_global_sorted_idx.size());

    //     for (std::size_t g = 0; g < numerical_global_sorted_idx.size(); ++g) {
    //         const auto& global_order = numerical_global_sorted_idx[g];
    //         auto& active_order = state.sorted_idx_by_num[g];

    //         active_order.reserve(global_order.size());

    //         for (int row : global_order) {
    //             if (mask_has_row_(mask, row)) {
    //                 active_order.push_back(row);
    //             }
    //         }
    //     }

    //     return train_greedy_continuous_numerical(
    //         mask,
    //         depth_budget,
    //         pk,
    //         state
    //     );
    // }

    int greedy_numerical_entry_point(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk
    ) {
        if (numerical_X_cols_for_greedy.size() != continuous_starts.size() ||
            numerical_global_sorted_idx.size() != continuous_starts.size() ||
            numerical_unique_values_for_greedy.size() != continuous_starts.size()) {
            throw std::logic_error(
                "Numerical greedy representation is not aligned with continuous_starts."
            );
        }

        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        // Important: check the greedy cache before paying to build the
        // NumericalGreedyState.
        if (proxy_caching_enabled) {
            const uint64_t kmask = key_of_subproblem(mask, pk);
            const K2 key{kmask, depth_budget};

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        NumericalGreedyState state;
        const std::size_t G = numerical_global_sorted_idx.size();
        state.sorted_idx_by_num.resize(G);

        // true:
        // collect active rows once, then sort those local rows separately for each numerical feature.
        //
        // false: scan each global pre-sorted row list and keep active rows.
        // O(N/64 + k n log n) vs O(k N)
        constexpr bool BUILD_STATE_BY_LOCAL_SORT = false;

        if constexpr (BUILD_STATE_BY_LOCAL_SORT) {
            std::vector<int> active_rows;
            active_rows.reserve((std::size_t)count_total(mask));

            for (int w = 0; w < n_words; ++w) {
                uint64_t bits = mask.w[(std::size_t)w];

                while (bits) {
#if defined(_MSC_VER)
                    unsigned long bidx;
                    _BitScanForward64(&bidx, bits);
                    const int b = static_cast<int>(bidx);
#else
                    const int b = __builtin_ctzll(bits);
#endif
                    const int row = (w << 6) + b;

                    if (row < n_samples) {
                        active_rows.push_back(row);
                    }

                    bits &= (bits - 1);
                }
            }

            for (std::size_t g = 0; g < G; ++g) {
                const std::vector<double>& x =
                    numerical_X_cols_for_greedy[g];

                auto& active_order = state.sorted_idx_by_num[g];
                active_order = active_rows;

                std::stable_sort(
                    active_order.begin(),
                    active_order.end(),
                    [&](int a, int b) {
                        const double xa = x[(std::size_t)a];
                        const double xb = x[(std::size_t)b];

                        if (xa < xb) return true;
                        if (xb < xa) return false;
                        return a < b;
                    }
                );
            }
        } else {
            for (std::size_t g = 0; g < G; ++g) {
                const auto& global_order = numerical_global_sorted_idx[g];
                auto& active_order = state.sorted_idx_by_num[g];

                active_order.clear();
                active_order.reserve(global_order.size());

                for (int row : global_order) {
                    if (mask_has_row_(mask, row)) {
                        active_order.push_back(row);
                    }
                }
            }
        }

        return train_greedy_continuous_numerical(
            mask,
            depth_budget,
            pk,
            state
        );
    }

    GainSplitResult best_numerical_score_split_(
        int num_group,
        const std::vector<int>& sorted_rows,
        int n_total,
        int pos_total,
        const std::vector<int>& parent_counts
    ) const {
        GainSplitResult best;
        best.score = -std::numeric_limits<double>::infinity();
        best.feat = -1;

        if (num_group < 0 || num_group >= (int)continuous_starts.size()) {
            return best;
        }

        if (sorted_rows.size() <= 1) {
            return best;
        }

        const int start_feat = continuous_starts[(std::size_t)num_group];
        const int end_feat = continuous_group_end_(num_group);

        if (start_feat >= end_feat) {
            return best;
        }

        const std::vector<double>& x =
            numerical_X_cols_for_greedy[(std::size_t)num_group];

        const std::vector<double>& vals =
        numerical_unique_values_for_greedy[(std::size_t)num_group];

        if (vals.size() <= 1) {
            return best;
        }

        const int expected_end_feat = start_feat + (int)vals.size() - 1;
        if (expected_end_feat != end_feat) {
            throw std::logic_error(
                "continuous_starts/end does not match numerical_unique_values_for_greedy."
            );
        }

        int nL = 0;
        int posL = 0;

        std::vector<int> left_counts;
        if (num_classes != 2) {
            left_counts.assign((std::size_t)num_classes, 0);
        }

        std::size_t i = 0;
        while (i < sorted_rows.size()) {
            const double value = x[(std::size_t)sorted_rows[i]];

            // move the whole tie block with this value to the left.
            std::size_t j = i;
            while (j < sorted_rows.size() &&
                   x[(std::size_t)sorted_rows[j]] == value) {
                const int row = sorted_rows[j];

                ++nL;

                if (num_classes == 2) {
                    if (y_train[(std::size_t)row] == 1) ++posL;
                } else {
                    const int c = y_train[(std::size_t)row];
                    if (c >= 0 && c < num_classes) {
                        ++left_counts[(std::size_t)c];
                    }
                }

                ++j;
            }


            // cannot split after the maximum value.
            if (j >= sorted_rows.size()) {
                break;
            }

            auto it = std::lower_bound(vals.begin(), vals.end(), value);

            if (it == vals.end() || *it != value) {
                throw std::logic_error(
                    "Active numerical value not found in global unique values."
                );
            }

            const int global_idx = (int)std::distance(vals.begin(), it);

            // Last global value has no threshold column.
            if (global_idx + 1 >= (int)vals.size()) {
                break;
            }

            const int feat = start_feat + global_idx;

            if (feat < start_feat || feat >= end_feat) {
                throw std::logic_error(
                    "Mapped numerical threshold feature index is out of group range."
                );
            }


            if (nL == 0 || nL == n_total) {
                i = j;
                continue;
            }

            double score;

            if (num_classes == 2) {
                const int nR = n_total - nL;
                const int posR = pos_total - posL;

                if (greedy_split_mode == 2) {
                    // greedy mode 2: choose split minimizing child leaf objective.
                    const int left_loss =
                        leaf_objective_binary_from_counts(nL, posL);
                    const int right_loss =
                        leaf_objective_binary_from_counts(nR, posR);
                    score = -(double)(left_loss + right_loss);
                } else {
                    // information gain
                    score = split_score_from_counts_binary_(
                        n_total,
                        pos_total,
                        nL,
                        posL
                    );
                }
            } else {
                if (greedy_split_mode == 2) {
                    const int left_loss =
                        leaf_objective_multiclass_from_counts_(left_counts);

                    std::vector<int> right_counts((std::size_t)num_classes, 0);
                    for (int c = 0; c < num_classes; ++c) {
                        right_counts[(std::size_t)c] =
                            parent_counts[(std::size_t)c] -
                            left_counts[(std::size_t)c];
                    }

                    const int right_loss =
                        leaf_objective_multiclass_from_counts_(right_counts);

                    score = -(double)(left_loss + right_loss);
                } else {
                    score = split_score_from_counts_multiclass_(
                        parent_counts,
                        left_counts,
                        n_total,
                        nL
                    );
                }
            }

            if (score > best.score) {
                best.score = score;
                best.feat = feat;
            }

            i = j;
        }

        return best;
    }

        static inline double entropy_binary_count_(int n, int pos) {
        if (n <= 0) return 0.0;

        const double p = (double)pos / (double)n;

        if (p <= 0.0 || p >= 1.0) return 0.0;

        return -p * std::log(p) - (1.0 - p) * std::log(1.0 - p);
    }

    double binary_impurity_gain_score_(
        int n_total,
        int pos_total,
        int nL,
        int posL
    ) const {
        const int nR = n_total - nL;
        const int posR = pos_total - posL;

        if (nL <= 0 || nR <= 0) {
            return -std::numeric_limits<double>::infinity();
        }

        const double parent =
            entropy_binary_count_(n_total, pos_total);

        const double left =
            entropy_binary_count_(nL, posL);

        const double right =
            entropy_binary_count_(nR, posR);

        return parent
            - ((double)nL / (double)n_total) * left
            - ((double)nR / (double)n_total) * right;
    }

    int leaf_objective_multiclass_from_counts_(
        const std::vector<int>& counts
    ) const {
        int n = 0;
        int best = 0;

        for (int c : counts) {
            n += c;
            if (c > best) best = c;
        }

        return gamma + (n - best);
    }

    void partition_numerical_state_(
        const NumericalGreedyState& parent_state,
        const Packed& L,
        const Packed& R,
        NumericalGreedyState& left_state,
        NumericalGreedyState& right_state
    ) const {
        const std::size_t G = parent_state.sorted_idx_by_num.size();

        const std::size_t nL = (std::size_t)L.count();
        const std::size_t nR = (std::size_t)R.count();

        left_state.sorted_idx_by_num.resize(G);
        right_state.sorted_idx_by_num.resize(G);

        for (std::size_t g = 0; g < G; ++g) {
            const auto& src = parent_state.sorted_idx_by_num[g];
            auto& dstL = left_state.sorted_idx_by_num[g];
            auto& dstR = right_state.sorted_idx_by_num[g];

            dstL.clear();
            dstR.clear();

            dstL.reserve(nL);
            dstR.reserve(nR);

            for (int row : src) {
                if (mask_has_row_(L, row)) {
                    dstL.push_back(row);
                } else {
                    dstR.push_back(row);
                }
            }
        }
    }

    int train_greedy_continuous_numerical(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const NumericalGreedyState& state
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        // at depth 1, modes 1/2 use exact stump optimization.
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depth1_numerical_solver_cached(mask, pk, state);
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        GainSplitResult best;
        best.score = -std::numeric_limits<double>::infinity();
        best.feat = -1;

        const int first_cont = first_continuous_feature_();

        // ordinary binary features still use the existing binary scan
        if (num_classes == 2) {
            GainSplitResult bres = best_binary_score_split_(
                mask,
                /*first_feat=*/0,
                /*end_feat=*/first_cont,
                n_sub,
                pos
            );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        } else {
            GainSplitResult bres = best_binary_score_split_multiclass_(
                mask,
                /*first_feat=*/0,
                /*end_feat=*/first_cont,
                n_sub,
                parent_counts
            );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        }

        // continuous features use numerical sorted lists.
        const int G = (int)continuous_starts.size();

        if ((int)state.sorted_idx_by_num.size() != G) {
            throw std::logic_error(
                "Numerical greedy state is not aligned with continuous_starts."
            );
        }

        for (int g = 0; g < G; ++g) {
            const auto& sorted_rows = state.sorted_idx_by_num[(std::size_t)g];

            GainSplitResult cres = best_numerical_score_split_(
                g,
                sorted_rows,
                n_sub,
                pos,
                parent_counts
            );

            if (cres.feat >= 0 && cres.score > best.score) {
                best = cres;
            }
        }

        if (best.feat < 0) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        Packed L(n_words), R(n_words);
        split_threshold_bits_(mask, best.feat, L, R);

        if (!L.any() || !R.any()) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        // special greedy numerical depth-2 case:
        // greedy already chose the root split best.feat.
        // now solve the two depth-1 child problems with the fixed-root-split depth 2 solver
        // if (depth_budget == 2) {
        //     const PathKey* pkLp = &empty_pk();
        //     const PathKey* pkRp = &empty_pk();

        //     PathKey pkL_local;
        //     PathKey pkR_local;

        //     make_child_pks_if_needed_(
        //         best.feat,
        //         pk,
        //         pkLp,
        //         pkRp,
        //         pkL_local,
        //         pkR_local
        //     );

        //     const int split_loss =
        //         special_depth2_fixed_root_split_sum_numerical_(
        //             L,
        //             R,
        //             *pkLp,
        //             *pkRp,
        //             state,
        //             leaf_loss
        //         );

        //     const int ans = std::min(leaf_loss, split_loss);

        //     if (proxy_caching_enabled) {
        //         greedy_cache.emplace(key, ans);
        //     }

        //     return ans;
        // }

        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();

        PathKey pkL_local;
        PathKey pkR_local;

        make_child_pks_if_needed_(
            best.feat,
            pk,
            pkLp,
            pkRp,
            pkL_local,
            pkR_local
        );

        NumericalGreedyState left_state;
        NumericalGreedyState right_state;

        partition_numerical_state_(
            state,
            L,
            R,
            left_state,
            right_state
        );

        const int left_loss = train_greedy_continuous_numerical(
            L,
            depth_budget - 1,
            *pkLp,
            left_state
        );

        const int right_loss = train_greedy_continuous_numerical(
            R,
            depth_budget - 1,
            *pkRp,
            right_state
        );

        const int split_loss = left_loss + right_loss;
        const int ans = std::min(leaf_loss, split_loss);

        if (proxy_caching_enabled) {
            greedy_cache.emplace(key, ans);
        }

        return ans;
    }

    int depth1_numerical_solver_cached(
        const Packed& mask,
        const PathKey& pk,
        const NumericalGreedyState& state
    ) {
        constexpr int8_t DEPTH = 1;

        uint64_t kmask = 0;
        K2 key{0, DEPTH};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            leaf_loss = leaf_objective(mask);
        }

        if (n_sub <= 1) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }
            return leaf_loss;
        }

        // any split has at least 2 leaves, so its regularization cost is at least 2 * gamma.
        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }
            return leaf_loss;
        }

        int best_sum = leaf_loss;

        Packed L(n_words), R(n_words);

        const int first_cont = first_continuous_feature_();

        // ordinary binary features: exact depth-1 split objective.
        for (int f = 0; f < first_cont; ++f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);

                if (left_n == 0 || left_n == n_sub) {
                    continue;
                }
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);

                if (!L.any() || !R.any()) {
                    continue;
                }
            }

            const int sum = leaf_objective(L) + leaf_objective(R);

            if (sum < best_sum) {
                best_sum = sum;
            }
        }

        // numerical continuous features: scan sorted rows exactly once per numerical feature, evaluate all valid threshold positions.
        const int G = (int)continuous_starts.size();

        if ((int)state.sorted_idx_by_num.size() != G) {
            throw std::logic_error(
                "Numerical greedy state is not aligned with continuous_starts."
            );
        }

        if ((int)numerical_X_cols_for_greedy.size() != G ||
            (int)numerical_unique_values_for_greedy.size() != G) {
            throw std::logic_error(
                "Numerical greedy arrays are not aligned with continuous_starts."
            );
        }

        for (int g = 0; g < G; ++g) {
            const int candidate = depth1_numerical_feature_best_sum_(
                g,
                state.sorted_idx_by_num[(std::size_t)g],
                n_sub,
                pos,
                parent_counts,
                best_sum
            );

            if (candidate < best_sum) {
                best_sum = candidate;
            }
        }

        if (proxy_caching_enabled) {
            greedy_cache.emplace(key, best_sum);
        }

        return best_sum;
    }

    int depth1_numerical_feature_best_sum_(
        int num_group,
        const std::vector<int>& sorted_rows,
        int n_total,
        int pos_total,
        const std::vector<int>& parent_counts,
        int incumbent
    ) const {
        if (num_group < 0 || num_group >= (int)continuous_starts.size()) {
            return incumbent;
        }

        if (sorted_rows.size() <= 1) {
            return incumbent;
        }

        const int start_feat = continuous_starts[(std::size_t)num_group];
        const int end_feat = continuous_group_end_(num_group);

        if (start_feat >= end_feat) {
            return incumbent;
        }

        const std::vector<double>& x =
            numerical_X_cols_for_greedy[(std::size_t)num_group];

        const std::vector<double>& vals =
            numerical_unique_values_for_greedy[(std::size_t)num_group];

        if (vals.size() <= 1) {
            return incumbent;
        }

        // in prepare_continuous_data, we make one threshold column for every unique value except the global maximum.
        const int expected_end_feat = start_feat + (int)vals.size() - 1;
        if (expected_end_feat != end_feat) {
            throw std::logic_error(
                "continuous_starts/end does not match numerical_unique_values_for_greedy."
            );
        }

        int best_sum = incumbent;

        int nL = 0;
        int posL = 0;

        std::vector<int> left_counts;
        if (num_classes != 2) {
            left_counts.assign((std::size_t)num_classes, 0);
        }

        std::size_t i = 0;

        while (i < sorted_rows.size()) {
            const double value = x[(std::size_t)sorted_rows[i]];

            // move the whole tie block for this value left.
            std::size_t j = i;
            while (j < sorted_rows.size() &&
                x[(std::size_t)sorted_rows[j]] == value) {
                const int row = sorted_rows[j];

                ++nL;

                if (num_classes == 2) {
                    if (y_train[(std::size_t)row] == 1) {
                        ++posL;
                    }
                } else {
                    const int c = y_train[(std::size_t)row];

                    if (c < 0 || c >= num_classes) {
                        throw std::logic_error(
                            "Class label out of range in numerical depth-1 solver."
                        );
                    }

                    ++left_counts[(std::size_t)c];
                }

                ++j;
            }

            // if all active rows are left, right child is empty, so no valid split.
            if (j >= sorted_rows.size()) {
                break;
            }

            // value -> global unique value index -> threshold-column index.
            auto it = std::lower_bound(vals.begin(), vals.end(), value);

            if (it == vals.end() || *it != value) {
                throw std::logic_error(
                    "Active numerical value not found in global unique values."
                );
            }

            const int global_idx = (int)std::distance(vals.begin(), it);

            // no threshold column exists for the global maximum.
            if (global_idx + 1 >= (int)vals.size()) {
                break;
            }

            const int feat = start_feat + global_idx;

            if (feat < start_feat || feat >= end_feat) {
                throw std::logic_error(
                    "Mapped numerical threshold feature index is out of group range."
                );
            }

            if (nL == 0 || nL == n_total) {
                i = j;
                continue;
            }

            int sum = 0;

            if (num_classes == 2) {
                const int nR = n_total - nL;
                const int posR = pos_total - posL;

                const int left_loss =
                    leaf_objective_binary_from_counts(nL, posL);

                const int right_loss =
                    leaf_objective_binary_from_counts(nR, posR);

                sum = left_loss + right_loss;
            } else {
                std::vector<int> right_counts((std::size_t)num_classes, 0);

                for (int c = 0; c < num_classes; ++c) {
                    right_counts[(std::size_t)c] =
                        parent_counts[(std::size_t)c] -
                        left_counts[(std::size_t)c];
                }

                const int left_loss =
                    leaf_objective_multiclass_from_counts_(left_counts);

                const int right_loss =
                    leaf_objective_multiclass_from_counts_(right_counts);

                sum = left_loss + right_loss;
            }

            if (sum < best_sum) {
                best_sum = sum;
            }

            i = j;
        }

        return best_sum;
    }

    int depth1_bitvector_solver_cached_continuous(
        const Packed& mask,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        constexpr int8_t DEPTH = 1;

        uint64_t kmask = 0;
        K2 key{0, DEPTH};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            leaf_loss = leaf_objective_multiclass_from_counts_(parent_counts);
        }

        if (n_sub <= 1) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }
            return leaf_loss;
        }

        // any split has at least two leaves, so no split can beat this.
        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }
            return leaf_loss;
        }

        int best_sum = leaf_loss;

        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        // ordinary binary features.
        if (num_classes == 2) {
            best_sum = depth1_bitvector_best_sum_binary_range_(
                mask,
                /*start_idx=*/0,
                /*end_idx=*/std::min(first_cont, F),
                n_sub,
                pos,
                best_sum,
                /*skip_duplicate_active_splits=*/false
            );
        } else {
            best_sum = depth1_bitvector_best_sum_multiclass_range_(
                mask,
                /*start_idx=*/0,
                /*end_idx=*/std::min(first_cont, F),
                n_sub,
                parent_counts,
                best_sum,
                /*skip_duplicate_active_splits=*/false
            );
        }

        // continuous threshold groups.
        // use cpath only to restrict the threshold interval, then scan bitvectors linearly.
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    cpath
                );

            if (start_idx >= end_idx) continue;

            if (num_classes == 2) {
                best_sum = depth1_bitvector_best_sum_binary_range_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    pos,
                    best_sum,
                    /*skip_duplicate_active_splits=*/true
                );
            } else {
                best_sum = depth1_bitvector_best_sum_multiclass_range_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    parent_counts,
                    best_sum,
                    /*skip_duplicate_active_splits=*/true
                );
            }
        }

        if (proxy_caching_enabled) {
            greedy_cache.emplace(key, best_sum);
        }

        return best_sum;
    }

    int depth1_bitvector_best_sum_binary_range_(
        const Packed& mask,
        int start_idx,
        int end_idx,
        int n_total,
        int pos_total,
        int incumbent,
        bool skip_duplicate_active_splits
    ) const {
        int best_sum = incumbent;

        if (start_idx >= end_idx) return best_sum;

        const Packed& Ypos = Y_bits[(size_t)1];

        int prev_nL = -1;
        int prev_posL = -1;

        for (int feat = start_idx; feat < end_idx; ++feat) {
            const Packed& Xf = X_bits[(size_t)feat];

            int nL = 0;
            int posL = 0;

            // fast bitvector count only, not materializing L/R.
            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] & Xf.w[(size_t)w];

                nL += popcnt64(left_bits);
                posL += popcnt64(left_bits & Ypos.w[(size_t)w]);
            }

            // adjacent continuous thresholds can induce the same active split
            // inside the current subproblem. skip duplicate count states.
            if (skip_duplicate_active_splits &&
                nL == prev_nL &&
                posL == prev_posL) {
                continue;
            }

            prev_nL = nL;
            prev_posL = posL;

            if (nL == 0 || nL == n_total) continue;

            const int nR = n_total - nL;
            const int posR = pos_total - posL;

            const int left_loss =
                leaf_objective_binary_from_counts(nL, posL);

            const int right_loss =
                leaf_objective_binary_from_counts(nR, posR);

            const int sum = left_loss + right_loss;

            if (sum < best_sum) {
                best_sum = sum;
            }
        }

        return best_sum;
    }

    int depth1_bitvector_best_sum_multiclass_range_(
        const Packed& mask,
        int start_idx,
        int end_idx,
        int n_total,
        const std::vector<int>& parent_counts,
        int incumbent,
        bool skip_duplicate_active_splits
    ) const {
        int best_sum = incumbent;

        if (start_idx >= end_idx) return best_sum;

        std::vector<int> left_counts((size_t)num_classes, 0);
        std::vector<int> right_counts((size_t)num_classes, 0);

        std::vector<int> prev_left_counts;
        if (skip_duplicate_active_splits) {
            prev_left_counts.assign((size_t)num_classes, -1);
        }

        int prev_nL = -1;

        for (int feat = start_idx; feat < end_idx; ++feat) {
            std::fill(left_counts.begin(), left_counts.end(), 0);

            const Packed& Xf = X_bits[(size_t)feat];

            int nL = 0;

            // fast bitvector count only, not materializing L/R.
            for (int w = 0; w < n_words; ++w) {
                const uint64_t left_bits =
                    mask.w[(size_t)w] & Xf.w[(size_t)w];

                nL += popcnt64(left_bits);

                for (int c = 0; c < num_classes; ++c) {
                    left_counts[(size_t)c] += popcnt64(
                        left_bits & Y_bits[(size_t)c].w[(size_t)w]
                    );
                }
            }

            if (skip_duplicate_active_splits &&
                nL == prev_nL &&
                left_counts == prev_left_counts) {
                continue;
            }

            prev_nL = nL;
            if (skip_duplicate_active_splits) {
                prev_left_counts = left_counts;
            }

            if (nL == 0 || nL == n_total) continue;

            for (int c = 0; c < num_classes; ++c) {
                right_counts[(size_t)c] =
                    parent_counts[(size_t)c] - left_counts[(size_t)c];
            }

            const int left_loss =
                leaf_objective_multiclass_from_counts_(left_counts);

            const int right_loss =
                leaf_objective_multiclass_from_counts_(right_counts);

            const int sum = left_loss + right_loss;

            if (sum < best_sum) {
                best_sum = sum;
            }
        }

        return best_sum;
    }

    int train_greedy_continuous(
        const Packed& mask,
        int8_t depth_budget,
        const PathKey& pk,
        const ContinuousPath& cpath = empty_continuous_path()
    ) {
        if (depth_budget <= 0) {
            return leaf_objective(mask);
        }

        // last split level: use exact stump optimization, continuous version
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk, cpath);
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        std::vector<int> parent_counts;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            count_per_class(mask, parent_counts);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        GainSplitResult best;
        best.score = -std::numeric_limits<double>::infinity();
        best.feat = -1;

        const int F = n_features;
        const int first_cont = first_continuous_feature_();

        // prdinary binary features.
        if (num_classes == 2) {
            GainSplitResult bres = best_binary_score_split_(
                mask,
                /*first_feat=*/0,
                /*end_feat=*/first_cont,
                n_sub,
                pos
            );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        } else {
            GainSplitResult bres = best_binary_score_split_multiclass_(
                mask,
                /*first_feat=*/0,
                /*end_feat=*/first_cont,
                n_sub,
                parent_counts
            );

            if (bres.feat >= 0 && bres.score > best.score) {
                best = bres;
            }
        }

        // continuous threshold groups
        for (int cont_pos = 0; cont_pos < (int)continuous_starts.size(); ++cont_pos) {
            const int raw_start = continuous_starts[(size_t)cont_pos];
            const int raw_end = continuous_group_end_(cont_pos);

            if (raw_start >= F) continue;

            auto [start_idx, end_idx] =
                tighten_continuous_interval_from_path_(
                    raw_start,
                    std::min(raw_end, F),
                    cpath
                );

            if (start_idx >= end_idx) continue;

            GainSplitResult cres;

            if (num_classes == 2) {
                cres = best_continuous_score_split_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    pos
                );
            } else {
                cres = best_continuous_score_split_multiclass_(
                    mask,
                    start_idx,
                    end_idx,
                    n_sub,
                    parent_counts
                );
            }

            if (cres.feat >= 0 && cres.score > best.score) {
                best = cres;
            }
        }

        if (best.feat < 0) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        Packed L(n_words), R(n_words);
        split_threshold_bits_(mask, best.feat, L, R);

        if (!L.any() || !R.any()) {
            if (proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }

            return leaf_loss;
        }

        // special greedy depth-2 case:
        // greedy already chose the root split best.feat.
        // now solve the two depth-1 child problems simultaneously.
        // if (depth_budget == 2) {
        //     const PathKey* pkLp = &empty_pk();
        //     const PathKey* pkRp = &empty_pk();

        //     PathKey pkL_local;
        //     PathKey pkR_local;

        //     make_child_pks_if_needed_(
        //         best.feat,
        //         pk,
        //         pkLp,
        //         pkRp,
        //         pkL_local,
        //         pkR_local
        //     );

        //     const int split_loss =
        //         special_depth2_fixed_root_split_sum_bitvector_(
        //             L,
        //             R,
        //             *pkLp,
        //             *pkRp,
        //             leaf_loss
        //         );

        //     const int ans = std::min(leaf_loss, split_loss);

        //     if (proxy_caching_enabled) {
        //         greedy_cache.emplace(key, ans);
        //     }

        //     return ans;
        // }

        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();

        PathKey pkL_local;
        PathKey pkR_local;

        make_child_pks_if_needed_(
            best.feat,
            pk,
            pkLp,
            pkRp,
            pkL_local,
            pkR_local
        );

        const ContinuousPath* cpathLp = &cpath;
        const ContinuousPath* cpathRp = &cpath;

        ContinuousPath cpathL_local;
        ContinuousPath cpathR_local;

        make_child_continuous_paths_if_needed_(
            best.feat,
            cpath,
            cpathLp,
            cpathRp,
            cpathL_local,
            cpathR_local
        );

        const int left_loss = train_greedy_continuous(
            L,
            depth_budget - 1,
            *pkLp,
            *cpathLp
        );

        const int right_loss = train_greedy_continuous(
            R,
            depth_budget - 1,
            *pkRp,
            *cpathRp
        );

        const int split_loss = left_loss + right_loss;
        const int ans = std::min(leaf_loss, split_loss);

        if (proxy_caching_enabled) {
            greedy_cache.emplace(key, ans);
        }

        return ans;
    }

    // end continuous land

    
    int train_greedy(const Packed& mask, int8_t depth_budget, const PathKey& pk) {
        if (depth_budget == 0) {
            return leaf_objective(mask);
        }
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depthd_exact_proxy_objective_(mask, 1, pk); 
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;
            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) return it->second; // the objective of the tree returned by the proxy
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }
            return leaf_loss;
        }

        // decide which split-selection heuristic to use
        bool use_entropy;
        if (greedy_split_mode == 0) {
            use_entropy = true;                  // always entropy-driven
        } else if (greedy_split_mode == 1) {
            use_entropy = (depth_budget != 1);   // special depth==1 solver
        } else { // greedy_split_mode == 2
            use_entropy = false;                 // always minimize child leaf objective
        }


        // choose split via entropy gain
        int best_feat;
        if (num_classes == 2) {
            best_feat = find_best_split_binary_known_counts(mask, n_sub, pos, use_entropy);
        } else {
            best_feat = find_best_split(mask, use_entropy);
        }
        if (best_feat < 0) {
            if (proxy_caching_enabled) greedy_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }
        
        Packed L(n_words), R(n_words);
        and_bits(mask, X_bits[best_feat], L);
        andnot_bits(mask, X_bits[best_feat], R);
       if (!L.any() || !R.any()) {
            throw std::logic_error(
                "Invalid split choice in greedy tree method"
            );
        }
        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;
        make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

        int left_obj  = train_greedy(L, depth_budget - 1, *pkLp);
        int right_obj = train_greedy(R, depth_budget - 1, *pkRp);
        int split_obj = left_obj + right_obj;

        int ans = min(leaf_loss, split_obj);
        if (proxy_caching_enabled) greedy_cache.emplace(key, ans);
        return ans;
    }

    int eval_with_lookahead(const Packed& m, int depth, int k, const PathKey& pk) {
        if (k <= 0) return greedy_proxy_objective_(m, depth, pk);
        return generalized_lickety_split(m, depth, k, pk);
    }


    inline int next_k_cycle(int8_t k) const {
            // cycle: ... 3->2->1->K->K-1->...
            return (k > 1) ? (k - 1) : lookahead_init;
        }

        inline void split_bits_count_left(
        const Packed& mask,
        const Packed& split,
        Packed& L,
        Packed& R,
        int& left_n
    ) const {
        left_n = popcount_and_make_split_words(
            mask.w.data(),
            split.w.data(),
            L.w.data(),
            R.w.data(),
            n_words,
            tail_mask
        );
    }

    int depth1_exact_solver_cached(const Packed& mask, const PathKey& pk) {
        // const uint64_t kmask = key_of_subproblem(mask, pk);
        // constexpr int DEPTH = 1;
        // constexpr int KTAG  = 0;

        // int cached;
        // if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;

        constexpr int8_t DEPTH = 1;
        constexpr int8_t KTAG  = 0;

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        if (num_classes == 2) {
            int n_sub, pos_total;
            count_total_pos_binary(mask, n_sub, pos_total);

            const int leaf_loss = leaf_objective_binary_from_counts(n_sub, pos_total);

            // only cache cheap subproblems if flag enabled
            if (leaf_loss <= 2 * gamma) {
                if (proxy_caching_enabled) {
                    cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
                }
                return leaf_loss;
            }

            int best_sum = std::numeric_limits<int>::max();

            const Packed& Ypos = Y_bits[(size_t)1];
            const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

            auto eval_feature = [&](int f) {
                const Packed& Xf = X_bits[(size_t)f];

                int left_n = 0;
                int left_pos = 0;

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n   += popcnt64(lw);
                    left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                const int right_pos = pos_total - left_pos;

                const int sum =
                    leaf_objective_binary_from_counts(left_n, left_pos)
                    +
                    leaf_objective_binary_from_counts(right_n, right_pos);

                if (sum < best_sum) best_sum = sum;
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) eval_feature(f);
            } else {
                for (int f : feats) eval_feature(f);
            }

            int ans = leaf_loss;
            if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

            if (proxy_caching_enabled) cache_lickety_if_true_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
            return ans;
        }

        const int leaf_loss = leaf_objective(mask);

        // only cache cheap subproblems if flag enabled
        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const int sum = leaf_objective(L) + leaf_objective(R);
            if (sum < best_sum) best_sum = sum;
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }


    int depth2_special_solver_cached(const Packed& mask, const PathKey& pk){
        constexpr int8_t DEPTH = 2;
        constexpr int8_t KTAG  = 1;

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

        auto eval_feature = [&](int f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);
                if (left_n == 0 || left_n == n_sub) return;
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);
                if (!L.any() || !R.any()) return;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depth1_exact_solver_cached(L, *pkLp);
            const int right_best = depth1_exact_solver_cached(R, *pkRp);
            const int sum = left_best + right_best;

            if (sum < best_sum) best_sum = sum;
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }

    int depth2_fixed_root_children_best_sum_bitvector_(
        const Packed& rootL,
        const Packed& rootR,
        const std::vector<int>& feats,
        int incumbent
    ) const {
        const int leafL = leaf_objective(rootL);
        const int leafR = leaf_objective(rootR);

        int bestL = leafL;
        int bestR = leafR;

        int nLroot = 0;
        int nRroot = 0;

        if (num_classes == 2) {
            int pos_tmp = 0;
            count_total_pos_binary(rootL, nLroot, pos_tmp);
            count_total_pos_binary(rootR, nRroot, pos_tmp);
        } else {
            nLroot = count_total(rootL);
            nRroot = count_total(rootR);
        }

        // if the one-leaf child is already <= 2 * gamma, no depth-1 split under that child can strictly improve it.
        const bool scanL = (nLroot > 1 && leafL > 2 * gamma);
        const bool scanR = (nRroot > 1 && leafR > 2 * gamma);

        if (!scanL && !scanR) {
            return bestL + bestR;
        }

        Packed LL(n_words), LR(n_words);
        Packed RL(n_words), RR(n_words);

        auto eval_second_feature = [&](int f2) {
            if (scanL) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(rootL, X_bits[(std::size_t)f2], LL, LR, left_n);

                    if (left_n != 0 && left_n != nLroot) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                } else {
                    and_bits(rootL, X_bits[(std::size_t)f2], LL);
                    andnot_bits(rootL, X_bits[(std::size_t)f2], LR);

                    if (LL.any() && LR.any()) {
                        const int candL =
                            leaf_objective(LL) + leaf_objective(LR);

                        if (candL < bestL) {
                            bestL = candL;
                        }
                    }
                }
            }

            if (scanR) {
                if (num_classes == 2) {
                    int left_n = 0;
                    split_bits_count_left(rootR, X_bits[(std::size_t)f2], RL, RR, left_n);

                    if (left_n != 0 && left_n != nRroot) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                } else {
                    and_bits(rootR, X_bits[(std::size_t)f2], RL);
                    andnot_bits(rootR, X_bits[(std::size_t)f2], RR);

                    if (RL.any() && RR.any()) {
                        const int candR =
                            leaf_objective(RL) + leaf_objective(RR);

                        if (candR < bestR) {
                            bestR = candR;
                        }
                    }
                }
            }
        };

        if (feats.empty()) {
            for (int f2 = 0; f2 < n_features; ++f2) {
                eval_second_feature(f2);

                // for a fixed nonempty root split, the final subtree has at least one leaf on each side, so 2 * gamma is a lower bound.
                if (bestL + bestR <= 2 * gamma) {
                    break;
                }
            }
        } else {
            for (int f2 : feats) {
                eval_second_feature(f2);

                if (bestL + bestR <= 2 * gamma) {
                    break;
                }
            }
        }

        (void)incumbent;
        return bestL + bestR;
    }

    int special_depth2_special_solver_cached(
        const Packed& mask,
        const PathKey& pk
    ) {
        constexpr int8_t DEPTH = 2;
        constexpr int8_t KTAG  = 1;

        uint64_t kmask = 0;
        int cached = 0;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);

            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) {
                return cached;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (n_sub <= 1) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(
                    kmask,
                    DEPTH,
                    KTAG,
                    leaf_loss,
                    /*allow_cache=*/true
                );
            }

            return leaf_loss;
        }

        // any root split creates at least two leaves. if the parent leaf is already <= 2 * gamma, no split can strictly improve it.
        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(
                    kmask,
                    DEPTH,
                    KTAG,
                    leaf_loss,
                    /*allow_cache=*/cache_cheap_subproblems
                );
            }

            return leaf_loss;
        }

        int best_sum = leaf_loss;

        Packed L(n_words), R(n_words);

        const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

        auto eval_root_feature = [&](int f1) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[(std::size_t)f1], L, R, left_n);

                if (left_n == 0 || left_n == n_sub) {
                    return;
                }
            } else {
                and_bits(mask, X_bits[(std::size_t)f1], L);
                andnot_bits(mask, X_bits[(std::size_t)f1], R);

                if (!L.any() || !R.any()) {
                    return;
                }
            }

            const int sum = depth2_fixed_root_children_best_sum_bitvector_(
                L,
                R,
                feats,
                best_sum
            );

            if (sum < best_sum) {
                best_sum = sum;
            }
        };

        if (feats.empty()) {
            for (int f1 = 0; f1 < n_features; ++f1) {
                eval_root_feature(f1);

                if (best_sum <= 2 * gamma) {
                    break;
                }
            }
        } else {
            for (int f1 : feats) {
                eval_root_feature(f1);

                if (best_sum <= 2 * gamma) {
                    break;
                }
            }
        }

        const int ans = best_sum;

        if (proxy_caching_enabled) {
            cache_lickety_if_true_(
                kmask,
                DEPTH,
                KTAG,
                ans,
                /*allow_cache=*/true
            );
        }

        return ans;
    }

    int depthd_exact_solver_cached(const Packed& mask, int8_t depth_budget, const PathKey& pk) {
        if (depth_budget <= 0) return leaf_objective(mask);
        if (depth_budget == 1) return depth1_exact_solver_cached(mask, pk);
        if (depth_budget == 2) return depth2_special_solver_cached(mask, pk);

        const int8_t DEPTH = depth_budget;
        const int8_t KTAG  = depth_budget - 1;


        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        int n_sub = 0;
        int pos = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_binary(mask, n_sub, pos);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const auto& feats = proxy_features_for_(ProxyLoopKind::DepthDExact);

        auto eval_feature = [&](int f) {
            if (num_classes == 2) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[f], L, R, left_n);
                if (left_n == 0 || left_n == n_sub) return;
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);
                if (!L.any() || !R.any()) return;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depthd_exact_solver_cached(L, depth_budget - 1, *pkLp);
            const int right_best = depthd_exact_solver_cached(R, depth_budget - 1, *pkRp);

            const int sum = left_best + right_best;
            if (sum < best_sum) best_sum = sum;
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, depth_budget, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }


    inline void cache_lickety_if_true_(uint64_t kmask, int8_t depth_budget, int8_t k, int val, bool allow_cache) {
        if (!allow_cache) return;
        const bool use_kla = use_kla_cache();
        if (use_kla) lickety_cache_kla.emplace(KLA{kmask, depth_budget, k}, val);
        else         lickety_cache_k2.emplace(K2 {kmask, depth_budget},     val);
    }

    inline bool try_get_lickety_cached_(uint64_t kmask, int8_t depth_budget, int8_t k, int& out_val) const {
        const bool use_kla = use_kla_cache();
        if (use_kla) {
            auto it = lickety_cache_kla.find(KLA{kmask, depth_budget, k});
            if (it == lickety_cache_kla.end()) return false;
            out_val = it->second;
            return true;
        } else {
            auto it = lickety_cache_k2.find(K2{kmask, depth_budget});
            if (it == lickety_cache_k2.end()) return false;
            out_val = it->second;
            return true;
        }
    }

    // our modified lickety_split algorithm that is O(nk^2d^2). 
    int lickety_split_k1(const Packed& mask, int8_t depth_budget, const PathKey& pk)
    {
        if (depth_budget == 0) return leaf_objective(mask);
        if (depth_budget == 1) return depthd_exact_proxy_objective_(mask, 1, pk);
        if (depth_budget == 2) return depthd_exact_proxy_objective_(mask, 2, pk);

        // ---- caching (k=1 => use K2 cache) ----
        uint64_t kmask = 0;
        K2 key2{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
                return it->second;
        }

        const int leaf_loss = leaf_objective(mask);

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                lickety_cache_k2.emplace(key2, leaf_loss);
            }
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        Packed bestL(n_words), bestR(n_words);

        const auto& feats = proxy_features_for_(ProxyLoopKind::Lickety);
        const int8_t child_depth = (int8_t)(depth_budget - 1);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int sum =
                greedy_proxy_objective_(L, child_depth, *pkLp) +
                greedy_proxy_objective_(R, child_depth, *pkRp);

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;
            }
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        int ans = leaf_loss;

        // recurse with constant k=1 (proxy_style=0 behavior)
        if (best_feat >= 0) {
            const int8_t next_depth = (int8_t)(depth_budget - 1);

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_loss  = lickety_split_k1(bestL, next_depth, *pkLp);
            const int right_loss = lickety_split_k1(bestR, next_depth, *pkRp);

            ans = std::min(ans, left_loss + right_loss);
            ans = std::min(ans, best_sum);
        }

        if (proxy_caching_enabled) {
            lickety_cache_k2.emplace(key2, ans);
        }
        return ans;
    }

    // a generalized lickety_split algorithm with a lookahead parameter k. we also support other proxy styles here (such as recursively applying split) that have the same flavor.
    int generalized_lickety_split(const Packed& mask, int8_t depth_budget, int8_t k, const PathKey& pk) {
        if (k == 1 && lookahead_init == 1 && proxy_style == 0) {
            return lickety_split_k1(mask, depth_budget, pk);
        }

        if (depth_budget == 0) {
            return leaf_objective(mask);
        }
        if (depth_budget == 1) return depthd_exact_proxy_objective_(mask, 1, pk);

        if (k > depth_budget - 1) k = depth_budget - 1;

        if (depth_budget == 2 && k == 1) return depthd_exact_proxy_objective_(mask, 2, pk);
        if (k == depth_budget - 1) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk);
        }

        uint64_t kmask = 0;
        K2  key2{0, depth_budget};
        KLA keyla{0, depth_budget, k};

        const bool use_kla = use_kla_cache();

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            keyla.k = kmask;

            if (use_kla) {
                if (auto it = lickety_cache_kla.find(keyla); it != lickety_cache_kla.end())
                    return it->second;
            } else {
                if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
                    return it->second;
            }
        }

        const int leaf_loss = leaf_objective(mask);
        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                if (use_kla) lickety_cache_kla.emplace(keyla, leaf_loss);
                else         lickety_cache_k2.emplace(key2,  leaf_loss);
            }
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = numeric_limits<int>::max();

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);

        const int child_k = k - 1;
        const auto& feats = proxy_features_for_(ProxyLoopKind::Lickety);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int sum =
                eval_with_lookahead(L, depth_budget - 1, child_k, *pkLp)
                +
                eval_with_lookahead(R, depth_budget - 1, child_k, *pkRp);

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
            }
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) eval_feature(f);
        } else {
            for (int f : feats) eval_feature(f);
        }

        if (best_feat >= 0) {
            and_bits(mask, X_bits[best_feat], bestL);
            andnot_bits(mask, X_bits[best_feat], bestR);
        }

        int ans = leaf_loss; 

        int8_t k_recurse;
        if (proxy_style == 0) {
            // style 0: constant k (recursively choosing based on lower tier LicketySPLIT)
            k_recurse = k;
        } else if (proxy_style == 3) {
            // style 3: if we're running SPLIT without postprocessing, we don't need to do further recursive calls (the tree is fully determined).
            ans = std::min(ans, best_sum);
            if (proxy_caching_enabled) {
                if (use_kla) lickety_cache_kla.emplace(keyla, ans);
                else         lickety_cache_k2.emplace(key2,  ans);
            }
            return ans;
        } else {
            // styles 1/2: restart when it hits 0 (recursively applying SPLIT, so we're cycling k, k-1, k-2, ... 2 1 k)
            k_recurse = (child_k == 0) ? lookahead_init : child_k;
        }


        if (best_feat >= 0) {
            const int next_depth = depth_budget - 1;

            int left_loss, right_loss;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

            left_loss  = generalized_lickety_split(bestL, next_depth, k_recurse, *pkLp);
            right_loss = generalized_lickety_split(bestR, next_depth, k_recurse, *pkRp);
            ans = std::min(ans, left_loss + right_loss); // do licketysplit and take the minimum of it and leaf, even if greedy doesn't perform better than the leaf.
            ans = std::min(ans, best_sum); // if greedy is allowed more features, or heuristic pruning happens, we never want to be worse than greedy

        }
        if (proxy_caching_enabled) {
            if (use_kla) lickety_cache_kla.emplace(keyla, ans);
            else lickety_cache_k2.emplace(key2,  ans);
        }
        
        return ans;
    }

    int split_algorithm(const Packed& mask, int8_t depth_budget, int8_t k, const PathKey& pk) {
        if (depth_budget <= 0) return leaf_objective(mask);
        if (k > depth_budget - 1) k = depth_budget - 1; // same amount of computation is optimal
        if (k == depth_budget - 1) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk);
        }

        // if lookahead is exhausted, switch to optimal at this remaining depth
        if (k <= 0) {
            return depthd_exact_proxy_objective_(mask, depth_budget, pk);
        }

        const int8_t child_d = (int8_t)(depth_budget - 1);
        const int8_t child_k = (int8_t)(k - 1);

        Packed L(n_words), R(n_words);

        int best_feat = -1;
        int best_score = std::numeric_limits<int>::max();

        Packed bestL(n_words), bestR(n_words);
        PathKey bestPkL, bestPkR;
        bool have_best_pks = false;

        const auto& feats = proxy_features_for_(ProxyLoopKind::Lickety);

        auto eval_feature = [&](int f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) return;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            // choose the best split based on generalized_lickety_split at (d-1, k-1). with the strategy 3/4 (SPLIT algorithm), this is tracing out the splits that the SPLIT algorithm without postprocessing chose.
            int left_score  = lickety_proxy_objective_(L, child_d, child_k, *pkLp);
            int right_score = lickety_proxy_objective_(R, child_d, child_k, *pkRp);
            int score = left_score + right_score;

            if (score < best_score) {
                best_score = score;
                best_feat = f;

                bestL.w = L.w;
                bestR.w = R.w;

                if (key_mode == KeyMode::LITS_EXACT) {
                    bestPkL = *pkLp;
                    bestPkR = *pkRp;
                    have_best_pks = true;
                } else {
                    have_best_pks = false;
                }
            }
        };

        if (feats.empty()) {
            for (int f = 0; f < n_features; ++f) {
                eval_feature(f);
            }
        } else {
            for (int f : feats) {
                eval_feature(f);
            }
        }

        if (best_feat < 0) {
            return leaf_objective(mask);
        }

        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;

        if (key_mode == KeyMode::LITS_EXACT) {
            if (have_best_pks) {
                pkLp = &bestPkL;
                pkRp = &bestPkR;
            } else {
                make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);
            }
        }

        // instead of recursing with a greedy tree, we apply postprocessing here.
        // we do optimal on the children because we chose a split that was already best for greedy completion (aka we are tracing out the top k splits of the SPLIT tree without postprocessing to then apply it)
        if (child_k <= 0) {
            int left_cost  = depthd_exact_proxy_objective_(bestL, child_d, *pkLp);
            int right_cost = depthd_exact_proxy_objective_(bestR, child_d, *pkRp);
            return left_cost + right_cost;
        }

        // otherwise recurse using split() itself to find the next split it made
        int left_cost  = lickety_proxy_objective_(bestL, child_d, child_k, *pkLp);
        int right_cost = lickety_proxy_objective_(bestR, child_d, child_k, *pkRp);
        return left_cost + right_cost;
    }

    int find_best_split_binary_known_counts(
        const Packed& mask,
        int n_sub,
        int pos_total,
        bool use_entropy
    ) const {
        if (n_sub <= 1) return -1;

        const Packed& Ypos = Y_bits[(size_t)1];

        int best_f = -1;
        const auto& feats = proxy_features_for_(ProxyLoopKind::Greedy);

        if (use_entropy) {
            double best_score = 1e300;

            auto eval_feature = [&](int f) {
                int left_n = 0;
                int left_pos = 0;

                const Packed& Xf = X_bits[(size_t)f];

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n   += popcnt64(lw);
                    left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                const int right_pos = pos_total - left_pos;

                const double wl = (double)left_n  / (double)n_sub;
                const double wr = (double)right_n / (double)n_sub;

                const double pl = (double)left_pos  / (double)left_n;
                const double pr = (double)right_pos / (double)right_n;

                const double score = wl * entropy(pl) + wr * entropy(pr);

                if (score < best_score) {
                    best_score = score;
                    best_f = f;
                }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;
        } else {
            int best_sum = std::numeric_limits<int>::max();

            auto eval_feature = [&](int f) {
                int left_n = 0;
                int left_pos = 0;

                const Packed& Xf = X_bits[(size_t)f];

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n   += popcnt64(lw);
                    left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                const int right_pos = pos_total - left_pos;

                const int left_loss =
                    leaf_objective_binary_from_counts(left_n, left_pos);

                const int right_loss =
                    leaf_objective_binary_from_counts(right_n, right_pos);

                const int sum = left_loss + right_loss;

                if (sum < best_sum) {
                    best_sum = sum;
                    best_f = f;
                }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;
        }
    }

    int find_best_split_binary(const Packed& mask, bool use_entropy) const {
        int n_sub, pos_total;
        count_total_pos_binary(mask, n_sub, pos_total);

        if (n_sub <= 1) return -1;

        const Packed& Ypos = Y_bits[(size_t)1];

        int best_f = -1;
        const auto& feats = proxy_features_for_(ProxyLoopKind::Greedy);

        if (use_entropy) {
            double best_score = 1e300;

            auto eval_feature = [&](int f) {
                int left_n = 0;
                int left_pos = 0;

                const Packed& Xf = X_bits[(size_t)f];

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n   += popcnt64(lw);
                    left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                const int right_pos = pos_total - left_pos;

                const double wl = (double)left_n  / (double)n_sub;
                const double wr = (double)right_n / (double)n_sub;

                const double pl = (double)left_pos  / (double)left_n;
                const double pr = (double)right_pos / (double)right_n;

                const double score = wl * entropy(pl) + wr * entropy(pr);

                if (score < best_score) {
                    best_score = score;
                    best_f = f;
                }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;
        } else {
            int best_sum = std::numeric_limits<int>::max();

            auto eval_feature = [&](int f) {
                int left_n = 0;
                int left_pos = 0;

                const Packed& Xf = X_bits[(size_t)f];

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n   += popcnt64(lw);
                    left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                const int right_pos = pos_total - left_pos;

                const int left_loss =
                    gamma + std::min(left_pos, left_n - left_pos);

                const int right_loss =
                    gamma + std::min(right_pos, right_n - right_pos);

                const int sum = left_loss + right_loss;

                if (sum < best_sum) {
                    best_sum = sum;
                    best_f = f;
                }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;
        }
    }


    int find_best_split(const Packed& mask, bool use_entropy) const {
        if (num_classes == 2) {
            return find_best_split_binary(mask, use_entropy);
        }
        const int n_sub = count_total(mask);
        if (n_sub <= 1) return -1;

        Packed L(n_words);

        if (use_entropy) {
            // total class counts under mask
            std::vector<int> total_cnt((size_t)num_classes, 0);
            for (int c = 0; c < num_classes; ++c) {
                total_cnt[(size_t)c] = popcount_and(mask, Y_bits[(size_t)c]);
            }

            int best_f = -1;
            double best_score = 1e300;

            std::vector<int> left_cnt((size_t)num_classes, 0);
            std::vector<int> right_cnt((size_t)num_classes, 0);

            const auto& feats = proxy_features_for_(ProxyLoopKind::Greedy);

            auto eval_feature = [&](int f) {
                // L = mask & X_bits[f]
                for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[(size_t)f].w[i];
                L.w[n_words - 1] &= tail_mask;

                const int left_n  = L.count();
                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                // left class counts: popcount_and(L, Y_bits[c])
                for (int c = 0; c < num_classes; ++c) {
                    left_cnt[(size_t)c] = popcount_and(L, Y_bits[(size_t)c]);
                }

                // right class counts = total - left (no need to build)
                for (int c = 0; c < num_classes; ++c) {
                    right_cnt[(size_t)c] = total_cnt[(size_t)c] - left_cnt[(size_t)c];
                }

                const double wl = (double)left_n  / (double)n_sub;
                const double wr = (double)right_n / (double)n_sub;

                const double Hl = entropy_multiclass(left_cnt,  left_n);
                const double Hr = entropy_multiclass(right_cnt, right_n);

                const double score = (wl * Hl + wr * Hr);
                if (score < best_score) { 
                    best_score = score; 
                    best_f = f; 
                }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;

        } else {
            // minimize child leaf objectives: leaf_objective(L)+leaf_objective(R)
            int best_f = -1;
            int best_sum = std::numeric_limits<int>::max();

            Packed R(n_words);
            const auto& feats = proxy_features_for_(ProxyLoopKind::Greedy);

            auto eval_feature = [&](int f) {
                // L = mask & X_bits[f]
                for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[f].w[i]; 
                L.w[n_words-1] &= tail_mask;

                const int left_n = L.count();
                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) return;

                // R = mask & ~X_bits[f]
                for (int i = 0; i < n_words; ++i) R.w[i] = mask.w[i] & ~X_bits[f].w[i];
                R.w[n_words-1] &= tail_mask;

                const int sum = leaf_objective(L) + leaf_objective(R);
                if (sum < best_sum) { best_sum = sum; best_f = f; }
            };

            if (feats.empty()) {
                for (int f = 0; f < n_features; ++f) {
                    eval_feature(f);
                }
            } else {
                for (int f : feats) {
                    eval_feature(f);
                }
            }

            return best_f;
        }
    }
    // not used by PRAXIS Rashomon mode, to support giving single decision tree algorithm results in package.
    shared_ptr<PredNode> build_best_tree_from_caches(const Packed& mask, int8_t depth_budget, const PathKey& pk) const {
        const int INF = std::numeric_limits<int>::max();

        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = 0;
            return t;
        }

        std::vector<int> cnts;
        count_per_class(mask, cnts);
        int best_c = 0;
        int best_cnt = cnts[0];
        for (int c = 1; c < num_classes; ++c) {
            int v = cnts[(size_t)c];
            if (v > best_cnt || (v == best_cnt && c > best_c)) {
                best_cnt = v;
                best_c = c;
            }
        }
        const int leaf_pred = best_c;
        const int mis = n_sub - best_cnt;
        const int leaf_loss = gamma + mis;

        if (depth_budget <= 0) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = leaf_pred;
            return t;
        }

        // helper: lookup min cached objective for (mask, depth, pk) across greedy + lickety caches
        auto best_cached_obj = [&](const Packed& m, int8_t d, const PathKey& pk_child) -> int {
            if (d < 0) return 0;
            if (d==0) return leaf_objective(m);
            if (!proxy_caching_enabled) return INF;

            const uint64_t km = key_of_subproblem(m, pk_child);

            int best = INF;

            // greedy cache: (subproblem, depth)
            {
                auto itg = greedy_cache.find(K2{km, d});
                if (itg != greedy_cache.end()) best = std::min(best, itg->second);
            }

            // lickety cache:
            if (use_kla_cache()) {
                // try all k = 0..d-1
                for (int kk = 0; kk <= (int)(d-1); ++kk) {
                    auto it = lickety_cache_kla.find(KLA{km, d, kk});
                    if (it != lickety_cache_kla.end()) best = std::min(best, it->second);
                }
            } else {
                // K2-form: no k needed
                auto it = lickety_cache_k2.find(K2{km, d});
                if (it != lickety_cache_k2.end()) best = std::min(best, it->second);
            }

            return best;
        };

        // choose best split using cached objectives
        const int8_t child_d = (int8_t)(depth_budget - 1);

        int best_feat = -1;
        int best_sum  = INF;

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);
        PathKey bestPkL, bestPkR;
        bool have_best_pks = false;

        const int F = n_features;
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_obj  = best_cached_obj(L, child_d, *pkLp);
            const int right_obj = best_cached_obj(R, child_d, *pkRp);
            if (left_obj == INF || right_obj == INF) continue;

            const int sum = left_obj + right_obj;
            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;

                if (key_mode == KeyMode::LITS_EXACT) {
                    bestPkL = *pkLp;
                    bestPkR = *pkRp;
                    have_best_pks = true;
                } else {
                    have_best_pks = false;
                }
            }
        }

        // If no split is yielded or it doesn't beat leaf, return leaf.
        if (best_feat < 0 || best_sum >= leaf_loss) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = leaf_pred;
            return t;
        }

        // --- recurse on chosen split ---
        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;

        if (key_mode == KeyMode::LITS_EXACT) {
            if (have_best_pks) {
                pkLp = &bestPkL;
                pkRp = &bestPkR;
            } else {
                make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);
            }
        } else {
            // non-literal mode: pk is ignored by key_of_subproblem anyway, so keep empty_pk()
            pkLp = &empty_pk();
            pkRp = &empty_pk();
        }

        auto left_tree  = build_best_tree_from_caches(bestL, child_d, *pkLp);
        auto right_tree = build_best_tree_from_caches(bestR, child_d, *pkRp);

        auto t = make_shared<PredNode>();
        t->feature = best_feat;
        t->prediction = -1;
        t->left = left_tree;
        t->right = right_tree;
        return t;
    }

    shared_ptr<PredNode> get_ith_tree(uint64_t i) const {
        if (!result) {
            throw runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }
        
        // count_trees will ensure that the histograms are built at the root and every child node (by building them if they are not yet built)
        uint64_t total = result->count_trees();
        if (i >= total) {
            throw out_of_range("Tree index out of range in get_ith_tree");
        }

        uint64_t cum = 0;
        int target_obj = -1;
        uint64_t k_within = 0;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                k_within = i - cum;
                break;
            }
            cum += e.cnt;
        }
        if (target_obj < 0) {
            throw runtime_error("Failed to locate objective bucket in get_ith_tree");
        }

        return get_kth_tree_with_objective(result.get(), target_obj, k_within);
    }

    shared_ptr<PredNode> get_kth_tree_with_objective(const TreeTrieNode* node, int target_obj, uint64_t k) const {
        if (!node) {
            throw runtime_error("Null node in get_kth_tree_with_objective");
        }

        // handle leaf-only trees at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss == target_obj) {
                if (k == 0) {
                    auto t = make_shared<PredNode>();
                    t->feature = -1;
                    t->prediction = leaf.prediction;
                    return t;
                }
                --k;
            }
        }

        // handle splits
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();

            // total_here = #trees under this split with exactly target_obj
            uint64_t total_here = 0;

            // for each L histogram entry, we go over it. R is sorted by obj, so we can binary search to find each r_obj that pairs to sum to exactly this target_obj.
            for (const auto& le : L->hist) {
                int l_obj = le.obj;
                uint64_t lc = le.cnt;
                int r_obj = target_obj - l_obj;
                // binary search r_obj in R->hist
                auto it = lower_bound(
                    R->hist.begin(), R->hist.end(),
                    HistEntry{r_obj, 0},
                    hist_less
                );
                if (it != R->hist.end() && it->obj == r_obj) {
                    uint64_t rc = it->cnt;
                    uint64_t pairs = lc * rc;

                    if (total_here + pairs > k) { // k is smaller than the culm amount in this split we've seen so far (for the first time), so we know that we want to recurse on this split (which was already known), with this particular l_obj and r_obj, but we also need what index within each objective to recurse 
                        uint64_t rel = k - total_here; // what index inside this block the tree lives (again, 0 indexed)
                        uint64_t left_idx  = rel / rc; // left contributes lc possibilities, right contributes rc, a cross product without filtering, this indexing scheme works to break ties
                        uint64_t right_idx = rel % rc;

                        auto left_tree  = get_kth_tree_with_objective(L, l_obj,  left_idx); // now we have all the information we need, recurse
                        auto right_tree = get_kth_tree_with_objective(R, r_obj, right_idx);

                        auto t = make_shared<PredNode>();
                        t->feature = split.feature;
                        t->prediction = -1;
                        t->left = left_tree;
                        t->right = right_tree;
                        return t;
                    }

                    total_here += pairs;
                }

            }
            // skip all trees from this split that achieve target_obj
            k -= total_here;
            
        }

        throw out_of_range("Index out of range for given objective in get_kth_tree_with_objective");
    }

    void predict_tree_recursive(const PredNode* node, const std::vector<std::vector<uint8_t>>& X_row_major, std::vector<uint8_t>& out, const std::vector<int>& idx) const {
        if (!node) return;

        // leaf: assign prediction to all indices in this subset.
        if (node->feature < 0) {
            uint8_t pred = static_cast<uint8_t>(node->prediction);
            for (int row : idx) {
                out[row] = pred;
            }
            return;
        }

        // internal node: split indices by feature
        int f = node->feature;
        std::vector<int> left_idx;
        std::vector<int> right_idx;
        left_idx.reserve(idx.size()); 
        right_idx.reserve(idx.size());

        for (int row : idx) {
            uint8_t v = X_row_major[row][f];
            if (v) left_idx.push_back(row); // 1 is left
            else   right_idx.push_back(row);
        }

        if (!left_idx.empty()) {
            predict_tree_recursive(node->left.get(), X_row_major, out, left_idx);
        }
        if (!right_idx.empty()) {
            predict_tree_recursive(node->right.get(), X_row_major, out, right_idx);
        }
    }

    void collect_paths(const PredNode* node, std::vector<int>& current, std::vector<std::vector<int>>& paths, std::vector<int>& preds) const {
        if (!node) {
            throw std::logic_error("collect_paths: encountered null node");
        }

        // leaf: record this path and prediction
        if (node->feature < 0) {
            paths.push_back(current); // current starts empty and is appended to along the dfs
            preds.push_back(node->prediction);
            return;
        }

        int f = node->feature;
        // IMPORTANT: we have to switch to 1-indexing here so that +- for the 0th (1st) feature means something

        // go left (true) -> +f or rather f+1
        current.push_back(f+1);
        collect_paths(node->left.get(), current, paths, preds);
        current.pop_back(); // backtrack after we complete a path so we have 1 vector that is updated in a nice way throughout this

        // go right (false) -> -f (technically -(f+1))
        current.push_back(-(f+1));
        collect_paths(node->right.get(), current, paths, preds);
        current.pop_back();
    }

// whole trie prediction for RID

private:
    static inline void and_bits_eval(
        const Packed& a,
        const Packed& b,
        Packed& out,
        int n_words,
        uint64_t tail_mask
    ) {
        if (n_words <= 0) return;

    #if PRAXIS_USE_AVX512
        int i = 0;
        for (; i + 8 <= n_words; i += 8) {
            __m512i va = _mm512_loadu_si512((const void*)(a.w.data() + i));
            __m512i vb = _mm512_loadu_si512((const void*)(b.w.data() + i));
            __m512i vc = _mm512_and_si512(va, vb);
            _mm512_storeu_si512((void*)(out.w.data() + i), vc);
        }
        for (; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & b.w[(size_t)i];
        }
    #else
        for (int i = 0; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & b.w[(size_t)i];
        }
    #endif

        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    static inline void andnot_bits_eval(
        const Packed& a,
        const Packed& b,
        Packed& out,
        int n_words,
        uint64_t tail_mask
    ) {
        if (n_words <= 0) return;

    #if PRAXIS_USE_AVX512
        int i = 0;
        for (; i + 8 <= n_words; i += 8) {
            __m512i va = _mm512_loadu_si512((const void*)(a.w.data() + i));
            __m512i vb = _mm512_loadu_si512((const void*)(b.w.data() + i));

            // _mm512_andnot_si512(x, y) computes ~x & y.
            // So this is ~b & a = a & ~b.
            __m512i vc = _mm512_andnot_si512(vb, va);

            _mm512_storeu_si512((void*)(out.w.data() + i), vc);
        }
        for (; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & ~b.w[(size_t)i];
        }
    #else
        for (int i = 0; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] & ~b.w[(size_t)i];
        }
    #endif

        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    static inline void or_bits_eval(
        const Packed& a,
        const Packed& b,
        Packed& out,
        int n_words,
        uint64_t tail_mask
    ) {
        if (n_words <= 0) return;

    #if PRAXIS_USE_AVX512
        int i = 0;
        for (; i + 8 <= n_words; i += 8) {
            __m512i va = _mm512_loadu_si512((const void*)(a.w.data() + i));
            __m512i vb = _mm512_loadu_si512((const void*)(b.w.data() + i));
            __m512i vc = _mm512_or_si512(va, vb);
            _mm512_storeu_si512((void*)(out.w.data() + i), vc);
        }
        for (; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] | b.w[(size_t)i];
        }
    #else
        for (int i = 0; i < n_words; ++i) {
            out.w[(size_t)i] = a.w[(size_t)i] | b.w[(size_t)i];
        }
    #endif

        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    static inline bool any_eval(const Packed& a) {
        const int n_words = (int)a.w.size();
        if (n_words <= 0) return false;

    #if PRAXIS_USE_AVX512
        int i = 0;
        __m512i accum = _mm512_setzero_si512();

        for (; i + 8 <= n_words; i += 8) {
            __m512i v = _mm512_loadu_si512((const void*)(a.w.data() + i));
            accum = _mm512_or_si512(accum, v);
        }

        // if any 64-bit lane is nonzero, this mask is nonzero.
        if (_mm512_test_epi64_mask(accum, accum) != 0) {
            return true;
        }

        for (; i < n_words; ++i) {
            if (a.w[(size_t)i]) return true;
        }

        return false;
    #else
        for (uint64_t t : a.w) {
            if (t) return true;
        }
        return false;
    #endif
    }

    static inline void clear_eval(Packed& a) {
        if (!a.w.empty()) {
            std::memset(a.w.data(), 0, a.w.size() * sizeof(uint64_t));
        }
    }

    static inline Packed zeros_eval(int n_words) {
        return Packed((size_t)n_words);
    }

    static inline Packed copy_eval_mask(
        const Packed& m,
        int n_words,
        uint64_t tail_mask
    ) {
        Packed out((size_t)n_words);

        if (n_words > 0) {
            std::memcpy(
                out.w.data(),
                m.w.data(),
                (size_t)n_words * sizeof(uint64_t)
            );

            out.w[(size_t)(n_words - 1)] &= tail_mask;
        }

        return out;
    }

    static inline PackedPredMulti zeros_predmulti(
        int n_words,
        int num_classes
    ) {
        PackedPredMulti pm;
        pm.by_class.reserve((size_t)num_classes);

        for (int c = 0; c < num_classes; ++c) {
            pm.by_class.emplace_back(Packed((size_t)n_words));
        }

        return pm;
    }

    static inline void clear_predmulti(PackedPredMulti& pm) {
        for (auto& p : pm.by_class) {
            if (!p.w.empty()) {
                std::memset(p.w.data(), 0, p.w.size() * sizeof(uint64_t));
            }
        }
    }

    // OR-combine two multiclass prediction packs into out
    static inline void or_predmulti(
        const PackedPredMulti& a,
        const PackedPredMulti& b,
        PackedPredMulti& out,
        int n_words,
        uint64_t tail_mask
    ) {
        const int C = (int)a.by_class.size();

        for (int c = 0; c < C; ++c) {
            const Packed& ac = a.by_class[(size_t)c];
            const Packed& bc = b.by_class[(size_t)c];
            Packed& oc = out.by_class[(size_t)c];

            or_bits_eval(ac, bc, oc, n_words, tail_mask);
        }
    }


    // build packed feature columns for EVAL X, turning into column major
    static inline EvalCtx build_eval_ctx_(const std::vector<std::vector<uint8_t>>& X_row_major, int n_features_expected) {
        EvalCtx ctx;

        ctx.n_eval = (int)X_row_major.size();
        if (ctx.n_eval == 0) {
            ctx.n_words = 0;
            ctx.tail_mask = ~0ULL;
            return ctx;
        }

        const int d = (int)X_row_major[0].size();
        if (d != n_features_expected) {
            throw std::runtime_error("Eval X has different number of features than training.");
        }

        ctx.n_words = (ctx.n_eval + 63) / 64;
        ctx.tail_mask = (ctx.n_eval % 64) ? ((1ULL << (ctx.n_eval % 64)) - 1ULL) : ~0ULL;

        ctx.X_bits_eval.assign((size_t)d, Packed((size_t)ctx.n_words));

        for (int f = 0; f < d; ++f) {
            Packed &col = ctx.X_bits_eval[f];
            clear_eval(col);
            for (int i = 0; i < ctx.n_eval; ++i) {
                if (X_row_major[(size_t)i][(size_t)f]) {
                    col.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
                }
            }
            col.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
        }

        return ctx;
    }

    // all 1s bitvector (for the evaluation passed in dataset not train)
    static inline Packed eval_root_mask_(int n_words, uint64_t tail_mask) {
        Packed m((size_t)n_words);
        if (n_words == 0) return m;
        for (int i = 0; i < n_words - 1; ++i) m.w[(size_t)i] = ~0ULL;
        m.w[(size_t)(n_words - 1)] = tail_mask;
        return m;
    }

    static inline std::vector<ObjBucketMulti> to_sorted_buckets_multi_(
        std::unordered_map<int, std::vector<PackedPredMulti>>& acc
    ) {
        std::vector<ObjBucketMulti> out;
        out.reserve(acc.size());
        for (auto &kv : acc) {
            ObjBucketMulti b;
            b.obj = kv.first;
            b.preds = std::move(kv.second);
            out.push_back(std::move(b));
        }
        std::sort(out.begin(), out.end(),
                [](const ObjBucketMulti& a, const ObjBucketMulti& b){ return a.obj < b.obj; });
        return out;
    }


    // core recursion: returns buckets of predictions grouped by objective for ALL trees rooted at node with obj <= budget.
    std::vector<ObjBucketMulti> collect_preds_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask, // does not decrease size, just gets sparser
        const EvalCtx& ctx
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        if (node->min_objective == std::numeric_limits<int>::max()) return {};
        if (node->min_objective > budget) return {};

        // accumulate as obj (training) -> list of preds on evaluation (Packed)
        //std::unordered_map<int, std::vector<Packed>> acc;
        std::unordered_map<int, std::vector<PackedPredMulti>> acc;
        // heuristic reserve
        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaves at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            PackedPredMulti pm = zeros_predmulti(ctx.n_words, num_classes);

            if (ctx.n_words > 0) {
                // predicted class gets eval_mask, others stay zero
                const int pc = leaf.prediction; // should be 0..num_classes-1
                pm.by_class[(size_t)pc].w = eval_mask.w;
                pm.by_class[(size_t)pc].w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
            }

            acc[leaf.loss].push_back(std::move(pm));
            
            // storing the predictions in the map with that objective.
        }

        // splits
        const int INF = std::numeric_limits<int>::max();

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;

            // cap child budgets using the other side's min objective so everything found will pair with exactly one subtree on the other side
            int bL = budget - minR;
            int bR = budget - minL;
            if (bL < 0 || bR < 0) continue;

            // also cap by the budgets actually used to build those trie nodes. should never change anything (assuming we do iterative budget refinement, otherwise this is needed for tightening).
            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            // evaluation dataset routing masks
            Packed Lmask((size_t)ctx.n_words), Rmask((size_t)ctx.n_words);
            if (ctx.n_words > 0) {
                and_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Lmask, ctx.n_words, ctx.tail_mask);
                andnot_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Rmask, ctx.n_words, ctx.tail_mask);
            }

            // recurse
            auto Lb = collect_preds_by_obj_(L, bL, Lmask, ctx); // these return sorted lists of objective bucket objects
            auto Rb = collect_preds_by_obj_(R, bR, Rmask, ctx);
            if (Lb.empty() || Rb.empty()) continue;

            // for filtering by <= budget, both Lb and Rb are sorted by obj.
            // we'll two-pointer for each left obj to find all right objs <= (budget - l_obj).
            size_t r_hi = 0; // exclusive upper bound index in Rb
            for (size_t li = 0; li < Lb.size(); ++li) {
                const int lo = Lb[li].obj; // smallest objective initially
                if (lo > budget) break;
                const int rem = budget - lo; // how far do we have to look

                while (r_hi < Rb.size() && Rb[r_hi].obj <= rem) ++r_hi; // getting the first invalid index. never look past the remainder because RHS is also sorted
                if (r_hi == 0) continue; // no right objs fit

                // cross product (filtered)
                for (size_t ri = 0; ri < r_hi; ++ri) { // r_hi is one past the last valid
                    const int ro = Rb[ri].obj; // objective value for that bucket
                    const int tot = lo + ro; // additivity of objectives

                    // combine each left pred with each right pred (disjoint masks so OR is correct)
                    const auto& Lpreds = Lb[li].preds;
                    const auto& Rpreds = Rb[ri].preds;

                    // reserve some space in this objective bucket to reduce reallocs
                    auto &dest = acc[tot]; // alias for simplicity
                    // rough reserve: only if currently empty
                    if (dest.empty()) {
                            dest.reserve(std::max(Lpreds.size(), Rpreds.size()));
                        }

                    for (const auto& lp : Lpreds) {
                        for (const auto& rp : Rpreds) {
                            PackedPredMulti comb = zeros_predmulti(ctx.n_words, num_classes);
                            if (ctx.n_words > 0) {
                                or_predmulti(lp, rp, comb, ctx.n_words, ctx.tail_mask);
                            }
                            dest.push_back(std::move(comb));
                        }
                    }

                }
            }
        }

        return to_sorted_buckets_multi_(acc);
    }

public:
    // main entry: enumerate ALL valid trees under the trie root (or budget_override if >=0),
    // returning (training_objective, prediction vector on evaluation dataset) for each tree.
    // NOTE: this can be extremely large in memory if the Rashomon set is huge.
    std::vector<PredPackWithObj> get_all_predictions_packed_trie(const std::vector<std::vector<uint8_t>>& X_row_major, int budget_override = -1) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features); // get the evaluation dataset in column major

        // decide budget
        int budget = (budget_override >= 0) ? budget_override : result->budget;

        // root eval mask = all eval rows are 1, with the padding 0s (eval_rows % 64)
        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        // collect grouped by objective
        auto buckets = collect_preds_by_obj_(result.get(), budget, root_mask, ctx);

        // flatten
        std::vector<PredPackWithObj> out;
        // compute total count for reserve
        size_t total = 0;
        for (const auto& b : buckets) total += b.preds.size();
        out.reserve(total);

        for (auto &b : buckets) {
            for (auto &p : b.preds) {
                out.push_back(PredPackWithObj{b.obj, std::move(p)});
            }
        }
        return out;
    }
};