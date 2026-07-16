// tt.cpp - Transposition table implementation.
#include "search/tt.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#  include <malloc.h>
#  include <intrin.h>
#else
#  include <xmmintrin.h>
#endif

namespace {
void* aligned_alloc_bytes(std::size_t bytes, std::size_t align) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) p = nullptr;
    return p;
#endif
}
void aligned_free_bytes(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
} // namespace

namespace chess {

Value value_to_tt(Value v, int ply) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return Value(v + ply);
    if (v <= -VALUE_MATE_IN_MAX_PLY) return Value(v - ply);
    return v;
}

Value value_from_tt(Value v, int ply) {
    if (v == VALUE_NONE) return VALUE_NONE;
    if (v >= VALUE_MATE_IN_MAX_PLY) return Value(v - ply);
    if (v <= -VALUE_MATE_IN_MAX_PLY) return Value(v + ply);
    return v;
}

TranspositionTable::~TranspositionTable() {
    aligned_free_bytes(table_);
}

void TranspositionTable::resize(std::size_t mb) {
    aligned_free_bytes(table_);
    table_ = nullptr;

    std::size_t bytes = mb * 1024 * 1024;
    std::size_t count = bytes / sizeof(Cluster);
    if (count == 0) count = 1;

    // Round down to a power of two for fast masking.
    std::size_t pow2 = 1;
    while (pow2 * 2 <= count) pow2 *= 2;
    clusterCount_ = pow2;

    // 64-byte aligned so each cluster occupies exactly one cache line.
    table_ = static_cast<Cluster*>(
        aligned_alloc_bytes(clusterCount_ * sizeof(Cluster), 64));
    clear();
}

void TranspositionTable::prefetch(Key key) const {
    _mm_prefetch(reinterpret_cast<const char*>(&table_[key & (clusterCount_ - 1)]),
                 _MM_HINT_T0);
}

void TranspositionTable::clear() {
    if (table_)
        std::memset(table_, 0, clusterCount_ * sizeof(Cluster));
    generation8_ = 0;
}

TTEntry* TranspositionTable::probe(Key key, bool& found) const {
    TTEntry* const first = table_[key & (clusterCount_ - 1)].entry;
    const std::uint16_t key16 = std::uint16_t(key >> 48);

    for (int i = 0; i < ClusterSize; ++i) {
        // genBound8 (generation | bound) is the "occupied" sentinel, not depth8:
        // qsearch now stores entries at depth 0 (see Search::qsearch), so depth8
        // alone can no longer distinguish "empty slot" from "valid depth-0
        // entry". genBound8 is never 0 for a real write (store() always ORs in
        // a nonzero Bound), so it remains a reliable empty-slot sentinel.
        if (first[i].key16 == key16 && first[i].genBound8 != 0) {
            found = true;
            return &first[i];
        }
    }

    // Not found: pick the entry to replace. Lower "worth" = more replaceable:
    //   worth = depth - age_bonus + cut_bonus
    //     age_bonus = (currentAge - entry.age) * 4   (generation steps by 4),
    //                 so older generations are preferred for replacement;
    //     cut_bonus protects entries that caused a beta cutoff (BOUND_LOWER).
    // Lower depth, older generation, and non-cut entries are evicted first.
    auto worth = [&](const TTEntry& e) {
        const int ageBonus = int((generation8_ - e.generation()) & 0xFC);
        const int cutBonus = (e.bound() == BOUND_LOWER) ? 4 : 0;
        return int(e.depth8) - ageBonus + cutBonus;
    };
    TTEntry* replace = first;
    for (int i = 1; i < ClusterSize; ++i)
        if (worth(first[i]) < worth(*replace))
            replace = &first[i];
    found = false;
    return replace;
}

void TranspositionTable::store(TTEntry* tte, Key key, Value v, Bound b,
                               int depth, Move m, Value eval) {
    const std::uint16_t key16 = std::uint16_t(key >> 48);

    // Preserve an existing move if the new store has none.
    if (m != Move::none() || tte->key16 != key16)
        tte->move16 = std::uint16_t(m.raw());

    // Replace if: different position, deeper search, or an exact bound.
    if (b == BOUND_EXACT || tte->key16 != key16 || depth + 4 > tte->depth8) {
        tte->key16     = key16;
        tte->value16   = std::int16_t(v);
        tte->eval16    = std::int16_t(eval);
        tte->depth8    = std::uint8_t(depth);
        tte->genBound8 = std::uint8_t(generation8_ | b);
    }
}

int TranspositionTable::hashfull() const {
    if (!table_ || clusterCount_ == 0) return 0;
    // Sample the first 1000 clusters (or fewer if the table is small) and report
    // the permille of slots holding a current-generation entry.
    const std::size_t clusters = std::min<std::size_t>(1000, clusterCount_);
    int cnt = 0;
    for (std::size_t i = 0; i < clusters; ++i)
        for (int j = 0; j < ClusterSize; ++j)
            if (table_[i].entry[j].genBound8   // occupied sentinel, see probe()
                && table_[i].entry[j].generation() == generation8_)
                ++cnt;
    return int(std::int64_t(cnt) * 1000 / (std::int64_t(clusters) * ClusterSize));
}

} // namespace chess
