// tt.h - Transposition table (Zobrist-keyed).
#pragma once

#include "core/types.h"
#include <vector>
#include <cstddef>

namespace chess {

enum Bound : std::uint8_t {
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,   // fail-low : value is an upper bound
    BOUND_LOWER = 2,   // fail-high: value is a lower bound
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

// 10-byte logical entry (packed). Several entries share a cache-line cluster.
struct TTEntry {
    std::uint16_t key16   = 0;
    std::uint16_t move16  = 0;
    std::int16_t  value16 = 0;
    std::int16_t  eval16  = 0;
    // depth8 == 0 is a legitimate stored depth (qsearch entries; see
    // Search::qsearch), not an emptiness marker -- genBound8 is the occupied
    // sentinel (see probe()), since it is never 0 for a real write.
    std::uint8_t  depth8  = 0;
    std::uint8_t  genBound8 = 0;   // bits 0-1 bound, bits 2-7 generation

    Bound bound() const { return Bound(genBound8 & 0x3); }
    int   depth() const { return int(depth8); }
    Move  move()  const { return Move(move16); }
    Value value() const { return Value(value16); }
    Value eval()  const { return Value(eval16); }
    std::uint8_t generation() const { return std::uint8_t(genBound8 & 0xFC); }
};

class TranspositionTable {
public:
    static constexpr int ClusterSize = 3;

    // 32-byte cluster (3*10 + 2 pad): two clusters fit a 64-byte cache line and
    // a 32-byte-aligned cluster never straddles a line. Minimal padding waste.
    struct alignas(32) Cluster {
        TTEntry entry[ClusterSize];
        char    padding[32 - ClusterSize * sizeof(TTEntry)];
    };
    static_assert(sizeof(Cluster) == 32, "TT cluster should be 32 bytes");

    ~TranspositionTable();

    void resize(std::size_t mb);   // allocate clusters for `mb` megabytes
    void clear();
    void new_search() { generation8_ += 4; }   // bump generation (low 2 bits = bound)
    std::uint8_t generation() const { return generation8_; }

    // Probe: returns pointer to the matching/replaceable entry. `found` is true
    // iff an entry with the same key already exists.
    TTEntry* probe(Key key, bool& found) const;

    // Prefetch the cluster for `key` into cache (overlaps memory latency).
    void prefetch(Key key) const;

    // Store/refresh an entry at the slot returned by probe().
    void store(TTEntry* tte, Key key, Value v, Bound b, int depth, Move m, Value eval);

    int hashfull() const;   // permille of slots used by current generation

private:
    Cluster*    table_     = nullptr;
    std::size_t clusterCount_ = 0;
    std::uint8_t generation8_ = 0;
};

// Mate-score <-> TT-storage adjustment (mates are stored as distance-to-mate
// from the current node, not from the root).
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply);

} // namespace chess
