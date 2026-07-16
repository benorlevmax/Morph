// nnue.h - NNUE evaluation (HalfKP, dual perspective, king-bucketed).
//
// Architecture (Bundle 3 upgrade):
//   * HalfKP feature set: (king-bucket, piece-square, own/opp piece type).
//   * Dual perspective: separate accumulators for White and Black; the output
//     concatenates [side-to-move, other-side] so the net is side-relative.
//   * King buckets: 16 coarse regions of the king square (per perspective).
//   * Hidden width NNUE_HL = 512.
//
// Accumulator maintenance: non-king piece moves are incremental (single-feature
// add/sub, see `add`/`sub` below), already wired into Position's low-level piece
// mutators. A king move invalidates only that side's own accumulator (feature
// indices only depend on the mover's king *bucket*, never the opponent's), and
// is handled by a per-Position king-bucket cache (`FinnyTable`, one slot per
// (perspective, bucket)): moving within the same bucket is free (nothing to
// recompute), and moving into a different bucket reuses that bucket's last
// cached accumulator plus an incremental catch-up diff instead of a full
// from-scratch rebuild, falling back to a full rebuild only on a cold entry.
// See `refresh_perspective_cached`. The network is currently untrained
// (PSQT-equivalent init), so the classical evaluator remains the production
// default and fallback.
#pragma once

#include "core/bitboard.h"
#include "core/types.h"

#include <cstdint>
#include <string>

namespace chess {

class Position;

constexpr int NNUE_HL          = 512;            // hidden / accumulator width
constexpr int NNUE_KING_BUCKETS = 16;            // coarse king-square regions
constexpr int NNUE_PIECE_REL   = 10;             // PNBRQ x {own, opp}
constexpr int NNUE_FEATURES    = NNUE_KING_BUCKETS * 64 * NNUE_PIECE_REL;  // 10240
constexpr int NNUE_OUT_BUCKETS = 8;              // output buckets by piece count

// Dual-perspective accumulator: [perspective][hidden].
struct alignas(32) Accumulator {
    std::int16_t v[COLOR_NB][NNUE_HL];
};

namespace NNUE {

void init();                              // build the default PSQT-equivalent net
bool load(const std::string& path);
bool save(const std::string& path);
bool write_net(const std::string& path,
               const std::int16_t* ftBias,    // [NNUE_HL]
               const std::int16_t* ftWeights, // [NNUE_FEATURES * NNUE_HL]
               const std::int16_t* outWeights,// [NNUE_OUT_BUCKETS * 2 * NNUE_HL]
               const std::int32_t* outBias,   // [NNUE_OUT_BUCKETS]
               std::int32_t scale);

bool enabled();
void set_enabled(bool on);

int  king_bucket(Square kingSq);                          // 0..15 (white frame)
int  king_bucket_of(Color persp, Square kingSq);           // orient(persp) + bucket
int  feature_index(Color persp, Square kingSq, Piece pc, Square s);
int  output_bucket(const Position& pos);                  // 0..7 by total piece count

// Incremental accumulator maintenance (called from Position make/unmake).
// `pc` must be a non-king piece; `wk`/`bk` are the white/black king squares.
void add(Accumulator& acc, Piece pc, Square s, Square wk, Square bk);
void sub(Accumulator& acc, Piece pc, Square s, Square wk, Square bk);
void refresh_perspective(const Position& pos, Accumulator& acc, Color persp);
void refresh(const Position& pos, Accumulator& acc);      // both perspectives

// ---------------------------------------------------------------------------
// King-bucket accumulator cache (Finny table): one slot per (perspective,
// king bucket), storing the last accumulator computed for that bucket plus the
// non-king piece placement it reflects. Owned per-Position (see Position::
// finnyCache_), so it is copied along with everything else when a search
// thread deep-copies its root Position -- no shared mutable state between
// threads, satisfying Lazy SMP thread-safety the same way `acc_` already does.
// ---------------------------------------------------------------------------
struct alignas(32) FinnyEntry {
    Bitboard      occ[COLOR_NB][PIECE_TYPE_NB] = {};   // snapshot: PAWN..QUEEN only
    std::int16_t  acc[NNUE_HL] = {};                    // must stay 32-byte aligned:
                                                         // add_feature/sub_feature use
                                                         // aligned AVX2 loads on this buffer
    bool          valid = false;
};
using FinnyTable = FinnyEntry[COLOR_NB][NNUE_KING_BUCKETS];

// Diagnostic counters for the cache (see cache_stats()/reset_cache_stats()).
struct CacheStats {
    std::uint64_t sameBucket = 0;   // king move stayed in its bucket: free, no-op
    std::uint64_t hits       = 0;   // bucket changed, cache entry warm: patched
    std::uint64_t misses     = 0;   // bucket changed, cache entry cold: full rebuild
};
CacheStats cache_stats();
void       reset_cache_stats();

// Called from Position::move_piece when the moved piece is a king. Brings
// acc.v[persp] up to date for the king's new bucket using `table`, checkpointing
// the old bucket's accumulator into `table` for future reuse. Bit-identical to
// calling refresh_perspective(pos, acc, persp) after the move -- this is a
// caching/incrementalization of that call, not a behavior change.
void refresh_perspective_cached(const Position& pos, Accumulator& acc, Color persp,
                                 Square fromSq, Square toSq, FinnyTable& table);

int  output(const Accumulator& acc, Color stm, int bucket);        // stm POV cp (SIMD)
int  output_scalar(const Accumulator& acc, Color stm, int bucket); // reference path
int  evaluate(const Position& pos);                                // refresh + output

} // namespace NNUE
} // namespace chess
