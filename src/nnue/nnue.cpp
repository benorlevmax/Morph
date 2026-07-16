// nnue.cpp - HalfKP dual-perspective NNUE implementation.
#include "nnue/nnue.h"
#include "core/position.h"
#include "core/bitboard.h"
#include "eval/evaluate.h"   // PieceValue (material-init weights)

#include <atomic>
#include <cassert>
#include <cstring>
#include <fstream>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace chess {
namespace NNUE {

namespace {

struct Net {
    std::int16_t ftBias[NNUE_HL];
    std::int16_t ftWeights[NNUE_FEATURES][NNUE_HL];
    std::int16_t outWeights[NNUE_OUT_BUCKETS][2 * NNUE_HL];
    std::int32_t outBias[NNUE_OUT_BUCKETS];
    std::int32_t scale;
};

Net  g_net;
bool g_enabled = false;

constexpr int CR_MIN = 0;
constexpr int CR_MAX = 32767;
constexpr std::int16_t LINEAR_BIAS = 16384;

constexpr std::uint32_t MAGIC   = 0x4B504E32;   // "2NPK" (HalfKP v2 marker)
constexpr std::uint32_t VERSION = 2;

// Orient a square to a perspective (Black sees the board vertically flipped so
// its own pieces sit on the low ranks, matching White's frame).
inline Square orient(Color persp, Square s) {
    return persp == WHITE ? s : Square(int(s) ^ 56);
}

} // namespace

int king_bucket(Square kingSq) {
    // 16 buckets: a 4x4 grid over the (already oriented) board.
    return (rank_of(kingSq) / 2) * 4 + (file_of(kingSq) / 2);
}

int feature_index(Color persp, Square kingSq, Piece pc, Square s) {
    const int kb  = king_bucket(orient(persp, kingSq));
    const int rel = int(orient(persp, s));
    const int pieceRel = (int(type_of(pc)) - 1) * 2 + (color_of(pc) == persp ? 0 : 1);
    return kb * (64 * NNUE_PIECE_REL) + rel * NNUE_PIECE_REL + pieceRel;
}

bool enabled() { return g_enabled; }
void set_enabled(bool on) { g_enabled = on; }

void init() {
    std::memset(&g_net, 0, sizeof(g_net));

    // Material-equivalent initialization: hidden neuron 0 (own perspective)
    // carries the side-relative material balance in the clipped-ReLU linear
    // region. Weight is king- and square-independent (pure material): own pieces
    // positive, opponent pieces negative. A trained net will replace these.
    for (int kb = 0; kb < NNUE_KING_BUCKETS; ++kb) {
        for (int rel = 0; rel < 64; ++rel) {
            for (int pr = 0; pr < NNUE_PIECE_REL; ++pr) {
                const int f = kb * (64 * NNUE_PIECE_REL) + rel * NNUE_PIECE_REL + pr;
                const PieceType pt = PieceType(pr / 2 + 1);
                const int sign = (pr % 2 == 0) ? +1 : -1;   // own / opp
                g_net.ftWeights[f][0] = std::int16_t(sign * PieceValue[pt]);
            }
        }
    }
    g_net.ftBias[0] = LINEAR_BIAS;
    // Every output bucket starts identical (own-perspective neuron 0 = material);
    // training differentiates them. Architecture-only change: eval is unchanged.
    for (int b = 0; b < NNUE_OUT_BUCKETS; ++b) {
        g_net.outWeights[b][0]        = 1;   // own-perspective neuron 0
        g_net.outWeights[b][NNUE_HL]  = 0;   // opp-perspective neuron 0 unused
        g_net.outBias[b]              = -LINEAR_BIAS;
    }
    g_net.scale = 1;
}

bool write_net(const std::string& path,
               const std::int16_t* ftBias,
               const std::int16_t* ftWeights,
               const std::int16_t* outWeights,
               const std::int32_t* outBias,
               std::int32_t scale) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    auto w32 = [&](std::uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
    w32(MAGIC); w32(VERSION);
    w32(NNUE_FEATURES); w32(NNUE_HL); w32(NNUE_OUT_BUCKETS);
    f.write(reinterpret_cast<char*>(&scale), 4);
    f.write(reinterpret_cast<const char*>(ftBias), sizeof(std::int16_t) * NNUE_HL);
    f.write(reinterpret_cast<const char*>(ftWeights),
            sizeof(std::int16_t) * std::size_t(NNUE_FEATURES) * NNUE_HL);
    f.write(reinterpret_cast<const char*>(outWeights),
            sizeof(std::int16_t) * NNUE_OUT_BUCKETS * 2 * NNUE_HL);
    f.write(reinterpret_cast<const char*>(outBias), sizeof(std::int32_t) * NNUE_OUT_BUCKETS);
    return bool(f);
}

bool save(const std::string& path) {
    return write_net(path, g_net.ftBias, &g_net.ftWeights[0][0],
                     &g_net.outWeights[0][0], g_net.outBias, g_net.scale);
}

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::uint32_t magic, version, feats, hl, buckets;
    auto r32 = [&](std::uint32_t& v) { f.read(reinterpret_cast<char*>(&v), 4); };
    r32(magic); r32(version); r32(feats); r32(hl); r32(buckets);
    if (magic != MAGIC || feats != NNUE_FEATURES || hl != NNUE_HL
        || buckets != NNUE_OUT_BUCKETS) return false;
    f.read(reinterpret_cast<char*>(&g_net.scale), 4);
    f.read(reinterpret_cast<char*>(g_net.ftBias), sizeof(g_net.ftBias));
    f.read(reinterpret_cast<char*>(g_net.ftWeights), sizeof(g_net.ftWeights));
    f.read(reinterpret_cast<char*>(g_net.outWeights), sizeof(g_net.outWeights));
    f.read(reinterpret_cast<char*>(g_net.outBias), sizeof(g_net.outBias));
    if (g_net.scale == 0) g_net.scale = 1;
    return bool(f);
}

namespace {
inline void add_feature(std::int16_t* col, const std::int16_t* w) {
#if defined(__AVX2__)
    // Unaligned loads/stores: col may be Accumulator::v (32-byte aligned) or
    // FinnyEntry::acc (alignment not guaranteed once nested inside containers).
    for (int i = 0; i < NNUE_HL; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(col + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(col + i), _mm256_add_epi16(a, b));
    }
#else
    for (int i = 0; i < NNUE_HL; ++i) col[i] = std::int16_t(col[i] + w[i]);
#endif
}
inline void sub_feature(std::int16_t* col, const std::int16_t* w) {
#if defined(__AVX2__)
    for (int i = 0; i < NNUE_HL; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<__m256i*>(col + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(col + i), _mm256_sub_epi16(a, b));
    }
#else
    for (int i = 0; i < NNUE_HL; ++i) col[i] = std::int16_t(col[i] - w[i]);
#endif
}
} // namespace

void add(Accumulator& acc, Piece pc, Square s, Square wk, Square bk) {
    add_feature(acc.v[WHITE], g_net.ftWeights[feature_index(WHITE, wk, pc, s)]);
    add_feature(acc.v[BLACK], g_net.ftWeights[feature_index(BLACK, bk, pc, s)]);
}

void sub(Accumulator& acc, Piece pc, Square s, Square wk, Square bk) {
    sub_feature(acc.v[WHITE], g_net.ftWeights[feature_index(WHITE, wk, pc, s)]);
    sub_feature(acc.v[BLACK], g_net.ftWeights[feature_index(BLACK, bk, pc, s)]);
}

void refresh_perspective(const Position& pos, Accumulator& acc, Color persp) {
    std::memcpy(acc.v[persp], g_net.ftBias, sizeof(g_net.ftBias));
    const Square ksq = pos.king_square(persp);
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        const Piece pc = pos.piece_on(s);
        if (pc == NO_PIECE || type_of(pc) == KING) continue;   // HalfKP excludes kings
        add_feature(acc.v[persp], g_net.ftWeights[feature_index(persp, ksq, pc, s)]);
    }
}

void refresh(const Position& pos, Accumulator& acc) {
    refresh_perspective(pos, acc, WHITE);
    refresh_perspective(pos, acc, BLACK);
}

int king_bucket_of(Color persp, Square kingSq) {
    return king_bucket(orient(persp, kingSq));
}

// ---------------------------------------------------------------------------
// King-bucket accumulator cache (Finny table).
// ---------------------------------------------------------------------------
namespace {
std::atomic<std::uint64_t> g_sameBucket{0};
std::atomic<std::uint64_t> g_hits{0};
std::atomic<std::uint64_t> g_misses{0};

#ifndef NDEBUG
std::atomic<std::uint64_t> g_verifyCounter{0};
constexpr std::uint64_t VERIFY_SAMPLE_STRIDE = 256;   // occasional, not every call
#endif

// Snapshot the 10 non-king (color, piece-type) bitboards into an entry.
void snapshot_occ(const Position& pos, FinnyEntry& e) {
    for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
        e.occ[WHITE][pt] = pos.pieces(WHITE, pt);
        e.occ[BLACK][pt] = pos.pieces(BLACK, pt);
    }
}
} // namespace

CacheStats cache_stats() {
    return CacheStats{g_sameBucket.load(std::memory_order_relaxed),
                      g_hits.load(std::memory_order_relaxed),
                      g_misses.load(std::memory_order_relaxed)};
}

void reset_cache_stats() {
    g_sameBucket.store(0, std::memory_order_relaxed);
    g_hits.store(0, std::memory_order_relaxed);
    g_misses.store(0, std::memory_order_relaxed);
}

void refresh_perspective_cached(const Position& pos, Accumulator& acc, Color persp,
                                 Square fromSq, Square toSq, FinnyTable& table) {
    const int oldBucket = king_bucket_of(persp, fromSq);
    const int newBucket = king_bucket_of(persp, toSq);

    if (oldBucket == newBucket) {
        // Feature indices for every other piece depend only on the mover's king
        // *bucket*, never its exact square or the opponent's king -- so nothing
        // in acc.v[persp] can have changed. Verify that assumption occasionally
        // rather than trusting it silently.
        g_sameBucket.fetch_add(1, std::memory_order_relaxed);
#ifndef NDEBUG
        if (g_verifyCounter.fetch_add(1, std::memory_order_relaxed) % VERIFY_SAMPLE_STRIDE == 0) {
            Accumulator scratch;
            refresh_perspective(pos, scratch, persp);
            assert(std::memcmp(scratch.v[persp], acc.v[persp], sizeof(scratch.v[persp])) == 0
                   && "Finny cache: same-bucket king move unexpectedly changed the accumulator");
        }
#endif
        return;
    }

    // Checkpoint the old bucket with the accumulator as it stood just before
    // this call (still sitting in acc.v[persp]) so it can be reused if the king
    // comes back to this bucket later. Only the king moved since this value was
    // last valid, so the current non-king occupancy is exactly what it reflects.
    FinnyEntry& oldEntry = table[persp][oldBucket];
    snapshot_occ(pos, oldEntry);
    std::memcpy(oldEntry.acc, acc.v[persp], sizeof(oldEntry.acc));
    oldEntry.valid = true;

    FinnyEntry& newEntry = table[persp][newBucket];
    if (!newEntry.valid) {
        g_misses.fetch_add(1, std::memory_order_relaxed);
        refresh_perspective(pos, acc, persp);      // full O(pieces) rebuild
        snapshot_occ(pos, newEntry);
        std::memcpy(newEntry.acc, acc.v[persp], sizeof(newEntry.acc));
        newEntry.valid = true;
        return;
    }

    g_hits.fetch_add(1, std::memory_order_relaxed);
    for (Color c : {WHITE, BLACK}) {
        for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
            const Bitboard cur    = pos.pieces(c, pt);
            const Bitboard cached = newEntry.occ[c][pt];
            Bitboard added   = cur & ~cached;
            Bitboard removed = cached & ~cur;
            while (added) {
                const Square s = pop_lsb(added);
                add_feature(newEntry.acc, g_net.ftWeights[feature_index(persp, toSq, make_piece(c, pt), s)]);
            }
            while (removed) {
                const Square s = pop_lsb(removed);
                sub_feature(newEntry.acc, g_net.ftWeights[feature_index(persp, toSq, make_piece(c, pt), s)]);
            }
            newEntry.occ[c][pt] = cur;
        }
    }
    std::memcpy(acc.v[persp], newEntry.acc, sizeof(newEntry.acc));

#ifndef NDEBUG
    if (g_verifyCounter.fetch_add(1, std::memory_order_relaxed) % VERIFY_SAMPLE_STRIDE == 0) {
        Accumulator scratch;
        refresh_perspective(pos, scratch, persp);
        assert(std::memcmp(scratch.v[persp], acc.v[persp], sizeof(scratch.v[persp])) == 0
               && "Finny cache: patched accumulator diverged from a full refresh");
    }
#endif
}

int output_bucket(const Position& pos) {
    const int n = popcount(pos.pieces());
    int b = (n - 1) / 4;                       // 2 pieces -> 0 ... 32 pieces -> 7
    return b < 0 ? 0 : (b >= NNUE_OUT_BUCKETS ? NNUE_OUT_BUCKETS - 1 : b);
}

int output_scalar(const Accumulator& acc, Color stm, int bucket) {
    const std::int16_t* own = acc.v[stm];
    const std::int16_t* opp = acc.v[~stm];
    const std::int16_t* w   = g_net.outWeights[bucket];
    std::int64_t sum = g_net.outBias[bucket];
    for (int i = 0; i < NNUE_HL; ++i) {
        int x = own[i]; if (x < CR_MIN) x = CR_MIN; if (x > CR_MAX) x = CR_MAX;
        sum += std::int64_t(x) * w[i];
    }
    for (int i = 0; i < NNUE_HL; ++i) {
        int x = opp[i]; if (x < CR_MIN) x = CR_MIN; if (x > CR_MAX) x = CR_MAX;
        sum += std::int64_t(x) * w[NNUE_HL + i];
    }
    return int(sum / g_net.scale);
}

int output(const Accumulator& acc, Color stm, int bucket) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i hi   = _mm256_set1_epi16(CR_MAX);
    const std::int16_t* w = g_net.outWeights[bucket];
    auto dot = [&](const std::int16_t* col, const std::int16_t* ws) {
        __m256i s = _mm256_setzero_si256();
        for (int i = 0; i < NNUE_HL; i += 16) {
            __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(col + i));
            x = _mm256_min_epi16(_mm256_max_epi16(x, zero), hi);   // clipped ReLU
            __m256i ww = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ws + i));
            s = _mm256_add_epi32(s, _mm256_madd_epi16(x, ww));
        }
        __m128i lo = _mm256_castsi256_si128(s);
        __m128i hh = _mm256_extracti128_si256(s, 1);
        __m128i t = _mm_add_epi32(lo, hh);
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
        return _mm_cvtsi128_si32(t);
    };
    std::int64_t sum = g_net.outBias[bucket];
    sum += dot(acc.v[stm], w);
    sum += dot(acc.v[~stm], w + NNUE_HL);
    return int(sum / g_net.scale);
#else
    return output_scalar(acc, stm, bucket);
#endif
}

int evaluate(const Position& pos) {
    thread_local Accumulator acc;
    refresh(pos, acc);
    return output(acc, pos.side_to_move(), output_bucket(pos));
}

} // namespace NNUE
} // namespace chess
