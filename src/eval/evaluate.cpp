// evaluate.cpp - Classical tapered evaluation.
//
// Components: material + PSQT (incremental, from Position::psq()), pawn
// structure (cached by pawn key), passed pawns, piece mobility & placement,
// king safety, endgame king-passer interaction, endgame scaling, mop-up, and
// a tempo bonus. The whole evaluation is white-POV internally and negated for
// the side to move at the very end.
#include "eval/evaluate.h"
#include "eval/psqt.h"
#include "core/bitboard.h"
#include "core/movegen.h"
#include "nnue/nnue.h"

#include <algorithm>
#include <cstdint>

namespace chess {

namespace {

constexpr Bitboard LightSquares = 0x55AA55AA55AA55AAULL;

// ---------------------------------------------------------------------------
// Tunable weights
// ---------------------------------------------------------------------------
constexpr Score BishopPair       = {25, 40};
constexpr Score RookOpenFile     = {25, 12};
constexpr Score RookSemiOpenFile = {12,  6};
constexpr Score IsolatedPawn     = {-8, -12};
constexpr Score DoubledPawn      = {-10, -18};
constexpr Score TempoBonus       = {18,  0};

constexpr Score PassedBonus[RANK_NB] = {
    {0,0}, {0,5}, {5,10}, {15,25}, {35,50}, {65,90}, {110,150}, {0,0}
};
constexpr Score ConnectedBonus[RANK_NB] = {
    {0,0}, {3,2}, {5,3}, {8,5}, {14,10}, {25,20}, {45,40}, {0,0}
};
constexpr int PassedKingWeight[RANK_NB] = {0, 0, 0, 2, 4, 7, 10, 0};

constexpr int KingAttackWeight[PIECE_TYPE_NB] = {0, 0, 2, 2, 3, 5, 0};
constexpr int ShelterByRank[RANK_NB] = {0, 8, 6, 3, 1, 0, 0, 0};  // relative rank

// Mobility tables, indexed by reachable-square count.
constexpr Score KnightMob[9] = {
    {-30,-30},{-20,-22},{-9,-12},{-2,-3},{4,4},{9,9},{14,13},{18,16},{20,18}
};
constexpr Score BishopMob[14] = {
    {-25,-30},{-14,-18},{0,-6},{6,2},{12,8},{18,14},{22,20},{26,25},
    {28,28},{30,30},{31,32},{33,34},{34,35},{35,36}
};
constexpr Score RookMob[15] = {
    {-25,-30},{-15,-12},{-6,2},{-2,10},{0,18},{3,26},{6,34},{9,40},
    {12,46},{14,50},{16,54},{18,57},{19,60},{20,62},{21,63}
};
constexpr Score QueenMob[28] = {
    {-15,-25},{-10,-15},{-6,-8},{-3,-2},{-1,4},{1,10},{3,16},{5,22},
    {7,28},{9,33},{10,38},{12,42},{13,46},{14,50},{15,53},{16,56},
    {17,59},{18,61},{19,63},{20,65},{20,67},{21,68},{22,69},{22,70},
    {23,71},{23,72},{24,73},{24,74}
};

inline Score mobility_bonus(PieceType pt, int m) {
    switch (pt) {
        case KNIGHT: return KnightMob[std::min(m, 8)];
        case BISHOP: return BishopMob[std::min(m, 13)];
        case ROOK:   return RookMob[std::min(m, 14)];
        case QUEEN:  return QueenMob[std::min(m, 27)];
        default:     return Score{};
    }
}

// ---------------------------------------------------------------------------
// Pawn-structure cache (keyed by the incremental pawn-only Zobrist key).
// ---------------------------------------------------------------------------
struct PawnEntry {
    Key      key = 0;
    Score    score{};               // white POV: structure + passer-rank bonus
    Bitboard passed[COLOR_NB] = {0, 0};
    bool     valid = false;
};

constexpr std::size_t PawnTableSize = 1 << 14;   // 16k entries
PawnEntry PawnTable[PawnTableSize];

// ---------------------------------------------------------------------------
// Per-evaluation scratch state.
// ---------------------------------------------------------------------------
struct EvalInfo {
    Bitboard pawnAttacks[COLOR_NB] = {0, 0};
    Bitboard mobilityArea[COLOR_NB] = {0, 0};
    Bitboard kingRing[COLOR_NB] = {0, 0};
    Bitboard passed[COLOR_NB] = {0, 0};
    int kingAttackersCount[COLOR_NB] = {0, 0};
    int kingAttackersWeight[COLOR_NB] = {0, 0};
};

inline Bitboard adjacent_files_bb(File f) {
    const Bitboard ff = file_bb(f);
    return ((ff & ~FILE_H_BB) << 1) | ((ff & ~FILE_A_BB) >> 1);
}

// All ranks strictly ahead of `s` from color c's perspective.
inline Bitboard forward_ranks(Color c, Square s) {
    Bitboard b = 0;
    if (c == WHITE)
        for (Rank r = Rank(rank_of(s) + 1); r <= RANK_8; r = Rank(r + 1)) b |= rank_bb(r);
    else
        for (int r = int(rank_of(s)) - 1; r >= 0; --r) b |= rank_bb(Rank(r));
    return b;
}

// ---------------------------------------------------------------------------
// Pawn structure (per color), filling passed-pawn bitboards.
// ---------------------------------------------------------------------------
template <Color Us>
Score pawns_eval(const Position& pos, Bitboard& passedOut) {
    constexpr Color Them = ~Us;
    const Bitboard ours   = pos.pieces(Us, PAWN);
    const Bitboard theirs = pos.pieces(Them, PAWN);

    Score s{};
    Bitboard passed = 0;
    Bitboard b = ours;
    while (b) {
        const Square sq = pop_lsb(b);
        const File f = file_of(sq);
        const Bitboard adj = adjacent_files_bb(f);
        const Bitboard ahead = forward_ranks(Us, sq);

        const bool isolated = !(ours & adj);
        const bool doubled  = ours & file_bb(f) & ahead;
        const Bitboard span = (file_bb(f) | adj) & ahead;
        const bool passedP  = !(theirs & span);
        const bool supported = ours & pawn_attacks(Them, sq);  // defended by a pawn
        const bool phalanx   = ours & adj & rank_bb(sq);

        if (passedP) { passed |= square_bb(sq); s += PassedBonus[relative_rank(Us, sq)]; }
        if (isolated) s += IsolatedPawn;
        if (doubled)  s += DoubledPawn;
        if (supported || phalanx) s += ConnectedBonus[relative_rank(Us, sq)];
    }
    passedOut = passed;
    return s;
}

Score probe_pawns(const Position& pos, EvalInfo& ei) {
    const Key key = pos.pawn_key();
    PawnEntry& e = PawnTable[key & (PawnTableSize - 1)];
    if (e.valid && e.key == key) {
        ei.passed[WHITE] = e.passed[WHITE];
        ei.passed[BLACK] = e.passed[BLACK];
        return e.score;
    }

    Bitboard wPassed = 0, bPassed = 0;
    Score s = pawns_eval<WHITE>(pos, wPassed) - pawns_eval<BLACK>(pos, bPassed);

    e.key = key;
    e.score = s;
    e.passed[WHITE] = wPassed;
    e.passed[BLACK] = bPassed;
    e.valid = true;
    ei.passed[WHITE] = wPassed;
    ei.passed[BLACK] = bPassed;
    return s;
}

// ---------------------------------------------------------------------------
// Piece mobility & placement (per color), accumulating king-attack pressure.
// ---------------------------------------------------------------------------
template <Color Us>
Score pieces_eval(const Position& pos, EvalInfo& ei) {
    constexpr Color Them = ~Us;
    const Bitboard occ = pos.pieces();
    Score s{};

    for (PieceType pt = KNIGHT; pt <= QUEEN; ++pt) {
        Bitboard b = pos.pieces(Us, pt);
        while (b) {
            const Square sq = pop_lsb(b);
            const Bitboard att = attacks_bb(pt, sq, occ);

            s += mobility_bonus(pt, popcount(att & ei.mobilityArea[Us]));

            if (att & ei.kingRing[Them]) {
                ei.kingAttackersCount[Us]++;
                ei.kingAttackersWeight[Us] += KingAttackWeight[pt];
            }

            if (pt == ROOK) {
                const Bitboard fileMask = file_bb(sq);
                if (!(pos.pieces(PAWN) & fileMask))           s += RookOpenFile;
                else if (!(pos.pieces(Us, PAWN) & fileMask))  s += RookSemiOpenFile;
            }
        }
    }

    if (popcount(pos.pieces(Us, BISHOP)) >= 2)
        s += BishopPair;

    return s;
}

// ---------------------------------------------------------------------------
// King safety (per color): pawn shelter + attacker pressure.
// ---------------------------------------------------------------------------
template <Color Us>
Score king_safety(const Position& pos, const EvalInfo& ei) {
    constexpr Color Them = ~Us;
    const Square ksq = pos.king_square(Us);
    const Bitboard ourPawns = pos.pieces(Us, PAWN);

    Score s{};

    // Pawn shelter on the king file and its neighbours.
    int kf = file_of(ksq);
    kf = std::clamp(kf, int(FILE_B), int(FILE_G));
    for (int f = kf - 1; f <= kf + 1; ++f) {
        const Bitboard fp = ourPawns & file_bb(File(f));
        if (fp) {
            const Square nearest = (Us == WHITE) ? lsb(fp) : msb(fp);
            s.mg += ShelterByRank[relative_rank(Us, nearest)];
        } else {
            s.mg -= 6;   // open file next to our king
        }
    }

    // Attacker pressure from enemy pieces near our king ring.
    const int weight = ei.kingAttackersWeight[Them];
    if (ei.kingAttackersCount[Them] >= 2)
        s.mg -= (weight * weight) / 6;
    else
        s.mg -= weight;

    return s;
}

// ---------------------------------------------------------------------------
// Endgame king/passed-pawn interaction (eg only, white POV).
// ---------------------------------------------------------------------------
Score passed_king_eval(const Position& pos, const EvalInfo& ei) {
    Score s{};
    for (Color us : {WHITE, BLACK}) {
        const Color them = ~us;
        Bitboard p = ei.passed[us];
        while (p) {
            const Square sq = pop_lsb(p);
            const int w = PassedKingWeight[relative_rank(us, sq)];
            if (!w) continue;
            const Square block = us == WHITE ? Square(sq + 8) : Square(sq - 8);
            if (!is_ok(block)) continue;
            const int dThem = distance(pos.king_square(them), block);
            const int dUs   = distance(pos.king_square(us), block);
            const int eg = (dThem * 5 - dUs * 5) * w / 10;
            if (us == WHITE) s.eg += eg; else s.eg -= eg;
        }
    }
    return s;
}

// Distance-from-center (0 center .. 6 corner), used for mop-up mating.
inline int center_distance(Square s) {
    const int f = file_of(s), r = rank_of(s);
    return std::max(3 - f, f - 4) + std::max(3 - r, r - 4);
}

// Mop-up: when one side has only its king, drive it to the edge/corner and
// bring the winning king closer.
Score mop_up(const Position& pos) {
    const bool whiteBare = pos.pieces(WHITE) == pos.pieces(WHITE, KING);
    const bool blackBare = pos.pieces(BLACK) == pos.pieces(BLACK, KING);
    Score s{};
    if (blackBare && !whiteBare) {
        const int d = distance(pos.king_square(WHITE), pos.king_square(BLACK));
        s.eg += center_distance(pos.king_square(BLACK)) * 12 + (7 - d) * 4;
    } else if (whiteBare && !blackBare) {
        const int d = distance(pos.king_square(WHITE), pos.king_square(BLACK));
        s.eg -= center_distance(pos.king_square(WHITE)) * 12 + (7 - d) * 4;
    }
    return s;
}

int game_phase(const Position& pos) {
    int p = PhaseKnight * popcount(pos.pieces(KNIGHT))
          + PhaseBishop * popcount(pos.pieces(BISHOP))
          + PhaseRook   * popcount(pos.pieces(ROOK))
          + PhaseQueen  * popcount(pos.pieces(QUEEN));
    return std::min(p, PhaseMax);
}

// Endgame scale factor (0..64); shrinks the eg component for drawish material.
int scale_factor(const Position& pos) {
    const Bitboard wb = pos.pieces(WHITE, BISHOP);
    const Bitboard bb = pos.pieces(BLACK, BISHOP);
    const bool noOther = !(pos.pieces(KNIGHT) | pos.pieces(ROOK) | pos.pieces(QUEEN));
    if (noOther && popcount(wb) == 1 && popcount(bb) == 1) {
        const bool wLight = wb & LightSquares;
        const bool bLight = bb & LightSquares;
        if (wLight != bLight)
            return 36;   // opposite-coloured bishops: drawish
    }
    return 64;
}

void init_eval(const Position& pos, EvalInfo& ei) {
    ei.pawnAttacks[WHITE] = pawn_attacks_bb<WHITE>(pos.pieces(WHITE, PAWN));
    ei.pawnAttacks[BLACK] = pawn_attacks_bb<BLACK>(pos.pieces(BLACK, PAWN));

    for (Color c : {WHITE, BLACK}) {
        const Square ksq = pos.king_square(c);
        ei.kingRing[c] = king_attacks(ksq) | square_bb(ksq);
        ei.mobilityArea[c] = ~(pos.pieces(c, KING) | pos.pieces(c, PAWN)
                               | ei.pawnAttacks[~c]);
    }
}

} // namespace

namespace {
bool g_materialOnly = false;
EvalMode g_mode = EvalMode::Classical;
}

namespace Eval {
void set_material_only(bool on) { g_materialOnly = on; }
bool material_only() { return g_materialOnly; }

void set_mode(EvalMode m) {
    g_mode = m;
    NNUE::set_enabled(m == EvalMode::NNUE);
}
EvalMode mode() { return g_mode; }

void on_search_start(Position& pos) {
    if (g_mode == EvalMode::NNUE) pos.nnue_refresh();
}
} // namespace Eval

Value evaluate(const Position& pos) {
    if (g_mode == EvalMode::NNUE) {
        // Dual-perspective net already returns a side-to-move-relative score.
        return Value(NNUE::output(pos.accumulator(), pos.side_to_move(),
                                  NNUE::output_bucket(pos)));
    }

    if (g_materialOnly) {
        int sc = 0;
        for (PieceType pt = PAWN; pt <= QUEEN; ++pt)
            sc += PieceValue[pt] * (pos.count(WHITE, pt) - pos.count(BLACK, pt));
        sc += pos.side_to_move() == WHITE ? 18 : -18;   // tempo
        return Value(pos.side_to_move() == WHITE ? sc : -sc);
    }

    EvalInfo ei;
    init_eval(pos, ei);

    Score score = pos.psq();                 // incremental material + PSQT
    score += probe_pawns(pos, ei);           // pawn structure + passers (cached)
    score += pieces_eval<WHITE>(pos, ei) - pieces_eval<BLACK>(pos, ei);
    score += king_safety<WHITE>(pos, ei) - king_safety<BLACK>(pos, ei);
    score += passed_king_eval(pos, ei);
    score += mop_up(pos);
    score += pos.side_to_move() == WHITE ? TempoBonus : -TempoBonus;

    const int phase = game_phase(pos);
    const int scale = scale_factor(pos);
    const Value v = tapered(score, phase, scale);

    return pos.side_to_move() == WHITE ? v : -v;
}

} // namespace chess
