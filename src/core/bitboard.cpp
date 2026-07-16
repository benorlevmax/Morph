// bitboard.cpp - Attack-table initialization.
#include "core/bitboard.h"

#include <algorithm>
#include <random>
#include <vector>

namespace chess::detail {

Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];
Bitboard RayAttacks[8][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
std::uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

Magic BishopMagics[SQUARE_NB];
Magic RookMagics[SQUARE_NB];

// Shared attack tables the Magic descriptors point into.
Bitboard BishopTable[5248];
Bitboard RookTable[102400];

} // namespace chess::detail

namespace chess {

namespace {

// File/rank deltas for the 8 compass directions, indexed to match
// the ray-index convention used in bitboard.h:
//   0=N 1=E 2=S 3=W 4=NE 5=NW 6=SE 7=SW
constexpr int RayDF[8] = { 0, +1,  0, -1, +1, -1, +1, -1 };
constexpr int RayDR[8] = {+1,  0, -1,  0, +1, +1, -1, -1 };

// Knight/king offsets as (df,dr) pairs.
constexpr int KnightDF[8] = {+1, +2, +2, +1, -1, -2, -2, -1};
constexpr int KnightDR[8] = {+2, +1, -1, -2, -2, -1, +1, +2};
constexpr int KingDF[8]   = {+1, +1,  0, -1, -1, -1,  0, +1};
constexpr int KingDR[8]   = { 0, +1, +1, +1,  0, -1, -1, -1};

inline bool on_board(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

// ---- Magic bitboard initialization ---------------------------------------
constexpr int RookDir[4][2]   = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
constexpr int BishopDir[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

inline Bitboard sq_bb(int f, int r) { return square_bb(make_square(File(f), Rank(r))); }

// On-the-fly sliding attacks (used only to build the magic tables).
Bitboard slide(Square sq, Bitboard occ, const int dir[4][2]) {
    Bitboard att = 0;
    const int f0 = file_of(sq), r0 = rank_of(sq);
    for (int d = 0; d < 4; ++d) {
        int f = f0 + dir[d][0], r = r0 + dir[d][1];
        while (on_board(f, r)) {
            Bitboard b = sq_bb(f, r);
            att |= b;
            if (occ & b) break;
            f += dir[d][0]; r += dir[d][1];
        }
    }
    return att;
}

// Relevant occupancy mask (interior squares only; edges never matter).
Bitboard relevant_mask(Square sq, bool rook) {
    Bitboard m = 0;
    const int f0 = file_of(sq), r0 = rank_of(sq);
    if (rook) {
        for (int r = r0 + 1; r <= 6; ++r) m |= sq_bb(f0, r);
        for (int r = r0 - 1; r >= 1; --r) m |= sq_bb(f0, r);
        for (int f = f0 + 1; f <= 6; ++f) m |= sq_bb(f, r0);
        for (int f = f0 - 1; f >= 1; --f) m |= sq_bb(f, r0);
    } else {
        for (int f = f0 + 1, r = r0 + 1; f <= 6 && r <= 6; ++f, ++r) m |= sq_bb(f, r);
        for (int f = f0 + 1, r = r0 - 1; f <= 6 && r >= 1; ++f, --r) m |= sq_bb(f, r);
        for (int f = f0 - 1, r = r0 + 1; f >= 1 && r <= 6; --f, ++r) m |= sq_bb(f, r);
        for (int f = f0 - 1, r = r0 - 1; f >= 1 && r >= 1; --f, --r) m |= sq_bb(f, r);
    }
    return m;
}

void init_magic_group(detail::Magic magics[], Bitboard table[],
                      const int dir[4][2], bool rook) {
    std::mt19937_64 rng(rook ? 0xD00DFEEDULL : 0xC0FFEE99ULL);
    auto sparse = [&]() { return rng() & rng() & rng(); };

    std::size_t offset = 0;
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        const Bitboard mask = relevant_mask(s, rook);
        const int bits = popcount(mask);
        const unsigned size = 1u << bits;

        detail::Magic& mg = magics[s];
        mg.mask = mask;
        mg.shift = unsigned(64 - bits);
        mg.attacks = table + offset;
        offset += size;

        // Enumerate every occupancy subset of the mask and its reference attack.
        std::vector<Bitboard> occ(size), ref(size);
        Bitboard b = 0;
        unsigned idx = 0;
        do { occ[idx] = b; ref[idx] = slide(s, b, dir); ++idx; b = (b - mask) & mask; }
        while (b);

        // Search for a collision-free magic.
        while (true) {
            Bitboard magic = sparse();
            std::fill(mg.attacks, mg.attacks + size, Bitboard(0));
            bool ok = true;
            for (unsigned k = 0; k < size; ++k) {
                unsigned i = unsigned((occ[k] * magic) >> mg.shift);
                if (mg.attacks[i] == 0) mg.attacks[i] = ref[k];
                else if (mg.attacks[i] != ref[k]) { ok = false; break; }
            }
            if (ok) { mg.magic = magic; break; }
        }
    }
}

void init_magics() {
    init_magic_group(detail::BishopMagics, detail::BishopTable, BishopDir, false);
    init_magic_group(detail::RookMagics, detail::RookTable, RookDir, true);
}

} // namespace

void Bitboards::init() {
    using namespace detail;

    // Distances and simple step attacks.
    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1) {
        const int f = file_of(s1), r = rank_of(s1);

        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2) {
            SquareDistance[s1][s2] = std::uint8_t(std::max(
                std::abs(file_of(s1) - file_of(s2)),
                std::abs(rank_of(s1) - rank_of(s2))));
        }

        // Knight attacks.
        Bitboard kn = 0;
        for (int i = 0; i < 8; ++i)
            if (on_board(f + KnightDF[i], r + KnightDR[i]))
                kn |= square_bb(make_square(File(f + KnightDF[i]), Rank(r + KnightDR[i])));
        KnightAttacks[s1] = kn;

        // King attacks.
        Bitboard kg = 0;
        for (int i = 0; i < 8; ++i)
            if (on_board(f + KingDF[i], r + KingDR[i]))
                kg |= square_bb(make_square(File(f + KingDF[i]), Rank(r + KingDR[i])));
        KingAttacks[s1] = kg;

        // Pawn attacks.
        PawnAttacks[WHITE][s1] = pawn_attacks_bb<WHITE>(square_bb(s1));
        PawnAttacks[BLACK][s1] = pawn_attacks_bb<BLACK>(square_bb(s1));

        // Ray attacks per direction (extend until board edge).
        for (int d = 0; d < 8; ++d) {
            Bitboard ray = 0;
            int cf = f + RayDF[d], cr = r + RayDR[d];
            while (on_board(cf, cr)) {
                ray |= square_bb(make_square(File(cf), Rank(cr)));
                cf += RayDF[d];
                cr += RayDR[d];
            }
            RayAttacks[d][s1] = ray;
        }
    }

    // Magic sliding-attack tables (must precede line/between, which query them).
    init_magics();

    // Line and between tables (depend on ray/sliding attacks above).
    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1) {
        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2) {
            BetweenBB[s1][s2] = 0;
            LineBB[s1][s2]    = 0;
            if (s1 == s2) continue;

            for (PieceType pt : {BISHOP, ROOK}) {
                Bitboard a1 = attacks_bb(pt, s1, 0);
                if (a1 & s2) {
                    // Full line through both squares.
                    LineBB[s1][s2] = (attacks_bb(pt, s1, 0) & attacks_bb(pt, s2, 0))
                                     | square_bb(s1) | square_bb(s2);
                    // Exclusive segment between them.
                    BetweenBB[s1][s2] =
                        attacks_bb(pt, s1, square_bb(s2)) &
                        attacks_bb(pt, s2, square_bb(s1));
                }
            }
        }
    }
}

} // namespace chess
