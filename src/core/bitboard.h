// bitboard.h - Bitboard utilities and attack generation.
#pragma once

#include "core/types.h"

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace chess {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
constexpr Bitboard FILE_C_BB = FILE_A_BB << 2;
constexpr Bitboard FILE_D_BB = FILE_A_BB << 3;
constexpr Bitboard FILE_E_BB = FILE_A_BB << 4;
constexpr Bitboard FILE_F_BB = FILE_A_BB << 5;
constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;

constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << (8 * 1);
constexpr Bitboard RANK_3_BB = RANK_1_BB << (8 * 2);
constexpr Bitboard RANK_4_BB = RANK_1_BB << (8 * 3);
constexpr Bitboard RANK_5_BB = RANK_1_BB << (8 * 4);
constexpr Bitboard RANK_6_BB = RANK_1_BB << (8 * 5);
constexpr Bitboard RANK_7_BB = RANK_1_BB << (8 * 6);
constexpr Bitboard RANK_8_BB = RANK_1_BB << (8 * 7);

constexpr Bitboard square_bb(Square s) { return Bitboard(1) << s; }

inline Bitboard  operator&(Bitboard b, Square s) { return b & square_bb(s); }
inline Bitboard  operator|(Bitboard b, Square s) { return b | square_bb(s); }
inline Bitboard  operator^(Bitboard b, Square s) { return b ^ square_bb(s); }
inline Bitboard& operator|=(Bitboard& b, Square s) { return b |= square_bb(s); }
inline Bitboard& operator^=(Bitboard& b, Square s) { return b ^= square_bb(s); }

constexpr Bitboard file_bb(File f) { return FILE_A_BB << f; }
constexpr Bitboard rank_bb(Rank r) { return RANK_1_BB << (8 * r); }
constexpr Bitboard file_bb(Square s) { return file_bb(file_of(s)); }
constexpr Bitboard rank_bb(Square s) { return rank_bb(rank_of(s)); }

// ---------------------------------------------------------------------------
// Population count / bit scan
// ---------------------------------------------------------------------------
inline int popcount(Bitboard b) {
#if defined(_MSC_VER)
    return int(__popcnt64(b));
#elif defined(__GNUC__)
    return __builtin_popcountll(b);
#else
    int c = 0;
    while (b) { b &= b - 1; ++c; }
    return c;
#endif
}

inline Square lsb(Bitboard b) {
    assert(b);
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return Square(idx);
#elif defined(__GNUC__)
    return Square(__builtin_ctzll(b));
#else
    int i = 0;
    while (!(b & 1)) { b >>= 1; ++i; }
    return Square(i);
#endif
}

inline Square msb(Bitboard b) {
    assert(b);
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return Square(idx);
#elif defined(__GNUC__)
    return Square(63 ^ __builtin_clzll(b));
#else
    int i = 63;
    while (!(b & (Bitboard(1) << 63))) { b <<= 1; --i; }
    return Square(i);
#endif
}

inline Square pop_lsb(Bitboard& b) {
    assert(b);
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr bool more_than_one(Bitboard b) { return b & (b - 1); }

// ---------------------------------------------------------------------------
// Shifting
// ---------------------------------------------------------------------------
template <Direction D>
constexpr Bitboard shift(Bitboard b) {
    return D == NORTH      ? b << 8
         : D == SOUTH      ? b >> 8
         : D == EAST       ? (b & ~FILE_H_BB) << 1
         : D == WEST       ? (b & ~FILE_A_BB) >> 1
         : D == NORTH_EAST ? (b & ~FILE_H_BB) << 9
         : D == NORTH_WEST ? (b & ~FILE_A_BB) << 7
         : D == SOUTH_EAST ? (b & ~FILE_H_BB) >> 7
         : D == SOUTH_WEST ? (b & ~FILE_A_BB) >> 9
         : 0;
}

// Pawn attacks for a whole bitboard of pawns of color C.
template <Color C>
constexpr Bitboard pawn_attacks_bb(Bitboard b) {
    return C == WHITE ? shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b)
                      : shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
}

// ---------------------------------------------------------------------------
// Precomputed attack tables (initialized by Bitboards::init()).
// ---------------------------------------------------------------------------
namespace detail {
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard RayAttacks[8][SQUARE_NB];          // 8 compass directions (init helper)
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];   // exclusive segment
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];      // full line through both
extern std::uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

// Magic bitboard descriptor for sliding attacks (O(1) lookup).
struct Magic {
    Bitboard  mask  = 0;       // relevant occupancy bits
    Bitboard  magic = 0;       // multiplier
    Bitboard* attacks = nullptr; // pointer into the shared attack table
    unsigned  shift = 0;
    unsigned  index(Bitboard occ) const {
        return unsigned(((occ & mask) * magic) >> shift);
    }
};
extern Magic BishopMagics[SQUARE_NB];
extern Magic RookMagics[SQUARE_NB];
} // namespace detail

namespace Bitboards {
void init();
}

// Non-sliding piece attacks ------------------------------------------------
inline Bitboard pawn_attacks(Color c, Square s)  { return detail::PawnAttacks[c][s]; }
inline Bitboard knight_attacks(Square s)         { return detail::KnightAttacks[s]; }
inline Bitboard king_attacks(Square s)           { return detail::KingAttacks[s]; }

// Sliding piece attacks (magic bitboards: O(1) table lookup) ---------------
inline Bitboard bishop_attacks(Square s, Bitboard occ) {
    const detail::Magic& m = detail::BishopMagics[s];
    return m.attacks[m.index(occ)];
}

inline Bitboard rook_attacks(Square s, Bitboard occ) {
    const detail::Magic& m = detail::RookMagics[s];
    return m.attacks[m.index(occ)];
}

inline Bitboard queen_attacks(Square s, Bitboard occ) {
    return bishop_attacks(s, occ) | rook_attacks(s, occ);
}

// Generic dispatch by piece type (occ ignored for non-sliders).
inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occ) {
    switch (pt) {
        case KNIGHT: return knight_attacks(s);
        case BISHOP: return bishop_attacks(s, occ);
        case ROOK:   return rook_attacks(s, occ);
        case QUEEN:  return queen_attacks(s, occ);
        case KING:   return king_attacks(s);
        default:     return 0;
    }
}

// Geometry helpers ---------------------------------------------------------
inline Bitboard between_bb(Square a, Square b) { return detail::BetweenBB[a][b]; }
inline Bitboard line_bb(Square a, Square b)    { return detail::LineBB[a][b]; }
inline bool aligned(Square a, Square b, Square c) {
    return line_bb(a, b) & square_bb(c);
}
inline int distance(Square a, Square b) { return detail::SquareDistance[a][b]; }

} // namespace chess
