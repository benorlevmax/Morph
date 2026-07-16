// types.h - Fundamental engine types: colors, pieces, squares, moves.
#pragma once

#include <cstdint>
#include <cassert>

namespace chess {

// ---------------------------------------------------------------------------
// Bitboard
// ---------------------------------------------------------------------------
using Bitboard = std::uint64_t;
using Key      = std::uint64_t;   // Zobrist hash key

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------
enum Color : int {
    WHITE, BLACK, COLOR_NB = 2
};

constexpr Color operator~(Color c) { return Color(c ^ BLACK); }

// ---------------------------------------------------------------------------
// Piece types and pieces
// ---------------------------------------------------------------------------
enum PieceType : int {
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    ALL_PIECES = 0,
    PIECE_TYPE_NB = 7
};

enum Piece : int {
    NO_PIECE,
    W_PAWN = PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = PAWN + 8, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};

constexpr Piece make_piece(Color c, PieceType pt) {
    return Piece((c << 3) + pt);
}
constexpr PieceType type_of(Piece pc) { return PieceType(pc & 7); }
constexpr Color color_of(Piece pc) {
    assert(pc != NO_PIECE);
    return Color(pc >> 3);
}

// ---------------------------------------------------------------------------
// Squares, files, ranks
// ---------------------------------------------------------------------------
enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,
    SQUARE_NB = 64
};

enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };

constexpr Square make_square(File f, Rank r) { return Square((r << 3) + f); }
constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

// Relative rank from a color's point of view (RANK_1 is own back rank).
constexpr Rank relative_rank(Color c, Rank r) { return Rank(r ^ (c * 7)); }
constexpr Rank relative_rank(Color c, Square s) { return relative_rank(c, rank_of(s)); }
constexpr Square relative_square(Color c, Square s) { return Square(s ^ (c * 56)); }

// ---------------------------------------------------------------------------
// Directions (square index deltas)
// ---------------------------------------------------------------------------
enum Direction : int {
    NORTH =  8, EAST =  1, SOUTH = -8, WEST = -1,
    NORTH_EAST = NORTH + EAST, SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST, NORTH_WEST = NORTH + WEST
};

// ---------------------------------------------------------------------------
// Castling rights (bitmask)
// ---------------------------------------------------------------------------
enum CastlingRights : int {
    NO_CASTLING,
    WHITE_OO  = 1,
    WHITE_OOO = 2,
    BLACK_OO  = 4,
    BLACK_OOO = 8,
    KING_SIDE      = WHITE_OO  | BLACK_OO,
    QUEEN_SIDE     = WHITE_OOO | BLACK_OOO,
    WHITE_CASTLING = WHITE_OO  | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO  | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,
    CASTLING_RIGHT_NB = 16
};

// ---------------------------------------------------------------------------
// Move encoding (16 bits)
//   bits  0- 5 : destination square
//   bits  6-11 : origin square
//   bits 12-13 : promotion piece type minus KNIGHT (0=N,1=B,2=R,3=Q)
//   bits 14-15 : move type (NORMAL / PROMOTION / EN_PASSANT / CASTLING)
// ---------------------------------------------------------------------------
enum MoveType : int {
    NORMAL,
    PROMOTION  = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING   = 3 << 14
};

class Move {
public:
    Move() = default;
    constexpr explicit Move(std::uint16_t d) : data_(d) {}

    constexpr Move(Square from, Square to) : data_(std::uint16_t((from << 6) + to)) {}

    template <MoveType T>
    static constexpr Move make(Square from, Square to, PieceType pt = KNIGHT) {
        return Move(std::uint16_t(T + ((pt - KNIGHT) << 12) + (from << 6) + to));
    }

    constexpr Square from_sq() const { return Square((data_ >> 6) & 0x3F); }
    constexpr Square to_sq()   const { return Square(data_ & 0x3F); }
    constexpr int    from_to() const { return data_ & 0xFFF; }
    constexpr MoveType type_of() const { return MoveType(data_ & (3 << 14)); }
    constexpr PieceType promotion_type() const {
        return PieceType(((data_ >> 12) & 3) + KNIGHT);
    }

    constexpr bool is_ok() const { return none().data_ != data_ && null().data_ != data_; }
    constexpr std::uint16_t raw() const { return data_; }

    static constexpr Move none() { return Move(0); }
    static constexpr Move null() { return Move(65); }   // from==to==A1 impossible

    constexpr bool operator==(const Move& m) const { return data_ == m.data_; }
    constexpr bool operator!=(const Move& m) const { return data_ != m.data_; }

private:
    std::uint16_t data_ = 0;
};

// ---------------------------------------------------------------------------
// Scores (centipawn-based; large mate band)
// ---------------------------------------------------------------------------
enum Value : int {
    VALUE_ZERO      = 0,
    VALUE_DRAW      = 0,
    VALUE_MATE      = 32000,
    VALUE_INFINITE  = 32001,
    VALUE_NONE      = 32002,
    VALUE_MATE_IN_MAX_PLY = VALUE_MATE - 256,
    VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY
};

// Negamax arithmetic on Value (keeps search code readable and type-safe).
constexpr Value operator-(Value d) { return Value(-int(d)); }
constexpr Value operator+(Value a, Value b) { return Value(int(a) + int(b)); }
constexpr Value operator-(Value a, Value b) { return Value(int(a) - int(b)); }
constexpr Value operator+(Value a, int b) { return Value(int(a) + b); }
constexpr Value operator-(Value a, int b) { return Value(int(a) - b); }
constexpr Value& operator+=(Value& a, Value b) { return a = a + b; }
constexpr Value& operator-=(Value& a, Value b) { return a = a - b; }

constexpr Value mate_in(int ply)  { return Value(VALUE_MATE - ply); }
constexpr Value mated_in(int ply) { return Value(-VALUE_MATE + ply); }

// ---------------------------------------------------------------------------
// Generic enum arithmetic helpers
// ---------------------------------------------------------------------------
#define ENABLE_INCR_OPERATORS(T)                                            \
    constexpr T& operator++(T& d) { return d = T(int(d) + 1); }            \
    constexpr T& operator--(T& d) { return d = T(int(d) - 1); }

ENABLE_INCR_OPERATORS(PieceType)
ENABLE_INCR_OPERATORS(Square)
ENABLE_INCR_OPERATORS(File)
ENABLE_INCR_OPERATORS(Rank)
#undef ENABLE_INCR_OPERATORS

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Square& operator+=(Square& s, Direction d) { return s = s + d; }
constexpr Square& operator-=(Square& s, Direction d) { return s = s - d; }
constexpr Direction operator+(Direction a, Direction b) { return Direction(int(a) + int(b)); }
constexpr Direction operator*(int i, Direction d) { return Direction(i * int(d)); }

} // namespace chess
