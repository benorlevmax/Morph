// movegen.cpp - Move generation.
//
// Strategy for Phase 1: generate pseudo-legal moves, then (for GenType LEGAL)
// filter through Position::is_legal. Castling path-attack checks are done here
// because is_legal only validates the king's destination square.
#include "core/movegen.h"

namespace chess {

namespace {

template <Color Us>
void generate_pawn_moves(const Position& pos, MoveList& list) {
    constexpr Color Them = ~Us;
    constexpr Direction Up      = (Us == WHITE ? NORTH : SOUTH);
    constexpr Direction UpRight = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft  = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);
    constexpr Bitboard  Rank7   = (Us == WHITE ? RANK_7_BB : RANK_2_BB);
    constexpr Bitboard  Rank3   = (Us == WHITE ? RANK_3_BB : RANK_6_BB);

    const Bitboard pawns      = pos.pieces(Us, PAWN);
    const Bitboard empty      = ~pos.pieces();
    const Bitboard enemies    = pos.pieces(Them);

    const Bitboard pawnsNot7  = pawns & ~Rank7;
    const Bitboard pawns7     = pawns & Rank7;

    // Single and double pushes.
    Bitboard b1 = shift<Up>(pawnsNot7) & empty;
    Bitboard b2 = shift<Up>(b1 & Rank3) & empty;
    while (b1) {
        Square to = pop_lsb(b1);
        list.add(Move(to - Up, to));
    }
    while (b2) {
        Square to = pop_lsb(b2);
        list.add(Move(to - Up - Up, to));
    }

    // Captures (non-promoting).
    Bitboard cr = shift<UpRight>(pawnsNot7) & enemies;
    Bitboard cl = shift<UpLeft>(pawnsNot7) & enemies;
    while (cr) { Square to = pop_lsb(cr); list.add(Move(to - UpRight, to)); }
    while (cl) { Square to = pop_lsb(cl); list.add(Move(to - UpLeft, to)); }

    // Promotions (push and capture).
    if (pawns7) {
        Bitboard pp = shift<Up>(pawns7) & empty;
        Bitboard pr = shift<UpRight>(pawns7) & enemies;
        Bitboard pl = shift<UpLeft>(pawns7) & enemies;
        auto emit = [&](Square from, Square to) {
            list.add(Move::make<PROMOTION>(from, to, QUEEN));
            list.add(Move::make<PROMOTION>(from, to, ROOK));
            list.add(Move::make<PROMOTION>(from, to, BISHOP));
            list.add(Move::make<PROMOTION>(from, to, KNIGHT));
        };
        while (pp) { Square to = pop_lsb(pp); emit(to - Up, to); }
        while (pr) { Square to = pop_lsb(pr); emit(to - UpRight, to); }
        while (pl) { Square to = pop_lsb(pl); emit(to - UpLeft, to); }
    }

    // En passant.
    if (pos.ep_square() != SQ_NONE) {
        const Square ep = pos.ep_square();
        Bitboard attackers = pawn_attacks(Them, ep) & pawnsNot7;
        while (attackers) {
            Square from = pop_lsb(attackers);
            list.add(Move::make<EN_PASSANT>(from, ep));
        }
    }
}

template <PieceType Pt>
void generate_piece_moves(const Position& pos, MoveList& list, Bitboard target) {
    const Color us = pos.side_to_move();
    const Bitboard occ = pos.pieces();
    Bitboard from = pos.pieces(us, Pt);
    while (from) {
        Square s = pop_lsb(from);
        Bitboard att = attacks_bb(Pt, s, occ) & target;
        while (att) list.add(Move(s, pop_lsb(att)));
    }
}

template <Color Us>
void generate_castling(const Position& pos, MoveList& list) {
    if (pos.in_check()) return;
    const int cr = pos.castling_rights();
    const Bitboard occ = pos.pieces();
    constexpr Color Them = ~Us;
    constexpr Square KingFrom = (Us == WHITE ? SQ_E1 : SQ_E8);

    auto try_castle = [&](int right, Square kingTo, Square rookFrom,
                          std::initializer_list<Square> emptySq,
                          std::initializer_list<Square> safeSq) {
        if (!(cr & right)) return;
        if (pos.piece_on(rookFrom) != make_piece(Us, ROOK)) return;
        for (Square s : emptySq) if (occ & square_bb(s)) return;
        for (Square s : safeSq)  if (pos.is_square_attacked(s, Them)) return;
        list.add(Move::make<CASTLING>(KingFrom, kingTo));
    };

    if (Us == WHITE) {
        try_castle(WHITE_OO,  SQ_G1, SQ_H1, {SQ_F1, SQ_G1}, {SQ_F1, SQ_G1});
        try_castle(WHITE_OOO, SQ_C1, SQ_A1, {SQ_B1, SQ_C1, SQ_D1}, {SQ_C1, SQ_D1});
    } else {
        try_castle(BLACK_OO,  SQ_G8, SQ_H8, {SQ_F8, SQ_G8}, {SQ_F8, SQ_G8});
        try_castle(BLACK_OOO, SQ_C8, SQ_A8, {SQ_B8, SQ_C8, SQ_D8}, {SQ_C8, SQ_D8});
    }
}

template <Color Us>
void generate_all(const Position& pos, MoveList& list, GenType type) {
    const Bitboard own     = pos.pieces(Us);
    const Bitboard enemies = pos.pieces(~Us);

    Bitboard target;
    switch (type) {
        case CAPTURES: target = enemies; break;
        case QUIETS:   target = ~pos.pieces(); break;
        default:       target = ~own; break;   // NON_EVASIONS / LEGAL handled by filter
    }

    // Pawns generate their own captures/quiets/promotions/ep internally; for a
    // pure CAPTURES request we still include pawn captures + promotions via the
    // full generator and rely on the caller for filtering when needed.
    generate_pawn_moves<Us>(pos, list);
    generate_piece_moves<KNIGHT>(pos, list, target);
    generate_piece_moves<BISHOP>(pos, list, target);
    generate_piece_moves<ROOK>(pos, list, target);
    generate_piece_moves<QUEEN>(pos, list, target);
    generate_piece_moves<KING>(pos, list, target);

    if (type != CAPTURES)
        generate_castling<Us>(pos, list);
}

} // namespace

void generate(const Position& pos, MoveList& list, GenType type) {
    if (type == LEGAL) {
        MoveList pseudo;
        if (pos.side_to_move() == WHITE) generate_all<WHITE>(pos, pseudo, NON_EVASIONS);
        else                             generate_all<BLACK>(pos, pseudo, NON_EVASIONS);
        for (const auto& sm : pseudo)
            if (pos.is_legal(sm.move))
                list.add(sm.move);
        return;
    }

    if (pos.side_to_move() == WHITE) generate_all<WHITE>(pos, list, type);
    else                             generate_all<BLACK>(pos, list, type);
}

} // namespace chess
