// position.h - Board state, FEN I/O, make/unmake, attack queries.
#pragma once

#include "core/bitboard.h"
#include "eval/score.h"
#include "nnue/nnue.h"

#include <cstring>
#include <string>
#include <vector>

namespace chess {

// Reversible per-move state, stored on an undo stack.
struct StateInfo {
    Key            key            = 0;
    Bitboard       checkers       = 0;
    Piece          captured       = NO_PIECE;
    Square         epSquare       = SQ_NONE;
    int            castlingRights = NO_CASTLING;
    int            halfmoveClock  = 0;
    int            pliesFromNull  = 0;
    int            repetition     = 0;   // ply-distance to a prior identical key
};

class Position {
public:
    Position();

    // Copies must re-point st_ into the copied state stack (the default copy
    // would leave st_ dangling into the source's vector).
    Position(const Position& o) { *this = o; }
    Position& operator=(const Position& o) {
        if (this == &o) return *this;
        std::memcpy(byType_, o.byType_, sizeof(byType_));
        std::memcpy(byColor_, o.byColor_, sizeof(byColor_));
        std::memcpy(board_, o.board_, sizeof(board_));
        sideToMove_ = o.sideToMove_;
        gamePly_ = o.gamePly_;
        psq_ = o.psq_;
        pawnKey_ = o.pawnKey_;
        acc_ = o.acc_;
        std::memcpy(finnyCache_, o.finnyCache_, sizeof(finnyCache_));
        states_ = o.states_;
        st_ = &states_.back();
        return *this;
    }

    // FEN ------------------------------------------------------------------
    // Returns false (and resets *this to the standard starting position)
    // on malformed input, instead of throwing -- see position.cpp. Safe to
    // ignore the return value for callers that only ever pass known-valid,
    // hardcoded FENs (tests, bench suites); callers parsing external input
    // (UCI, dataset files) should check it -- see uci.cpp, trainer.cpp.
    bool set(const std::string& fen);
    std::string fen() const;

    static void init_static();   // initializes bitboard + zobrist tables once

    // Make / unmake --------------------------------------------------------
    void do_move(Move m);
    void undo_move(Move m);
    void do_null_move();
    void undo_null_move();

    // Board queries --------------------------------------------------------
    Piece    piece_on(Square s) const { return board_[s]; }
    bool     empty(Square s)    const { return board_[s] == NO_PIECE; }
    Color    side_to_move()     const { return sideToMove_; }
    Square   ep_square()        const { return st_->epSquare; }
    int      castling_rights()  const { return st_->castlingRights; }
    Key      key()              const { return st_->key; }
    int      game_ply()         const { return gamePly_; }
    int      halfmove_clock()   const { return st_->halfmoveClock; }

    // Incremental evaluation accumulator (material + PSQT, white POV). This is
    // the integration seam: a future NNUE accumulator is maintained at the same
    // put/remove/move_piece sites that maintain this score.
    Score    psq()              const { return psq_; }
    Key      pawn_key()         const { return pawnKey_; }

    // NNUE accumulator (incrementally maintained when NNUE is enabled).
    const Accumulator& accumulator() const { return acc_; }
    void     nnue_refresh() { NNUE::refresh(*this, acc_); }

    Bitboard pieces() const { return byColor_[WHITE] | byColor_[BLACK]; }
    Bitboard pieces(Color c) const { return byColor_[c]; }
    Bitboard pieces(PieceType pt) const { return byType_[pt]; }
    Bitboard pieces(Color c, PieceType pt) const { return byColor_[c] & byType_[pt]; }
    Bitboard pieces(PieceType pt1, PieceType pt2) const { return byType_[pt1] | byType_[pt2]; }

    Square   king_square(Color c) const { return lsb(pieces(c, KING)); }
    int      count(Color c, PieceType pt) const { return popcount(pieces(c, pt)); }

    // Attack / check queries ----------------------------------------------
    Bitboard attackers_to(Square s, Bitboard occ) const;
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    bool     is_square_attacked(Square s, Color by) const;
    Bitboard checkers() const { return st_->checkers; }
    bool     in_check() const { return st_->checkers != 0; }

    // Legality (used by movegen filter) -----------------------------------
    bool     is_legal(Move m) const;
    bool     gives_check(Move m) const;

    // Draw detection -------------------------------------------------------
    bool     is_repetition() const;          // 3-fold (incl. current)
    // Cuckoo-hash upcoming-repetition detection: is a repetition reachable in a
    // few plies (a game cycle) from this node? `ply` is the search ply.
    bool     has_game_cycle(int ply) const;
    bool     is_fifty_move_draw() const { return st_->halfmoveClock >= 100; }
    bool     is_insufficient_material() const;
    bool     is_draw() const;

    std::string to_string() const;

private:
    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    void set_check_info();

    Bitboard byType_[PIECE_TYPE_NB];   // index 0 == ALL_PIECES (all occupancy)
    Bitboard byColor_[COLOR_NB];
    Piece    board_[SQUARE_NB];
    Color    sideToMove_;
    int      gamePly_;
    Score    psq_{};                   // incremental material + PSQT (white POV)
    Key      pawnKey_ = 0;             // incremental pawns-only Zobrist key
    Accumulator acc_{};                // incremental NNUE accumulator
    NNUE::FinnyTable finnyCache_{};     // king-bucket accumulator cache (per-Position)

    std::vector<StateInfo> states_;    // undo stack
    StateInfo* st_;
};

std::string square_to_string(Square s);
std::string move_to_uci(Move m);

} // namespace chess
