// position.cpp - Board state implementation.
#include "core/position.h"
#include "core/zobrist.h"
#include "eval/psqt.h"

#include <algorithm>
#include <sstream>
#include <cctype>

namespace chess {

namespace {
constexpr char PieceChar[PIECE_NB + 1] = " PNBRQK  pnbrqk";

// Castling-rights mask removed when a piece touches a given square.
int CastlingMask[SQUARE_NB];

void init_castling_masks() {
    for (int s = 0; s < SQUARE_NB; ++s) CastlingMask[s] = ANY_CASTLING;
    CastlingMask[SQ_A1] = ANY_CASTLING & ~WHITE_OOO;
    CastlingMask[SQ_H1] = ANY_CASTLING & ~WHITE_OO;
    CastlingMask[SQ_E1] = ANY_CASTLING & ~WHITE_CASTLING;
    CastlingMask[SQ_A8] = ANY_CASTLING & ~BLACK_OOO;
    CastlingMask[SQ_H8] = ANY_CASTLING & ~BLACK_OO;
    CastlingMask[SQ_E8] = ANY_CASTLING & ~BLACK_CASTLING;
}

// ---- Cuckoo hashing for upcoming-repetition (game cycle) detection ----------
// A table of the Zobrist signatures of every reversible (non-pawn) move, stored
// with cuckoo hashing so a key->move lookup is O(1). Built once in init_static.
Key  Cuckoo[8192];
Move CuckooMove[8192];

inline int H1(Key h) { return int(h & 0x1FFF); }
inline int H2(Key h) { return int((h >> 16) & 0x1FFF); }

void init_cuckoo() {
    for (int i = 0; i < 8192; ++i) { Cuckoo[i] = 0; CuckooMove[i] = Move::none(); }
    int count = 0;
    for (Color c : {WHITE, BLACK})
        for (PieceType pt = KNIGHT; pt <= KING; ++pt)   // non-pawn reversible movers
            for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
                for (Square s2 = Square(int(s1) + 1); s2 <= SQ_H8; ++s2)
                    if (attacks_bb(pt, s1, 0) & square_bb(s2)) {
                        const Piece pc = make_piece(c, pt);
                        Move move = Move(s1, s2);
                        Key  key  = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2]
                                  ^ Zobrist::side;
                        int  i = H1(key);
                        while (true) {                  // cuckoo displacement
                            std::swap(Cuckoo[i], key);
                            std::swap(CuckooMove[i], move);
                            if (move == Move::none()) break;
                            i = (i == H1(key)) ? H2(key) : H1(key);
                        }
                        ++count;
                    }
    (void)count;   // 3668 entries for standard chess
}
} // namespace

std::string square_to_string(Square s) {
    if (!is_ok(s)) return "-";
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string move_to_uci(Move m) {
    if (m == Move::none()) return "(none)";
    if (m == Move::null()) return "0000";
    std::string s = square_to_string(m.from_sq()) + square_to_string(m.to_sq());
    if (m.type_of() == PROMOTION)
        s += " pnbrqk"[m.promotion_type()];
    return s;
}

void Position::init_static() {
    Bitboards::init();
    Zobrist::init();
    psqt_init();
    NNUE::init();
    init_castling_masks();
    init_cuckoo();   // needs Zobrist keys + attack tables (initialized above)
}

Position::Position() {
    states_.reserve(512);
    set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

// ---------------------------------------------------------------------------
// Low-level piece manipulation (also updates Zobrist key)
// ---------------------------------------------------------------------------
void Position::put_piece(Piece pc, Square s) {
    const PieceType pt = type_of(pc);
    const Color c = color_of(pc);
    board_[s] = pc;
    byType_[ALL_PIECES] |= s;
    byType_[pt]         |= s;
    byColor_[c]         |= s;
    st_->key ^= Zobrist::psq[pc][s];
    psq_ += PSQTable[pc][s];                         // incremental eval accumulator
    if (pt == PAWN) pawnKey_ ^= Zobrist::psq[pc][s];
    // Incremental NNUE accumulator (both kings must be present for valid feature
    // indexing; during set() the final refresh fixes any partial state).
    if (NNUE::enabled() && byColor_[WHITE] & byType_[KING]
                        && byColor_[BLACK] & byType_[KING]) {
        if (pt == KING) NNUE::refresh_perspective(*this, acc_, c);
        else NNUE::add(acc_, pc, s, king_square(WHITE), king_square(BLACK));
    }
}

void Position::remove_piece(Square s) {
    const Piece pc = board_[s];
    const PieceType pt = type_of(pc);
    const Color c = color_of(pc);
    byType_[ALL_PIECES] ^= s;
    byType_[pt]         ^= s;
    byColor_[c]         ^= s;
    board_[s] = NO_PIECE;
    st_->key ^= Zobrist::psq[pc][s];
    psq_ -= PSQTable[pc][s];
    if (pt == PAWN) pawnKey_ ^= Zobrist::psq[pc][s];
    if (NNUE::enabled() && pt != KING && byColor_[WHITE] & byType_[KING]
                        && byColor_[BLACK] & byType_[KING])
        NNUE::sub(acc_, pc, s, king_square(WHITE), king_square(BLACK));
}

void Position::move_piece(Square from, Square to) {
    const Piece pc = board_[from];
    const PieceType pt = type_of(pc);
    const Color c = color_of(pc);
    const Bitboard fromTo = square_bb(from) | square_bb(to);
    byType_[ALL_PIECES] ^= fromTo;
    byType_[pt]         ^= fromTo;
    byColor_[c]         ^= fromTo;
    board_[from] = NO_PIECE;
    board_[to]   = pc;
    st_->key ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    psq_ += PSQTable[pc][to] - PSQTable[pc][from];
    if (pt == PAWN) pawnKey_ ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    if (NNUE::enabled()) {
        if (pt == KING) {
            // King moved: only that side's own accumulator can be affected (see
            // nnue.h). Use the king-bucket cache instead of an unconditional
            // full rebuild -- free if the bucket didn't change, a cheap patch
            // against a cached snapshot if it did, and a full rebuild only on a
            // cold cache entry.
            NNUE::refresh_perspective_cached(*this, acc_, c, from, to, finnyCache_);
        } else {
            const Square wk = king_square(WHITE), bk = king_square(BLACK);
            NNUE::sub(acc_, pc, from, wk, bk);
            NNUE::add(acc_, pc, to, wk, bk);
        }
    }
}

// ---------------------------------------------------------------------------
// FEN parsing
//
// Returns true and leaves *this set to `fen` on success. Returns false on
// any malformed input (bad piece character, wrong side-to-move field,
// missing/duplicated king) *without* throwing and *without* leaving the
// position half-mutated: on failure this resets *this to the standard
// starting position before returning, so a caller that ignores the return
// value still ends up with a legal, playable position rather than a
// corrupt one (e.g. a board with zero kings, which would later hit
// undefined behavior in king_square()'s lsb() on an empty bitboard).
// Callers that need to reject the input outright (e.g. the UCI `position`
// command) should check the return value themselves; see uci.cpp.
// ---------------------------------------------------------------------------
namespace {
constexpr const char* kStandardStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}

bool Position::set(const std::string& fen) {
    for (auto& b : byType_)  b = 0;
    for (auto& b : byColor_) b = 0;
    for (auto& p : board_)   p = NO_PIECE;
    psq_ = Score{};
    pawnKey_ = 0;
    // A new position invalidates every cached king-bucket accumulator: they were
    // snapshotted against a different, unrelated piece placement.
    for (auto& perColor : finnyCache_)
        for (auto& e : perColor)
            e.valid = false;

    states_.clear();
    states_.emplace_back();
    st_ = &states_.back();
    *st_ = StateInfo{};

    std::istringstream ss(fen);
    std::string boardStr, stm, castle, ep;
    int halfmove = 0, fullmove = 1;
    ss >> boardStr >> stm >> castle >> ep;
    ss >> halfmove >> fullmove;

    bool ok = !boardStr.empty() && (stm == "w" || stm == "b");

    // Piece placement (rank 8 first).
    File f = FILE_A;
    Rank r = RANK_8;
    for (char ch : boardStr) {
        if (ch == '/') { f = FILE_A; --r; }
        else if (std::isdigit(static_cast<unsigned char>(ch))) {
            f = File(f + (ch - '0'));
        } else {
            Piece pc = NO_PIECE;
            for (int i = 0; i < PIECE_NB; ++i)
                if (PieceChar[i] == ch) { pc = Piece(i); break; }
            if (pc == NO_PIECE) { ok = false; continue; }   // keep scanning, still rejected below
            if (f < FILE_A || f > FILE_H || r < RANK_1 || r > RANK_8) { ok = false; continue; }
            put_piece(pc, make_square(f, r));
            ++f;
        }
    }

    ok = ok && count(WHITE, KING) == 1 && count(BLACK, KING) == 1;

    if (!ok) {
        // Malformed input: reset to a known-legal position instead of
        // leaving *this half-built (e.g. missing a king), then report
        // failure. kStandardStartFen always validates, so this can't recurse.
        set(kStandardStartFen);
        return false;
    }

    sideToMove_ = (stm == "w") ? WHITE : BLACK;
    if (sideToMove_ == BLACK) st_->key ^= Zobrist::side;

    st_->castlingRights = NO_CASTLING;
    for (char ch : castle) {
        switch (ch) {
            case 'K': st_->castlingRights |= WHITE_OO;  break;
            case 'Q': st_->castlingRights |= WHITE_OOO; break;
            case 'k': st_->castlingRights |= BLACK_OO;  break;
            case 'q': st_->castlingRights |= BLACK_OOO; break;
            default: break;
        }
    }
    st_->key ^= Zobrist::castling[st_->castlingRights];

    st_->epSquare = SQ_NONE;
    if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h') {
        st_->epSquare = make_square(File(ep[0] - 'a'), Rank(ep[1] - '1'));
        st_->key ^= Zobrist::enpassant[file_of(st_->epSquare)];
    }

    st_->halfmoveClock = halfmove;
    gamePly_ = 2 * (fullmove - 1) + (sideToMove_ == BLACK ? 1 : 0);

    // Build the NNUE accumulator from scratch now that all pieces and both kings
    // are placed (incremental updates during piece placement may be partial).
    if (NNUE::enabled()) NNUE::refresh(*this, acc_);

    set_check_info();
    return true;
}

std::string Position::fen() const {
    std::ostringstream ss;
    for (Rank r = RANK_8; r >= RANK_1; --r) {
        int empty = 0;
        for (File f = FILE_A; f <= FILE_H; ++f) {
            Piece pc = piece_on(make_square(f, r));
            if (pc == NO_PIECE) { ++empty; continue; }
            if (empty) { ss << empty; empty = 0; }
            ss << PieceChar[pc];
        }
        if (empty) ss << empty;
        if (r > RANK_1) ss << '/';
    }
    ss << (sideToMove_ == WHITE ? " w " : " b ");

    const int cr = st_->castlingRights;
    if (!cr) ss << '-';
    else {
        if (cr & WHITE_OO)  ss << 'K';
        if (cr & WHITE_OOO) ss << 'Q';
        if (cr & BLACK_OO)  ss << 'k';
        if (cr & BLACK_OOO) ss << 'q';
    }
    ss << ' ' << (st_->epSquare == SQ_NONE ? "-" : square_to_string(st_->epSquare));
    ss << ' ' << st_->halfmoveClock << ' ' << (gamePly_ / 2 + 1);
    return ss.str();
}

// ---------------------------------------------------------------------------
// Attack queries
// ---------------------------------------------------------------------------
Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (pawn_attacks(BLACK, s) & pieces(WHITE, PAWN))
         | (pawn_attacks(WHITE, s) & pieces(BLACK, PAWN))
         | (knight_attacks(s)      & pieces(KNIGHT))
         | (king_attacks(s)        & pieces(KING))
         | (bishop_attacks(s, occ) & pieces(BISHOP, QUEEN))
         | (rook_attacks(s, occ)   & pieces(ROOK, QUEEN));
}

bool Position::is_square_attacked(Square s, Color by) const {
    const Bitboard occ = pieces();
    return (pawn_attacks(~by, s) & pieces(by, PAWN))
        || (knight_attacks(s)    & pieces(by, KNIGHT))
        || (king_attacks(s)      & pieces(by, KING))
        || (bishop_attacks(s, occ) & (pieces(by, BISHOP) | pieces(by, QUEEN)))
        || (rook_attacks(s, occ)   & (pieces(by, ROOK)   | pieces(by, QUEEN)));
}

void Position::set_check_info() {
    const Square ksq = king_square(sideToMove_);
    st_->checkers = attackers_to(ksq, pieces()) & pieces(~sideToMove_);
}

// ---------------------------------------------------------------------------
// Legality: a move is legal iff our king is not attacked in the resulting
// position. We compute the post-move occupancy and the set of surviving enemy
// pieces, then run a full attacker test against the king square. This is
// uniformly correct for quiet moves, captures, evasions (incl. knight/pawn
// checks), pinned pieces, and en passant. King moves are handled separately
// because the king itself relocates.
// ---------------------------------------------------------------------------
bool Position::is_legal(Move m) const {
    const Color us = sideToMove_;
    const Square from = m.from_sq();
    const Square to = m.to_sq();
    const MoveType mt = m.type_of();

    // King moves (incl. castling target): the destination must be unattacked,
    // computed with the king removed from its origin so it cannot shield itself.
    if (type_of(piece_on(from)) == KING) {
        // For castling, intermediate squares were validated in movegen.
        const Bitboard occ = pieces() ^ square_bb(from);   // king vacates origin
        return !( (pawn_attacks(us, to) & pieces(~us, PAWN))
               || (knight_attacks(to)   & pieces(~us, KNIGHT))
               || (king_attacks(to)     & pieces(~us, KING))
               || (bishop_attacks(to, occ) & (pieces(~us, BISHOP) | pieces(~us, QUEEN)))
               || (rook_attacks(to, occ)   & (pieces(~us, ROOK)   | pieces(~us, QUEEN))) );
    }

    const Square ksq = king_square(us);

    // Build post-move occupancy and surviving-enemy mask.
    Bitboard occ  = (pieces() ^ square_bb(from)) | square_bb(to);
    Bitboard them = pieces(~us) & ~square_bb(to);   // a capture removes the target
    if (mt == EN_PASSANT) {
        const Square capsq = make_square(file_of(to), rank_of(from));
        occ  ^= square_bb(capsq);
        them &= ~square_bb(capsq);
    }

    const Bitboard pawns   = pieces(PAWN)          & them;
    const Bitboard knights = pieces(KNIGHT)        & them;
    const Bitboard kings   = pieces(KING)          & them;
    const Bitboard bishops = pieces(BISHOP, QUEEN) & them;
    const Bitboard rooks   = pieces(ROOK, QUEEN)   & them;

    return !( (pawn_attacks(us, ksq) & pawns)
           || (knight_attacks(ksq)   & knights)
           || (king_attacks(ksq)     & kings)
           || (bishop_attacks(ksq, occ) & bishops)
           || (rook_attacks(ksq, occ)   & rooks) );
}

bool Position::gives_check(Move m) const {
    // Generic (not fully incremental) check test: apply, test, revert via copy
    // of the relevant occupancy. Used outside the hot search path in Phase 1.
    const Color us = sideToMove_;
    const Square from = m.from_sq();
    const Square to = m.to_sq();
    const Square theirKing = king_square(~us);

    Bitboard occ = (pieces() ^ square_bb(from)) | square_bb(to);
    PieceType pt = type_of(piece_on(from));
    if (m.type_of() == PROMOTION) pt = m.promotion_type();
    if (m.type_of() == EN_PASSANT)
        occ ^= square_bb(make_square(file_of(to), rank_of(from)));

    // Direct check by the moved piece.
    Bitboard att = (pt == PAWN)   ? pawn_attacks(us, to)
                 : (pt == KNIGHT) ? knight_attacks(to)
                 : (pt == KING)   ? king_attacks(to)
                 : attacks_bb(pt, to, occ);
    if (att & square_bb(theirKing)) return true;

    // Discovered check from sliders.
    if (bishop_attacks(theirKing, occ) & (pieces(us, BISHOP) | pieces(us, QUEEN)) & occ) return true;
    if (rook_attacks(theirKing, occ)   & (pieces(us, ROOK)   | pieces(us, QUEEN)) & occ) return true;
    return false;
}

// ---------------------------------------------------------------------------
// do_move / undo_move
// ---------------------------------------------------------------------------
void Position::do_move(Move m) {
    const Color us = sideToMove_;
    const Color them = ~us;
    const Square from = m.from_sq();
    const Square to = m.to_sq();
    const Piece movedPiece = piece_on(from);
    const MoveType mt = m.type_of();

    // Push new state, copying the parts that persist by default.
    StateInfo prev = *st_;
    states_.emplace_back(prev);
    st_ = &states_.back();
    st_->captured = NO_PIECE;
    st_->pliesFromNull = prev.pliesFromNull + 1;
    ++st_->halfmoveClock;

    // Clear old ep / castling from the key (re-added below if applicable).
    if (prev.epSquare != SQ_NONE)
        st_->key ^= Zobrist::enpassant[file_of(prev.epSquare)];
    st_->epSquare = SQ_NONE;

    if (mt == CASTLING) {
        // King already at 'from'; 'to' is the king target square.
        const bool kingSide = file_of(to) == FILE_G;
        const Square rookFrom = kingSide ? make_square(FILE_H, rank_of(from))
                                         : make_square(FILE_A, rank_of(from));
        const Square rookTo   = kingSide ? make_square(FILE_F, rank_of(from))
                                         : make_square(FILE_D, rank_of(from));
        move_piece(from, to);
        move_piece(rookFrom, rookTo);
    } else {
        // Handle capture (including en passant).
        Square capSq = to;
        if (mt == EN_PASSANT) capSq = make_square(file_of(to), rank_of(from));

        if (!empty(capSq) && capSq != from) {
            st_->captured = piece_on(capSq);
            remove_piece(capSq);
            st_->halfmoveClock = 0;
        }

        move_piece(from, to);

        if (type_of(movedPiece) == PAWN) {
            st_->halfmoveClock = 0;
            // Double push -> set ep square.
            if ((int(from) ^ int(to)) == 16) {
                Square epsq = Square((from + to) / 2);
                // Only set ep if an enemy pawn can actually capture (still set
                // unconditionally for FEN/zobrist correctness vs known perft).
                st_->epSquare = epsq;
                st_->key ^= Zobrist::enpassant[file_of(epsq)];
            } else if (mt == PROMOTION) {
                remove_piece(to);
                put_piece(make_piece(us, m.promotion_type()), to);
            }
        }
    }

    // Update castling rights.
    const int newCr = prev.castlingRights & CastlingMask[from] & CastlingMask[to];
    if (newCr != prev.castlingRights) {
        st_->key ^= Zobrist::castling[prev.castlingRights];
        st_->key ^= Zobrist::castling[newCr];
        st_->castlingRights = newCr;
    }

    sideToMove_ = them;
    st_->key ^= Zobrist::side;
    ++gamePly_;

    set_check_info();

    // Record repetition distance (used by has_game_cycle and draw detection):
    // the ply-distance back to a position with the same key, negative if that
    // earlier position was itself a repetition.
    st_->repetition = 0;
    const int rend = std::min(st_->halfmoveClock, st_->pliesFromNull);
    if (rend >= 4) {
        const std::size_t n = states_.size();
        for (int i = 4; i <= rend && std::size_t(i) < n; i += 2) {
            const StateInfo& back = states_[n - 1 - std::size_t(i)];
            if (back.key == st_->key) {
                st_->repetition = back.repetition ? -i : i;
                break;
            }
        }
    }
    (void)us;
}

void Position::undo_move(Move m) {
    sideToMove_ = ~sideToMove_;
    const Color us = sideToMove_;
    const Square from = m.from_sq();
    const Square to = m.to_sq();
    const MoveType mt = m.type_of();
    const Piece captured = st_->captured;

    if (mt == CASTLING) {
        const bool kingSide = file_of(to) == FILE_G;
        const Square rookFrom = kingSide ? make_square(FILE_H, rank_of(from))
                                         : make_square(FILE_A, rank_of(from));
        const Square rookTo   = kingSide ? make_square(FILE_F, rank_of(from))
                                         : make_square(FILE_D, rank_of(from));
        move_piece(to, from);
        move_piece(rookTo, rookFrom);
    } else {
        if (mt == PROMOTION) {
            remove_piece(to);
            put_piece(make_piece(us, PAWN), to);
        }
        move_piece(to, from);

        if (captured != NO_PIECE) {
            Square capSq = to;
            if (mt == EN_PASSANT) capSq = make_square(file_of(to), rank_of(from));
            put_piece(captured, capSq);
        }
    }

    states_.pop_back();
    st_ = &states_.back();
    --gamePly_;
}

void Position::do_null_move() {
    StateInfo prev = *st_;
    states_.emplace_back(prev);
    st_ = &states_.back();
    if (prev.epSquare != SQ_NONE) {
        st_->key ^= Zobrist::enpassant[file_of(prev.epSquare)];
        st_->epSquare = SQ_NONE;
    }
    st_->key ^= Zobrist::side;
    st_->pliesFromNull = 0;
    ++st_->halfmoveClock;
    sideToMove_ = ~sideToMove_;
    ++gamePly_;
    set_check_info();
}

void Position::undo_null_move() {
    states_.pop_back();
    st_ = &states_.back();
    sideToMove_ = ~sideToMove_;
    --gamePly_;
}

// ---------------------------------------------------------------------------
// Draw detection
// ---------------------------------------------------------------------------
bool Position::is_repetition() const {
    const int end = st_->pliesFromNull;
    if (end < 4) return false;
    int count = 0;
    // Walk back through the state stack in steps of 2 plies.
    const std::size_t n = states_.size();
    for (int i = 4; i <= end && std::size_t(i) < n; i += 2) {
        if (states_[n - 1 - i].key == st_->key) {
            if (++count >= 2) return true;   // current + 2 earlier = 3-fold
        }
    }
    return false;
}

bool Position::has_game_cycle(int ply) const {
    const int end = std::min(st_->halfmoveClock, st_->pliesFromNull);
    if (end < 3) return false;

    const Key originalKey = st_->key;
    const std::size_t n = states_.size();

    for (int i = 3; i <= end && std::size_t(i) < n; i += 2) {
        const Key moveKey = originalKey ^ states_[n - 1 - std::size_t(i)].key;
        int j;
        if ((j = H1(moveKey), Cuckoo[j] == moveKey)
         || (j = H2(moveKey), Cuckoo[j] == moveKey)) {
            const Move m = CuckooMove[j];
            const Square s1 = m.from_sq(), s2 = m.to_sq();
            // The squares between s1 and s2 (exclusive) must be empty for the
            // reversible move to be playable now.
            if (!(between_bb(s1, s2) & pieces())) {
                // A cycle whose far end is inside the search tree is an
                // immediate (2-fold) draw we can claim.
                if (ply > i) return true;
                // Otherwise the cycle reaches into game history: it only counts
                // if the moving piece belongs to the side to move and that
                // earlier position had itself already repeated.
                const Square occ = empty(s1) ? s2 : s1;
                if (color_of(piece_on(occ)) != sideToMove_) continue;
                if (states_[n - 1 - std::size_t(i)].repetition) return true;
            }
        }
    }
    return false;
}

bool Position::is_insufficient_material() const {
    if (pieces(PAWN) | pieces(ROOK) | pieces(QUEEN)) return false;
    // Only kings and minor pieces remain.
    const int minors = popcount(pieces(KNIGHT) | pieces(BISHOP));
    return minors <= 1;   // KK, KNK, KBK
}

bool Position::is_draw() const {
    return is_fifty_move_draw() || is_repetition() || is_insufficient_material();
}

std::string Position::to_string() const {
    std::ostringstream ss;
    ss << "\n +---+---+---+---+---+---+---+---+\n";
    for (Rank r = RANK_8; r >= RANK_1; --r) {
        for (File f = FILE_A; f <= FILE_H; ++f) {
            Piece pc = piece_on(make_square(f, r));
            ss << " | " << (pc == NO_PIECE ? ' ' : PieceChar[pc]);
        }
        ss << " | " << (r + 1) << "\n +---+---+---+---+---+---+---+---+\n";
    }
    ss << "   a   b   c   d   e   f   g   h\n";
    ss << "\nFen: " << fen() << "\nKey: " << std::hex << key() << std::dec << "\n";
    return ss.str();
}

} // namespace chess
