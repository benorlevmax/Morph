// tablebases.cpp - Syzygy probing via Fathom, with graceful fallback.
//
// When built with -DCHESS_USE_FATHOM (and Fathom's tbprobe.c/.h, tbchess.c and
// stdendian.h vendored), this performs real WDL probing during search and DTZ
// probing at the root. Without that flag, available() reports false and every
// probe declines, so search transparently uses its normal evaluation.
#include "syzygy/tablebases.h"
#include "core/bitboard.h"
#include "core/movegen.h"

#include <sstream>

#ifdef CHESS_USE_FATHOM
extern "C" {
#include "tbprobe.h"
}
#endif

namespace chess::Tablebases {

namespace {
std::string g_paths   = "<empty>";
bool        g_enabled = false;     // true only when a real decoder is active
int         g_maxPieces = 0;

#ifdef CHESS_USE_FATHOM
// Build Fathom's bitboard arguments from a Position (Fathom and this engine
// share the A1=0..H8=63 square / bit convention).
struct FathomArgs {
    std::uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned rule50, castling, ep;
    bool turn;
};
FathomArgs to_fathom(const Position& pos) {
    FathomArgs a;
    a.white   = pos.pieces(WHITE);
    a.black   = pos.pieces(BLACK);
    a.kings   = pos.pieces(KING);
    a.queens  = pos.pieces(QUEEN);
    a.rooks   = pos.pieces(ROOK);
    a.bishops = pos.pieces(BISHOP);
    a.knights = pos.pieces(KNIGHT);
    a.pawns   = pos.pieces(PAWN);
    a.rule50  = unsigned(pos.halfmove_clock());
    a.castling = unsigned(pos.castling_rights());   // Fathom needs 0; declines otherwise
    a.ep      = pos.ep_square() == SQ_NONE ? 0u : unsigned(pos.ep_square());
    a.turn    = pos.side_to_move() == WHITE;
    return a;
}

WDLResult map_wdl(unsigned wdl) {
    switch (wdl) {
        case TB_WIN:          return WDLResult::Win;
        case TB_CURSED_WIN:   return WDLResult::CursedWin;
        case TB_DRAW:         return WDLResult::Draw;
        case TB_BLESSED_LOSS: return WDLResult::BlessedLoss;
        case TB_LOSS:         return WDLResult::Loss;
        default:              return WDLResult::Fail;
    }
}
#endif // CHESS_USE_FATHOM

// Hook for a real decoder. Returns true and sets `out` if it can probe `pos`.
bool probe_wdl_impl(const Position& pos, WDLResult& out) {
#ifdef CHESS_USE_FATHOM
    const FathomArgs a = to_fathom(pos);
    // tb_probe_wdl declines unless rule50 == 0 and castling == 0 (WDL is only
    // meaningful at a reset 50-move counter), giving correct 50-move handling.
    unsigned res = tb_probe_wdl(a.white, a.black, a.kings, a.queens, a.rooks,
                                a.bishops, a.knights, a.pawns,
                                a.rule50, a.castling, a.ep, a.turn);
    if (res == TB_RESULT_FAILED) return false;
    out = map_wdl(res);
    return out != WDLResult::Fail;
#else
    (void)pos; (void)out;
    return false;
#endif
}
} // namespace

void init(const std::string& paths) {
    g_paths = paths;
    g_enabled   = false;
    g_maxPieces = 0;
#ifdef CHESS_USE_FATHOM
    if (paths.empty() || paths == "<empty>") {
        tb_free();
        return;
    }
    if (tb_init(paths.c_str()) && TB_LARGEST > 0) {
        g_enabled   = true;
        g_maxPieces = int(TB_LARGEST);
    }
#endif
}

bool available()  { return g_enabled && g_maxPieces > 0; }
int  max_pieces() { return g_maxPieces; }

std::string status() {
    std::ostringstream ss;
    if (!available())
#ifdef CHESS_USE_FATHOM
        ss << "unavailable (path=" << g_paths << ", no files found) -> normal search";
#else
        ss << "unavailable (path=" << g_paths
           << ", built without Fathom) -> using normal search";
#endif
    else
        ss << "ready up to " << g_maxPieces << " pieces (path=" << g_paths << ")";
    return ss.str();
}

WDLResult probe_wdl(const Position& pos) {
    if (!available())
        return WDLResult::Fail;
    if (popcount(pos.pieces()) > g_maxPieces)
        return WDLResult::Fail;

    WDLResult r;
    return probe_wdl_impl(pos, r) ? r : WDLResult::Fail;
}

Value wdl_to_value(WDLResult r, int ply) {
    switch (r) {
        case WDLResult::Win:          return Value(VALUE_MATE_IN_MAX_PLY - ply - 1);
        case WDLResult::CursedWin:    return Value(1);
        case WDLResult::Draw:         return VALUE_DRAW;
        case WDLResult::BlessedLoss:  return Value(-1);
        case WDLResult::Loss:         return Value(-VALUE_MATE_IN_MAX_PLY + ply + 1);
        default:                      return VALUE_NONE;
    }
}

RootProbe probe_root(const Position& pos) {
    RootProbe rp;
    if (!available() || popcount(pos.pieces()) > g_maxPieces)
        return rp;
#ifdef CHESS_USE_FATHOM
    const FathomArgs a = to_fathom(pos);
    unsigned res = tb_probe_root(a.white, a.black, a.kings, a.queens, a.rooks,
                                 a.bishops, a.knights, a.pawns,
                                 a.rule50, a.castling, a.ep, a.turn, nullptr);
    if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE
        || res == TB_RESULT_STALEMATE)
        return rp;   // let normal search handle terminal/failed cases

    const unsigned from  = TB_GET_FROM(res);
    const unsigned to    = TB_GET_TO(res);
    const unsigned promo = TB_GET_PROMOTES(res);   // 0=none, 1=Q .. 4=N
    const WDLResult wdl  = map_wdl(TB_GET_WDL(res));

    // Match the suggested (from,to,promotion) to one of our legal moves so the
    // returned Move is fully consistent with the engine's move encoding.
    MoveList list;
    generate(pos, list, LEGAL);
    for (const auto& sm : list) {
        const Move m = sm.move;
        if (unsigned(m.from_sq()) != from || unsigned(m.to_sq()) != to) continue;
        const bool isPromo = (m.type_of() == PROMOTION);
        if ((promo != TB_PROMOTES_NONE) != isPromo) continue;
        if (isPromo) {
            const PieceType pt = promo == TB_PROMOTES_QUEEN  ? QUEEN
                               : promo == TB_PROMOTES_ROOK   ? ROOK
                               : promo == TB_PROMOTES_BISHOP ? BISHOP : KNIGHT;
            if (m.promotion_type() != pt) continue;
        }
        rp.ok   = true;
        rp.best = m;
        rp.wdl  = wdl;
        return rp;
    }
#endif
    return rp;
}

} // namespace chess::Tablebases
