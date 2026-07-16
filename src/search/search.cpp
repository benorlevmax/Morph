// search.cpp - Iterative deepening, alpha-beta with PVS, quiescence search,
// Phase-4 refinements, and (Phase 6B) Lazy SMP parallel search.
//
// Threading model: each helper thread owns a ThreadState (its own root copy,
// history/killers/countermoves/pv). The transposition table is shared and
// accessed lockless (the 16-bit key check makes torn reads benign, and moves
// are always re-validated against the legal move list, so play stays correct).
// The main thread (id 0) is authoritative for the returned move and controls
// time; helpers only deepen and fill the shared TT. threads==1 runs
// synchronously and is bit-for-bit identical to the single-threaded engine.
#include "search/search.h"
#include "eval/evaluate.h"
#include "core/movegen.h"
#include "syzygy/tablebases.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace chess {

namespace {

bool is_capture(const Position& p, Move m) {
    const MoveType mt = m.type_of();
    if (mt == CASTLING)   return false;
    if (mt == EN_PASSANT) return true;
    return !p.empty(m.to_sq());
}
bool is_noisy(const Position& p, Move m) {
    return is_capture(p, m) || m.type_of() == PROMOTION;
}

// Static Exchange Evaluation: is the capture/threat at least `threshold` cp?
bool see_ge(const Position& pos, Move m, int threshold) {
    const Square from = m.from_sq();
    const Square to   = m.to_sq();

    if (m.type_of() != NORMAL && m.type_of() != EN_PASSANT)
        return 0 >= threshold;

    const PieceType captured = (m.type_of() == EN_PASSANT)
                             ? PAWN : type_of(pos.piece_on(to));
    int swap = (captured == NO_PIECE_TYPE ? 0 : PieceValue[captured]) - threshold;
    if (swap < 0) return false;

    swap = PieceValue[type_of(pos.piece_on(from))] - swap;
    if (swap <= 0) return true;

    Bitboard occ = pos.pieces() ^ square_bb(from) ^ square_bb(to);
    if (m.type_of() == EN_PASSANT)
        occ ^= square_bb(make_square(file_of(to), rank_of(from)));

    Bitboard attackers = pos.attackers_to(to, occ) & occ;
    Color stm = color_of(pos.piece_on(from));
    int res = 1;

    const Bitboard bishops = pos.pieces(BISHOP, QUEEN);
    const Bitboard rooks   = pos.pieces(ROOK, QUEEN);

    while (true) {
        stm = ~stm;
        attackers &= occ;
        Bitboard stmAtt = attackers & pos.pieces(stm);
        if (!stmAtt) break;

        res ^= 1;

        Bitboard bb;
        if ((bb = stmAtt & pos.pieces(PAWN))) {
            if ((swap = PieceValue[PAWN] - swap) < res) break;
            occ ^= (bb & (~bb + 1));
            attackers |= bishop_attacks(to, occ) & bishops;
        } else if ((bb = stmAtt & pos.pieces(KNIGHT))) {
            if ((swap = PieceValue[KNIGHT] - swap) < res) break;
            occ ^= (bb & (~bb + 1));
        } else if ((bb = stmAtt & pos.pieces(BISHOP))) {
            if ((swap = PieceValue[BISHOP] - swap) < res) break;
            occ ^= (bb & (~bb + 1));
            attackers |= bishop_attacks(to, occ) & bishops;
        } else if ((bb = stmAtt & pos.pieces(ROOK))) {
            if ((swap = PieceValue[ROOK] - swap) < res) break;
            occ ^= (bb & (~bb + 1));
            attackers |= rook_attacks(to, occ) & rooks;
        } else if ((bb = stmAtt & pos.pieces(QUEEN))) {
            if ((swap = PieceValue[QUEEN] - swap) < res) break;
            occ ^= (bb & (~bb + 1));
            attackers |= (bishop_attacks(to, occ) & bishops)
                       | (rook_attacks(to, occ) & rooks);
        } else {  // king
            return (attackers & ~pos.pieces(stm)) ? bool(res ^ 1) : bool(res);
        }
    }
    return bool(res);
}

void pick_next(MoveList& list, std::size_t i) {
    ScoredMove* a = list.begin();
    const std::size_t n = list.size();
    std::size_t best = i;
    for (std::size_t j = i + 1; j < n; ++j)
        if (a[j].score > a[best].score) best = j;
    if (best != i) std::swap(a[i], a[best]);
}

void update_pv(Move* pv, Move move, const Move* childPv) {
    *pv++ = move;
    while (childPv && *childPv != Move::none())
        *pv++ = *childPv++;
    *pv = Move::none();
}

constexpr int TT_SCORE       = 1 << 30;
constexpr int GOOD_CAPTURE   = 1 << 28;
constexpr int KILLER0_SCORE  = (1 << 27) + 1;
constexpr int KILLER1_SCORE  = 1 << 27;
constexpr int COUNTER_SCORE  = 1 << 26;
constexpr int BAD_CAPTURE    = -(1 << 28);
constexpr int HISTORY_MAX    = 1 << 24;

// History update with "gravity": keeps values bounded in [-HISTORY_MAX, MAX]
// while letting frequently-good moves accumulate signal.
inline void hist_update(int& h, int bonus) {
    h += bonus - h * std::abs(bonus) / HISTORY_MAX;
}

// Multi-ply continuation history: sum/update the table at the moves played
// 1, 2, 4 and 6 plies ago (skipping null-move and out-of-range contexts).
inline int cont_hist_score(const ThreadState& ts, int ply, Piece pc, Square to) {
    int s = 0;
    for (int off : {1, 2, 4, 6}) {
        const int p = ply - off;
        if (p >= 0 && ts.spOk[p])
            s += ts.contHist[ts.spPiece[p]][ts.spTo[p]][pc][to];
    }
    return s;
}
inline void cont_hist_update(ThreadState& ts, int ply, Piece pc, Square to, int bonus) {
    for (int off : {1, 2, 4, 6}) {
        const int p = ply - off;
        if (p >= 0 && ts.spOk[p])
            hist_update(ts.contHist[ts.spPiece[p]][ts.spTo[p]][pc][to], bonus);
    }
}

// Material signature key for correction history (counts of each piece type).
inline Key material_key(const Position& pos) {
    Key k = 0;
    for (Color c : {WHITE, BLACK})
        for (PieceType pt = PAWN; pt <= QUEEN; ++pt)
            k = k * 31 + Key(pos.count(c, pt));
    return k;
}

} // namespace

Search::Search() {
    init_reductions();
    tt_.resize(16);
    tt_.clear();
}

void Search::init_reductions() {
    // Log-based reduction:  r = base + ln(depth) * ln(moveCount) / divisor
    //   base    = 0.5   (flat base reduction)
    //   divisor = 2.0   (Stockfish-style scaling)
    constexpr double kBase    = 0.5;
    constexpr double kDivisor = 2.0;
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            reductions_[d][m] =
                int(kBase + std::log(double(d)) * std::log(double(m)) / kDivisor);
    for (int m = 0; m < 64; ++m) reductions_[0][m] = 0;
    for (int d = 0; d < 64; ++d) reductions_[d][0] = 0;
}

void Search::clear() {
    tt_.clear();
}

std::uint64_t Search::nodes() const {
    std::uint64_t n = 0;
    for (const auto& t : ts_) n += t.nodes;
    return n;
}

// ---------------------------------------------------------------------------
// Time management
// ---------------------------------------------------------------------------
void Search::compute_time_budget(const Position& pos) {
    optimumMs_ = maximumMs_ = 0;

    if (limits_.movetime > 0) {
        optimumMs_ = maximumMs_ = limits_.movetime;
        return;
    }
    if (limits_.infinite || limits_.depth > 0 || limits_.nodes > 0)
        return;

    const Color us = pos.side_to_move();
    const int t   = limits_.time[us];
    const int inc = limits_.inc[us];
    if (t <= 0 && inc <= 0)
        return;

    const int mtg = limits_.movestogo > 0 ? limits_.movestogo : 30;
    // Move overhead: reserve time for GUI/network latency so we never flag.
    const std::int64_t overhead = moveOverhead_;
    const std::int64_t total = std::max<std::int64_t>(1, t - overhead);

    std::int64_t opt = total / mtg + (inc * 3) / 4;
    opt = std::max<std::int64_t>(1, std::min(opt, total));
    std::int64_t mx = std::min<std::int64_t>(total, opt * 4);

    optimumMs_ = opt;
    maximumMs_ = std::max(opt, mx);
}

bool Search::time_up() {
    if (maximumMs_ <= 0) return false;
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start_).count();
    // Hard limit: never exceed 95% of the computed maximum (extra safety margin).
    return ms >= (maximumMs_ * 95) / 100;
}

void Search::check_time(ThreadState& ts) {
    if (ts.id != 0) return;                    // only the main thread enforces limits
    if (limits_.nodes && nodes() >= limits_.nodes) stop();
    else if (time_up()) stop();
}

void Search::report(ThreadState& ts, int depth, Value score) {
    if (quiet_) return;
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start_).count();
    const std::uint64_t n = nodes();
    const std::uint64_t nps = ms > 0 ? (n * 1000ULL) / std::uint64_t(ms) : n;

    std::cout << "info depth " << depth << " score ";
    if (std::abs(int(score)) >= VALUE_MATE_IN_MAX_PLY) {
        const int mate = (score > 0 ? VALUE_MATE - score + 1 : -VALUE_MATE - score) / 2;
        std::cout << "mate " << mate;
    } else {
        std::cout << "cp " << int(score);
    }
    std::cout << " nodes " << n << " nps " << nps << " time " << ms
              << " hashfull " << tt_.hashfull() << " pv";
    // Emit only the legal prefix of the PV: replay it from the root and stop at
    // the first move that is not legal (the triangular PV table can retain a
    // stale tail from a sibling line). Output-only; search state is untouched.
    {
        Position pv = ts.pos;   // copy of the root (safe deep copy)
        for (int i = 0; i < MAX_PLY; ++i) {
            const Move m = ts.pv[0][i];
            if (m == Move::none()) break;
            MoveList legalMoves;
            generate(pv, legalMoves, LEGAL);
            bool legalHere = false;
            for (const auto& sm : legalMoves)
                if (sm.move == m) { legalHere = true; break; }
            if (!legalHere) break;        // truncate at first illegal PV move
            std::cout << ' ' << move_to_uci(m);
            pv.do_move(m);
        }
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Quiescence search
// ---------------------------------------------------------------------------
Value Search::qsearch(ThreadState& ts, Value alpha, Value beta, int ply, bool genChecks) {
    Position& pos = ts.pos;
    ++ts.nodes;
    if ((ts.nodes & 1023) == 0) check_time(ts);   // check the clock every 1024 nodes
    if (stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;

    if (pos.is_draw()) return VALUE_DRAW;
    if (ply >= MAX_PLY - 1) return evaluate(pos);

    const Value alphaOrig = alpha;              // for the final store's bound
    const bool  pvNode    = (beta - alpha) > 1; // qsearch has no PvNode template arg

    // Transposition table (Stockfish-style qsearch usage). Entries are always
    // stored at depth 0 -- the lowest depth this TT ever stores at, since
    // Search::search() redirects to qsearch whenever depth <= 0, before it
    // ever reaches its own TT probe/store (so a real main-search probe/store
    // site never sees depth <= 0). That guarantees a qsearch-origin entry can
    // never satisfy a main-search depth cutoff (`tte->depth() >= depth` with
    // depth >= 1 there), while qsearch itself can still reuse its own prior
    // result, or a deeper main-search result, at this position.
    constexpr int QS_TT_DEPTH = 0;

    bool ttHit = false;
    TTEntry* tte = tt_.probe(pos.key(), ttHit);
    const Value ttValue = ttHit ? value_from_tt(tte->value(), ply) : VALUE_NONE;
    const Move  ttMove  = ttHit ? tte->move() : Move::none();

    // Non-PV cutoff only (mirrors main search / Stockfish: a true PV window is
    // never short-circuited by a TT hit here).
    if (!pvNode && ttHit && ttValue != VALUE_NONE
        && (tte->bound() == BOUND_EXACT
         || (tte->bound() == BOUND_LOWER && ttValue >= beta)
         || (tte->bound() == BOUND_UPPER && ttValue <= alpha)))
        return ttValue;

    const bool inCheck = pos.in_check();

    Value bestValue;
    Value rawEval = VALUE_NONE;   // stored alongside the result, as main search does
    if (inCheck) {
        bestValue = -VALUE_INFINITE;
    } else {
        rawEval = evaluate(pos);   // evaluation itself is unchanged
        bestValue = rawEval;
        if (bestValue >= beta) {
            tt_.store(tte, pos.key(), value_to_tt(bestValue, ply), BOUND_LOWER,
                      QS_TT_DEPTH, Move::none(), rawEval);
            return bestValue;
        }
        if (bestValue > alpha) alpha = bestValue;
    }
    const Value standPat = bestValue;   // for delta pruning

    MoveList list;
    // Optimization: outside check, and when quiet checking moves are not being
    // generated (genChecks is only ever true on the single entry call from main
    // search / razoring -- every recursive qsearch-to-qsearch call passes
    // false), the per-move filter below can only ever keep captures and
    // promotions. Generating the much smaller pseudo-legal CAPTURES set instead
    // of the fully legal-filtered NON_EVASIONS set (and legality-checking just
    // that smaller set below) searches the exact same moves, just cheaper to
    // produce -- this covers the large majority of qsearch nodes (every
    // recursive descent). When in check, or when quiet checks are still in
    // play, keep the original full legal generation so behavior is unchanged.
    const bool capturesOnly = !inCheck && !genChecks;
    if (capturesOnly)
        generate(pos, list, CAPTURES);
    else
        generate(pos, list, LEGAL);

    for (auto& sm : list) {
        Move m = sm.move;
        if (m == ttMove) {
            sm.score = TT_SCORE;
        } else if (is_capture(pos, m) || m.type_of() == PROMOTION) {
            PieceType victim = (m.type_of() == EN_PASSANT) ? PAWN
                             : is_capture(pos, m) ? type_of(pos.piece_on(m.to_sq()))
                             : NO_PIECE_TYPE;
            PieceType attacker = type_of(pos.piece_on(m.from_sq()));
            int mvv = 64 * PieceValue[victim] - PieceValue[attacker];
            if (m.type_of() == PROMOTION) mvv += PieceValue[m.promotion_type()];
            // MVV-LVA dominates; capture history is a learned tie-break (mirrors
            // the main search's capture ordering).
            const PieceType capIdx = (victim == NO_PIECE_TYPE) ? PAWN : victim;
            const int ch = ts.captureHistory[pos.piece_on(m.from_sq())][m.to_sq()][capIdx];
            sm.score = mvv * 16 + ch / 128;
        } else {
            sm.score = 0;
        }
    }

    Move bestMove = Move::none();
    int legal = 0;
    for (std::size_t i = 0; i < list.size(); ++i) {
        pick_next(list, i);
        Move m = list.begin()[i].move;

        if (!inCheck) {
            // The CAPTURES generator above is pseudo-legal (unlike LEGAL), so
            // legality must be checked explicitly here for that path.
            if (capturesOnly && !pos.is_legal(m)) continue;
            const bool noisy = is_noisy(pos, m);
            // At the first quiescence ply also search quiet checking moves.
            const bool quietCheck = genChecks && !noisy && pos.gives_check(m);
            if (!noisy && !quietCheck) continue;
            if ((is_capture(pos, m) || quietCheck) && !see_ge(pos, m, 0)) continue;
            // Delta pruning: a capture/promotion that cannot lift the stand-pat
            // to alpha even after the best-case material gain (plus a margin) is
            // hopeless. Promotions use a higher margin and add the promotion gain.
            if (is_capture(pos, m) || m.type_of() == PROMOTION) {
                const PieceType victim = !is_capture(pos, m) ? NO_PIECE_TYPE
                                       : (m.type_of() == EN_PASSANT) ? PAWN
                                       : type_of(pos.piece_on(m.to_sq()));
                int gain   = (victim == NO_PIECE_TYPE) ? 0 : PieceValue[victim];
                int margin = 200;
                if (m.type_of() == PROMOTION) {
                    gain  += PieceValue[m.promotion_type()] - PieceValue[PAWN];
                    margin = 350;   // promotions are more volatile -> wider margin
                }
                if (standPat + gain + margin < alpha) continue;
            }
        }
        ++legal;

        pos.do_move(m);
        Value score = -qsearch(ts, -beta, -alpha, ply + 1, false);
        pos.undo_move(m);

        if (stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;

        if (score > bestValue) {
            bestValue = score;
            if (score > alpha) {
                bestMove = m;
                if (score >= beta) {
                    tt_.store(tte, pos.key(), value_to_tt(bestValue, ply), BOUND_LOWER,
                              QS_TT_DEPTH, bestMove, rawEval);
                    return score;
                }
                alpha = score;
            }
        }
    }

    if (inCheck && legal == 0)
        return mated_in(ply);

    // pvNode-gated EXACT, matching main search / Stockfish: a non-PV node only
    // ever stores UPPER here (the LOWER/cutoff cases already returned above).
    const Bound bound = (pvNode && bestValue > alphaOrig) ? BOUND_EXACT : BOUND_UPPER;
    tt_.store(tte, pos.key(), value_to_tt(bestValue, ply), bound, QS_TT_DEPTH, bestMove, rawEval);
    return bestValue;
}

// ---------------------------------------------------------------------------
// Main alpha-beta / PVS search
// ---------------------------------------------------------------------------
template <bool PvNode>
Value Search::search(ThreadState& ts, Value alpha, Value beta, int depth, int ply,
                     Move prevMove) {
    Position& pos = ts.pos;
    const bool rootNode = PvNode && ply == 0;

    if (depth <= 0)
        return qsearch(ts, alpha, beta, ply, true);

    if (PvNode) ts.pv[ply][0] = Move::none();

    ++ts.nodes;
    if ((ts.nodes & 1023) == 0) check_time(ts);   // check the clock every 1024 nodes
    if (stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;

    if (!rootNode) {
        if (pos.is_draw()) return VALUE_DRAW;

        // Upcoming-repetition (cuckoo) detection: if a draw by repetition is
        // reachable, treat it as a draw lower bound. A winning side (alpha above
        // draw) ignores it and keeps searching for the win; a losing side (alpha
        // below draw) can claim the draw immediately.
        if (alpha < VALUE_DRAW && pos.has_game_cycle(ply)) {
            alpha = VALUE_DRAW;
            if (alpha >= beta) return alpha;
        }

        if (Tablebases::available()) {
            WDLResult wdl = Tablebases::probe_wdl(pos);
            if (wdl != WDLResult::Fail)
                return Tablebases::wdl_to_value(wdl, ply);
        }

        alpha = std::max(alpha, mated_in(ply));
        beta  = std::min(beta, mate_in(ply + 1));
        if (alpha >= beta) return alpha;
        if (ply >= MAX_PLY - 1) return evaluate(pos);
    }

    bool ttHit = false;
    TTEntry* tte = tt_.probe(pos.key(), ttHit);
    const Value ttValue = ttHit ? value_from_tt(tte->value(), ply) : VALUE_NONE;
    const Move  ttMove  = ttHit ? tte->move() : Move::none();
    const Move  excludedMove = ts.excluded[ply];

    if (!PvNode && excludedMove == Move::none() && ttHit && tte->depth() >= depth && ttValue != VALUE_NONE) {
        if (tte->bound() == BOUND_EXACT
         || (tte->bound() == BOUND_LOWER && ttValue >= beta)
         || (tte->bound() == BOUND_UPPER && ttValue <= alpha))
            return ttValue;
    }

    const bool inCheck = pos.in_check();

    // Internal iterative reductions: with no TT move the move ordering is
    // unreliable, so reduce depth and let a shallower search populate the TT.
    if (depth >= 4 && ttMove == Move::none() && !inCheck)
        --depth;

    const Value rawEval = inCheck ? VALUE_NONE
                        : (ttHit && tte->eval() != VALUE_NONE ? tte->eval()
                                                             : evaluate(pos));
    // Correction history: shift the raw eval by a learned, pawn-structure-keyed
    // residual between past search results and static eval.
    Value staticEval = rawEval;
    if (!inCheck) {
        const Color stm = pos.side_to_move();
        const int corr = (ts.pawnCorr[stm][pos.pawn_key() & 16383]
                        + ts.matCorr[stm][material_key(pos) & 16383]) / 256;
        staticEval = Value(std::clamp(int(rawEval) + corr,
                                      int(-VALUE_MATE_IN_MAX_PLY) + 1,
                                      int(VALUE_MATE_IN_MAX_PLY) - 1));
    }
    ts.evalStack[ply] = staticEval;

    if (cfg_.reverseFutility && !PvNode && !inCheck && depth <= 8
        && std::abs(int(beta)) < VALUE_MATE_IN_MAX_PLY
        && staticEval - 80 * depth >= beta)
        return staticEval;

    {
        const Color stmNmp = pos.side_to_move();
        const Bitboard nonPawn = pos.pieces(stmNmp)
                               & ~pos.pieces(stmNmp, PAWN) & ~pos.pieces(stmNmp, KING);
        const int nonPawnCount = popcount(nonPawn);
        // Zugzwang guard: never null-move with only pawns + king, and skip
        // near-bare-king endgames (few pieces) where zugzwang is common.
        const bool nmpMaterial = nonPawnCount >= 1
                              && !(nonPawnCount == 1 && pos.count(stmNmp, PAWN) <= 1);
        if (cfg_.nullMove && !PvNode && excludedMove == Move::none() && !inCheck && depth >= 3
            && prevMove != Move::null() && staticEval >= beta && nmpMaterial
            && std::abs(int(beta)) < VALUE_MATE_IN_MAX_PLY) {
            // Adaptive reduction: deeper and further-above-beta -> reduce more.
            const int R = 3 + depth / 4 + std::min(int(staticEval - beta) / 200, 3);
            ts.spOk[ply] = false;   // null move: no 1-ply continuation context
            pos.do_null_move();
            Value nv = -search<false>(ts, -beta, -(beta - 1),
                                      depth - 1 - R, ply + 1, Move::null());
            pos.undo_null_move();
            if (stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
            if (nv >= beta) {
                if (std::abs(int(nv)) >= VALUE_MATE_IN_MAX_PLY) nv = beta;
                if (depth < 10) return nv;
                // Verification at high depth (null disabled for this node) to
                // guard against zugzwang fails.
                Value v = search<false>(ts, beta - 1, beta, depth - 1 - R, ply, Move::null());
                if (v >= beta) return nv;
            }
        }
    }

    if (cfg_.razoring && !PvNode && !inCheck && depth <= 3
        && staticEval + 200 * depth < alpha) {
        Value rv = qsearch(ts, alpha, beta, ply, true);
        if (rv < alpha)
            return rv;
    }

    MoveList list;
    generate(pos, list, LEGAL);

    if (list.empty())
        return inCheck ? mated_in(ply) : VALUE_DRAW;

    // ProbCut: if a capture, verified by a shallow search, beats a beta raised
    // by a margin, the node almost certainly fails high. Stockfish-style, guarded
    // by the TT (skip when it already says the node is worse than probCutBeta),
    // by SEE (only captures that plausibly reach probCutBeta), and disabled in
    // PV/check/singular contexts so it never perturbs the principal variation.
    // It reuses the already-generated legal `list` (no extra move generation).
    const Value probCutBeta = Value(int(beta) + 200);
    if (!PvNode && !inCheck && excludedMove == Move::none() && depth >= 5
        && std::abs(int(beta)) < VALUE_MATE_IN_MAX_PLY
        && !(ttHit && tte->depth() >= depth - 3 && ttValue != VALUE_NONE
             && ttValue < probCutBeta)) {
        const int seeThresh = int(probCutBeta) - int(staticEval);
        for (const auto& sm : list) {
            Move m = sm.move;
            if (m == ttMove || !is_capture(pos, m)) continue;
            // Cheap pre-gate: the captured material alone must be able to bridge
            // the margin, else skip the (costlier) SEE and verification entirely.
            const PieceType victim = (m.type_of() == EN_PASSANT)
                                   ? PAWN : type_of(pos.piece_on(m.to_sq()));
            if (PieceValue[victim] < seeThresh) continue;
            if (!see_ge(pos, m, seeThresh)) continue;

            ts.spPiece[ply] = pos.piece_on(m.from_sq());
            ts.spTo[ply]    = m.to_sq();
            ts.spOk[ply]    = true;
            pos.do_move(m);
            Value v = -qsearch(ts, -probCutBeta, -(probCutBeta - 1), ply + 1, false);
            if (v >= probCutBeta)
                v = -search<false>(ts, -probCutBeta, -(probCutBeta - 1),
                                   depth - 4, ply + 1, m);
            pos.undo_move(m);
            if (stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
            if (v >= probCutBeta) {
                tt_.store(tte, pos.key(), value_to_tt(v, ply), BOUND_LOWER,
                          depth - 3, m, rawEval);
                return v;
            }
        }
    }

    const Color us = pos.side_to_move();
    const bool  hasPrev = prevMove != Move::none() && prevMove != Move::null();
    const Piece prevPc  = hasPrev ? pos.piece_on(prevMove.to_sq()) : NO_PIECE;
    const Square prevTo = hasPrev ? prevMove.to_sq() : SQ_A1;
    const Move counter = hasPrev ? ts.counterMoves[prevPc][prevTo] : Move::none();

    for (auto& sm : list) {
        Move m = sm.move;
        if (m == ttMove) {
            sm.score = TT_SCORE;
        } else if (is_capture(pos, m) || m.type_of() == PROMOTION) {
            PieceType victim = (m.type_of() == EN_PASSANT) ? PAWN
                             : is_capture(pos, m) ? type_of(pos.piece_on(m.to_sq()))
                             : NO_PIECE_TYPE;
            PieceType attacker = type_of(pos.piece_on(m.from_sq()));
            int mvv = 64 * PieceValue[victim] - PieceValue[attacker];
            if (m.type_of() == PROMOTION) mvv += PieceValue[m.promotion_type()];
            const bool bad = cfg_.advancedOrdering && !see_ge(pos, m, 0);
            const PieceType capIdx = (victim == NO_PIECE_TYPE) ? PAWN : victim;
            const int ch = ts.captureHistory[pos.piece_on(m.from_sq())][m.to_sq()][capIdx];
            // MVV-LVA dominates; capture history is a learned tie-break.
            sm.score = (bad ? BAD_CAPTURE : GOOD_CAPTURE) + mvv * 16 + ch / 128;
        } else if (m == ts.killers[ply][0]) {
            sm.score = KILLER0_SCORE;
        } else if (m == ts.killers[ply][1]) {
            sm.score = KILLER1_SCORE;
        } else if (cfg_.advancedOrdering && m == counter) {
            sm.score = COUNTER_SCORE;
        } else {
            sm.score = ts.history[us][m.from_sq()][m.to_sq()]
                     + cont_hist_score(ts, ply, pos.piece_on(m.from_sq()), m.to_sq());
        }
    }

    const bool improving = !inCheck && ply >= 2
        && ts.evalStack[ply - 2] != VALUE_NONE && staticEval != VALUE_NONE
        && staticEval > ts.evalStack[ply - 2];

    Value bestValue = -VALUE_INFINITE;
    Move  bestMove  = Move::none();
    Bound bound     = BOUND_UPPER;
    int   moveCount = 0;

    // Moves actually searched at this node (for history bonus/malus on cutoff).
    Move searchedQuiets[64];   int nQuiets = 0;
    Move searchedCaptures[32]; int nCaptures = 0;

    for (std::size_t i = 0; i < list.size(); ++i) {
        pick_next(list, i);
        Move move = list.begin()[i].move;
        if (move == excludedMove) continue;   // singular verification skips it
        ++moveCount;

        const bool quiet = !is_noisy(pos, move);
        const bool givesCheck = pos.gives_check(move);

        // A node is a pruning candidate only when it is non-PV, not in check, at
        // shallow depth, past the first move, and not in a losing-mate line.
        // Checking moves are always exempt from pruning so forcing tactics and
        // sacrifices (e.g. mating attacks) are never discarded.
        const bool prunable = !PvNode && !inCheck && depth <= 8 && moveCount > 1
                              && bestValue > VALUE_MATED_IN_MAX_PLY;
        const bool canPrune = prunable && !givesCheck;

        // Combined history (butterfly + continuation) for this quiet move; drives
        // history-based pruning and the LMR reduction adjustment below.
        const int moveHist = quiet
            ? ts.history[us][move.from_sq()][move.to_sq()]
              + cont_hist_score(ts, ply, pos.piece_on(move.from_sq()), move.to_sq())
            : 0;

        // Futility pruning: drop late quiets that cannot realistically raise alpha.
        if (cfg_.futility && canPrune && quiet && depth <= 6
            && staticEval + 100 + 90 * depth <= alpha)
            continue;

        // Late move pruning: once enough quiet moves have been tried at shallow
        // depth, the remaining (worse-ordered) quiets are skipped outright.
        if (canPrune && quiet) {
            const int lmpLimit = improving ? (3 + depth * depth)
                                           : (3 + depth * depth) / 2;
            if (moveCount > lmpLimit) continue;
        }

        // History-based pruning: skip quiets with very poor combined history.
        if (canPrune && quiet && depth <= 4 && moveHist < -4000 * depth)
            continue;

        // SEE-based pruning: skip moves that lose too much material by static
        // exchange (quiets hanging material, or clearly losing captures).
        if (canPrune) {
            const int seeMargin = quiet ? -20 * depth * depth : -80 * depth;
            if (!see_ge(pos, move, seeMargin)) continue;
        }

        // Singular extension: if the TT move is much better than all others
        // (a reduced search of the rest fails low below a margin), extend it.
        // If that reduced search instead beats beta, the whole node fails high
        // (multicut). Plus a check extension for forcing moves.
        int extension = 0;
        if (move == ttMove && excludedMove == Move::none() && depth >= 8 && ttHit
            && (tte->bound() == BOUND_LOWER || tte->bound() == BOUND_EXACT)
            && tte->depth() >= depth - 3
            && std::abs(int(ttValue)) < VALUE_MATE_IN_MAX_PLY) {
            const Value sBeta = Value(int(ttValue) - 2 * depth);
            ts.excluded[ply] = ttMove;
            Value sv = search<false>(ts, Value(int(sBeta) - 1), sBeta,
                                     (depth - 1) / 2, ply, prevMove);
            ts.excluded[ply] = Move::none();
            if (stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
            if (sv < sBeta)                 extension = 1;        // singular
            else if (int(sBeta) >= int(beta)) return sBeta;      // multicut
        }
        if (extension == 0 && givesCheck && PvNode && see_ge(pos, move, 0))
            extension = 1;                                        // check extension (PV only)
        const int newDepth = depth - 1 + extension;

        // Record searched moves for the post-cutoff history updates.
        if (quiet) { if (nQuiets < 64) searchedQuiets[nQuiets++] = move; }
        else if (is_capture(pos, move)) {
            if (nCaptures < 32) searchedCaptures[nCaptures++] = move;
        }

        // Record this move on the continuation-history stack for child nodes.
        ts.spPiece[ply] = pos.piece_on(move.from_sq());
        ts.spTo[ply]    = move.to_sq();
        ts.spOk[ply]    = true;

        const std::uint64_t nodesBefore = ts.nodes;   // for root node-effort
        pos.do_move(move);
        tt_.prefetch(pos.key());   // overlap TT memory latency (helps at 8+ threads)

        Value score;
        if (cfg_.lmr && depth >= 3 && moveCount > 1 && quiet && !inCheck) {
            int r = reductions_[std::min(depth, 63)][std::min(moveCount, 63)];
            if (PvNode) --r;
            if (!improving) ++r;
            // Stat-based adjustments: reduce less for killer/counter moves and
            // for checking moves; the history term reduces good-history moves
            // less and bad-history moves more.
            if (move == ts.killers[ply][0] || move == ts.killers[ply][1] || move == counter)
                --r;
            if (givesCheck) --r;    // checking moves are reduced less
            r -= moveHist / 8192;   // reduce good-history moves less, poor ones more
            // Clamp: minimum reduction 1, maximum (newDepth - 1) so the reduced
            // search keeps at least depth 1.
            r = std::clamp(r, 1, newDepth - 1);

            score = -search<false>(ts, -(alpha + 1), -alpha, newDepth - r, ply + 1, move);
            if (score > alpha && r > 0)
                score = -search<false>(ts, -(alpha + 1), -alpha, newDepth, ply + 1, move);
            if (score > alpha && PvNode && score < beta)
                score = -search<true>(ts, -beta, -alpha, newDepth, ply + 1, move);
        } else if (PvNode && moveCount == 1) {
            score = -search<true>(ts, -beta, -alpha, newDepth, ply + 1, move);
        } else {
            score = -search<false>(ts, -(alpha + 1), -alpha, newDepth, ply + 1, move);
            if (score > alpha && PvNode && score < beta)
                score = -search<true>(ts, -beta, -alpha, newDepth, ply + 1, move);
        }

        pos.undo_move(move);

        if (stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;

        if (score > bestValue) {
            bestValue = score;
            if (score > alpha) {
                bestMove = move;
                if (PvNode) update_pv(ts.pv[ply], move, ts.pv[ply + 1]);
                if (rootNode) {
                    ts.rootBest = move;
                    ts.bestMoveNodes = ts.nodes - nodesBefore;  // effort on best move
                }

                if (score >= beta) {
                    bound = BOUND_LOWER;
                    break;
                }
                alpha = score;
                bound = BOUND_EXACT;
            }
        }
    }

    // No move searched: only reachable inside a singular search whose sole move
    // was the excluded one -> fail low so the TT move is judged singular.
    if (moveCount == 0)
        return excludedMove != Move::none() ? alpha : (inCheck ? mated_in(ply) : VALUE_DRAW);

    if (bound == BOUND_LOWER && bestMove != Move::none()) {
        const int bonus = std::min(depth * depth, 400);   // malus is the negation

        if (!is_noisy(pos, bestMove)) {
            // Quiet cutoff: reward the cutoff move, penalize the other quiets
            // that were tried and failed (history malus).
            if (ts.killers[ply][0] != bestMove) {
                ts.killers[ply][1] = ts.killers[ply][0];
                ts.killers[ply][0] = bestMove;
            }
            if (prevMove != Move::none() && prevMove != Move::null())
                ts.counterMoves[pos.piece_on(prevMove.to_sq())][prevMove.to_sq()] = bestMove;

            hist_update(ts.history[us][bestMove.from_sq()][bestMove.to_sq()], bonus);
            cont_hist_update(ts, ply, pos.piece_on(bestMove.from_sq()), bestMove.to_sq(), bonus);
            for (int q = 0; q < nQuiets; ++q) {
                Move m = searchedQuiets[q];
                if (m != bestMove) {
                    hist_update(ts.history[us][m.from_sq()][m.to_sq()], -bonus);
                    cont_hist_update(ts, ply, pos.piece_on(m.from_sq()), m.to_sq(), -bonus);
                }
            }
        } else if (is_capture(pos, bestMove)) {
            // Capture cutoff: reward in capture history.
            const Piece pc = pos.piece_on(bestMove.from_sq());
            const PieceType cap = (bestMove.type_of() == EN_PASSANT)
                                ? PAWN : type_of(pos.piece_on(bestMove.to_sq()));
            hist_update(ts.captureHistory[pc][bestMove.to_sq()][cap], bonus);
        }

        // Penalize every other tried capture (history malus for captures).
        for (int c = 0; c < nCaptures; ++c) {
            Move m = searchedCaptures[c];
            if (m == bestMove) continue;
            const Piece pc = pos.piece_on(m.from_sq());
            const PieceType cap = (m.type_of() == EN_PASSANT)
                                ? PAWN : type_of(pos.piece_on(m.to_sq()));
            hist_update(ts.captureHistory[pc][m.to_sq()][cap], -bonus);
        }
    }

    // Update correction history toward the residual (search result vs raw eval)
    // for quiet outcomes, bounded and depth-weighted.
    if (excludedMove == Move::none() && !inCheck && rawEval != VALUE_NONE
        && std::abs(int(bestValue)) < VALUE_MATE_IN_MAX_PLY
        && (bestMove == Move::none() || !is_capture(pos, bestMove))
        && (bound == BOUND_EXACT
            || (bound == BOUND_LOWER && bestValue > staticEval)
            || (bound == BOUND_UPPER && bestValue < staticEval))) {
        const Color stm = pos.side_to_move();
        const int diff = int(bestValue) - int(rawEval);
        for (int* c : {&ts.pawnCorr[stm][pos.pawn_key() & 16383],
                       &ts.matCorr[stm][material_key(pos) & 16383]}) {
            *c += (diff * 256 - *c) * std::min(depth + 1, 8) / 1024;  // gentle learning
            *c = std::clamp(*c, -16 * 256, 16 * 256);                 // small, safe correction
        }
    }

    if (excludedMove == Move::none())
        tt_.store(tte, pos.key(), value_to_tt(bestValue, ply), bound, depth, bestMove, rawEval);
    return bestValue;
}

// ---------------------------------------------------------------------------
// Per-thread iterative deepening
// ---------------------------------------------------------------------------
void Search::iterative_deepening(ThreadState& ts) {
    Eval::on_search_start(ts.pos);   // refresh this thread's NNUE accumulator

    const bool main = (ts.id == 0);
    const int maxDepth = limits_.depth > 0 ? limits_.depth : MAX_PLY - 1;

    Value prevScore = VALUE_NONE;
    Move  prevBest  = Move::none();
    Value prevIterScore = VALUE_NONE;
    int   stableCount = 0;          // consecutive iterations with the same best move
    const Value rootEval = evaluate(ts.pos);          // for complexity scaling

    // Lazy SMP diversification: helper threads skip certain depths on a
    // per-thread pattern so they explore at different rates and fill the shared
    // TT more diversely (the main thread, id 0, is never skipped).
    static const int SkipSize[]  = {1, 1, 2, 2, 2, 3, 3, 3};
    static const int SkipPhase[] = {0, 1, 0, 1, 2, 0, 1, 2};

    for (int d = 1; d <= maxDepth; ++d) {
        if (!main) {
            const int i = (ts.id - 1) % 8;
            if ((d + SkipPhase[i]) % (2 * SkipSize[i]) >= SkipSize[i])
                continue;
        }

        Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
        int delta = 0;
        if (cfg_.aspiration && d >= 4 && prevScore != VALUE_NONE
            && std::abs(int(prevScore)) < VALUE_MATE_IN_MAX_PLY) {
            delta = 18;
            alpha = std::max(prevScore - delta, -VALUE_INFINITE);
            beta  = std::min(prevScore + delta,  VALUE_INFINITE);
        }

        const std::uint64_t iterStartNodes = ts.nodes;
        Value v;
        while (true) {
            for (int p = 0; p < MAX_PLY; ++p) ts.pv[p][0] = Move::none();
            v = search<true>(ts, alpha, beta, d, 0, Move::none());

            if (stop_.load(std::memory_order_relaxed)) break;

            if (v <= alpha) {
                beta  = Value((int(alpha) + int(beta)) / 2);
                alpha = std::max(v - delta, -VALUE_INFINITE);
            } else if (v >= beta) {
                beta = std::min(v + delta, VALUE_INFINITE);
            } else {
                break;
            }
            delta += delta / 2 + 5;
        }

        if (stop_.load(std::memory_order_relaxed))
            break;

        prevScore = v;
        if (ts.pv[0][0] != Move::none()) ts.rootBest = ts.pv[0][0];
        ts.ponder = ts.pv[0][1];
        ts.score = v;
        ts.completedDepth = d;

        if (main) {
            report(ts, d, v);

            // Adaptive time scaling of the optimum (soft) limit. Combines four
            // signals; the hard maximum (maximumMs_) is never exceeded, so this
            // cannot cause a time forfeit.
            if (ts.rootBest == prevBest) ++stableCount; else stableCount = 0;
            double scale = 1.0;
            if (d >= 4) {
                // (1) Best-move stability bonus: a long-stable best move -> spend
                //     less time (and exit early once optimum is reached).
                scale = std::max(0.55, 1.30 - 0.12 * stableCount);

                // (2) Score-drop instability: a falling score -> spend more.
                if (prevIterScore != VALUE_NONE && v < prevIterScore - 30)
                    scale += 0.5;

                // (3) Complexity scaling: a large gap between the root static
                //     eval and the search score signals a complex/tactical
                //     position -> spend more; quiet/simple positions -> less.
                if (rootEval != VALUE_NONE)
                    scale *= 1.0 + std::min(std::abs(int(v) - int(rootEval)) / 400.0, 0.40);

                // (4) Node-effort instability: if the best move dominates the
                //     node count the choice is easy -> spend less; if effort is
                //     spread across moves the position is unstable -> spend more.
                const std::uint64_t iterNodes = ts.nodes - iterStartNodes;
                if (iterNodes > 0) {
                    const double effort =
                        std::clamp(double(ts.bestMoveNodes) / double(iterNodes), 0.0, 1.0);
                    scale *= 1.15 - 0.50 * effort;           // high effort -> ~0.65x
                }
                // (5) Instability bonus: if the best move just changed from the
                //     previous iteration, spend more time (x1.1). A stable best
                //     move leaves the scale untouched so the soft-limit early
                //     exit below can fire.
                if (ts.rootBest != prevBest)
                    scale *= 1.1;

                scale = std::clamp(scale, 0.40, 2.6);

                // Surface the same instability signals this block already
                // computes for time management, on ThreadState, so
                // Search::think() can copy them into the returned
                // SearchResult (see search.h's SearchResult::scoreSwing /
                // bestMoveChanges) -- no extra search work, just recording
                // what iterative deepening already knows.
                if (ts.rootBest != prevBest)
                    ++ts.bestMoveChanges;
            }
            if (prevIterScore != VALUE_NONE)
                ts.lastScoreSwing = Value(std::abs(int(v) - int(prevIterScore)));
            prevBest = ts.rootBest;
            prevIterScore = v;

            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - start_).count();
            // Soft limit: the engine tries to finish within optimum * 0.75 (scaled
            // by the instability signals); the hard limit (time_up) is separate.
            const std::int64_t soft = std::int64_t(optimumMs_ * 0.75 * scale);
            if (!limits_.infinite && optimumMs_ > 0 && ms >= soft) break;
            if (limits_.nodes && nodes() >= limits_.nodes) break;
        }

        if (std::abs(int(v)) >= VALUE_MATE_IN_MAX_PLY) break;
    }
}

// ---------------------------------------------------------------------------
// Coordinator: spawn helper threads, run main, pick the result.
// ---------------------------------------------------------------------------
SearchResult Search::think(Position& pos, const SearchLimits& limits) {
    limits_ = limits;
    // stop_ is cleared by the caller via arm() before launching.
    start_ = Clock::now();
    compute_time_budget(pos);
    tt_.new_search();

    MoveList rootMoves;
    generate(pos, rootMoves, LEGAL);
    if (rootMoves.empty())
        return SearchResult{};
    const Move firstLegal = rootMoves.begin()[0].move;

    if (rootMoves.size() == 1 && optimumMs_ > 0 && !limits_.infinite
        && limits_.depth == 0) {
        SearchResult r; r.best = firstLegal; r.depth = 1;
        return r;
    }

    // Root DTZ probe: if the position is in the tablebases, play the DTZ-optimal
    // move directly (it is 50-move-rule aware). No-op when TBs are unavailable.
    if (Tablebases::available()) {
        Tablebases::RootProbe rp = Tablebases::probe_root(pos);
        if (rp.ok) {
            SearchResult r;
            r.best  = rp.best;
            r.score = Tablebases::wdl_to_value(rp.wdl, 0);
            r.depth = 1;
            return r;
        }
    }

    const int n = threads_;
    ts_.resize(std::size_t(n));
    for (int i = 0; i < n; ++i) {
        ThreadState& t = ts_[std::size_t(i)];
        t.id = i;
        t.pos = pos;                       // safe deep copy (incl. NNUE acc)
        t.nodes = 0;
        t.rootBest = firstLegal;
        t.ponder = Move::none();
        t.score = VALUE_ZERO;
        t.completedDepth = 0;
        t.lastScoreSwing = VALUE_ZERO;
        t.bestMoveChanges = 0;
        std::memset(t.killers, 0, sizeof(t.killers));
        std::memset(t.history, 0, sizeof(t.history));
        std::memset(t.counterMoves, 0, sizeof(t.counterMoves));
        std::memset(t.captureHistory, 0, sizeof(t.captureHistory));
        std::memset(t.contHist, 0, sizeof(t.contHist));
        std::memset(t.pawnCorr, 0, sizeof(t.pawnCorr));
        std::memset(t.matCorr, 0, sizeof(t.matCorr));
        std::memset(t.spOk, 0, sizeof(t.spOk));
        std::memset(t.excluded, 0, sizeof(t.excluded));
    }

    std::vector<std::thread> pool;
    pool.reserve(std::size_t(n - 1));
    for (int i = 1; i < n; ++i)
        pool.emplace_back([this, i] { iterative_deepening(ts_[std::size_t(i)]); });

    iterative_deepening(ts_[0]);           // main thread (authoritative)

    stop();                                // tell helpers to finish
    for (auto& th : pool) th.join();

    SearchResult r;
    r.best   = ts_[0].rootBest;
    r.ponder = ts_[0].ponder;
    r.score  = ts_[0].score;
    r.depth  = ts_[0].completedDepth;
    r.scoreSwing      = ts_[0].lastScoreSwing;
    r.bestMoveChanges = ts_[0].bestMoveChanges;
    return r;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
std::uint64_t Search::bench(int depth) {
    static const char* suite[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
        "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
        "8/8/8/3k4/8/8/8/Q3K3 w - - 0 1",
        "r4rk1/1b2bppp/ppq1p3/2pp3n/5P2/1PNBP3/PBPPQ1PP/R4RK1 w - - 0 1",
    };

    clear();
    NNUE::reset_cache_stats();
    std::uint64_t total = 0;
    const auto t0 = Clock::now();

    for (const char* fen : suite) {
        Position pos;
        pos.set(fen);
        SearchLimits lim;
        lim.depth = depth;
        arm();
        think(pos, lim);
        total += nodes();
    }

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - t0).count();
    const std::uint64_t nps = ms > 0 ? (total * 1000ULL) / std::uint64_t(ms) : total;
    std::cout << "bench: " << total << " nodes " << nps << " nps "
              << ms << " ms depth " << depth << std::endl;
    if (NNUE::enabled()) {
        const NNUE::CacheStats cs = NNUE::cache_stats();
        const std::uint64_t crossBucket = cs.hits + cs.misses;
        const double hitRate = crossBucket ? (100.0 * double(cs.hits) / double(crossBucket)) : 0.0;
        std::cout << "finny: sameBucket " << cs.sameBucket << " hits " << cs.hits
                   << " misses " << cs.misses << " hitRate " << hitRate << "%" << std::endl;
    }
    return total;
}

// Explicit template instantiations.
template Value Search::search<true>(ThreadState&, Value, Value, int, int, Move);
template Value Search::search<false>(ThreadState&, Value, Value, int, int, Move);

} // namespace chess
