// search.h - Iterative deepening alpha-beta / PVS search with Lazy SMP.
#pragma once

#include "core/position.h"
#include "search/tt.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace chess {

constexpr int MAX_PLY = 128;

struct SearchLimits {
    int      depth      = 0;        // max depth (0 = unlimited)
    std::int64_t movetime = 0;      // fixed ms per move (0 = unused)
    std::uint64_t nodes  = 0;       // node cap (0 = unused)
    bool     infinite   = false;

    int      time[COLOR_NB] = {0, 0};
    int      inc[COLOR_NB]  = {0, 0};
    int      movestogo  = 0;
};

struct SearchResult {
    Move  best  = Move::none();
    Move  ponder = Move::none();
    Value score = VALUE_ZERO;
    int   depth = 0;
    // Cheap, already-computed-internally search-instability signals (see
    // iterative_deepening()'s existing time-management heuristics, which
    // track the same quantities) -- surfaced here so callers outside the
    // search (self-play data generation, distributed difficulty mining)
    // can use them as a training-data quality/difficulty signal without
    // running any extra search. Both are 0 for a position resolved in a
    // single iteration (e.g. a forced/tablebase move).
    Value scoreSwing      = VALUE_ZERO;  // |score at final depth - score at the previous depth|
    int   bestMoveChanges = 0;           // # of iterations (depth >= 4) where the root best move changed
};

// Runtime toggles for search features (A/B testing, version emulation).
struct SearchConfig {
    bool aspiration       = true;
    bool reverseFutility  = true;
    bool nullMove         = true;
    bool razoring         = true;
    bool futility         = true;
    bool lmr              = true;
    bool advancedOrdering = true;   // SEE capture split + countermoves
};

// All per-thread search state. Helper threads each own one; nothing here is
// shared, so history / killers / countermoves are race-free by construction.
struct ThreadState {
    int          id = 0;
    Position     pos;                 // this thread's own root copy
    std::uint64_t nodes = 0;

    Move  pv[MAX_PLY][MAX_PLY];
    Move  killers[MAX_PLY][2];
    int   history[COLOR_NB][SQUARE_NB][SQUARE_NB];
    Move  counterMoves[PIECE_NB][SQUARE_NB];
    // Capture history: [moving piece][to-square][captured piece type].
    int   captureHistory[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB];
    // Continuation history: [prev piece][prev to][cur piece][cur to], indexed at
    // several look-back offsets (1/2/4/6 ply) via the move stack below.
    int   contHist[PIECE_NB][SQUARE_NB][PIECE_NB][SQUARE_NB];
    // Per-ply record of the move played to reach the next ply (for cont-hist).
    Piece  spPiece[MAX_PLY + 8];
    Square spTo[MAX_PLY + 8];
    bool   spOk[MAX_PLY + 8];
    // Singular-extension excluded move, per ply.
    Move   excluded[MAX_PLY + 8];
    // Correction history (corrects static eval), keyed by pawn structure and by
    // material signature, per color.
    int   pawnCorr[COLOR_NB][16384];
    int   matCorr[COLOR_NB][16384];
    Value evalStack[MAX_PLY + 2];

    // Last completed-iteration result for this thread.
    Move  rootBest = Move::none();
    Move  ponder   = Move::none();
    Value score    = VALUE_ZERO;
    int   completedDepth = 0;
    // Search-instability tracking (main thread only; see SearchResult's
    // scoreSwing/bestMoveChanges -- these accumulate across iterations and
    // are copied into the final SearchResult in Search::think()).
    Value lastScoreSwing  = VALUE_ZERO;
    int   bestMoveChanges = 0;
    // Nodes spent in the best root move's subtree this iteration (time mgmt).
    std::uint64_t bestMoveNodes = 0;
};

class Search {
public:
    Search();

    void set_hash_size(std::size_t mb) { tt_.resize(mb); }
    void set_threads(int n) { threads_ = n < 1 ? 1 : (n > 256 ? 256 : n); }
    int  threads() const { return threads_; }
    void clear();                     // ucinewgame: wipe TT
    void stop() { stop_.store(true, std::memory_order_relaxed); }
    // Clear the stop flag. MUST be called on the controlling thread before
    // launching a search worker, so a racing stop() is not lost.
    void arm() { stop_.store(false, std::memory_order_relaxed); }

    void set_config(const SearchConfig& c) { cfg_ = c; }
    void set_quiet(bool q) { quiet_ = q; }
    // Reserved time (ms) for GUI/network latency, subtracted from the clock.
    void set_move_overhead(int ms) { moveOverhead_ = ms < 0 ? 0 : ms; }

    // Run a search on `pos` (left unmodified). Returns the best move.
    SearchResult think(Position& pos, const SearchLimits& limits);

    // Fixed-depth benchmark over a built-in position suite (returns total nodes).
    std::uint64_t bench(int depth);

    std::uint64_t nodes() const;

private:
    using Clock = std::chrono::steady_clock;

    void iterative_deepening(ThreadState& ts);

    template <bool PvNode>
    Value search(ThreadState& ts, Value alpha, Value beta, int depth, int ply, Move prevMove);
    Value qsearch(ThreadState& ts, Value alpha, Value beta, int ply, bool genChecks);

    void init_reductions();

    bool  time_up();
    void  check_time(ThreadState& ts);
    void  compute_time_budget(const Position& pos);
    void  report(ThreadState& ts, int depth, Value score);

    // Shared, read-mostly during a search.
    TranspositionTable tt_;
    std::atomic<bool>  stop_{false};
    SearchLimits limits_{};
    Clock::time_point start_{};
    std::int64_t      optimumMs_ = 0;
    std::int64_t      maximumMs_ = 0;
    std::int64_t      moveOverhead_ = 50;   // ms reserved for GUI/network latency
    int  reductions_[64][64];
    SearchConfig cfg_{};
    bool quiet_ = false;
    int  threads_ = 1;

    // Per-thread states (index 0 is the main/authoritative thread).
    std::vector<ThreadState> ts_;
};

} // namespace chess
