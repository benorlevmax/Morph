// test_search.cpp - Search correctness: mate detection, tactics, determinism.
// (Phase 4 search-refinement regression coverage.)
#include "core/position.h"
#include "search/search.h"

#include <cmath>
#include <iostream>
#include <string>

using namespace chess;

namespace {
int failures = 0;

void check(bool cond, const std::string& what) {
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!cond) ++failures;
}

SearchResult run(const std::string& fen, int depth) {
    Position pos;
    pos.set(fen);
    Search search;
    SearchLimits limits;
    limits.depth = depth;
    return search.think(pos, limits);
}
} // namespace

int main() {
    Position::init_static();

    // 1. Start position: legal move, near-equal score.
    {
        SearchResult r = run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6);
        check(r.best != Move::none(), "startpos returns a legal move");
        check(std::abs(int(r.score)) < 200, "startpos score is roughly balanced");
    }

    // 2. Mate in 1: Rh1-h8#.
    {
        SearchResult r = run("4k3/8/4K3/8/8/8/8/7R w - - 0 1", 4);
        check(move_to_uci(r.best) == "h1h8", "finds mate-in-1 move (Rh8#)");
        check(r.score == mate_in(1), "reports mate score for mate-in-1");
    }

    // 3. Tactics: capture a hanging queen (Rd1xd4).
    {
        SearchResult r = run("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1", 6);
        check(move_to_uci(r.best) == "d1d4", "captures hanging queen (Rxd4)");
        // After Rxd4 white is left a rook up vs a bare king (~+500 cp).
        check(int(r.score) > 300, "score reflects winning material");
    }

    // 4. Avoid stalemate trap / detect being mated: black to move, already mated.
    {
        // Black king h8 mated by Qg7 supported by Kg6.
        SearchResult r = run("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 2);
        check(r.best == Move::none(), "no legal move when checkmated");
    }

    // 5. Tactical regression guards (catch pruning-induced blindness).
    {
        // WAC.001: the famous Qg6!! mating attack (mate in 2). With LMR this
        // deep quiet sacrifice surfaces at depth 13 (a few ms).
        SearchResult r = run("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", 13);
        check(move_to_uci(r.best) == "g3g6",
              "WAC.001 finds Qg6 (got " + move_to_uci(r.best) + ")");
        check(r.score >= VALUE_MATE_IN_MAX_PLY, "WAC.001 scored as a forced mate");
    }
    {
        // Back-rank mate in 1: Ra8# (king boxed by its own f7/g7/h7 pawns).
        SearchResult r = run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 4);
        check(move_to_uci(r.best) == "a1a8", "back-rank Ra8# found");
        check(r.score == mate_in(1), "back-rank mate scored as mate-in-1");
    }

    // 6. Determinism: identical searches give identical results.
    {
        SearchResult a = run("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 7);
        SearchResult b = run("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 7);
        check(a.best == b.best && a.score == b.score, "search is deterministic");
    }

    std::cout << "\n" << (failures == 0 ? "ALL SEARCH TESTS PASSED"
                                        : std::to_string(failures) + " SEARCH FAILURE(S)")
              << "\n";
    return failures == 0 ? 0 : 1;
}
