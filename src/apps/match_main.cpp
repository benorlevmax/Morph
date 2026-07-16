// match_main.cpp - Engine-vs-engine / self-play match driver with Elo + SPRT.
//
// Usage:
//   chess_match [--games N] [--depthA d] [--depthB d]
//               [--movetimeA ms] [--movetimeB ms]
//               [--pgn out.pgn] [--sprt elo0 elo1]
#include "match/match.h"
#include "match/stats.h"
#include "core/position.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace chess;

int main(int argc, char** argv) {
    Position::init_static();

    MatchSettings s;
    s.games = 10;
    EngineConfig a, b;
    a.name = "A"; b.name = "B";
    a.limits.depth = 6; b.limits.depth = 6;
    std::string pgnPath;
    bool useSprt = false;
    double elo0 = 0, elo1 = 10;

    // Apply a named feature configuration to an engine.
    auto applyConfig = [](EngineConfig& e, const std::string& name) {
        e.name = name;
        if (name == "phase2") {                  // material eval, no refinements
            e.materialOnly = true;
            e.evalMode = EvalMode::Classical;
            e.cfg = SearchConfig{false, false, false, false, false, false, false};
        } else if (name == "phase3") {           // full eval, no refinements
            e.materialOnly = false;
            e.evalMode = EvalMode::Classical;
            e.cfg = SearchConfig{false, false, false, false, false, false, false};
        } else if (name == "classical") {        // full classical eval + refinements
            e.materialOnly = false;
            e.evalMode = EvalMode::Classical;
            e.cfg = SearchConfig{};
        } else if (name == "nnue") {             // NNUE eval + refinements
            e.materialOnly = false;
            e.evalMode = EvalMode::NNUE;
            e.cfg = SearchConfig{};
        } else {                                  // "current": classical + refinements
            e.materialOnly = false;
            e.evalMode = EvalMode::Classical;
            e.cfg = SearchConfig{};
        }
    };

    for (int i = 1; i < argc; ++i) {
        std::string t = argv[i];
        auto next = [&](int d) { return (i + 1 < argc) ? std::atoi(argv[++i]) : d; };
        if      (t == "--games")     s.games = next(s.games);
        else if (t == "--depthA")    a.limits.depth = next(a.limits.depth);
        else if (t == "--depthB")    b.limits.depth = next(b.limits.depth);
        else if (t == "--movetimeA") { a.limits.movetime = next(0); a.limits.depth = 0; }
        else if (t == "--movetimeB") { b.limits.movetime = next(0); b.limits.depth = 0; }
        else if (t == "--nodesA")    { a.limits.nodes = std::uint64_t(next(0)); a.limits.depth = 0; }
        else if (t == "--nodesB")    { b.limits.nodes = std::uint64_t(next(0)); b.limits.depth = 0; }
        else if (t == "--threadsA")  a.threads = next(1);
        else if (t == "--threadsB")  b.threads = next(1);
        else if (t == "--configA")   { if (i + 1 < argc) applyConfig(a, argv[++i]); }
        else if (t == "--configB")   { if (i + 1 < argc) applyConfig(b, argv[++i]); }
        else if (t == "--pgn")       { if (i + 1 < argc) pgnPath = argv[++i]; }
        else if (t == "--sprt")      { useSprt = true;
                                       if (i + 1 < argc) elo0 = std::atof(argv[++i]);
                                       if (i + 1 < argc) elo1 = std::atof(argv[++i]); }
    }

    // Built-in balanced opening suite (1-2 plies in) for game diversity.
    s.openings = {
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 2",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
    };
    if (s.games < int(s.openings.size()) * 2)
        s.games = int(s.openings.size()) * 2;   // each opening, both colors

    std::cout << "Match: " << a.name << " vs " << b.name
              << "  games=" << s.games << "\n";

    MatchResult mr = play_match(a, b, s);

    const int n = mr.aWins + mr.bWins + mr.draws;
    std::cout << "Result (A): +" << mr.aWins << " -" << mr.bWins
              << " =" << mr.draws << "  (" << n << " games)\n";

    EloEstimate e = elo_estimate(mr.aWins, mr.bWins, mr.draws);
    std::cout << "Elo(A-B): " << e.elo << " +/- " << e.margin << "\n";

    if (useSprt) {
        SprtResult sr = sprt(mr.aWins, mr.bWins, mr.draws, elo0, elo1);
        std::cout << "SPRT[" << elo0 << "," << elo1 << "] LLR=" << sr.llr
                  << " (" << sr.lowerBound << "," << sr.upperBound << ") -> "
                  << (sr.verdict == SprtVerdict::AcceptH1 ? "H1"
                    : sr.verdict == SprtVerdict::AcceptH0 ? "H0" : "continue")
                  << "\n";
    }

    if (!pgnPath.empty()) {
        std::ofstream os(pgnPath);
        for (const auto& g : mr.games) write_pgn(os, g);
        std::cout << "Wrote " << mr.games.size() << " games to " << pgnPath << "\n";
    }
    return 0;
}
