// match.h - In-process engine-vs-engine match runner and self-play.
#pragma once

#include "core/position.h"
#include "search/search.h"
#include "eval/evaluate.h"
#include "io/pgn.h"

#include <string>
#include <vector>

namespace chess {

struct EngineConfig {
    std::string  name   = "Engine";
    SearchLimits limits;            // depth / movetime / time control
    std::size_t  hashMB = 16;
    SearchConfig cfg{};             // search-feature toggles
    bool         materialOnly = false;  // material-only evaluation
    EvalMode     evalMode = EvalMode::Classical;
    int          threads = 1;
};

struct MatchSettings {
    int  games        = 2;          // total games (colors alternate)
    int  maxPlies     = 400;        // adjudicate draw beyond this
    int  resignScore  = 0;          // |cp| threshold to adjudicate (0 = off)
    int  resignPlies  = 6;          // consecutive plies above threshold
    bool alternate    = true;       // swap colors each game
    bool collectPgn   = true;
    std::vector<std::string> openings;  // start FENs (empty -> startpos)
};

struct MatchResult {
    int aWins = 0, bWins = 0, draws = 0;   // from engine A's perspective
    std::vector<GameRecord> games;
};

// Play a single game; returns "1-0","0-1","1/2-1/2". `white`/`black` are the
// engine configs; `out` receives the game record if non-null.
std::string play_game(EngineConfig white, EngineConfig black,
                      const std::string& startFen, const MatchSettings& s,
                      GameRecord* out);

// Play a full match between A and B, alternating colors.
MatchResult play_match(const EngineConfig& a, const EngineConfig& b,
                       const MatchSettings& s);

} // namespace chess
