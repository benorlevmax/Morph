// tablebases.h - Syzygy endgame tablebase framework.
//
// This provides the integration architecture and a graceful-fallback probe.
// A full WDL/DTZ decoder (e.g. linking Fathom) drops in behind this interface
// without changing search or UCI: enable real probing by implementing
// probe_wdl_impl() and reporting available() == true after init().
#pragma once

#include "core/position.h"

#include <string>

namespace chess {

// Result of a Win/Draw/Loss probe, from the side-to-move's perspective.
enum class WDLResult {
    Loss,
    BlessedLoss,   // loss but drawn under the 50-move rule
    Draw,
    CursedWin,     // win but drawn under the 50-move rule
    Win,
    Fail           // probe unavailable / position not in tablebases
};

namespace Tablebases {

// Configure with a path list ("<empty>" or "" disables). Safe to call repeatedly.
void init(const std::string& paths);

bool        available();      // true only when real tablebases are loaded
int         max_pieces();     // largest TB cardinality available (0 if none)
std::string status();         // human-readable status for `info string`

// Probe Win/Draw/Loss for `pos`. Returns WDLResult::Fail when unavailable or
// when the position has more pieces than the loaded tablebases support.
WDLResult probe_wdl(const Position& pos);

// Convert a successful WDL result into a search Value at the given ply
// (mate-scaled wins/losses; draw == 0). Caller must check != Fail first.
Value wdl_to_value(WDLResult r, int ply);

// Result of a root DTZ probe: the DTZ-optimal best move (50-move-rule aware).
struct RootProbe {
    bool      ok   = false;        // true only if a real DTZ probe succeeded
    Move      best = Move::none();
    WDLResult wdl  = WDLResult::Fail;
};

// Probe the DTZ tables at the root for best-move selection. Returns ok==false
// when tablebases are unavailable or the position is not probable, so callers
// fall back to normal search. DTZ accounts for the 50-move rule internally.
RootProbe probe_root(const Position& pos);

} // namespace Tablebases
} // namespace chess
