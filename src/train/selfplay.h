// selfplay.h - Self-play training-data generation using the current engine.
#pragma once

#include "train/dataset.h"
#include "search/search.h"

#include <cstdint>

namespace chess::train {

struct SelfPlayConfig {
    int          games      = 100;
    int          maxPlies   = 200;
    int          randomPlies = 8;     // random opening plies for diversity
    // Beyond the fixed randomPlies-ply opening prefix, move selection was
    // previously fully deterministic (always engine.think()'s single best
    // move) for the rest of the game. That's fine while the accumulated
    // position database is small, but once it grows large (observed in
    // practice: DATA_GENERATION batches coming back 100% duplicate at
    // ~500K stored positions even with a generous randomPlies), independent
    // games' differing openings increasingly transpose back into a shared
    // position -- and from there, deterministic search means every
    // subsequent move (and therefore every subsequent recorded position) is
    // byte-identical between them, no matter how different the opening was.
    // randomMoveProb is the fix: at every ply *after* the opening prefix too
    // (not just during it), roll this probability each ply; on a hit, play
    // a uniformly random legal move instead of running the search (not
    // recorded as a training sample, same as an opening-phase random move,
    // since its position wasn't actually evaluated by real search). This
    // reintroduces divergence throughout the whole game, not just at the
    // start, which is what actually breaks the long deterministic tails
    // causing the duplicate batches. 0.0 (default) preserves the exact old
    // behavior. Kept deliberately small by convention where enabled (e.g.
    // 0.03) -- too high starts trading away realistic play quality for
    // diversity, since these injected moves are position-blind, not
    // reduced-depth-search-informed.
    double       randomMoveProb = 0.0;
    SearchLimits limits;              // per-move search budget (depth/nodes/time)
    std::uint64_t seed      = 0xC0FFEE;
    bool         verbose    = false;
};

// Play `cfg.games` self-play games and append labelled positions to `out`.
// Returns the number of samples generated.
std::size_t generate_selfplay(const SelfPlayConfig& cfg, Dataset& out);

} // namespace chess::train
