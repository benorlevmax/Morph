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
    SearchLimits limits;              // per-move search budget (depth/nodes/time)
    std::uint64_t seed      = 0xC0FFEE;
    bool         verbose    = false;
};

// Play `cfg.games` self-play games and append labelled positions to `out`.
// Returns the number of samples generated.
std::size_t generate_selfplay(const SelfPlayConfig& cfg, Dataset& out);

} // namespace chess::train
