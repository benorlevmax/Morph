// perft.h - Perft node counting for movegen validation.
#pragma once

#include "core/position.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace chess {

// Count leaf nodes at the given depth.
std::uint64_t perft(Position& pos, int depth);

// Perft with per-root-move breakdown (UCI 'go perft' style output).
std::pair<std::uint64_t, std::vector<std::pair<std::string, std::uint64_t>>>
perft_divide(Position& pos, int depth);

} // namespace chess
