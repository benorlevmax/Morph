// perft.cpp - Perft implementation.
#include "core/perft.h"
#include "core/movegen.h"

namespace chess {

std::uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    MoveList list;
    generate(pos, list, LEGAL);

    // Bulk counting at depth 1 (standard perft optimization).
    if (depth == 1) return list.size();

    std::uint64_t nodes = 0;
    for (const auto& sm : list) {
        pos.do_move(sm.move);
        nodes += perft(pos, depth - 1);
        pos.undo_move(sm.move);
    }
    return nodes;
}

std::pair<std::uint64_t, std::vector<std::pair<std::string, std::uint64_t>>>
perft_divide(Position& pos, int depth) {
    std::vector<std::pair<std::string, std::uint64_t>> breakdown;
    std::uint64_t total = 0;

    MoveList list;
    generate(pos, list, LEGAL);
    for (const auto& sm : list) {
        pos.do_move(sm.move);
        std::uint64_t n = depth <= 1 ? 1 : perft(pos, depth - 1);
        pos.undo_move(sm.move);
        breakdown.emplace_back(move_to_uci(sm.move), n);
        total += n;
    }
    return {total, breakdown};
}

} // namespace chess
