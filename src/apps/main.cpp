// main.cpp - Perft CLI driver.
//
// Usage:
//   chess_perft                         -> perft(6) on the start position
//   chess_perft <depth>                 -> perft(depth) on the start position
//   chess_perft <depth> "<fen>"         -> perft(depth) on a custom FEN
//   chess_perft divide <depth> ["fen"]  -> per-move breakdown
#include "core/perft.h"

#include <chrono>
#include <iostream>
#include <string>

using namespace chess;

int main(int argc, char** argv) {
    Position::init_static();

    const std::string startFen =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    bool divide = false;
    int depth = 6;
    std::string fen = startFen;
    int argIdx = 1;

    if (argc > argIdx && std::string(argv[argIdx]) == "divide") {
        divide = true;
        ++argIdx;
    }
    if (argc > argIdx) depth = std::stoi(argv[argIdx++]);
    if (argc > argIdx) fen = argv[argIdx++];

    Position pos;
    pos.set(fen);
    std::cout << pos.to_string() << "\nPerft depth " << depth << "\n";

    const auto t0 = std::chrono::steady_clock::now();

    std::uint64_t total;
    if (divide) {
        auto [tot, breakdown] = perft_divide(pos, depth);
        for (const auto& [mv, n] : breakdown)
            std::cout << mv << ": " << n << "\n";
        total = tot;
    } else {
        total = perft(pos, depth);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\nNodes: " << total << "\n";
    std::cout << "Time:  " << secs << " s\n";
    if (secs > 0)
        std::cout << "Speed: " << std::uint64_t(total / secs) << " nps\n";
    return 0;
}
