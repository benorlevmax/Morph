// test_perft.cpp - Validates move generation against known perft node counts.
#include "core/perft.h"

#include <iostream>
#include <string>
#include <vector>

using namespace chess;

namespace {

struct Case {
    std::string fen;
    std::vector<std::uint64_t> expected;  // expected[d-1] = perft(d)
    const char* name;
};

// Canonical perft test positions (Chess Programming Wiki).
const std::vector<Case> kCases = {
    { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      {20, 400, 8902, 197281, 4865609}, "startpos" },

    { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      {48, 2039, 97862, 4085603}, "kiwipete" },

    { "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      {14, 191, 2812, 43238, 674624}, "position3" },

    { "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
      {6, 264, 9467, 422333}, "position4" },

    { "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      {44, 1486, 62379, 2103487}, "position5" },

    { "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
      {46, 2079, 89890, 3894594}, "position6" },
};

} // namespace

int main() {
    Position::init_static();

    int failures = 0;
    Position pos;

    for (const auto& c : kCases) {
        pos.set(c.fen);
        for (std::size_t i = 0; i < c.expected.size(); ++i) {
            const int depth = int(i) + 1;
            const std::uint64_t got = perft(pos, depth);
            const std::uint64_t exp = c.expected[i];
            const bool ok = (got == exp);
            std::cout << (ok ? "[PASS] " : "[FAIL] ")
                      << c.name << " perft(" << depth << ") = " << got;
            if (!ok) {
                std::cout << "  expected " << exp;
                ++failures;
            }
            std::cout << "\n";
            if (!ok) break;   // deeper depths will also fail; move on
        }
    }

    std::cout << "\n" << (failures == 0 ? "ALL PERFT TESTS PASSED"
                                        : std::to_string(failures) + " PERFT FAILURE(S)")
              << "\n";
    return failures == 0 ? 0 : 1;
}
