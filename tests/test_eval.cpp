// test_eval.cpp - Classical evaluation: symmetry, incremental accumulator,
// tapering sanity, and sensible move selection.
#include "core/position.h"
#include "core/movegen.h"
#include "eval/evaluate.h"
#include "search/search.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace chess;

namespace {
int failures = 0;
void check(bool cond, const std::string& what) {
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!cond) ++failures;
}

// Vertically mirror a FEN and swap colors (eval must be invariant from the
// side-to-move's perspective).
std::string mirror_fen(const std::string& fen) {
    std::istringstream ss(fen);
    std::string board, stm, castle, ep, rest;
    ss >> board >> stm >> castle >> ep;
    std::string hm = "0", fm = "1";
    ss >> hm >> fm;

    std::vector<std::string> ranks;
    std::string r;
    std::istringstream bs(board);
    while (std::getline(bs, r, '/')) ranks.push_back(r);
    std::reverse(ranks.begin(), ranks.end());
    std::string nb;
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        for (char c : ranks[i])
            nb += char(std::islower((unsigned char)c) ? std::toupper(c)
                       : std::isupper((unsigned char)c) ? std::tolower(c) : c);
        if (i + 1 < ranks.size()) nb += '/';
    }

    std::string ns = (stm == "w") ? "b" : "w";
    std::string nc;
    if (castle == "-") nc = "-";
    else for (char c : castle)
        nc += char(std::islower((unsigned char)c) ? std::toupper(c) : std::tolower(c));

    std::string ne = "-";
    if (ep != "-" && ep.size() == 2)
        ne = std::string{ep[0], char('1' + ('8' - ep[1]))};

    return nb + " " + ns + " " + nc + " " + ne + " " + hm + " " + fm;
}

Score negate(Score s) { return Score{-s.mg, -s.eg}; }
} // namespace

int main() {
    Position::init_static();

    const std::vector<std::string> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
        "8/2k5/3p4/p2P1p2/P2P1P2/8/8/4K3 w - - 0 1",
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    };

    // 1. Mirror symmetry: eval invariant under color/rank flip.
    for (const auto& fen : fens) {
        Position a, b;
        a.set(fen);
        b.set(mirror_fen(fen));
        check(evaluate(a) == evaluate(b), "mirror symmetry: " + fen);
        check(a.psq() == negate(b.psq()), "psq accumulator mirrors: " + fen);
    }

    // 2. Start position: only the tempo term remains (=18 cp).
    {
        Position p;
        p.set(fens[0]);
        check(int(evaluate(p)) == 18, "startpos eval equals tempo (18 cp)");
    }

    // 3. Incremental accumulator matches a from-scratch recompute after moves.
    {
        Position p;
        p.set(fens[1]);
        MoveList list;
        generate(p, list, LEGAL);
        bool ok = true;
        for (const auto& sm : list) {
            p.do_move(sm.move);
            Position fresh;
            fresh.set(p.fen());
            if (p.psq() != fresh.psq() || p.pawn_key() != fresh.pawn_key()) ok = false;
            p.undo_move(sm.move);
            if (!ok) break;
        }
        check(ok, "incremental psq + pawn key match from-scratch after each move");
    }

    // 4. Material understanding: up a clean rook is clearly winning.
    {
        Position p;
        p.set("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
        check(int(evaluate(p)) > 350, "extra rook evaluated as winning");
    }

    // 5. Sensible opening: search prefers a central / developing move.
    {
        Position p;
        p.set(fens[0]);
        Search s;
        SearchLimits lim; lim.depth = 9;
        SearchResult r = s.think(p, lim);
        const std::array<std::string, 7> good =
            {"e2e4", "d2d4", "g1f3", "c2c4", "e2e3", "d2d3", "b1c3"};
        const std::string mv = move_to_uci(r.best);
        check(std::find(good.begin(), good.end(), mv) != good.end(),
              "startpos search picks a principled move (got " + mv + ")");
    }

    std::cout << "\n" << (failures == 0 ? "ALL EVAL TESTS PASSED"
                                        : std::to_string(failures) + " EVAL FAILURE(S)")
              << "\n";
    return failures == 0 ? 0 : 1;
}
