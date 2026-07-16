// test_fen.cpp - Validates FEN round-trips and Zobrist key consistency.
#include "core/position.h"
#include "core/movegen.h"

#include <iostream>
#include <string>
#include <vector>

using namespace chess;

namespace {
int failures = 0;

void check(bool cond, const std::string& what) {
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (!cond) ++failures;
}
} // namespace

int main() {
    Position::init_static();

    const std::vector<std::string> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    };

    Position pos;
    for (const auto& fen : fens) {
        pos.set(fen);
        check(pos.fen() == fen, "FEN round-trip: " + fen);
    }

    // Zobrist key must be restored after do/undo of every legal move.
    pos.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    const Key before = pos.key();
    MoveList list;
    generate(pos, list, LEGAL);
    bool keyOk = true;
    for (const auto& sm : list) {
        pos.do_move(sm.move);
        pos.undo_move(sm.move);
        if (pos.key() != before) { keyOk = false; break; }
    }
    check(keyOk, "Zobrist key restored after do/undo for all legal moves");

    // Null move must also restore the key.
    const Key k2 = pos.key();
    pos.do_null_move();
    pos.undo_null_move();
    check(pos.key() == k2, "Zobrist key restored after null move");

    // Recomputing the key from scratch (via set) must match incremental key.
    pos.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    Key incremental = pos.key();
    pos.do_move(*list.begin());
    std::string afterFen = pos.fen();
    Position fresh;
    fresh.set(afterFen);
    check(fresh.key() == pos.key(), "Incremental key matches from-scratch key after a move");
    (void)incremental;

    std::cout << "\n" << (failures == 0 ? "ALL FEN TESTS PASSED"
                                        : std::to_string(failures) + " FEN FAILURE(S)")
              << "\n";
    return failures == 0 ? 0 : 1;
}
