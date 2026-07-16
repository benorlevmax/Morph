// test_pgn.cpp - SAN notation and PGN round-trip. (Phase 5 IO coverage.)
#include "core/position.h"
#include "core/movegen.h"
#include "io/pgn.h"

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
} // namespace

int main() {
    Position::init_static();

    // 1. Basic SAN generation.
    {
        Position p;
        p.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        check(move_to_san(p, Move(SQ_E2, SQ_E4)) == "e4", "pawn move SAN e4");
        check(move_to_san(p, Move(SQ_G1, SQ_F3)) == "Nf3", "knight move SAN Nf3");
    }

    // 2. Disambiguation by file: two knights (d4,f4) both reach e6 -> Nde6.
    {
        Position p;
        p.set("8/8/8/8/3N1N2/8/8/k1K5 w - - 0 1");
        check(move_to_san(p, Move(SQ_D4, SQ_E6)) == "Nde6", "knight disambiguation Nde6");
        check(san_to_move(p, "Nde6") == Move(SQ_D4, SQ_E6), "parse Nde6");
        check(san_to_move(p, "Nfe6") == Move(SQ_F4, SQ_E6), "parse Nfe6");
    }

    // 3. Castling + check SAN.
    {
        Position p;
        p.set("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        check(move_to_san(p, Move::make<CASTLING>(SQ_E1, SQ_G1)) == "O-O", "kingside O-O");
        check(move_to_san(p, Move::make<CASTLING>(SQ_E1, SQ_C1)) == "O-O-O", "queenside O-O-O");
        check(san_to_move(p, "0-0") == Move::make<CASTLING>(SQ_E1, SQ_G1), "parse 0-0 spelling");
    }

    // 4. san_to_move round-trips every legal move at a complex position.
    {
        Position p;
        p.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        MoveList list;
        generate(p, list, LEGAL);
        bool ok = true;
        for (const auto& sm : list)
            if (san_to_move(p, move_to_san(p, sm.move)) != sm.move) { ok = false; break; }
        check(ok, "SAN <-> move round-trips for all legal moves");
    }

    // 5. PGN write then read reproduces the move list.
    {
        const char* sans[] = {"e4","e5","Nf3","Nc6","Bb5","a6","Ba4","Nf6","O-O","Be7"};
        Position p;
        p.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        GameRecord g;
        for (const char* s : sans) {
            Move m = san_to_move(p, s);
            g.moves.push_back(m);
            p.do_move(m);
        }
        g.result = "1/2-1/2";
        g.tags["White"] = "EngineA";
        g.tags["Black"] = "EngineB";

        std::ostringstream os;
        write_pgn(os, g);

        std::istringstream is(os.str());
        GameRecord r;
        bool read = read_pgn(is, r);
        check(read, "read_pgn returns a game");
        check(r.moves == g.moves, "PGN round-trip preserves moves");
        check(r.result == "1/2-1/2", "PGN round-trip preserves result");
        check(r.tags["White"] == "EngineA", "PGN round-trip preserves tags");
    }

    std::cout << "\n" << (failures == 0 ? "ALL PGN TESTS PASSED"
                                        : std::to_string(failures) + " PGN FAILURE(S)")
              << "\n";
    return failures == 0 ? 0 : 1;
}
