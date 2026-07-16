// test_opening_book.cpp - engine-analysis opening book (src/book/).
//
// Covers: round-trip save/load of the binary format, hash-based probing
// (including out-of-book and illegal-move-filtered cases), best-move
// selection determinism, and randomness selection staying within its
// score-loss tolerance window and being reproducible for a fixed seed.
// Does NOT test tools/opening_book/generate_book.py (a Python script,
// exercised separately -- see docs/opening_book.md's "reproduce results"
// section for the manual/CI invocation).
#include "book/opening_book.h"
#include "core/position.h"
#include "core/movegen.h"

#include <cstdio>
#include <iostream>
#include <string>

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

    const std::string path = "test_opening_book.bin";

    // 1. Build a tiny two-ply book in memory: startpos -> e4 (best) / d4
    //    (alternative, lower eval), then after e4 -> e5.
    Position start;
    start.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Position afterE4 = start;
    afterE4.do_move(Move(SQ_E2, SQ_E4));

    {
        OpeningBook book;
        BookMove e4{};
        e4.move = Move(SQ_E2, SQ_E4); e4.evalCp = 35; e4.depth = 20;
        e4.visits = 5; e4.confidence = 100; e4.frequency = 5;
        book.add(start.key(), e4);

        BookMove d4{};
        d4.move = Move(SQ_D2, SQ_D4); d4.evalCp = 30; d4.depth = 20;
        d4.visits = 3; d4.confidence = 100; d4.frequency = 3;
        book.add(start.key(), d4);

        BookMove e5{};
        e5.move = Move(SQ_E7, SQ_E5); e5.evalCp = -10; e5.depth = 18;
        e5.visits = 4; e5.confidence = 90; e5.frequency = 4;
        book.add(afterE4.key(), e5);

        book.finalize();
        check(book.save(path), "save() writes a book file");
    }

    // 2. Load it back and confirm every field round-trips exactly.
    OpeningBook loaded;
    check(OpeningBook::looks_like_book_file(path), "looks_like_book_file detects our own format");
    check(loaded.load(path), "load() reads the file back");
    check(loaded.size() == 3, "loaded entry count matches what was written");

    auto cand = loaded.probe(start);
    check(cand.size() == 2, "startpos probe returns both candidate moves");
    check(!cand.empty() && cand.front().move == Move(SQ_E2, SQ_E4),
          "probe() sorts strongest eval first (e4 before d4)");
    check(!cand.empty() && cand.front().evalCp == 35, "evalCp round-trips exactly");
    check(!cand.empty() && cand.front().depth == 20, "depth round-trips exactly");
    check(!cand.empty() && cand.front().visits == 5, "visits round-trips exactly");
    check(!cand.empty() && cand.front().confidence == 100, "confidence round-trips exactly");
    check(!cand.empty() && cand.front().frequency == 5, "frequency round-trips exactly");

    auto cand2 = loaded.probe(afterE4);
    check(cand2.size() == 1 && cand2.front().move == Move(SQ_E7, SQ_E5),
          "probe() after e4 finds the e5 reply");

    // 3. Out-of-book position (never added) returns empty, not a crash/garbage move.
    Position elsewhere;
    elsewhere.set("8/8/8/3k4/8/8/8/Q3K3 w - - 0 1");
    check(loaded.probe(elsewhere).empty(), "out-of-book position returns no candidates");

    // 4. A stored move that is no longer legal in a *different* position with
    //    the same piece layout minus castling rights would still share a
    //    different hash (castling rights are part of the key), so instead we
    //    directly check the illegal-move-filtering path: probing a position
    //    that legitimately has 2 legal moves stored but where one has been
    //    invalidated is exercised implicitly by every probe() call already
    //    (it always intersects against live movegen) -- covered by test 2/3
    //    above; this case explicitly checks a corrupt/hand-edited hash
    //    collision-style entry is dropped rather than returned.
    {
        OpeningBook trap;
        BookMove bogus{};
        bogus.move = Move(SQ_A1, SQ_A8);  // illegal from the startpos (blocked, and not even
                                           // a legal-shape move for whatever piece a real
                                           // collision might imply)
        bogus.evalCp = 999;
        trap.add(start.key(), bogus);
        trap.finalize();
        check(trap.save(path), "save() writes the trap book");
        OpeningBook loadedTrap;
        check(loadedTrap.load(path), "trap book loads");
        check(loadedTrap.probe(start).empty(),
              "an illegal stored move is filtered out, never returned");
    }

    // 5. select_book_move: randomness=0 is always the best move, deterministically.
    check(select_book_move(cand, 0, 12345) == Move(SQ_E2, SQ_E4),
          "randomness=0 always picks the best move");
    check(select_book_move(cand, 0, 999) == Move(SQ_E2, SQ_E4),
          "randomness=0 ignores the seed entirely (fully deterministic)");

    // 6. select_book_move: same seed -> same pick, reproducibly, at any
    //    randomness level (this is the "deterministic mode reproduces the
    //    same choice" contract -- determinism comes from seeding by
    //    position, not from a special mode flag).
    {
        Move a = select_book_move(cand, 50, 0xABCDEF);
        Move b = select_book_move(cand, 50, 0xABCDEF);
        check(a == b, "same seed reproduces the same randomized pick");
    }

    // 7. select_book_move never returns a move outside the candidate set,
    //    across a spread of seeds (a cheap randomized-but-bounded check).
    {
        bool allValid = true;
        for (std::uint64_t seed = 0; seed < 200; ++seed) {
            Move m = select_book_move(cand, 100, seed);
            if (m != Move(SQ_E2, SQ_E4) && m != Move(SQ_D2, SQ_D4)) { allValid = false; break; }
        }
        check(allValid, "select_book_move always returns one of the stored candidates");
    }

    // 8. A large score gap excludes the weaker move even at max randomness.
    {
        std::vector<BookMove> wide;
        BookMove best{}; best.move = Move(SQ_E2, SQ_E4); best.evalCp = 300; best.frequency = 1;
        BookMove bad{};  bad.move  = Move(SQ_D2, SQ_D4); bad.evalCp  = -50; bad.frequency = 100;
        wide.push_back(best); wide.push_back(bad);
        bool everPickedBad = false;
        for (std::uint64_t seed = 0; seed < 500; ++seed)
            if (select_book_move(wide, 100, seed) == bad.move) { everPickedBad = true; break; }
        check(!everPickedBad,
              "a move far below the best eval is never selected, even with high weight/randomness");
    }

    std::remove(path.c_str());

    std::cout << "\n" << (failures == 0 ? "ALL OPENING BOOK TESTS PASSED"
                                        : std::to_string(failures) + " OPENING BOOK FAILURE(S)")
              << "\n";
    return failures == 0 ? 0 : 1;
}
