// selftest.cpp - In-engine self-tests for Phase 5 infrastructure.
#include "test/selftest.h"

#include "core/position.h"
#include "core/movegen.h"
#include "io/pgn.h"
#include "io/book.h"
#include "syzygy/tablebases.h"
#include "match/stats.h"
#include "match/match.h"
#include "eval/evaluate.h"
#include "eval/psqt.h"
#include "nnue/nnue.h"
#include "train/dataset.h"
#include "train/encoding.h"
#include "train/selfplay.h"
#include "train/trainer.h"

#include <cmath>
#include <cstring>

#include <cstdio>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>

namespace chess {

namespace {

struct Ctx {
    std::ostream& os;
    int failures = 0;
    void check(bool cond, const std::string& what) {
        os << (cond ? "  [PASS] " : "  [FAIL] ") << what << "\n";
        if (!cond) ++failures;
    }
};

// -------------------------------------------------------------------------
// PGN / SAN
// -------------------------------------------------------------------------
void test_pgn(Ctx& c) {
    c.os << "[PGN / SAN]\n";
    Position p;
    p.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    c.check(move_to_san(p, Move(SQ_E2, SQ_E4)) == "e4", "pawn SAN e4");
    c.check(move_to_san(p, Move(SQ_G1, SQ_F3)) == "Nf3", "knight SAN Nf3");

    p.set("8/8/8/8/3N1N2/8/8/k1K5 w - - 0 1");
    c.check(move_to_san(p, Move(SQ_D4, SQ_E6)) == "Nde6", "disambiguation Nde6");
    c.check(san_to_move(p, "Nfe6") == Move(SQ_F4, SQ_E6), "parse Nfe6");

    p.set("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    c.check(move_to_san(p, Move::make<CASTLING>(SQ_E1, SQ_G1)) == "O-O", "castling O-O");
    c.check(san_to_move(p, "0-0") == Move::make<CASTLING>(SQ_E1, SQ_G1), "parse 0-0");

    // Round-trip every legal move at a complex position.
    p.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    {
        MoveList list; generate(p, list, LEGAL);
        bool ok = true;
        for (const auto& sm : list)
            if (san_to_move(p, move_to_san(p, sm.move)) != sm.move) { ok = false; break; }
        c.check(ok, "SAN round-trips for all legal moves");
    }

    // PGN write -> read.
    {
        const char* sans[] = {"e4","e5","Nf3","Nc6","Bb5","a6","Ba4","Nf6","O-O","Be7"};
        Position q; q.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        GameRecord g;
        for (const char* s : sans) { Move m = san_to_move(q, s); g.moves.push_back(m); q.do_move(m); }
        g.result = "1/2-1/2"; g.tags["White"] = "EngineA";
        std::ostringstream out; write_pgn(out, g);
        std::istringstream in(out.str()); GameRecord r;
        bool read = read_pgn(in, r);
        c.check(read && r.moves == g.moves, "PGN write/read preserves moves");
        c.check(r.result == "1/2-1/2", "PGN preserves result");
        c.check(r.tags["White"] == "EngineA", "PGN preserves tags");
    }
}

// -------------------------------------------------------------------------
// Opening book (generate from PGN -> load -> probe).
// -------------------------------------------------------------------------
void test_book(Ctx& c) {
    c.os << "[Opening book]\n";
    book_init();

    const std::string pgn =
        "[Event \"t\"]\n\n1. e4 e5 2. Nf3 Nc6 *\n\n"
        "[Event \"t\"]\n\n1. e4 e5 2. Nf3 Nc6 *\n\n"
        "[Event \"t\"]\n\n1. d4 d5 *\n\n";
    const std::string path = "selftest_book.bin";

    std::istringstream in(pgn);
    bool built = build_book_from_pgn(in, path, 8);
    c.check(built, "build_book_from_pgn writes a book");

    Book book;
    c.check(book.load(path), "book loads");

    Position p;
    p.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move first = book.probe(p, /*pickBest=*/true);
    c.check(first == Move(SQ_E2, SQ_E4),
            "book picks most-weighted first move (e4)");

    p.do_move(Move(SQ_E2, SQ_E4));
    c.check(book.probe(p) == Move(SQ_E7, SQ_E5), "book replies e5 after e4");
    p.do_move(Move(SQ_E7, SQ_E5));
    c.check(book.probe(p) == Move(SQ_G1, SQ_F3), "book plays Nf3");

    // Out-of-book position falls back gracefully.
    Position q;
    q.set("8/8/8/3k4/8/8/8/Q3K3 w - - 0 1");
    c.check(book.probe(q) == Move::none(), "out-of-book probe returns none");

    std::remove(path.c_str());
}

// -------------------------------------------------------------------------
// Syzygy framework (graceful fallback when no decoder/files).
// -------------------------------------------------------------------------
void test_syzygy(Ctx& c) {
    c.os << "[Syzygy framework]\n";
    Tablebases::init("<empty>");
    c.check(!Tablebases::available(), "tablebases unavailable without files/decoder");
    c.check(Tablebases::max_pieces() == 0, "max_pieces is 0 when unavailable");

    Position p;
    p.set("8/8/8/3k4/8/8/8/Q3K3 w - - 0 1");
    c.check(Tablebases::probe_wdl(p) == WDLResult::Fail, "probe fails gracefully");
    c.check(!Tablebases::status().empty(), "status string is reported");

    // A bogus path must not enable probing.
    Tablebases::init("C:/does/not/exist");
    c.check(!Tablebases::available(), "bogus path stays unavailable");
}

// -------------------------------------------------------------------------
// Match statistics: Elo + SPRT.
// -------------------------------------------------------------------------
void test_stats(Ctx& c) {
    c.os << "[Match stats]\n";
    c.check(std::abs(elo_diff(0.5)) < 1e-6, "elo_diff(0.5) == 0");
    c.check(elo_diff(0.75) > 150 && elo_diff(0.75) < 250, "elo_diff(0.75) ~ 191");

    EloEstimate even = elo_estimate(50, 50, 0);
    c.check(std::abs(even.elo) < 1e-6 && even.margin > 0, "even score -> 0 Elo with margin");
    EloEstimate strong = elo_estimate(80, 20, 0);
    c.check(strong.elo > 200, "80/20 -> clearly positive Elo");

    SprtResult sStrong = sprt(80, 20, 0, 0.0, 50.0);
    c.check(sStrong.llr > 0, "SPRT LLR positive when A strong");
    SprtResult sEven = sprt(50, 50, 0, 0.0, 50.0);
    c.check(sEven.llr < 0, "SPRT LLR negative when results favor H0");
    c.check(sStrong.upperBound > 0 && sStrong.lowerBound < 0, "SPRT bounds bracket zero");
}

// -------------------------------------------------------------------------
// Self-play match: games complete and produce legally-replayable PGN.
// -------------------------------------------------------------------------
void test_match(Ctx& c) {
    c.os << "[Self-play match]\n";
    EngineConfig a, b;
    a.name = "A"; b.name = "B";
    a.limits.depth = 3; b.limits.depth = 3;
    MatchSettings s;
    s.games = 4;
    s.maxPlies = 60;

    MatchResult mr = play_match(a, b, s);
    const int n = mr.aWins + mr.bWins + mr.draws;
    c.check(n == s.games, "all games produced a result");
    c.check(int(mr.games.size()) == s.games, "PGN record per game");

    // Every recorded game must replay legally to its end.
    bool replayOk = true;
    for (const auto& g : mr.games) {
        Position p; p.set(g.startFen);
        for (Move m : g.moves) {
            MoveList legal; generate(p, legal, LEGAL);
            bool found = false;
            for (const auto& sm : legal) if (sm.move == m) { found = true; break; }
            if (!found) { replayOk = false; break; }
            p.do_move(m);
        }
        if (!replayOk) break;
    }
    c.check(replayOk, "all match games replay legally");
}

// -------------------------------------------------------------------------
// NNUE: encoding correctness, incremental accumulator, SIMD, weight I/O.
// -------------------------------------------------------------------------
void test_nnue(Ctx& c) {
    c.os << "[NNUE]\n";
    const EvalMode prev = Eval::mode();
    Eval::set_mode(EvalMode::NNUE);

    const char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const char* kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    const char* fens[] = {startpos, kiwipete,
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9"};

    // 1. The material-initialized dual-perspective net outputs the side-to-move
    //    material balance; and the SIMD path matches the scalar reference.
    bool eqAll = true, simdAll = true;
    for (const char* fen : fens) {
        Position p; p.set(fen);
        int matW = 0;
        for (PieceType pt = PAWN; pt <= QUEEN; ++pt)
            matW += PieceValue[pt] * (p.count(WHITE, pt) - p.count(BLACK, pt));
        const int ref = (p.side_to_move() == WHITE) ? matW : -matW;
        const Color stm = p.side_to_move();
        const int out = NNUE::output(p.accumulator(), stm, NNUE::output_bucket(p));
        if (out != ref) eqAll = false;
        if (NNUE::output_scalar(p.accumulator(), stm, NNUE::output_bucket(p)) != out) simdAll = false;
    }
    c.check(eqAll, "NNUE output == side-to-move material (material-init net)");
    c.check(simdAll, "SIMD inference == scalar inference");

    // 2. Incremental accumulator matches a full refresh after every move,
    //    and is restored after unmake.
    {
        Position p; p.set(kiwipete);
        MoveList list; generate(p, list, LEGAL);
        bool incOk = true;
        for (const auto& sm : list) {
            p.do_move(sm.move);
            Accumulator tmp; NNUE::refresh(p, tmp);
            if (std::memcmp(tmp.v, p.accumulator().v, sizeof(tmp.v)) != 0) incOk = false;
            p.undo_move(sm.move);
            if (!incOk) break;
        }
        c.check(incOk, "incremental accumulator == full refresh after each move");
        Accumulator root; NNUE::refresh(p, root);
        c.check(std::memcmp(root.v, p.accumulator().v, sizeof(root.v)) == 0,
                "accumulator restored after unmake");
    }

    // 3. Weight save/load round-trip preserves evaluation.
    {
        Position p; p.set(kiwipete);
        const Color stm = p.side_to_move();
        const std::string path = "selftest_net.nnue";
        c.check(NNUE::save(path), "NNUE weights saved");
        const int before = NNUE::output(p.accumulator(), stm, NNUE::output_bucket(p));
        c.check(NNUE::load(path), "NNUE weights loaded");
        c.check(NNUE::output(p.accumulator(), stm, NNUE::output_bucket(p)) == before, "save/load preserves eval");
        std::remove(path.c_str());
    }

    // 4. Sanity: balanced start, decisive material edge.
    {
        Position s1; s1.set(startpos);
        c.check(std::abs(NNUE::output(s1.accumulator(), s1.side_to_move(), NNUE::output_bucket(s1))) < 60,
                "startpos NNUE ~ balanced");
        Position s2; s2.set("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
        c.check(NNUE::output(s2.accumulator(), s2.side_to_move(), NNUE::output_bucket(s2)) > 300,
                "up a rook -> positive NNUE");
    }

    // 5. NNUE-guided search solves tactics (synchronous; no UCI worker thread).
    {
        struct T { const char* fen; const char* best; };
        // Tactics solvable by a material-aware evaluator (the untrained
        // midgame-PSQT net lacks king-safety terms, so deep positional
        // sacrifices like WAC.001 Qg6 are out of its reach until trained).
        const T tac[] = {
            {"6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"},          // back-rank mate
            {"4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1", "d1d4"},          // win the queen
        };
        for (const T& t : tac) {
            Position p; p.set(t.fen);
            Search s; s.set_quiet(true);
            SearchLimits lim; lim.depth = 11;
            SearchResult r = s.think(p, lim);
            c.check(move_to_uci(r.best) == t.best,
                    std::string("NNUE search solves ") + t.fen
                    + " (want " + t.best + ", got " + move_to_uci(r.best) + ")");
        }
    }

    Eval::set_mode(prev);
}

// -------------------------------------------------------------------------
// Lazy SMP: legality consistency 1 vs N threads, determinism single-thread.
// -------------------------------------------------------------------------
void test_smp(Ctx& c) {
    c.os << "[Lazy SMP]\n";
    const char* fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    auto is_legal_move = [&](Move m) {
        Position p; p.set(fen);
        MoveList l; generate(p, l, LEGAL);
        for (const auto& sm : l) if (sm.move == m) return true;
        return false;
    };

    SearchLimits lim; lim.depth = 10;

    // Single-thread determinism (reproducible from a clean state: the TT is
    // cleared between runs so both start cold, as in a fresh search).
    Search s1; s1.set_quiet(true); s1.set_threads(1);
    s1.clear(); Position pa; pa.set(fen); s1.arm(); SearchResult r1 = s1.think(pa, lim);
    s1.clear(); Position pb; pb.set(fen); s1.arm(); SearchResult r2 = s1.think(pb, lim);
    c.check(r1.best == r2.best && r1.score == r2.score, "1-thread search is deterministic");
    c.check(is_legal_move(r1.best), "1-thread best move is legal");

    // Multi-thread: must not crash/corrupt and must return a legal move.
    for (int n : {2, 4}) {
        Search s; s.set_quiet(true); s.set_threads(n);
        Position p; p.set(fen); s.arm();
        SearchResult r = s.think(p, lim);
        c.check(is_legal_move(r.best),
                std::to_string(n) + "-thread best move is legal");
        c.check(s.nodes() > 0, std::to_string(n) + "-thread visited nodes");
        c.check(std::abs(int(r.score)) <= VALUE_INFINITE,
                std::to_string(n) + "-thread score in range");
    }
}

// -------------------------------------------------------------------------
// Training subsystem: self-play datagen, dataset I/O, reference training,
// checkpoint integrity. (Research-only; does not touch NNUE/search behavior.)
// -------------------------------------------------------------------------
void test_train(Ctx& c) {
    c.os << "[Training subsystem]\n";
    using namespace chess::train;

    // 1. Self-play generation.
    SelfPlayConfig sp;
    sp.games = 6; sp.maxPlies = 60; sp.randomPlies = 4;
    sp.limits.depth = 4;
    Dataset ds;
    std::size_t n = generate_selfplay(sp, ds);
    c.check(n > 0 && ds.size() == n, "self-play produced labelled samples");

    // 2. Dataset correctness: legal positions, sane labels.
    bool ok = true;
    Position p;
    for (std::size_t i = 0; i < ds.size(); ++i) {
        const Sample& s = ds[i];
        p.set(s.fen);
        if (p.fen() != s.fen) ok = false;                       // round-trips
        if (s.result < -1 || s.result > 1) ok = false;          // valid outcome
        if (std::abs(int(s.eval)) > 32000) ok = false;          // sane eval
    }
    c.check(ok, "every sample has a legal position and valid labels");

    // 3. Dataset save/load round-trip.
    {
        const std::string path = "selftest_data.dat";
        c.check(ds.save(path), "dataset saved");
        Dataset re;
        c.check(re.load(path) && re.size() == ds.size(), "dataset reloaded");
        bool same = re.size() == ds.size();
        for (std::size_t i = 0; same && i < ds.size(); ++i)
            if (re[i].fen != ds[i].fen || re[i].eval != ds[i].eval
                || re[i].result != ds[i].result) same = false;
        c.check(same, "dataset round-trip preserves all samples");
        std::remove(path.c_str());
    }

    // 4. Reference training reduces loss.
    TrainConfig tc;
    tc.epochs = 15; tc.hidden = 32; tc.verbose = false; tc.valFraction = 0.2f;
    RefTrainer trainer(tc.hidden, 1);
    TrainReport rep = trainer.train(ds, tc);
    c.check(rep.finalTrainLoss < rep.initialTrainLoss,
            "training loss decreases (" + std::to_string(rep.initialTrainLoss)
            + " -> " + std::to_string(rep.finalTrainLoss) + ")");

    // 5. Checkpoint save/load integrity (identical predictions after reload).
    {
        const std::string ck = "selftest_model.ckpt";
        c.check(trainer.save(ck), "checkpoint saved");
        RefTrainer reloaded(tc.hidden, 999);   // different init
        c.check(reloaded.load(ck), "checkpoint loaded");
        Position q; q.set(ds[0].fen);
        auto feat = encode_sparse(q);
        c.check(std::abs(reloaded.predict(feat) - trainer.predict(feat)) < 1e-6f,
                "reloaded model predicts identically");
        std::remove(ck.c_str());
    }
}

// -------------------------------------------------------------------------
// Phase 6 integration: NNUE+SMP together, and the full self-play -> train ->
// distill -> load -> play workflow producing usable NNUE weights.
// -------------------------------------------------------------------------
void test_integration(Ctx& c) {
    c.os << "[Phase 6 integration]\n";
    using namespace chess::train;
    const char* kiwi = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    auto legal_in = [&](const char* fen, Move m) {
        Position p; p.set(fen);
        MoveList l; generate(p, l, LEGAL);
        for (const auto& sm : l) if (sm.move == m) return true;
        return false;
    };
    const EvalMode prev = Eval::mode();

    // 1. Multithreaded Lazy SMP search driven by the NNUE evaluator.
    {
        Eval::set_mode(EvalMode::NNUE);
        Search s; s.set_quiet(true); s.set_threads(4);
        Position p; p.set(kiwi); SearchLimits lim; lim.depth = 9; s.arm();
        SearchResult r = s.think(p, lim);
        c.check(legal_in(kiwi, r.best), "NNUE + 4-thread search returns a legal move");
        Eval::set_mode(prev);
    }

    // 2. New NNUE architecture: incremental accumulator == full refresh after a
    //    sequence of moves (incl. king moves) — the key correctness property.
    {
        Eval::set_mode(EvalMode::NNUE);
        Position q; q.set("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        MoveList l; generate(q, l, LEGAL);
        bool ok = true;
        for (const auto& sm : l) {
            q.do_move(sm.move);
            Accumulator fresh;
            NNUE::refresh(q, fresh);                 // recompute from scratch
            const Accumulator& inc = q.accumulator(); // incrementally maintained
            for (int c = 0; c < 2 && ok; ++c)
                for (int h = 0; h < NNUE_HL; ++h)
                    if (inc.v[c][h] != fresh.v[c][h]) { ok = false; break; }
            // Also exercise a king move two plies deep.
            MoveList l2; generate(q, l2, LEGAL);
            if (!l2.empty()) {
                q.do_move(l2.begin()[0].move);
                Accumulator f2; NNUE::refresh(q, f2);
                const Accumulator& i2 = q.accumulator();
                for (int c = 0; c < 2 && ok; ++c)
                    for (int h = 0; h < NNUE_HL; ++h)
                        if (i2.v[c][h] != f2.v[c][h]) { ok = false; break; }
                q.undo_move(l2.begin()[0].move);
            }
            q.undo_move(sm.move);
            if (!ok) break;
        }
        c.check(ok, "NNUE incremental accumulator matches full refresh");
        Eval::set_mode(prev);
    }

    // 3. Material sanity: with the material-initialized net, being up a rook is
    //    evaluated as clearly winning for the side to move.
    {
        Eval::set_mode(EvalMode::NNUE);
        Position q; q.set("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");   // White: K+R vs k
        c.check(int(evaluate(q)) > 300, "NNUE (material init) sees an extra rook as winning");
        Eval::set_mode(prev);
    }

    // 4. Save / load round-trip preserves evaluation, and the engine plays
    //    legally with a loaded net (single-thread and 4-thread / SMP).
    {
        const std::string nn = "selftest_net.nnue";
        c.check(NNUE::save(nn), "NNUE saves new-format weights");
        c.check(NNUE::load(nn), "NNUE loads new-format weights");
        Eval::set_mode(EvalMode::NNUE);
        Position q; q.set(kiwi);
        const int e1 = int(evaluate(q));
        NNUE::init(); NNUE::load(nn);
        Position q2; q2.set(kiwi);
        c.check(int(evaluate(q2)) == e1, "NNUE eval identical after save/load");
        Search s; s.set_quiet(true); Position q3; q3.set(kiwi);
        SearchLimits lim; lim.depth = 8; s.arm();
        SearchResult r = s.think(q3, lim);
        c.check(legal_in(kiwi, r.best), "engine plays legally with loaded NNUE");
        Eval::set_mode(prev);
        std::remove(nn.c_str());
    }

    NNUE::init();                 // restore the default bundled net
}

} // namespace

int run_selftests(std::ostream& os) {
    Ctx c{os};
    test_pgn(c);
    test_book(c);
    test_syzygy(c);
    test_stats(c);
    test_match(c);
    test_nnue(c);
    test_smp(c);
    test_train(c);
    test_integration(c);
    os << (c.failures == 0 ? "SELFTEST: ALL PASSED"
                           : "SELFTEST: " + std::to_string(c.failures) + " FAILURE(S)")
       << std::endl;
    return c.failures;
}

} // namespace chess
