// uci.cpp - Minimal UCI implementation.
#include "uci/uci.h"
#include "eval/evaluate.h"
#include "core/movegen.h"
#include "core/perft.h"
#include "test/selftest.h"
#include "syzygy/tablebases.h"
#include "nnue/nnue.h"

#include <iostream>
#include <sstream>
#include <string>

namespace chess {

namespace {
const char* kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
} // namespace

UCIEngine::UCIEngine() {
    Position::init_static();
    book_init();
    // NNUE is fully integrated and is the intended primary evaluator, but the
    // bundled net is only PSQT-initialized (untrained). Until a trained net is
    // available it is ~280 Elo weaker than the full classical eval, so the
    // shipped default stays Classical. Switch with `setoption name Use NNUE`.
    Eval::set_mode(EvalMode::Classical);
    pos_.set(kStartFen);
    search_.set_hash_size(hashMB_);
}

UCIEngine::~UCIEngine() {
    cmd_stop();
}

Move UCIEngine::parse_move(const std::string& uci) const {
    MoveList list;
    generate(pos_, list, LEGAL);
    for (const auto& sm : list)
        if (move_to_uci(sm.move) == uci)
            return sm.move;
    return Move::none();
}

void UCIEngine::start_search(const SearchLimits& limits) {
    wait_search();
    search_.arm();   // clear stop on this thread before launching the worker
    worker_ = std::thread([this, limits]() {
        SearchResult r = search_.think(pos_, limits);
        std::cout << "bestmove " << move_to_uci(r.best);
        // Only report a ponder move if it is legal in the position after the
        // best move (the PV's 2nd entry can be a stale tail from search).
        if (r.best != Move::none() && r.ponder != Move::none()) {
            Position after = pos_;
            after.do_move(r.best);
            MoveList legal;
            generate(after, legal, LEGAL);
            for (const auto& sm : legal)
                if (sm.move == r.ponder) {
                    std::cout << " ponder " << move_to_uci(r.ponder);
                    break;
                }
        }
        std::cout << std::endl;
    });
}

void UCIEngine::wait_search() {
    if (worker_.joinable())
        worker_.join();
}

void UCIEngine::cmd_uci() {
    std::cout << "id name Morph 0.5\n";
    std::cout << "id author benor\n";
    std::cout << "option name Hash type spin default 16 min 1 max 65536\n";
    std::cout << "option name Threads type spin default 1 min 1 max 256\n";
    std::cout << "option name MoveOverhead type spin default 50 min 0 max 5000\n";
    std::cout << "option name OwnBook type check default false\n";
    std::cout << "option name OpeningBook type check default false\n";
    // One shared BookFile option: a UCI option name must be unique, and the
    // loader auto-detects which of the two book formats `value` actually is
    // (see cmd_setoption). OwnBook/OpeningBook independently gate whichever
    // one ends up loaded.
    std::cout << "option name BookFile type string default <empty>\n";
    std::cout << "option name BookDepth type spin default 20 min 0 max 400\n";
    std::cout << "option name BookTimeLimit type spin default 0 min 0 max 5000\n";
    std::cout << "option name BookRandomness type spin default 0 min 0 max 100\n";
    std::cout << "option name SyzygyPath type string default <empty>\n";
    std::cout << "option name Use NNUE type check default false\n";
    std::cout << "option name EvalFile type string default <internal>\n";
    std::cout << "uciok" << std::endl;
}

void UCIEngine::cmd_isready() {
    std::cout << "readyok" << std::endl;
}

void UCIEngine::cmd_newgame() {
    cmd_stop();
    search_.clear();
    pos_.set(kStartFen);
}

void UCIEngine::cmd_setoption(std::istream& is) {
    std::string token, name, value;
    is >> token;                 // "name"
    while (is >> token && token != "value") {
        if (!name.empty()) name += ' ';
        name += token;
    }
    while (is >> token) {
        if (!value.empty()) value += ' ';
        value += token;
    }
    if (name == "Hash") {
        hashMB_ = std::size_t(std::stoul(value));
        search_.set_hash_size(hashMB_);
    } else if (name == "Threads") {
        search_.set_threads(std::stoi(value));
    } else if (name == "MoveOverhead") {
        search_.set_move_overhead(std::stoi(value));
    } else if (name == "OwnBook") {
        ownBook_ = (value == "true" || value == "1");
    } else if (name == "OpeningBook") {
        openingBookEnabled_ = (value == "true" || value == "1");
    } else if (name == "BookDepth") {
        bookDepthPlies_ = std::stoi(value);
    } else if (name == "BookTimeLimit") {
        bookTimeLimitMs_ = std::stoi(value);
    } else if (name == "BookRandomness") {
        bookRandomness_ = std::stoi(value);
    } else if (name == "BookFile") {
        // Shared by both book systems: try the new engine-analysis format
        // first (cheap 4-byte magic check), fall back to legacy Polyglot.
        // This keeps `setoption name OwnBook value true` + `BookFile
        // old.bin` working exactly as before for anyone already using it.
        bookFile_ = value;
        book_.clear();
        openingBook_.clear();
        if (value != "<empty>" && !value.empty()) {
            if (OpeningBook::looks_like_book_file(value)) {
                bool ok = openingBook_.load(value);
                std::cout << "info string opening book " << (ok ? "loaded " : "failed to load ")
                          << value << " (" << openingBook_.size() << " entries)" << std::endl;
            } else {
                bool ok = book_.load(value);
                std::cout << "info string book " << (ok ? "loaded " : "not found ")
                          << value << std::endl;
            }
        }
    } else if (name == "SyzygyPath") {
        syzygyPath_ = value;
        Tablebases::init(value);
        std::cout << "info string syzygy " << Tablebases::status() << std::endl;
    } else if (name == "Use NNUE") {
        const bool on = (value == "true" || value == "1");
        Eval::set_mode(on ? EvalMode::NNUE : EvalMode::Classical);
        pos_.nnue_refresh();
        std::cout << "info string eval " << (on ? "NNUE" : "Classical") << std::endl;
    } else if (name == "EvalFile") {
        if (value != "<internal>" && !value.empty()) {
            bool ok = NNUE::load(value);
            if (ok) pos_.nnue_refresh();
            std::cout << "info string nnue " << (ok ? "loaded " : "load failed ")
                      << value << std::endl;
        }
    }
}

void UCIEngine::cmd_position(std::istream& is) {
    std::string token;
    is >> token;

    if (token == "startpos") {
        pos_.set(kStartFen);
        is >> token;             // expect "moves" (or eof)
    } else if (token == "fen") {
        std::string fen;
        while (is >> token && token != "moves")
            fen += token + ' ';
        if (!pos_.set(fen)) {
            // Malformed FEN: pos_.set() already reset itself to the standard
            // starting position rather than leaving anything half-built.
            // Report it and stop -- do not apply a `moves` list that was
            // computed against the FEN we just rejected, and never crash
            // the process over bad UCI input (see position.cpp's set()).
            std::cout << "info string invalid FEN, ignoring: " << fen << std::endl;
            return;
        }
    }

    if (token == "moves")
        while (is >> token) {
            Move m = parse_move(token);
            if (m == Move::none()) break;
            pos_.do_move(m);
        }
}

void UCIEngine::cmd_go(std::istream& is) {
    SearchLimits limits;
    std::string token;
    while (is >> token) {
        if      (token == "depth")     is >> limits.depth;
        else if (token == "movetime")  is >> limits.movetime;
        else if (token == "nodes")     is >> limits.nodes;
        else if (token == "infinite")  limits.infinite = true;
        else if (token == "wtime")     is >> limits.time[WHITE];
        else if (token == "btime")     is >> limits.time[BLACK];
        else if (token == "winc")      is >> limits.inc[WHITE];
        else if (token == "binc")      is >> limits.inc[BLACK];
        else if (token == "movestogo") is >> limits.movestogo;
        else if (token == "perft")     { cmd_perft(is); return; }
    }

    // Engine-analysis opening book (src/book/): tried first when enabled.
    // See try_opening_book_move() for the exact policy (ply-depth gating,
    // optional verification search, randomness). A false return here means
    // "did not play a book move", not "book disabled" -- fall through.
    if (try_opening_book_move(limits))
        return;

    // Legacy Polyglot / human-PGN book: unchanged behavior from before this
    // feature existed, kept for backward compatibility.
    if (ownBook_ && book_.loaded()) {
        Move bm = book_.probe(pos_);
        if (bm != Move::none()) {
            std::cout << "bestmove " << move_to_uci(bm) << std::endl;
            return;
        }
    }

    start_search(limits);
}

bool UCIEngine::try_opening_book_move(const SearchLimits& limits) {
    (void)limits;   // the book path does its own (much shorter) verification search, if any

    if (!openingBookEnabled_ || !openingBook_.loaded())
        return false;
    if (pos_.game_ply() > bookDepthPlies_)
        return false;   // past the configured opening horizon: always search normally

    std::vector<BookMove> candidates = openingBook_.probe(pos_);
    if (candidates.empty())
        return false;

    Move bm = select_book_move(candidates, bookRandomness_, pos_.key());

    // BookTimeLimit == 0 (default): trust the pre-computed book move outright
    // -- it was already analyzed at (typically) far greater depth than any
    // single real-time search could afford, so this is "choose the highest
    // quality move", not "skip analysis". BookTimeLimit > 0: run a quick
    // verification search first, and only override the book if the live
    // search strongly disagrees -- this is the "must never override a
    // strong search result if configured not to" requirement: with
    // verification on, a clearly stronger live result DOES win.
    if (bookTimeLimitMs_ > 0) {
        SearchLimits quick;
        quick.movetime = bookTimeLimitMs_;
        wait_search();
        search_.arm();
        SearchResult r = search_.think(pos_, quick);
        constexpr int kBookOverrideMarginCp = 150;   // named, tunable, not a magic-number hack
        if (r.best != Move::none() && r.best != bm &&
            int(r.score) > candidates.front().evalCp + kBookOverrideMarginCp) {
            return false;   // fall through to the normal, fully-budgeted search below
        }
    }

    std::cout << "bestmove " << move_to_uci(bm) << std::endl;
    return true;
}

void UCIEngine::cmd_stop() {
    search_.stop();
    wait_search();
}

void UCIEngine::cmd_perft(std::istream& is) {
    int depth = 1;
    is >> depth;
    auto [total, breakdown] = perft_divide(pos_, depth);
    for (const auto& [mv, n] : breakdown)
        std::cout << mv << ": " << n << "\n";
    std::cout << "\nNodes searched: " << total << std::endl;
}

void UCIEngine::execute(const std::string& line) {
    std::istringstream is(line);
    std::string cmd;
    is >> cmd;

    if      (cmd == "uci")        cmd_uci();
    else if (cmd == "isready")    cmd_isready();
    else if (cmd == "ucinewgame") cmd_newgame();
    else if (cmd == "setoption")  cmd_setoption(is);
    else if (cmd == "position")   cmd_position(is);
    else if (cmd == "go")         cmd_go(is);
    else if (cmd == "stop")       cmd_stop();
    else if (cmd == "ponderhit")  {}   // accepted; ponder search continues
    else if (cmd == "debug")      { std::string v; is >> v; (void)v; }
    else if (cmd == "d")          std::cout << pos_.to_string() << std::endl;
    else if (cmd == "eval")       std::cout << "eval " << int(evaluate(pos_)) << " cp" << std::endl;
    else if (cmd == "perft")      cmd_perft(is);
    else if (cmd == "bench")      { int d = 12; is >> d; search_.bench(d); }
    else if (cmd == "selftest")   run_selftests(std::cout);
    // unknown commands are ignored, per UCI convention
}

int UCIEngine::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit") { cmd_stop(); return 0; }
        execute(line);
    }
    // EOF (e.g. piped input): let a running finite search finish rather than
    // aborting it, so the final bestmove is still emitted.
    wait_search();
    return 0;
}

} // namespace chess
