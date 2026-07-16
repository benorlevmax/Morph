// uci.h - Minimal UCI protocol front-end.
#pragma once

#include "core/position.h"
#include "search/search.h"
#include "io/book.h"
#include "book/opening_book.h"

#include <string>
#include <thread>

namespace chess {

class UCIEngine {
public:
    UCIEngine();
    ~UCIEngine();

    // Read commands from stdin until "quit". Returns process exit code.
    int loop();

    // Parse a single command line (also usable for scripted testing).
    void execute(const std::string& line);

private:
    void cmd_uci();
    void cmd_isready();
    void cmd_newgame();
    void cmd_setoption(std::istream& is);
    void cmd_position(std::istream& is);
    void cmd_go(std::istream& is);
    void cmd_stop();
    void cmd_perft(std::istream& is);

    void start_search(const SearchLimits& limits);
    void wait_search();

    // Try the engine-analysis book first (probe -> optional verification
    // search -> select). Returns true and has already printed "bestmove ..."
    // if a book move was played; false means fall through to normal search
    // (either out of book, book disabled, or a verification search strongly
    // disagreed with the stored move -- see opening_book.h/cpp and
    // docs/opening_book.md for the exact policy).
    bool try_opening_book_move(const SearchLimits& limits);

    Move parse_move(const std::string& uci) const;

    Position    pos_;
    Search      search_;
    std::thread worker_;
    std::size_t hashMB_ = 16;

    // Legacy Polyglot / human-PGN book (src/io/book.h). Unmodified behavior;
    // kept for backward compatibility and third-party .bin interop.
    Book        book_;
    bool        ownBook_ = false;

    // Engine-analysis book (src/book/opening_book.h), gated by the new
    // OpeningBook/BookDepth/BookTimeLimit/BookRandomness options. See
    // cmd_setoption's BookFile handling for how the two loaders share one
    // "BookFile" UCI option via format auto-detection.
    OpeningBook openingBook_;
    bool        openingBookEnabled_ = false;
    int         bookDepthPlies_     = 20;   // BookDepth: max game ply to consult the book
    int         bookTimeLimitMs_    = 0;    // BookTimeLimit: 0 = trust book, >0 = verify first
    int         bookRandomness_     = 0;    // BookRandomness: 0 = always strongest move

    std::string bookFile_;
    std::string syzygyPath_ = "<empty>";
};

} // namespace chess
