// book.h - Polyglot-format opening book: key, reader, probe, and generator.
//
// The on-disk layout is the standard Polyglot 16-byte big-endian entry
// (key, move, weight, learn) sorted by key. The 781-entry random array used to
// hash positions is isolated in book_init(); swapping in the canonical Polyglot
// constants makes third-party .bin files interoperable without other changes.
#pragma once

#include "core/position.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace chess {

void book_init();                       // fill the position-hash random array
Key  book_key(const Position& pos);     // Polyglot-style position hash

struct BookEntry {
    Key           key    = 0;
    std::uint16_t move   = 0;           // Polyglot move encoding
    std::uint16_t weight = 0;
    std::uint32_t learn  = 0;
};

class Book {
public:
    bool load(const std::string& path);
    bool loaded() const { return !entries_.empty(); }
    void clear() { entries_.clear(); }

    // Returns a book move for `pos`, or Move::none() if out of book.
    // pickBest=true selects the highest-weight move; otherwise weight-random.
    Move probe(const Position& pos, bool pickBest = true) const;

private:
    std::vector<BookEntry> entries_;     // sorted by key ascending
};

// Encode a (legal) move in Polyglot form for the given position.
std::uint16_t encode_polyglot(const Position& pos, Move m);

// Build a book from a PGN stream and write it to `path` (Polyglot format).
// Counts move occurrences (as weights) up to `maxPlies` from each game.
bool build_book_from_pgn(std::istream& pgn, const std::string& path, int maxPlies = 16);

} // namespace chess
