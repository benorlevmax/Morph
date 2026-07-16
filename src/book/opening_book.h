// opening_book.h - Engine-analysis-generated opening book.
//
// Design decision (see docs/opening_book.md for the full writeup): this is a
// *separate* subsystem from src/io/book.h's existing Book class. That class
// reads/writes the standard Polyglot format and is populated from PGN game
// *frequency* (build_book_from_pgn counts how often humans played a move) --
// exactly the "hardcoded human openings" pattern this new system exists to
// avoid. Polyglot's fixed 16-byte (key, move, weight, learn) record also has
// no room for an evaluation score, search depth, or confidence, which every
// entry here is required to carry. Rather than bolt those fields onto a
// format defined by a third-party spec, this is a small from-scratch binary
// format in the same spirit as src/nnue/nnue.cpp's "NNU2" format: a magic +
// version header, then a flat, sorted, fixed-size-record array.
//
// Position identity uses the engine's own native Zobrist key (Position::key(),
// the same key the transposition table uses) rather than Polyglot's separate
// hashing scheme -- this format is engine-native only (not meant to be
// read by other engines), so there is no reason to pay for a second hash.
// Moves are stored using this format's own simple from/to/promotion encoding
// (see encode_book_move below), not the engine's internal Move::raw() bit
// packing -- so a book file can be written by a plain external tool (the
// Python generator) without reproducing engine-internal bit layout.
//
// The legacy Polyglot Book/OwnBook/BookFile machinery in src/io/book.h is
// left completely unmodified and still works for anyone loading a real
// Polyglot .bin file or a PGN-derived human book. See uci.cpp's cmd_setoption
// for how the (now shared) BookFile UCI option auto-detects which format a
// file is and routes it to the right loader.
#pragma once

#include "core/position.h"

#include <cstdint>
#include <string>
#include <vector>

namespace chess {

// One candidate move for a position, as recorded by the generator.
struct BookMove {
    Move          move;
    int           evalCp     = 0;     // White-relative centipawns, engine's own search score
    int           depth      = 0;     // search depth reached when this was analyzed
    std::uint32_t visits     = 1;     // number of times this position was reached/re-analyzed
    int           confidence = 0;     // 0-100, depth-relative heuristic (see .cpp), NOT a
                                       // statistical guarantee -- documented plainly as such
    int           frequency  = 0;     // optional selection weight among sibling candidates
                                       // (defaults to visits if the file didn't set it)
};

// Sorted-by-hash, fixed-size on-disk record. 8 + 2 + 2 + 1 + 4 + 1 + 2 = 20 bytes,
// written/read with explicit byte-level I/O (see .cpp) so the on-disk layout
// never depends on struct packing/alignment or host endianness.
constexpr std::uint32_t BOOK_MAGIC   = 0x314B4243;  // on-disk bytes spell "CBK1"
constexpr std::uint32_t BOOK_VERSION = 1;
constexpr int           BOOK_RECORD_BYTES = 20;

// Design decision: the on-disk move field is NOT Move::raw() (the engine's
// internal 16-bit packing, with its own move-type bits at 14-15 that a
// promotion/castling/en-passant move sets in a way specific to this engine's
// implementation). Requiring an external tool (the Python generator) to
// reproduce that exact bit-packing would be a fragile, easy-to-silently-get-
// wrong dependency. Instead, book moves are stored as this format's own
// trivial, from-scratch encoding -- from-square, to-square, and a 3-bit
// promotion code (0=none,1=N,2=B,3=R,4=Q) -- which is unambiguous for any
// legal chess move (see the .cpp comment on why castling/en-passant need no
// extra flag bits here) and is exactly as easy to compute in Python as in
// C++. This is the same reasoning src/io/book.cpp's encode_polyglot() uses
// for the *existing* legacy format, applied to a from-scratch encoding
// instead of adopting Polyglot's.
std::uint16_t encode_book_move(Square from, Square to, int promoCode);
bool          move_matches_code(Move m, std::uint16_t code);

class OpeningBook {
public:
    // Loads a book previously written by tools/opening_book/generate_book.py
    // (or OpeningBook::save(), used by the C++ test suite). Returns false if
    // the file doesn't exist or its header doesn't match BOOK_MAGIC -- the
    // caller (uci.cpp) uses that to fall back to the legacy Polyglot loader.
    bool load(const std::string& path);
    bool save(const std::string& path) const;   // used by tests + generator's C++ side (none yet: generator is Python-only, but kept here for symmetry/testability)
    bool loaded() const { return !entries_.empty(); }
    void clear() { entries_.clear(); }
    std::size_t size() const { return entries_.size(); }

    // Quick pre-check without doing a full parse: does `path` look like our
    // format (first 4 bytes match BOOK_MAGIC)? Used by uci.cpp to decide
    // which loader to try first.
    static bool looks_like_book_file(const std::string& path);

    // All candidate moves stored for this exact position, sorted strongest
    // eval first. Each is verified legal in `pos` via live move generation
    // before being returned (defensive, matches src/io/book.cpp's Book class);
    // an entry whose move is no longer legal (shouldn't happen for a
    // correctly-generated book, but a hand-edited or corrupted file could)
    // is silently skipped rather than returned. Empty if out of book.
    std::vector<BookMove> probe(const Position& pos) const;

    // Build in-memory entries directly (used by the C++ test suite to write
    // small books without going through the Python generator).
    void add(Key hash, const BookMove& mv);
    void finalize();   // sort by hash; call once after add()-ing everything

private:
    // On-disk-shaped entry: `code` (see encode_book_move) rather than a
    // `Move`, since a stored code cannot be turned into a fully-typed
    // Move (CASTLING/EN_PASSANT/PROMOTION flags) without a live position
    // to resolve it against -- that resolution happens once, in probe(),
    // against that call's real legal move list.
    struct Entry {
        Key           hash;
        std::uint16_t code;
        int           evalCp;
        int           depth;
        std::uint32_t visits;
        int           confidence;
        int           frequency;
    };
    std::vector<Entry> entries_;   // sorted by hash ascending after load()/finalize()
};

// Choose one move from probe()'s results.
//   randomness == 0   -> always candidates.front() (already sorted best-first).
//                        Fully deterministic: same book + same position always
//                        picks the same move.
//   randomness in 1..100 -> among candidates within a score-loss tolerance
//                        window (tolerance scales with randomness, see .cpp),
//                        pick one weighted by `frequency` (falling back to
//                        `visits` if frequency is 0), using a seed derived
//                        from `seed` (typically the position's own Zobrist
//                        key) so the choice is still reproducible for a given
//                        position + seed, never a hidden global RNG.
Move select_book_move(const std::vector<BookMove>& candidates, int randomness, std::uint64_t seed);

} // namespace chess
