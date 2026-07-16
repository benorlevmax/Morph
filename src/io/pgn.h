// pgn.h - SAN move notation and PGN game read/write.
#pragma once

#include "core/position.h"

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace chess {

// Standard Algebraic Notation. `pos` is mutated and restored internally.
std::string move_to_san(Position& pos, Move m);

// Parse a SAN token against the legal moves of `pos`. Returns Move::none()
// if it matches no legal move. Tolerant of check/annotation suffixes and
// '0-0' castling spelling.
Move san_to_move(Position& pos, const std::string& san);

// A complete game: starting FEN, the moves played, the result, and PGN tags.
struct GameRecord {
    std::string startFen =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::vector<Move> moves;
    std::string result = "*";                 // "1-0", "0-1", "1/2-1/2", "*"
    std::map<std::string, std::string> tags;  // Event, Site, White, Black, ...
};

// Write a single game in PGN (tags + SAN movetext + result).
void write_pgn(std::ostream& os, const GameRecord& game);

// Read the next game from a PGN stream. Returns false at end of stream.
bool read_pgn(std::istream& is, GameRecord& out);

} // namespace chess
