// movegen.h - Pseudo-legal and legal move generation.
#pragma once

#include "core/position.h"

namespace chess {

constexpr int MAX_MOVES = 256;

struct ScoredMove {
    Move move;
    int  score = 0;
    operator Move() const { return move; }
};

// Fixed-capacity move container (no heap allocation in the hot path).
class MoveList {
public:
    ScoredMove* begin() { return moves_; }
    ScoredMove* end()   { return last_; }
    const ScoredMove* begin() const { return moves_; }
    const ScoredMove* end()   const { return last_; }
    std::size_t size() const { return std::size_t(last_ - moves_); }
    bool empty() const { return last_ == moves_; }

    void add(Move m) { (last_++)->move = m; }

private:
    ScoredMove moves_[MAX_MOVES];
    ScoredMove* last_ = moves_;
};

enum GenType { CAPTURES, QUIETS, EVASIONS, NON_EVASIONS, LEGAL };

// Append generated moves of the requested type to `list`.
void generate(const Position& pos, MoveList& list, GenType type);

} // namespace chess
