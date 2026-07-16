// psqt.h - Piece-square tables (material + position, tapered).
//
// PSQTable[pc][sq] is the white-POV tapered Score contributed by a piece `pc`
// standing on square `sq` (white pieces positive, black pieces negative). The
// Position keeps a running sum of these, updated incrementally on every piece
// add/remove/move -- the same seam where a future NNUE accumulator will hook in.
#pragma once

#include "eval/score.h"

namespace chess {

extern Score PSQTable[PIECE_NB][SQUARE_NB];

void psqt_init();

} // namespace chess
