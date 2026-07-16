// zobrist.h - Zobrist hashing keys for transposition / repetition detection.
#pragma once

#include "core/types.h"

namespace chess::Zobrist {

extern Key psq[PIECE_NB][SQUARE_NB];   // piece on square
extern Key enpassant[FILE_NB];         // ep file
extern Key castling[CASTLING_RIGHT_NB];// castling rights mask
extern Key side;                       // side to move (black)

void init();

} // namespace chess::Zobrist
