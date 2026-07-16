// zobrist.cpp - Deterministic Zobrist key generation (xorshift64* PRNG).
#include "core/zobrist.h"

namespace chess::Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side;

namespace {
// Deterministic PRNG so hashes are reproducible across runs/platforms.
class PRNG {
public:
    explicit PRNG(Key seed) : s_(seed) {}
    Key next() {
        s_ ^= s_ >> 12;
        s_ ^= s_ << 25;
        s_ ^= s_ >> 27;
        return s_ * 0x2545F4914F6CDD1DULL;
    }
private:
    Key s_;
};
} // namespace

void init() {
    PRNG rng(0x9D39247E33776D41ULL);

    for (int pc = 0; pc < PIECE_NB; ++pc)
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            psq[pc][s] = rng.next();

    for (int f = 0; f < FILE_NB; ++f)
        enpassant[f] = rng.next();

    for (int cr = 0; cr < CASTLING_RIGHT_NB; ++cr)
        castling[cr] = rng.next();

    side = rng.next();
}

} // namespace chess::Zobrist
