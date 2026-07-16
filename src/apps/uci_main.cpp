// uci_main.cpp - Engine entry point (UCI).
#include "uci/uci.h"
#include "core/position.h"

int main() {
    // Initialize global attack/zobrist/NNUE tables BEFORE constructing the
    // engine: UCIEngine's pos_ member is built before the constructor body, and
    // its set() touches magic-bitboard tables that must already be populated.
    chess::Position::init_static();

    chess::UCIEngine engine;
    return engine.loop();
}
