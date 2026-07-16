// selfplay.cpp - Generate labelled positions by self-play.
#include "train/selfplay.h"
#include "core/movegen.h"

#include <iostream>
#include <random>

namespace chess::train {

namespace {
const char* kStart = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
} // namespace

std::size_t generate_selfplay(const SelfPlayConfig& cfg, Dataset& out) {
    std::mt19937_64 rng(cfg.seed);
    Search engine;
    engine.set_quiet(true);

    std::size_t produced = 0;

    for (int g = 0; g < cfg.games; ++g) {
        Position pos;
        pos.set(kStart);

        // Per-game samples (result filled in after the game).
        std::vector<Sample> gameSamples;
        int resultWhite = 0;   // default draw

        int ply = 0;
        for (; ply < cfg.maxPlies; ++ply) {
            MoveList legal;
            generate(pos, legal, LEGAL);
            if (legal.empty()) {
                // Checkmate or stalemate.
                resultWhite = pos.in_check()
                    ? (pos.side_to_move() == WHITE ? -1 : +1)   // side to move is mated
                    : 0;
                break;
            }
            if (pos.is_draw()) { resultWhite = 0; break; }

            Move move;
            if (ply < cfg.randomPlies) {
                // Random move for opening diversity (not recorded).
                std::uniform_int_distribution<std::size_t> d(0, legal.size() - 1);
                move = legal.begin()[d(rng)].move;
            } else {
                engine.arm();
                SearchResult r = engine.think(pos, cfg.limits);
                move = r.best;
                if (move == Move::none()) { resultWhite = 0; break; }
                // Record the position with its White-POV eval (skip noisy mates).
                const int evalWhite =
                    pos.side_to_move() == WHITE ? int(r.score) : -int(r.score);
                if (std::abs(evalWhite) < VALUE_MATE_IN_MAX_PLY) {
                    Sample s;
                    s.fen = pos.fen();
                    s.eval = std::int16_t(std::clamp(evalWhite, -32000, 32000));
                    // Search-instability signal (see search.h's
                    // SearchResult::scoreSwing / bestMoveChanges) -- already
                    // computed by this same search call, just surfaced here
                    // as a training-data quality/difficulty signal.
                    s.scoreSwing = std::int16_t(std::clamp(int(r.scoreSwing), 0, 32000));
                    s.bestMoveChanges =
                        std::uint8_t(std::clamp(r.bestMoveChanges, 0, 255));
                    gameSamples.push_back(s);
                }
            }
            pos.do_move(move);
        }

        for (Sample& s : gameSamples) {
            s.result = std::int8_t(resultWhite);
            out.add(s);
            ++produced;
        }

        if (cfg.verbose && (g + 1) % 10 == 0)
            std::cout << "selfplay: " << (g + 1) << " games, "
                      << produced << " samples\n";
    }

    return produced;
}

} // namespace chess::train
