// train_main.cpp - CLI for self-play data generation and reference training.
//
//   chess_train gen   --games N --depth D [--nodes K] --out data.dat
//   chess_train train --data data.dat --epochs E [--hidden H] --ckpt model.ckpt
#include "core/position.h"
#include "train/selfplay.h"
#include "train/trainer.h"
#include "train/dataset.h"
#include "nnue/nnue.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

using namespace chess;
using namespace chess::train;

int main(int argc, char** argv) {
    Position::init_static();
    if (argc < 2) {
        std::cerr << "usage: chess_train <gen|train> [options]\n";
        return 2;
    }
    const std::string cmd = argv[1];

    auto opt_i = [&](const char* name, int def) {
        for (int i = 2; i + 1 < argc; ++i)
            if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
        return def;
    };
    auto opt_s = [&](const char* name, const std::string& def) {
        for (int i = 2; i + 1 < argc; ++i)
            if (std::strcmp(argv[i], name) == 0) return std::string(argv[i + 1]);
        return def;
    };
    auto opt_u64 = [&](const char* name, std::uint64_t def) -> std::uint64_t {
        for (int i = 2; i + 1 < argc; ++i)
            if (std::strcmp(argv[i], name) == 0)
                return std::strtoull(argv[i + 1], nullptr, 10);
        return def;
    };
    auto opt_d = [&](const char* name, double def) -> double {
        for (int i = 2; i + 1 < argc; ++i)
            if (std::strcmp(argv[i], name) == 0) return std::atof(argv[i + 1]);
        return def;
    };
    // A fresh, process-unique random seed for --seed's default. NOT used
    // when --seed is passed explicitly (reproducible runs still work).
    // This matters for distributed data generation specifically: many
    // workers/tasks invoke `chess_train gen` with identical --games/--depth/
    // --randomplies concurrently, and a fixed default seed here previously
    // made every one of those runs produce byte-identical games -- which
    // the server's content-hash position dedup then silently discarded in
    // full (100% duplicates, 0 accepted) for every worker after the first.
    // See platform/worker/data_generation.py, which additionally passes an
    // explicit --seed derived from task/worker identity for defense in depth.
    auto fresh_random_seed = [&]() -> std::uint64_t {
        std::random_device rd;
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::uint64_t s = (std::uint64_t(rd()) << 32) ^ std::uint64_t(rd());
        s ^= std::uint64_t(now) + 0x9E3779B97F4A7C15ull + (s << 6) + (s >> 2);
        return s ? s : 0xC0FFEEull;
    };

    if (cmd == "gen") {
        SelfPlayConfig cfg;
        cfg.games = opt_i("--games", 100);
        cfg.limits.depth = opt_i("--depth", 6);
        int nodes = opt_i("--nodes", 0);
        if (nodes > 0) { cfg.limits.nodes = std::uint64_t(nodes); cfg.limits.depth = 0; }
        cfg.randomPlies = opt_i("--randomplies", 8);
        // See SelfPlayConfig::randomMoveProb (selfplay.h) -- 0.0 default
        // preserves the exact old fixed-opening-only-randomness behavior;
        // pass e.g. --random-move-prob 0.03 to also randomize ~3% of
        // post-opening plies, which is what actually stops long
        // deterministic-tail duplicate games once the dataset is large.
        cfg.randomMoveProb = opt_d("--random-move-prob", 0.0);
        cfg.seed = opt_u64("--seed", fresh_random_seed());
        cfg.verbose = true;
        // "dat" (binary, versioned, includes instability fields since v2) |
        // "bullet" (text, 3-field, unchanged -- the exact format bullet's
        // `convert` utility expects) | "bullet-ext" (text, same 3 fields plus
        // trailing scoreSwing/bestMoveChanges columns -- for distributed data
        // generation / difficulty mining, not for feeding bullet directly).
        const std::string fmt = opt_s("--format", "dat");
        const std::string out = opt_s("--out",
            fmt == "dat" ? "selfplay.dat" : "selfplay.txt");

        Dataset ds;
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t produced = generate_selfplay(cfg, ds);
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        if (fmt == "bullet")          ds.save_bullet(out);
        else if (fmt == "bullet-ext") ds.save_bullet_ext(out);
        else                          ds.save(out);
        std::cout << "generated " << produced << " samples from " << cfg.games
                  << " games in " << secs << " s ("
                  << (secs > 0 ? std::size_t(produced / secs) : produced)
                  << " samples/s) -> " << out << "\n";
        return 0;
    }

    if (cmd == "train") {
        const std::string data = opt_s("--data", "selfplay.dat");
        const std::string ckpt = opt_s("--ckpt", "model.ckpt");
        TrainConfig tc;
        tc.epochs = opt_i("--epochs", 20);
        tc.hidden = opt_i("--hidden", 64);

        Dataset ds;
        if (!ds.load(data)) { std::cerr << "cannot load " << data << "\n"; return 1; }
        std::cout << "loaded " << ds.size() << " samples\n";

        RefTrainer trainer(tc.hidden, tc.seed);
        TrainReport rep = trainer.train(ds, tc);
        trainer.save(ckpt);
        std::cout << "initial train loss " << rep.initialTrainLoss
                  << " -> final " << rep.finalTrainLoss
                  << " (val " << rep.finalValLoss << ")\n";
        std::cout << "saved checkpoint -> " << ckpt << "\n";
        return 0;
    }

    if (cmd == "distill") {
        // Train the flat reference net (research) and emit a loadable net in the
        // new HalfKP/dual-perspective format. NOTE: faithfully distilling the
        // learned signal into the HalfKP net requires a HalfKP-shaped trainer
        // (the upcoming training phase); until then the emitted net is the
        // material-initialized baseline of the new architecture.
        const std::string data = opt_s("--data", "selfplay.dat");
        const std::string out  = opt_s("--out", "distilled.nnue");
        TrainConfig tc;
        tc.epochs = opt_i("--epochs", 30);

        Dataset ds;
        if (!ds.load(data)) { std::cerr << "cannot load " << data << "\n"; return 1; }
        std::cout << "loaded " << ds.size() << " samples; training reference net\n";

        RefTrainer trainer(tc.hidden, tc.seed);
        TrainReport rep = trainer.train(ds, tc);
        std::cout << "train loss " << rep.initialTrainLoss << " -> " << rep.finalTrainLoss
                  << " (val " << rep.finalValLoss << ")\n";

        NNUE::init();
        if (!NNUE::save(out)) { std::cerr << "nnue write failed\n"; return 1; }
        std::cout << "wrote new-architecture NNUE (material baseline) -> " << out
                  << "\n  (load with: setoption name EvalFile value " << out
                  << "; HalfKP distillation pending the training phase)\n";
        return 0;
    }

    std::cerr << "unknown command: " << cmd << "\n";
    return 2;
}
