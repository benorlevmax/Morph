// trainer.h - Reference CPU trainer (MLP, manual backprop) for the training
// subsystem. Verifies the end-to-end pipeline (encode -> forward -> loss ->
// backprop -> checkpoint) without external dependencies. This is a
// self-contained correctness check, not the production NNUE trainer: its
// own `chess_train distill` CLI command (src/apps/train_main.cpp) trains
// this net but then discards the learned weights and writes the fixed
// material-baseline NNUE instead, because faithfully distilling a flat
// MLP into the engine's actual HalfKP architecture (src/nnue/nnue.h) was
// never implemented -- a structural dead end, documented where it's
// actually reached (platform/trainer/train_network.py's module docstring).
// The real production NNUE trainer -- HalfKP-shaped, quantizes and writes
// the exact binary format src/nnue/nnue.cpp reads -- lives in
// tools/nnue_pipeline/ (CPU reference) and tools/nnue_training/bullet_trainer
// (Rust/GPU via bullet_lib), driven by the distributed platform's
// TRAIN_NETWORK task (platform/trainer/train_network.py). There used to
// also be a LibTorch-based trainer here (trainer_torch.cpp); it was removed
// because it was never wired into that real pipeline either and trained a
// plain dense MLP architecturally incompatible with HalfKP -- see
// platform/docs/TRAINING.md.
#pragma once

#include "train/dataset.h"

#include <cstdint>
#include <string>
#include <vector>

namespace chess::train {

struct TrainConfig {
    int    epochs      = 10;
    float  lr          = 0.05f;
    int    hidden      = 64;
    float  lambda      = 0.5f;     // eval/result blend in the target
    float  valFraction = 0.1f;
    std::uint64_t seed = 1;
    bool   verbose     = true;
};

struct TrainReport {
    float initialTrainLoss = 0.0f;
    float finalTrainLoss   = 0.0f;
    float finalValLoss     = 0.0f;
    std::vector<float> trainLossPerEpoch;
    std::vector<float> valLossPerEpoch;
};

// 768 -> hidden (ReLU) -> 1 (sigmoid) MLP trained on win-probability MSE.
class RefTrainer {
public:
    explicit RefTrainer(int hidden = 64, std::uint64_t seed = 1);

    float predict(const std::vector<int>& activeFeatures) const;  // win prob [0,1]
    TrainReport train(const Dataset& data, const TrainConfig& cfg);

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // Distillation: quantize this trained net into NNUE .nnue weights so the
    // engine can use it as its production evaluator. Requires hidden == NNUE_HL.
    // The NNUE output then reproduces cpScale * logit(predict) in centipawns.

    int hidden() const { return H_; }

private:
    void init_weights(std::uint64_t seed);

    int H_;
    std::vector<float> W1_;   // [N_FEATURES * H]
    std::vector<float> b1_;   // [H]
    std::vector<float> W2_;   // [H]
    float b2_;
};

} // namespace chess::train
