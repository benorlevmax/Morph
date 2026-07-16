// trainer.cpp - Reference CPU MLP trainer.
#include "train/trainer.h"
#include "train/encoding.h"
#include "core/position.h"
#include "nnue/nnue.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace chess::train {

namespace {
constexpr std::uint32_t CKPT_MAGIC = 0x434B5054;   // "CKPT"
constexpr std::uint32_t CKPT_VER   = 1;

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Encoded {
    std::vector<int> active;
    float target;
};
} // namespace

RefTrainer::RefTrainer(int hidden, std::uint64_t seed) : H_(hidden) {
    init_weights(seed);
}

void RefTrainer::init_weights(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    W1_.assign(std::size_t(N_FEATURES) * H_, 0.0f);
    b1_.assign(H_, 0.0f);
    W2_.assign(H_, 0.0f);
    for (auto& w : W1_) w = nd(rng);
    for (auto& w : W2_) w = nd(rng);
    b2_ = 0.0f;
}

float RefTrainer::predict(const std::vector<int>& active) const {
    std::vector<float> h(b1_);
    for (int i : active) {
        const float* row = &W1_[std::size_t(i) * H_];
        for (int j = 0; j < H_; ++j) h[j] += row[j];
    }
    float o = b2_;
    for (int j = 0; j < H_; ++j) o += W2_[j] * relu(h[j]);
    return sigmoidf(o);
}

TrainReport RefTrainer::train(const Dataset& data, const TrainConfig& cfg) {
    // Pre-encode every sample once (FEN -> active features + target).
    std::vector<Encoded> enc;
    enc.reserve(data.size());
    Position pos;
    for (std::size_t i = 0; i < data.size(); ++i) {
        const Sample& s = data[i];
        // pos.set() never throws; on a malformed FEN it resets pos to the
        // standard starting position and returns false. Skip such samples
        // outright rather than silently training on a substituted position
        // mislabeled with the original (bogus) sample's eval/result.
        if (!pos.set(s.fen)) continue;
        enc.push_back({encode_sparse(pos),
                       win_prob_target(s.eval, s.result, cfg.lambda)});
    }

    // Train/val split.
    std::vector<std::size_t> order(enc.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(cfg.seed);
    std::shuffle(order.begin(), order.end(), rng);
    const std::size_t valN = std::size_t(enc.size() * cfg.valFraction);
    std::vector<std::size_t> val(order.begin(), order.begin() + valN);
    std::vector<std::size_t> trn(order.begin() + valN, order.end());

    auto dataset_loss = [&](const std::vector<std::size_t>& ids) {
        if (ids.empty()) return 0.0f;
        double sum = 0.0;
        for (std::size_t id : ids) {
            float p = predict(enc[id].active);
            float d = p - enc[id].target;
            sum += double(d) * d;
        }
        return float(sum / ids.size());
    };

    TrainReport rep;
    rep.initialTrainLoss = dataset_loss(trn);

    std::vector<float> h(H_), preact(H_);
    for (int e = 0; e < cfg.epochs; ++e) {
        std::shuffle(trn.begin(), trn.end(), rng);
        for (std::size_t id : trn) {
            const auto& a = enc[id].active;
            const float t = enc[id].target;

            // Forward.
            for (int j = 0; j < H_; ++j) preact[j] = b1_[j];
            for (int i : a) {
                const float* row = &W1_[std::size_t(i) * H_];
                for (int j = 0; j < H_; ++j) preact[j] += row[j];
            }
            float o = b2_;
            for (int j = 0; j < H_; ++j) { h[j] = relu(preact[j]); o += W2_[j] * h[j]; }
            float p = sigmoidf(o);

            // Backward (MSE on win prob).
            float dLdo = 2.0f * (p - t) * p * (1.0f - p);

            // Output layer.
            for (int j = 0; j < H_; ++j) {
                float gW2 = dLdo * h[j];
                float dh  = (preact[j] > 0.0f) ? dLdo * W2_[j] : 0.0f;
                W2_[j] -= cfg.lr * gW2;
                b1_[j] -= cfg.lr * dh;
                // Hidden-layer weights only change for active inputs (== 1.0).
                preact[j] = dh;   // stash grad-into-hidden for the W1 update below
            }
            b2_ -= cfg.lr * dLdo;
            for (int i : a) {
                float* row = &W1_[std::size_t(i) * H_];
                for (int j = 0; j < H_; ++j) row[j] -= cfg.lr * preact[j];
            }
        }

        float tl = dataset_loss(trn), vl = dataset_loss(val);
        rep.trainLossPerEpoch.push_back(tl);
        rep.valLossPerEpoch.push_back(vl);
        if (cfg.verbose)
            std::cout << "epoch " << (e + 1) << "/" << cfg.epochs
                      << "  train_loss " << tl << "  val_loss " << vl << "\n";
    }

    rep.finalTrainLoss = rep.trainLossPerEpoch.empty() ? rep.initialTrainLoss
                                                       : rep.trainLossPerEpoch.back();
    rep.finalValLoss   = rep.valLossPerEpoch.empty() ? 0.0f : rep.valLossPerEpoch.back();
    return rep;
}

bool RefTrainer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    auto w32 = [&](std::uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
    w32(CKPT_MAGIC); w32(CKPT_VER);
    w32(std::uint32_t(N_FEATURES)); w32(std::uint32_t(H_));
    f.write(reinterpret_cast<const char*>(W1_.data()), std::streamsize(W1_.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(b1_.data()), std::streamsize(b1_.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(W2_.data()), std::streamsize(W2_.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(&b2_), sizeof(float));
    return bool(f);
}

bool RefTrainer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::uint32_t magic, ver, feats, hid;
    auto r32 = [&](std::uint32_t& v) { return bool(f.read(reinterpret_cast<char*>(&v), 4)); };
    if (!r32(magic) || !r32(ver) || !r32(feats) || !r32(hid)) return false;
    if (magic != CKPT_MAGIC || feats != N_FEATURES) return false;
    H_ = int(hid);
    W1_.assign(std::size_t(N_FEATURES) * H_, 0.0f);
    b1_.assign(H_, 0.0f);
    W2_.assign(H_, 0.0f);
    f.read(reinterpret_cast<char*>(W1_.data()), std::streamsize(W1_.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(b1_.data()), std::streamsize(b1_.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(W2_.data()), std::streamsize(W2_.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(&b2_), sizeof(float));
    return bool(f);
}

} // namespace chess::train
