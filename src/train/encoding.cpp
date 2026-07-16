// encoding.cpp - Feature encoding and training targets.
#include "train/encoding.h"

#include <cmath>

namespace chess::train {

int feature_index(Piece pc, Square s) {
    const int colorBase = (color_of(pc) == WHITE) ? 0 : 384;
    return colorBase + (int(type_of(pc)) - 1) * 64 + int(s);
}

std::vector<int> encode_sparse(const Position& pos) {
    std::vector<int> idx;
    idx.reserve(32);
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        const Piece pc = pos.piece_on(s);
        if (pc != NO_PIECE) idx.push_back(feature_index(pc, s));
    }
    return idx;
}

void encode_dense(const Position& pos, float* out) {
    for (int i = 0; i < N_FEATURES; ++i) out[i] = 0.0f;
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        const Piece pc = pos.piece_on(s);
        if (pc != NO_PIECE) out[feature_index(pc, s)] = 1.0f;
    }
}

float sigmoid_eval(int cp) {
    return 1.0f / (1.0f + std::exp(-float(cp) / 400.0f));
}

float win_prob_target(int evalCpWhite, int resultWhite, float lambda) {
    const float evalP = sigmoid_eval(evalCpWhite);
    const float resP  = 0.5f * float(resultWhite) + 0.5f;   // -1/0/+1 -> 0/0.5/1
    return lambda * evalP + (1.0f - lambda) * resP;
}

} // namespace chess::train
