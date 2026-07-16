// encoding.h - Position -> feature encoding for the training subsystem.
//
// Independent of the NNUE module (kept separate per the modularity constraint),
// but uses the same (color, piece-type, square) feature layout so a trained net
// can later be distilled into NNUE.
#pragma once

#include "core/position.h"

#include <vector>

namespace chess::train {

constexpr int N_FEATURES = 768;   // 2 colors * 6 piece types * 64 squares

int feature_index(Piece pc, Square s);

// Sparse encoding: indices of active (==1.0) features, White-oriented absolute.
std::vector<int> encode_sparse(const Position& pos);

// Dense encoding into a caller-provided buffer of size N_FEATURES (for CNN/MLP).
void encode_dense(const Position& pos, float* out);

// Training target as a win probability in [0,1] (White POV): a blend of the
// engine eval (via a sigmoid) and the final game result. This is the standard
// NNUE-distillation objective.
float win_prob_target(int evalCpWhite, int resultWhite, float lambda = 0.5f);

float sigmoid_eval(int cp);   // sigmoid(cp / scale)

} // namespace chess::train
