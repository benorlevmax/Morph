// evaluate.h - Classical evaluation entry point.
//
// This single `evaluate(pos)` symbol is the stable seam that search depends on.
// Phase 6 turns it into a dispatcher (NNUE vs classical) without touching any
// search code.
#pragma once

#include "core/position.h"

namespace chess {

// Simple per-type values for SEE / MVV-LVA move ordering (midgame scale).
constexpr int PieceValue[PIECE_TYPE_NB] = {0, 100, 320, 330, 500, 900, 0};

// Static evaluation from the side-to-move's point of view (negamax convention).
Value evaluate(const Position& pos);

enum class EvalMode { Classical, NNUE };

namespace Eval {
// When enabled, evaluate() returns a material-only score (for emulating early
// engine versions in A/B match testing).
void set_material_only(bool on);
bool material_only();

// Select the active evaluator (Classical is the fallback; NNUE is primary).
void set_mode(EvalMode m);
EvalMode mode();

// Must be called once per search root (refreshes the NNUE accumulator so it is
// correct regardless of prior mode switches). No-op in classical mode.
void on_search_start(Position& pos);
} // namespace Eval

} // namespace chess
