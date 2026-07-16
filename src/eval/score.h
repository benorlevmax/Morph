// score.h - Tapered (midgame/endgame) score type and game-phase machinery.
#pragma once

#include "core/types.h"

namespace chess {

// A Score carries separate midgame (mg) and endgame (eg) components which are
// interpolated by game phase at the end of evaluation (tapered eval).
struct Score {
    int mg = 0;
    int eg = 0;
};

constexpr Score make_score(int mg, int eg) { return Score{mg, eg}; }

constexpr Score operator+(Score a, Score b) { return {a.mg + b.mg, a.eg + b.eg}; }
constexpr Score operator-(Score a, Score b) { return {a.mg - b.mg, a.eg - b.eg}; }
constexpr Score operator-(Score a)          { return {-a.mg, -a.eg}; }
constexpr Score operator*(Score a, int i)   { return {a.mg * i, a.eg * i}; }
constexpr Score operator*(int i, Score a)   { return {a.mg * i, a.eg * i}; }

constexpr Score& operator+=(Score& a, Score b) { a.mg += b.mg; a.eg += b.eg; return a; }
constexpr Score& operator-=(Score& a, Score b) { a.mg -= b.mg; a.eg -= b.eg; return a; }

constexpr bool operator==(Score a, Score b) { return a.mg == b.mg && a.eg == b.eg; }
constexpr bool operator!=(Score a, Score b) { return !(a == b); }

// ---------------------------------------------------------------------------
// Game phase: 24 (full material) down to 0 (bare endgame). Weighted by
// non-pawn material so the eval tapers smoothly from mg to eg tables.
// ---------------------------------------------------------------------------
constexpr int PhaseKnight = 1;
constexpr int PhaseBishop = 1;
constexpr int PhaseRook   = 2;
constexpr int PhaseQueen  = 4;
constexpr int PhaseMax    = 4 * PhaseKnight + 4 * PhaseBishop
                          + 4 * PhaseRook   + 2 * PhaseQueen;   // = 24

// Interpolate a tapered Score into a single centipawn value.
// `phase` is clamped to [0, PhaseMax]; `scale` (0..64) shrinks the endgame
// component for drawish material distributions.
constexpr Value tapered(Score s, int phase, int scale = 64) {
    const int egScaled = s.eg * scale / 64;
    return Value((s.mg * phase + egScaled * (PhaseMax - phase)) / PhaseMax);
}

} // namespace chess
