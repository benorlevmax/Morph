# Search Architecture

Audit date: 2026-07-01. Baseline: `bench 12` = **951,119 nodes**, `ctest` 5/5 green.

This document audits the search against modern top engines (Stockfish, Ethereal,
Berserk, RubiChess) for **Project A — Modern Search Completion**. The headline
finding: **the search is already at or near modern completeness.** Every major
technique targeted by Project A is present and interacting correctly. No new
subsystems were added, because doing so would duplicate existing heuristics.

All line references are into `src/search/search.cpp` unless noted.

## Iterative deepening & root

- Iterative deepening with per-thread `ThreadState` (own `Position`, history,
  killers, countermoves, PV, eval stack). Lazy SMP; thread 0 authoritative.
- **Aspiration windows** (`d >= 4`): initial delta 18, symmetric; fail-low pulls
  beta to midpoint, fail-high widens beta; growth `delta += delta/2 + 5`.
- Legality-validated PV output; ponder move validated against the post-best
  position.

## Transposition table (`tt.{h,cpp}`)

- 3-entry, 32-byte, cache-line-aligned clusters; power-of-two masking; prefetch.
- Aging via generation (`+= 4` per search). Replacement victim =
  `min(depth − age·4 + cutNodeBonus)`; BOUND_LOWER (cut) entries preserved.
- `hashfull()` samples up to 1000 clusters → permille (bounds-safe).
- Mate-score `value_to_tt` / `value_from_tt` ply adjustment.

## Node-level pruning & reductions (main search)

| Technique | Location | Notes |
|---|---|---|
| TT cutoff | ~413 | non-PV, `tte->depth() >= depth`, exact-or-bound match |
| **IIR** | ~424 | `depth >= 4 && !ttMove && !inCheck → --depth` |
| Reverse futility (RFP) | ~430 | `depth <= 8`, `staticEval − 80·depth >= beta` |
| **Null-move pruning** | ~459 | adaptive `R = 3 + depth/4 + min(3,(eval−beta)/200)`; zugzwang material guard (non-pawn + not near-bare-king); verification search at `depth >= 10` |
| Razoring | ~481 | `depth <= 3`, `staticEval + 200·depth < alpha` → qsearch |
| **ProbCut** | ~494 | `depth >= 5`, `probCutBeta = beta + 200`; TT-guarded, SEE-gated, qsearch + verification re-search; reuses generated list |
| **Futility pruning** | ~606 | `depth <= 6`, quiet, non-check |
| **Late Move Pruning** | ~613 | `lmpLimit = improving ? 3+d² : (3+d²)/2` |
| **SEE pruning (main tree)** | ~626 | quiet margin `−20·d²`, capture margin `−80·d` |
| History-based pruning | ~617 | skip quiets with `moveHist < −4000·depth` at low depth |

## Extensions

- **Singular extension** (~632): `depth >= 8`, TT move, `sBeta = ttValue − 2·depth`,
  excluded-move reduced search; **multicut** when `sBeta >= beta`.
- **Check extension** (~647): PV-only, `see_ge(move, 0)` (significant checks only).
- *(Not present: double/negative extensions — see "Remaining differences".)*

## Late Move Reductions (~665)

Log table `r = int(0.75 + ln(d)·ln(m)/2.25)` with stat adjustments: `−1` PV,
`+1` not-improving, `−1` killer/counter, `−1` gives-check, `−= moveHist/8192`.
Re-search on fail-high; PV re-search inside window.

## Move ordering (~540)

TT move → captures (`MVV-LVA·16 + captureHistory/128`, SEE splits good/bad) →
killers (2) → countermove → quiets (`history + continuation-history`). Single
additive score space picked incrementally (`pick_next`).

## History tables (all with gravity)

- `hist_update`: `h += bonus − h·|bonus|/HISTORY_MAX`, `HISTORY_MAX = 1<<24`.
- **Butterfly** `history[color][from][to]`.
- **Capture history** `captureHistory[piece][to][captured]` (ordering + bonus/malus).
- **Continuation history** `contHist[…]` at look-back **1/2/4/6 ply**; used in
  ordering, LMR, and history pruning.
- **Correction history** (~437, 765): pawn-structure key + material-signature key,
  per color; gentle (`clamp ±16·256`, learn `min(depth+1,8)/… `), adjusts static eval.
- **History malus**: non-cutoff quiets and captures receive `−bonus`
  (`bonus = min(depth², 400)`).

## Quiescence search (~272)

Stand-pat with beta cutoff; **check evasions fully searched when in check**;
non-check nodes search captures + (first-ply) quiet checks; **SEE-negative
captures skipped**; **delta pruning** margin 200 (350 for promotions, promotion
gain added); MVV-LVA + capture-history ordering.

## Draw detection (`position.cpp`)

3-fold repetition, fifty-move, insufficient material (KK/KNK/KBK); **cuckoo
upcoming-repetition (game-cycle) detection**; draws scored flat `VALUE_DRAW`.

## Time management (~191)

- Configurable **`MoveOverhead`** UCI option (default 50 ms) subtracted from clock.
- Soft limit `optimum · 0.75 · scale`; hard limit `maximum · 0.95` (never exceeded).
- Instability `scale` from five signals: best-move stability (`stableCount`),
  score drop, complexity (eval-vs-score gap), node-effort concentration, and
  best-move-change bonus (×1.1). Clock checked every 1024 nodes.
- Single-legal-move fast path; movestogo / increment handling.

## Verdict

Search is **no longer a primary weakness**. Relative to the four reference
engines, the remaining gaps are incremental tuning, not missing subsystems.
