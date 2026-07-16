# Learned Endgame Oracle (8–10 Pieces) — Feasibility Study

Research only. No code changed. No implementation performed.

---

## 1. GO / NO-GO decision

**NO-GO, for the goal of "fastest path to ~3200 Elo."** Not because the idea
is silly — it's a legitimate research direction and the engineering
integration point is genuinely clean (§6) — but because the premise has a
load-bearing flaw (§5, Finding #1) and even world-class engines with vastly
more data/compute than this project has today measurably fail at a *simpler*
version of this exact problem (§3). Two much cheaper, better-evidenced levers
(finish training the NNUE that doesn't exist yet; turn on the exact
tablebase support that's already written but disabled) sit ahead of this in
priority (§9) and should be done first regardless.

**Conditional GO for a much narrower, different feature**: enabling real
Syzygy probing (≤7 pieces, exact, already-written integration, currently
inert — §2 Finding #6) is cheap, low-risk, and should happen this week,
independent of anything neural.

## 2. Expected Elo gain

- **8–10 piece neural oracle, as specified**: low confidence, plausibly
  **0 to +10 Elo**, with a real chance of being **slightly negative** if
  miscalibrated confidence lets a wrong oracle call override correct search
  in a critical position (see §5 Q3/Q6). This is a guess bounded mostly by
  "how rarely do real games reach 8-10-piece positions where the oracle and
  normal search would have disagreed" — for an engine at this project's
  current strength, that's a rare and low-leverage event compared to
  everyday middlegame/opening play.
- **Turning on exact Syzygy TB (≤7 pieces)**, for comparison: Stockfish's own
  historical testing found this worth low-single-digit to ~10 Elo in
  engine-vs-engine SPRT testing at normal time controls — small but real,
  free once the build flag and net files are in place, and *exact*, not
  approximate.
- **Training the currently-untrained production NNUE** (already proven
  end-to-end this session, `docs/phaseA_nnue_bullet_audit.md`): **+150 to
  +400 Elo**, based on Stockfish's own NNUE-adoption jump in 2020 and this
  session's own architecture audit finding the net is fully wired but has
  never been trained.

The gap between these numbers is the whole argument.

## 3. Development time estimate

- Neural endgame oracle, done properly (not a toy): **4–8 weeks** of
  genuine research/engineering, not counting the fact that "properly" is
  itself unresolved — see §5 Q1/Q2. Realistic breakdown: 1–2 weeks data
  pipeline + label-quality investigation (there is no exact teacher at this
  piece count, §5 Finding #1, so this alone is open-ended research, not
  engineering), 1–2 weeks architecture/training iteration, 1–2 weeks search
  integration + confidence-gating + regression testing, 1+ week Elo
  validation (SPRT needs many thousands of games to detect a ≤10 Elo effect
  reliably).
- Turning on Syzygy TB: **1–2 days** (vendor `tbchess.c`, flip
  `CHESS_USE_FATHOM` on, download the public Syzygy 3-7 piece set, SPRT-test).
- Training the NNUE: **3–7 days**, already scoped and proven this session.

## 4. Recommended architecture (if this were pursued anyway)

Not recommended to build now, but if a later, stronger version of this
engine revisits it: **Option D (root/leaf-only, confidence-gated) is the
only option that doesn't fight the search architecture** (§7). Input: same
feature representation as the production NNUE (§6) plus explicit
side-to-move and a WDL-style 3-way softmax output rather than a single
scalar (§8) — a calibrated probability distribution is what makes
confidence-gating (§5 Q6) possible at all, and a single blended scalar
(Option B) is not. Do not attempt Option A (replace NNUE) or Option E
(pruning/extensions) — both fail search-soundness in ways detailed in §7.

## 5. Training strategy

**Not recommended to execute given Finding #1 below, but if pursued**: treat
it as *distillation from noisy teachers* (deep search + self-play, per §8),
explicitly NOT as "compressed exact tablebase knowledge," and build
confidence estimation as a first-class model output from day one, not a
bolted-on filter (§5 Q6, §8). Validate against the free, already-solved
7-piece Syzygy set as a *held-out sanity check* on the boundary (positions
with 7 pieces should be learnable near-perfectly if the pipeline works at
all, before trusting it at 8-10).

## 6. Integration strategy

**The cleanest integration point already exists and requires no new
architecture to reach**: `search.cpp`'s per-node tablebase probe
(`if (Tablebases::available()) { WDLResult wdl = Tablebases::probe_wdl(pos); ... }`,
right after the repetition/draw check, before the TT probe, in every
non-root PVS node) is exactly where a piece-count-gated oracle call would
slot in — same location, same pattern, one new branch: *if TB probe fails
(piece count > `max_pieces()`) and piece count ≤ 10, try the oracle before
falling through to normal search.* `popcount(pos.pieces())` (already used
identically inside `Tablebases::probe_wdl`/`probe_root` and in classical
eval's `mop_up`/`scale_factor`) is the exact, free piece-count gate needed —
no new detection machinery required. `EvalMode`/`Eval::set_mode` and the
`SearchConfig` boolean-flag pattern (`cfg_.reverseFutility` etc.) are the
established, idiomatic way to make it toggleable. This is a genuinely clean
integration point — the engineering side of this proposal is *not* the
problem.

## 7. Biggest risks

1. **No exact teacher exists at 8-10 pieces** (§5 Q1 / Finding #1) — the
   entire "exact solver → training positions" premise in the requested
   pipeline diagram doesn't apply past 7 pieces. Whatever labels the
   training data gets are themselves approximate, which caps the oracle's
   achievable accuracy below "tablebase-like" by construction, not by
   engineering effort.
2. **Alpha-beta soundness**: mixing an approximate, non-monotonic oracle
   into a search algorithm that assumes internally-consistent evaluations
   (for correct pruning/bounds) risks search instability exactly like
   inconsistent NNUE-vs-classical eval swaps already do elsewhere in this
   codebase (see this session's own "benign mate-detection instability"
   finding from the qsearch-TT work) — except an oracle disagreement near a
   pruning decision could silently produce a *wrong* result, not just a
   slower-to-find correct one. Options A/E are the highest-risk here (§ options
   table below); Option D is the safest.
3. **State-space explosion**: going from 7→10 pieces isn't a 3-piece
   increment in difficulty, it's several orders of magnitude more distinct
   material configurations and positions, right at the point where even
   Stockfish-2850/Lc0's own *learned* policies (trained on vastly more
   compute than this project has) still measurably err on far simpler 4-5
   piece endgames (§3 data below) — evidence the problem gets harder, not
   easier, for a smaller model with less data at higher piece counts.
4. **Opportunity cost**: every week spent here is a week not spent training
   the NNUE that's fully built but has literally never been trained (§2) —
   the single highest-leverage, lowest-risk, already-proven-working lever
   available in this codebase right now.

## 8. Comparison against improving NNUE

See the alternatives table in §11 — training NNUE has a **10-40x better
expected-Elo-per-week ratio** than the oracle, is already proven
end-to-end in this codebase (not hypothetical), and is architecturally
risk-free (it doesn't touch search at all).

## 9. Recommended priority order for reaching ~3200 Elo fastest

1. **Train the production NNUE** (proven pipeline, `docs/phaseA_nnue_bullet_audit.md`) — highest Elo/week by a wide margin.
2. **Turn on real Syzygy TB probing** (≤7 pieces, exact, ~1-2 days, integration code already written and dormant) — small, free, zero-risk Elo, and a real correctness improvement (perfect endgame play up to 7 pieces) independent of any neural work.
3. **Continue the classical search/eval optimization queue** already underway this session — proven, incremental, low-risk.
4. **Offline transformer-teacher distillation into NNUE** (per the earlier architecture report) — stacks on #1, zero production-code risk.
5. *(Not recommended for now)* **8-10 piece neural endgame oracle** — revisit only after 1-4 are done and Elo gains from them have plateaued, and only with a clear-eyed acceptance that the "exact solver" framing doesn't hold at this piece count (§7 Finding #1) — it would be a genuine research bet, not a proven lever like the others.

---

## Supporting detail

### A. Engine audit (current architecture, as it actually is today)

- **Eval interface**: single stable seam, `Value evaluate(const Position&)`
  (`src/eval/evaluate.h`), dispatching classical vs. NNUE via a private
  `EvalMode` flag (`Eval::set_mode`/`Eval::mode`). Search code never
  branches on eval mode itself — this is the same seam an oracle would need
  to hook, and it's clean.
- **Eval call path**: `evaluate(pos)` is called from exactly three places in
  `search.cpp` — qsearch's stand-pat (~line 316), main-search's
  max-ply cutoff (~line 467), and main-search's `rawEval` computation
  (~line 492, itself TT-cached via `tte->eval()` so repeated calls at the
  same position within a search are avoided). A correction-history system
  (`ts.pawnCorr`/`ts.matCorr`, keyed by pawn/material Zobrist-style hashes)
  already applies a small *learned residual* on top of raw eval — precedent
  in this codebase for "a learned correction gated by material signature,"
  architecturally similar in spirit to a confidence-gated oracle blend,
  worth reusing that pattern rather than inventing a new one.
- **Material counting / piece-count detection**: `Position::count(Color,
  PieceType)` (`popcount` of a bitboard) and `popcount(pos.pieces())` for
  total piece count are the existing, cheap primitives — already used
  identically for tablebase gating, mop-up eval, and endgame scale-factor
  logic. No new detection machinery is needed for an 8-10-piece gate.
- **Existing tablebase support**: `src/syzygy/tablebases.h`/`.cpp` is a
  well-designed Fathom-backed WDL/DTZ probing layer with graceful fallback
  — **but it is compiled out in the default build.** `CMakeLists.txt`'s
  `CHESS_USE_FATHOM` option defaults `OFF`, and even turning it on would
  currently fail: the option's own `FATAL_ERROR` check requires
  `tbchess.c` vendored next to `tbprobe.c`/`tbprobe.h` at the repo root,
  and only `tbprobe.c`/`.h` are present — `tbchess.c` is missing. **This
  means the "≤7 pieces: Syzygy exact tablebase" leg of the proposed
  architecture does not currently exist in the shipped engine at all**, a
  more fundamental and easier-to-fix finding than anything about the neural
  oracle itself (see §9 priority #2).
- **Search architecture / alpha-beta integration points**: PVS with qsearch,
  Lazy SMP, iterative deepening; the TB probe happens at the very top of
  every non-root node (`search()`, before the TT probe): `if
  (Tablebases::available()) { auto wdl = Tablebases::probe_wdl(pos); if (wdl
  != Fail) return wdl_to_value(wdl, ply); }` — an exact result short-circuits
  the entire subtree immediately. Root uses a separate DTZ probe
  (`Tablebases::probe_root`) for 50-move-rule-aware best-move selection
  before normal iterative deepening even starts. This exact pattern (probe →
  short-circuit on success → fall through to normal search on failure) is
  the natural template for an oracle call, gated by piece count instead of
  by `available()`.
- **UCI architecture**: `setoption name <X> value <Y>` dispatch in
  `uci.cpp::cmd_setoption`, already the mechanism for `Use NNUE` /
  `EvalFile` / (implicitly) tablebase path configuration — a hypothetical
  `EndgameOracleFile` option would follow the exact same, already-proven
  pattern as `EvalFile`.
- **Endgame handling today**: entirely classical-eval-side heuristics —
  `mop_up()` (drive a bare king to the edge), `passed_king_eval()`
  (king-vs-passed-pawn race distance), `scale_factor()` (opposite-bishop
  drawishness detection), plus a zugzwang guard that already disables
  null-move pruning in near-bare-king material configurations
  (`nmpMaterial` check, `search.cpp` ~line 518) — i.e. **the engine already
  has explicit, hand-written awareness that low-material positions need
  different search/eval treatment than the middlegame**, which is exactly
  the kind of position-class-specific handling an oracle would extend, not
  invent from nothing.

### B. Research: existing approaches

- **Syzygy** (Ronald de Man): WDL (win/draw/loss) + DTZ (distance-to-zero,
  50-move-rule-aware) bitbases, up to 7 pieces solved and freely available;
  this project already has a from-scratch-written, currently-dormant Fathom
  integration for it.
- **Gaviota** (Miguel Ballicora): DTM (distance-to-mate) tables, generally
  larger on disk than Syzygy for the same piece count since DTM carries more
  information than DTZ; less commonly used by modern top engines than
  Syzygy for exactly that storage-cost reason.
- **Stockfish's own endgame handling**: exact Syzygy probing during search
  (same pattern this project's dormant code already mirrors) plus a small
  number of hand-coded specialized endgame evaluation functions for known
  drawish/winning material patterns (KBNvK, KPvK, etc.) — notably, even
  Stockfish, one of the two strongest engines in the world, does **not** use
  a learned neural approximator for the 8+-piece gap beyond its tablebases;
  it relies on hand-written endgame knowledge plus its general NNUE plus
  deep search.
- **Lc0's endgame handling**: probes Syzygy for WDL only (not DTZ) during
  MCTS search; its own developers and independent academic study (Haque et
  al. 2021, cited below) document that the *lack* of DTZ awareness causes
  "weird endgame effects and losing play" — Lc0 sometimes simplifies into a
  tablebase-confirmed "won" position it then struggles to actually convert
  without DTZ guidance, i.e. **even a strong neural policy/value net,
  combined with real (if partial) exact-tablebase access, still exhibits
  measurable endgame-conversion problems** — direct evidence against the
  idea that "a neural net should just be good enough" past the exact-solved
  range.
- **Direct quantitative evidence** (Sadmine, Husna & Müller, "Stockfish or
  Leela Chess Zero? A Comparison Against Endgame Tablebases," ACG 2023,
  University of Alberta): using Stockfish 15.1 (rated 2850) and Lc0 0.29
  (GPU-backed) raw policy nets — i.e. **the two strongest, most heavily
  trained chess-playing neural/search systems that exist**, evaluated
  against ground-truth Syzygy tablebases on **3-, 4-, and a 1%-sample of
  5-piece** endgames (far simpler than this proposal's 8-10 piece target):
  raw-policy mistake rates up to **9.6%** (Stockfish, KQkr draw positions)
  and **4.6%** (Lc0, same), and even in 5-piece endgames, up to **7.5%**
  (Stockfish, KBNkp draws) and **3.55%** (Lc0). A tiny 400-node search
  reduces but does not eliminate these errors. **If the two best-resourced
  chess AI systems on Earth still measurably misplay 4-5 piece endgames on
  their raw learned evaluation, a smaller, far-less-trained, from-scratch
  8-10-piece oracle for this project should be expected to be meaningfully
  less accurate, not more** — 8-10 pieces is a much larger, sparser,
  higher-dimensional space than 4-5 pieces, not a small extrapolation.
- **Neural tablebase compression / knowledge distillation** (general
  literature, e.g. the 1998 Reading University KPK studies, and general
  ML-distillation practice): this body of work is about **compressing
  ALREADY-SOLVED, exactly-known endgames** for storage-size reasons (a
  network reproducing known-perfect KPK classification in far less disk
  space than a lookup table) — **not** about extending exact solving power
  into unsolved territory. This is the crucial distinction from what's
  proposed here: "compress a 5-piece tablebase into a small net" and "invent
  a tablebase-quality oracle for 8-10 pieces that were never solved" are
  different problems with very different risk profiles, and the existing
  literature only supports the former.
- **WDL/DTZ prediction networks / learned value functions from exact
  solvers**: exist, but every credible example found is trained *within* the
  solved range (≤7 pieces, i.e. redundant with Syzygy itself, useful only
  for storage/speed, not new knowledge) — no primary source was found
  describing a production chess engine successfully extending a *learned*
  oracle usefully beyond the exact-solved frontier into 8+ pieces at
  tablebase-comparable reliability.

### C. Architecture design — input/output and Option A-E evaluation

**Input** (if built): reuse the production NNUE's existing feature
philosophy (§A) — piece locations + king positions (already the core of the
16-king-bucket HalfKP-style feature set), side to move (already handled via
the dual-perspective accumulator convention), pawn structure (material this
sparse rarely has complex structure, but passed-pawn distance-to-promotion
is exactly the kind of feature classical eval's `passed_king_eval` already
hand-computes and an oracle would need too), castling/en passant (irrelevant
at 8-10 pieces in the vast majority of realistic positions — castling rights
are almost always long gone by the time the board is this empty; safe to
omit for simplicity, unlike full-game NNUE).

**Output**: a genuine 3-way WDL softmax (not a single scalar) plus an
explicit confidence/entropy signal derived from that distribution — a single
blended scalar (as Option B implies) cannot support the confidence-gating
design in §5 Q6, since "how sure is this number" isn't recoverable from the
number alone. A distance-to-conversion regression head is a reasonable
addition (mirrors DTZ) but adds real training complexity for uncertain
payoff and should be a v2 feature, not a v1 requirement. A move-policy head
is not justified for this proposal (Option C, evaluated below, is the
weakest of the five).

| Option | Expected Elo (this engine) | Speed impact | Impl. difficulty | Search compatibility | Biggest failure mode | Training reqs |
|---|---|---|---|---|---|---|
| **A: replace NNUE in 8-10 piece positions** | Likely negative-to-flat | None (swap, not addition) | Medium | **Poor** — an approximate evaluator replacing a consistent one mid-search breaks the implicit assumption that nearby nodes' evals are comparable, exactly the kind of inconsistency this session's TT-qsearch work already showed can cause search instability, except here it could change *which side is judged better*, not just *when* that's discovered | A wrong replacement eval directly misguides the whole subtree's pruning/alpha-beta bounds | High — must be as reliable as NNUE everywhere it's used, an unrealistic bar given §B evidence |
| **B: blend oracle + NNUE** | Marginal, hard to predict sign | Small (one extra forward pass per gated node) | Medium | **Poor-to-medium** — blending two differently-calibrated scalars needs careful weighting or you get worse-than-either results, and that weighting itself becomes a tuning problem | Miscalibrated blend weight silently degrades eval quality in exactly the endgames that matter most | High |
| **C: move ordering only** | Near-zero | Small | Medium | Good (soft signal, doesn't change search correctness, only which moves get tried first) | Wasted effort: this engine's own profiling (recommendation #3/#4 work, this session) already showed 90-92% first-move cutoff rates — ordering is near-ceiling already, so there's little room for this to help | Medium |
| **D: root/leaf only, confidence-gated** | **Best of the five, still small** (§2) | Smallest (called rarely — root nodes and confirmed-low-material leaves only) | Medium-high (needs the confidence-gating machinery, §5 Q6) | **Best** — never overrides an in-flight alpha-beta bound, only informs root move choice or a leaf's static eval the same way TB-probe-failure-fallback already works | Low-confidence positions still fall through to normal search/NNUE, bounding the downside | Medium-high (needs calibrated confidence, not just accuracy) |
| **E: pruning/extensions** | Unpredictable, real regression risk | Small per-call, but gates a hot path | High | **Worst** — an oracle-driven pruning/extension decision that's wrong doesn't just misjudge a value, it can make search skip a refutation entirely, a silent, hard-to-detect correctness bug class this codebase's own zugzwang-guard precedent (§A) shows the authors are already careful about avoiding | Silently pruned wins/missed refutations in exactly the sharp endgames where correctness matters most | Highest — needs near-tablebase reliability specifically in the positions where pruning decisions are made |

**Recommendation within this table** (not recommended overall, §1): Option D
only, and only as a later-stage project.

### D. Training pipeline design (if pursued)

Since no exact solver exists at 8-10 pieces (§7 Finding #1), the realistic
pipeline is:

```
Syzygy 7-piece (exact, boundary/validation set)
  +
Deep Stockfish-class search OR this engine's own deep search (noisy teacher)
  +
Self-play endgames (noisy teacher, same caveats as this session's Phase B)
        |
        v
Training positions (labels are APPROXIMATE past 7 pieces, by construction)
        |
        v
Small neural network (3-way WDL head + confidence)
        |
        v
Confidence-gated endgame module (NOT "compressed exact knowledge")
```

- **Dataset size**: likely needs to be very large (tens of millions of
  positions) to cover the material-configuration combinatorics of 8-10
  pieces at all, let alone learn it reliably — no confident estimate is
  possible without a pilot study, itself weeks of work.
- **Model parameters**: can be small (the whole appeal is a compact,
  fast-inference specialist net) — comparable to or smaller than the
  production NNUE's 512-wide accumulator, plausible.
- **Training time / GPU requirements**: modest once data exists (small net,
  small-ish input space relative to full NNUE) — the dataset generation and
  label-quality work dominates total cost, not the training run itself.
- **Inference cost**: must be cheap enough for root/leaf-only calls
  (Option D) not to matter — plausible for a small net, but every non-NNUE
  forward pass is still orders of magnitude more expensive than the
  incremental NNUE accumulator update this engine's search relies on for
  its per-node speed, reinforcing why Options A/E (per-node use) are
  unrealistic (§C).
- **Expected accuracy**: unknown, and per §B's evidence from Stockfish/Lc0
  struggling at 4-5 pieces, likely materially below "tablebase quality" —
  this is the crux of the NO-GO.
- **Storage savings**: not a relevant metric here — this isn't a
  storage-compression project (§B distinction), it's a knowledge-extension
  project, so "smaller than a hypothetical 8-10 piece tablebase" isn't a
  meaningful comparison since no such tablebase is being built or replaced.

### E. Critical questions, answered directly

1. **Can a neural network realistically approximate 8-10 piece tablebase
   knowledge?** Partially, with real, hard-to-bound error — not to
   tablebase-equivalent reliability. No primary source was found
   demonstrating this at production quality, and the closest available
   evidence (§B, Sadmine/Husna/Müller 2023) shows measurable error from
   far-better-resourced systems at an easier (4-5 piece) version of the
   problem.
2. **What accuracy is achievable?** Not confidently estimable without a
   pilot study; extrapolating from §B's 4-5-piece numbers pessimistically
   (larger space, less compute/data than Stockfish/Lc0 have), plausibly
   single-digit-to-low-double-digit percent mistake rates in exactly the
   positions where being wrong matters most (won/drawn boundary cases).
3. **Would errors be acceptable in alpha-beta search?** Depends entirely on
   the integration option (§C table) — acceptable for Option D
   (confidence-gated, falls back to normal search), unacceptable for
   Options A/E (directly corrupts search-tree correctness).
4. **How much Elo would this realistically add?** Low — see §2. Bounded
   above by how often real games (at this engine's current, pre-NNUE-training
   strength) even reach 8-10-piece positions where the oracle and normal
   play would meaningfully diverge; bounded below by zero or slightly
   negative if confidence-gating isn't done carefully.
5. **Is this better than spending the same time improving NNUE?** No — not
   close. NNUE training is proven, working, and estimated at 10-40x better
   Elo-per-week (§2, §11).
6. **Could confidence estimation prevent harmful mistakes?** Yes, this is
   the one part of the proposal that's genuinely sound and worth keeping if
   this is ever revisited — a calibrated WDL-softmax-derived confidence
   score, falling back to NNUE/search whenever confidence is low, is exactly
   the right shape of safety mechanism, and this codebase already has a
   working precedent for "small learned correction, applied conditionally"
   in its correction-history system (§A) to model it on.

### F. Comparison against alternatives

| Option | Elo potential | Development time | Compute cost | Risk |
|---|---|---|---|---|
| **A: Train current NNUE properly** | **+150 to +400** (Stockfish's own historical precedent; net is fully built, never trained) | 3-7 days | Low-moderate (Bullet + one rented GPU, hours) | Low — proven this session, doesn't touch search |
| **B: Improve search** | +10-40 per change (this session's own measured recommendation #1/#2 results) | Days per change, many changes available | None | Low — proven, incremental, this session's established discipline |
| **C: Build neural endgame oracle** | **0 to +10, plausibly negative if mishandled** | 4-8 weeks | Moderate (large dataset generation dominates) | **High** — no exact teacher past 7 pieces, real search-soundness risk depending on integration option |
| **D: Transformer teacher → NNUE distillation** | +50-150 additive on top of A (per the earlier architecture report) | 3-5 days on top of A | Moderate (batched offline GPU inference over an existing free checkpoint) | Low — zero production-code risk, doesn't touch search |

**Elo-gain-per-week, roughly**: A ≈ 30-80/week, B ≈ 5-15/week per item
(many available), D ≈ 15-30/week (stacks on A), **C ≈ 0-2/week** — the
worst ratio of the four by a wide margin, which is the whole basis for the
§1 NO-GO.
