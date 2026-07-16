# Phase A: Production NNUE Training Pipeline — Audit, Build, and Proof

Status: **chain proven end-to-end.** DATA → FEATURES → TRAIN → QUANTIZE → .NNUE → C++ INFERENCE
was built and verified in this sandbox using a hand-written reference trainer
(numpy), because the sandbox has no internet access, no Rust toolchain, and
no GPU (see Finding #7). The real Bullet trainer config is written and ready
to run on a properly provisioned machine, but has **not** been compiled or
executed anywhere — that is the one link in the chain still unverified.

No code in `src/` was modified. Every change is additive, under
`tools/nnue_training/`.

---

## 1. Production NNUE format (`src/nnue/nnue.h`, `nnue.cpp`)

- **Feature set**: HalfKP-style but NOT the classical HalfKP. 16 king buckets
  (a 4×4 grid: `(rank/2)*4 + (file/2)` over the king's *oriented* square, not
  all 64 king squares individually), × 64 squares × 10 piece-relative types
  (PAWN..QUEEN × own/opp; kings excluded from the feature set entirely) =
  **10,240 features**.
- **Perspective orientation**: `orient(persp, s) = persp==WHITE ? s : s^56`
  — a **rank flip only**. There is no file mirroring for the king's side
  (queenside vs kingside), unlike some engines' bucket schemes (e.g. Bullet's
  built-in `ChessBucketsMirrored`).
- **Accumulator**: `int16[COLOR_NB][512]`, dual-perspective, 32-byte aligned.
- **Activation**: plain (non-squared) clipped ReLU, range **[0, 32767]**
  (`CR_MIN=0, CR_MAX=32767`), applied to the raw int16 accumulator before the
  dot product with output weights.
- **Output layer**: 8 buckets selected by `(popcount(pieces)-1)/4`, clamped
  to `[0,7]`. Each bucket has its own `int16[2*512]` weight row (own-then-opp
  concatenation) and `int32` bias.
- **Final scalar**: `cp = (outBias[bucket] + Σ clipped(acc)·w) / scale`,
  where `scale` is a single `int32` stored in the file header — integer
  division, **truncating toward zero** (C++ semantics for `int64_t/int32_t`).
- **Binary layout** (`write_net`/`load`): `u32 MAGIC(0x4B504E32) | u32 VERSION(2)
  | u32 NNUE_FEATURES | u32 NNUE_HL | u32 NNUE_OUT_BUCKETS | i32 scale |
  i16 ftBias[512] | i16 ftWeights[10240][512] (feature-major) |
  i16 outWeights[8][1024] (bucket-major, own-then-opp) | i32 outBias[8]`.
- **Current state**: no `.nnue` file exists anywhere in the repo.
  `NNUE::init()` builds a PSQT-equivalent default in memory; `load()`/`save()`
  exist and work (verified, see §4) but had never been exercised against a
  real trained network before this session.
- `EvalFile` is a live UCI option (`src/uci/uci.cpp:135-137`) that calls
  `NNUE::load(value)` — loading a trained net into the real engine requires
  no code changes, just `setoption name EvalFile value <path>`.

## 2. Training-side code audit (`src/train/`)

- `Dataset::save_bullet()` (`dataset.cpp:33-43`) emits `<FEN> | <score> |
  <wdl>` text, white-relative centipawns and white-relative `1.0/0.5/0.0`.
  **This is byte-for-byte the format the current, official Bullet expects**
  (confirmed directly against `docs/3-data.md`, quoted in Finding #6 below)
  — no changes needed on the data side.
- `chess_train gen --format bullet --out <path>` (already-existing CLI,
  `src/apps/train_main.cpp:39-63`) generates this format directly from
  self-play with **zero new C++ code**. Used to generate the smoke dataset
  (§5): 300 self-play games, depth 5, 6 random opening plies → **39,218
  samples in 21.3s** (1,841 samples/sec, single-threaded, this sandbox).
- `src/train/encoding.h`/`trainer.cpp`/`trainer_torch.cpp` are a **separate,
  disconnected 768-feature flat MLP** (2×6×64 one-hot, not king-relative),
  explicitly NOT the production NNUE shape. Per the user's explicit
  instruction, **none of this was touched or trained.**

## 3. Bullet compatibility findings

Researched against the current `jw1912/bullet` repo directly (fetched
2026-07-14; MIT license, ~215 GitHub stars, actively maintained, used by
Halogen/PlentyChess/Reckless/Akimbo and many other current top open-source
engines per TalkChess/CCRL discussion).

**Finding #1 — data format: fully compatible, no changes needed.**
`docs/3-data.md` (quoted): *"you can convert to this data type from a text
file... each line is of the form `<FEN> | <score> | <result>` — score is
white relative and in centipawns — result is white relative and of the form
`1.0` for win, `0.5` for draw, `0.0` for loss."* This is exactly
`Dataset::save_bullet()`'s output. Confirmed compatible without inspecting a
single byte of Bullet's own source — the formats were designed independently
and happen to match, which is a lucky, not guaranteed, coincidence worth
flagging for future changes to either side.

**Finding #2 — architecture: no built-in preset matches ours; a custom
`SparseInputType`/`.inputs()` implementation is required, and IS supported.**
Bullet's docs are explicit that "custom bucket schemes often serve better
with less data" and that arbitrary architectures are first-class (the
`ValueTrainerBuilder` pattern, `.inputs(...)`, `.output_buckets(...)`,
`.save_format(...)`, `.build(|builder, stm, ntm| {...})` closure). No
built-in input type (`Chess768`, `ChessBucketsMirrored`) matches our exact
16-bucket/64-square/no-file-mirror/10-piece-relative-type scheme — the
closest built-in, `ChessBucketsMirrored`, mirrors by file, which ours never
does. **A custom feature-index implementation is required** (drafted in
`tools/nnue_training/bullet_trainer/src/main.rs`, marked unverified —
see Finding #7).

**Finding #3 — output format is fully general and can match our binary
layout exactly, with no repacking needed at conversion time, IF the
`SavedFormat` field order is chosen correctly.** Per `docs/4-saved-networks.md`:
weights are written in the order you list them, each field independently
`.quantise::<T>(Q)`-able, `.round()`-able, `.transpose()`-able. Setting
`.save_format(&[l0b, l0w, l1w, l1b])` in that order produces a `quantised.bin`
byte-for-byte matching `ftBias | ftWeights | outWeights | outBias` — our
converter (`bullet_checkpoint_to_nnue.py`) only needs to prepend our 24-byte
header, no reordering.

**Finding #4 (real bug/mismatch, engineering-relevant) — `CR_MAX=32767` makes
the engine's "clipped ReLU" functionally plain ReLU for any sane quantization
factor.** Bullet's own reference inference code (`examples/simple.rs`) clips
the accumulator to `[0, QA]` — the clip bound IS the quantization factor, by
design (that's what makes clipped-ReLU meaningful as a regularizer/overflow
guard). Our engine hardcodes `CR_MAX=32767` independent of whatever `QA` a
trained net used. For any reasonable `QA` (64–512), accumulator values will
essentially never approach 32767, so clipping never fires. **This is not a
correctness bug** (both the reference trainer and the real engine apply the
exact same, consistently-defined activation, so training/inference stay
matched — verified in §5) but it is a real architectural divergence from
standard NNUE practice, worth fixing in a future revision if clipping-based
regularization is desired (change `CR_MAX` to a named constant matching the
chosen `QA`, or make it net-format-versioned).

**Finding #5 (real subtlety, resolved) — Bullet's `eval_scale` training
parameter must be folded into the exported quantization scale correctly, or
every evaluation will be off by a constant multiplicative factor.** Bullet's
reference inference (`examples/simple.rs`, bottom) computes `cp = raw_sum *
EVAL_SCALE / (QA*QB)`; our engine computes `cp = raw_sum / scale` with no
separate `EVAL_SCALE` multiply. Equating the two: `scale = (QA*QB)/EVAL_SCALE`.
This is only an exact integer if `EVAL_SCALE` divides `QA*QB` cleanly.
**Resolution**: choose `QA == EVAL_SCALE` (both 400, matching this project's
own `sigmoid_eval(cp) = sigmoid(cp/400)` convention in `encoding.h`), which
makes `scale == QB` exactly — clean, no rounding bias. This is baked into
`bullet_trainer/src/main.rs` (`QA=400, EVAL_SCALE=400.0, ENGINE_STORED_SCALE=QB=128`)
with the derivation spelled out in comments. **This is exactly the kind of
architecture mismatch the task asked to be surfaced** — it would have
silently produced a net that evaluates every position at the wrong
magnitude (off by a constant factor, `EVAL_SCALE/QA` if mishandled) while
still "working" (loading fine, producing plausible-looking cp values), which
is the most dangerous kind of bug for this task.

**Finding #6 (data-loader detail).** Bullet recommends `bulletformat`/
`ChessBoard` (loaded via `DirectSequentialDataLoader`) only for small
networks / small datasets, and recommends binpack-style formats
(`SfBinpackLoader`, `ViriBinpackLoader`) for anything data-loading-bottlenecked
at scale. Our `save_bullet()` text format is the ChessBoard/ `bulletformat`
family (via `bullet-utils`' `convert`), which is the right choice for Phase
A/B-scale work and should be revisited (convert to a binpack, or generate
directly in binpack form) if/when a full-scale training run's data volume
makes text-format loading the bottleneck.

**Finding #7 (environment, not architecture) — the real Bullet trainer has
NOT been compiled or run anywhere.** This sandbox has: no outbound network
access from the shell (`curl` to `static.rust-lang.org` → `403 Forbidden,
X-Proxy-Error: blocked-by-allowlist`; confirmed for `crates.io` and
`raw.githubusercontent.com` too), no `cargo`/`rustc` installed, no GPU
(`nvidia-smi`: not found), and only 2 CPU cores / 3.8GB RAM. Real Bullet
(Rust + optionally CUDA/ROCm/Metal) cannot be built or exercised here at
all. `tools/nnue_training/bullet_trainer/` contains a structurally-correct,
heavily-commented draft written directly against the current API examples
fetched from `github.com/jw1912/bullet` (`examples/simple.rs`,
`examples/progression/4_multi_layer.rs`), with the one genuinely
hand-derived part — the custom `SparseInputType` feature-index
implementation — left as verified-correct pseudocode (ported 1:1 from
`reference_nnue.py`, which IS verified, see §5) pending a real `cargo doc`
check on a machine that has Rust. **Do not run this file without first
confirming the trait signatures against `cargo doc -p bullet_lib` on your
own machine.**

## 4. Reference-verification harness

`tools/nnue_training/reference_nnue.py` is a line-by-line Python port of
`nnue.cpp`'s `feature_index`/`king_bucket`/`orient`/`output_scalar`/`load`/
`save`, including exact C++ truncating-integer-division semantics (Python's
`//` floors; a `truncating_div` helper was written to match C++'s
truncate-toward-zero behavior for negative sums — this matters, and get it
wrong and you get silent, small, sign-dependent evaluation errors).

`tools/nnue_training/verify_against_engine.py` computes centipawn evals for
10 diverse FENs (opening, middlegame, endgame, mate-adjacent, castling
rights, en passant square) via the Python oracle, then drives the real
compiled `chess` UCI binary with the same `.nnue` file loaded via
`EvalFile`, and asserts **exact integer equality** (not "close enough" —
these are bit-identical integer computations on both sides; any drift means
a real bug).

## 5. Smoke run: chain proven end-to-end

Ran with `tools/nnue_training/train_reference.py` (a plain-numpy, CPU-only,
**not-Bullet** stand-in trainer — see Finding #7 for why the real Bullet
config couldn't be executed here) against the exact production shape
(10,240 features, 512-wide accumulator, 8 output buckets):

1. **Data**: `chess_train gen --games 300 --depth 5 --randomplies 6 --format
   bullet` → 39,218 samples, 21.3s.
2. **Train**: 3,000-sample subset, 2 epochs, batch 256, Adam, `QA=256,
   QB=256` → MSE 0.0721 → 0.0661 (win-prob target, `λ=0.5` blend of
   `sigmoid(eval/400)` and game result, matching `encoding.h`'s existing
   convention), 12.9s wall time.
3. **Quantize**: all weights/biases fit `int16`/`int32` cleanly at this
   scale (overflow check passed; the trainer fails loudly and refuses to
   write the file otherwise, per the explicit instruction).
4. **Export**: `smoke_A.nnue`, 10.5MB, in the exact production binary
   format (archived at `tools/nnue_training/smoke_run_evidence/smoke_A.nnue`).
5. **Verify**: `verify_against_engine.py` against 10 FENs (opening,
   middlegame, endgame, mate-adjacent) — **10/10 exact matches**
   (`python=-3 engine=-3`, `python=8 engine=8`, ... `python=-23 engine=-23`).
   Zero mismatches. This is the strongest evidence available that feature
   indexing, king bucketing, perspective orientation, output-bucket
   selection, quantization, and the binary format are all correct and
   mutually consistent between training and inference.
6. **Engine integration**: loaded via `EvalFile`, ran `bench 12` —
   17,263,198 nodes, 1,860,859 NPS, Finny cache hit rate 99.99%, no crashes,
   no assertion failures. All 5 CTest suites still pass (perft/fen/
   search/eval/pgn — unaffected, since no `src/` code changed).
7. **Sanity match**: 6 quick games (0.2s/move) vs. the classical evaluator
   via `smoke_sanity_match.py` — `+0 −6 =0` for the smoke net. **Expected
   and unconcerning**: 3,000 positions / 2 CPU-epochch is far too little
   data/training for a barely-initialized 10,240-input net to compete with
   a hand-tuned classical evaluator; this was a functional smoke test
   (no crashes/hangs/illegal moves), not a strength measurement, exactly as
   scoped ("small Elo sanity match", not SPRT).

## 6. What's proven vs. what's still open

**Proven**: the full architectural chain (feature format, king bucketing,
perspective handling, output bucketing, quantization scheme, binary layout,
UCI loading path) is internally consistent and bit-exact-verified end to end,
using a real dataset generated by this engine's own self-play and a real
(if small and CPU-bound) gradient-descent training run.

**Not yet proven** (all environment-blocked, not architecture-blocked):
- The real Bullet trainer (`bullet_trainer/src/main.rs`) has not been
  compiled — needs a machine with internet + Rust. Its `SparseInputType`
  impl is pseudocode pending a `cargo doc` check.
- `bullet_checkpoint_to_nnue.py` has not been run against a real Bullet
  `quantised.bin` — only against `reference_nnue.py`'s own output.
- No GPU-scale training (real Bullet run would be orders of magnitude
  faster and would use far more data) has happened; the numpy stand-in
  trained on 3,000 positions for 2 epochs is a pipeline proof, not a
  strength result.

## 7. Recommended next step (not yet executed — awaiting direction)

On a machine with internet + Rust (+ ideally a GPU): `cargo check` the
trainer, fix up the `SparseInputType` implementation against real trait
signatures (port `reference_nnue.py`'s formula verbatim), point it at a
larger self-play + Lichess-derived dataset, run it, then run
`bullet_checkpoint_to_nnue.py` and `verify_against_engine.py` on the
resulting `quantised.bin` before trusting it in any Elo test — exactly the
same discipline used here, just with real Bullet instead of the numpy
stand-in.
