# NNUE Training Pipeline

`tools/nnue_pipeline/` is a complete, automated generate -> train -> export ->
benchmark -> Elo-test loop for this engine's NNUE network. It is a thin
orchestration layer around the *existing, unmodified* engine: it drives the
compiled `chess_train`/`chess` binaries as subprocesses and never touches
`src/search`, `src/eval`, or NNUE inference. Nothing about how the engine
searches, orders moves, prunes, or evaluates changed to build this pipeline.

For the deeper story on real [Bullet](https://github.com/jw1912/bullet)
integration (the GPU trainer this pipeline can optionally shell out to), see
`docs/NNUE_TRAINING_BULLET.md`. This document covers the pipeline you'll
actually run day to day.

## Quickstart: one command

```sh
cd tools/nnue_pipeline
python3 train_nnue.py
```

(or `./train_nnue` on macOS/Linux, or `python tools\nnue_pipeline\train_nnue.py`
on Windows). This runs, with small smoke-test-sized defaults:

```
generate data  ->  train network  ->  export network  ->  benchmark  ->  Elo test
```

and writes everything under `tools/nnue_pipeline/runs/<run_id>/`:
`data.jsonl` (the dataset), `checkpoints/latest.npz` (raw trained weights),
`net.nnue` (the production network), `test_report.json` (benchmark + Elo
results), and `run_summary.json` (stage timings and paths).

A realistic-scale run:

```sh
python3 train_nnue.py --games 5000 --gen-depth 7 --epochs 15 \
    --max-samples 2000000 --match-games 200
```

Every flag is documented via `--help` on each stage script; `train_nnue.py
--help` lists the subset it forwards.

## Prerequisites

The engine must already be built (this pipeline never invokes the build
system):

```sh
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The pipeline looks for `chess_train(.exe)` and `chess(.exe)` in
`build/bin/Release/` or `build/bin/`; pass `--bin-dir` to point elsewhere.

Python side: Python 3.9+, `numpy` (required), `python-chess` (optional but
recommended -- used by `test.py`'s Elo match to detect checkmate/stalemate/
draws; without it, games run to a fixed ply cap with no adjudication). Install
with:

```sh
pip install numpy python-chess
```

## The four stages

### 1. `generate.py` -- self-play data generation

Wraps `chess_train gen --format bullet` (the engine playing itself at a fixed
search depth, already implemented in `src/train/selfplay.cpp`) and re-emits
its output as JSONL, one training sample per line:

```json
{"fen": "...", "result": 0.5, "eval": 17, "depth": 6,
 "engine_version": "Morph 0.5", "generated_at": "...", "run_id": "..."}
```

`result` is the eventual game outcome (White-relative, 1.0/0.5/0.0); `eval`
is the engine's own search score at generation time. Every run also appends a
provenance line to `data/manifest.jsonl`.

```sh
python3 generate.py --games 2000 --depth 7 --randomplies 6 --out data/run1.jsonl
```

### 2. `train.py` -- train a network

Trains the exact production architecture (HalfKP, 10,240 features, 16 king
buckets, 512-wide dual-perspective accumulator, clipped-ReLU, 8 output
buckets -- see `src/nnue/nnue.h`) against one or more `generate.py` datasets.

Two engines:

* `--engine reference` (default) -- a NumPy Adam trainer bundled in this
  pipeline (`nnue_format.py` + the training loop in `train.py`). Runs
  anywhere Python does, CPU-only. This is a **correctness-first stand-in for
  Bullet**, not a competitor to it at scale (see Hardware below).
* `--engine bullet` -- shells out to `cargo run --release` in a real Bullet
  trainer crate (`--bullet-dir`, default `tools/nnue_training/bullet_trainer`,
  the custom HalfKP crate written for this project). Requires the Rust
  toolchain and is only sensible with a GPU. Raises a clear error if `cargo`
  isn't found rather than silently falling back.

Checkpoints (`.npz`: raw float32 weights + Adam state + step/epoch counters,
**not yet quantized**) are written every epoch to `--out`, with `latest.npz`
always pointing at the newest one:

```sh
python3 train.py --data data/run1.jsonl --out checkpoints/run1 --epochs 15
# ...later, continue training the same net:
python3 train.py --data data/run2.jsonl --out checkpoints/run1 \
    --resume checkpoints/run1/latest.npz --epochs 5
```

### 3. `export.py` -- quantize to `.nnue`

Converts a `train.py` checkpoint (or a real Bullet `quantised.bin`, via
`--bullet-quantised`) into the engine's production binary format ("NNU2",
`src/nnue/nnue.cpp`'s `write_net()`/`load()`). Every weight is range-checked
against int16 (int32 for `outBias`) and the export **fails loudly** on
overflow rather than silently clamping a trained value:

```sh
python3 export.py --checkpoint checkpoints/run1/latest.npz --out nets/run1.nnue \
    --qa 256 --qb 256
```

It also round-trips the file it just wrote and prints two sanity evaluations
before exiting, so a badly-formed export is caught immediately.

### 4. `test.py` -- validate before you trust it

Four gated checks:

1. **load** -- parses the `.nnue` header and confirms it matches the compiled
   engine's architecture constants.
2. **verify** -- drives the real `chess` binary via UCI (`setoption name
   EvalFile`, then `eval` on 8 fixed positions) and asserts every score
   matches the pure-Python reference implementation *exactly* (integer
   centipawns). This is the same technique used to verify the original NNUE
   architecture change and it catches feature-indexing, quantization, or
   byte-layout bugs before they reach a slow match.
3. **benchmark** -- runs the engine's built-in `bench` command with the
   candidate net loaded vs. the baseline (a previous `.nnue`, or the
   classical evaluator if none given), reporting nodes/NPS/time for both.
4. **Elo match** -- plays an automated match (candidate vs. baseline) and
   reports W/L/D, an Elo estimate with a 95% margin, and a GSPRT verdict.

```sh
python3 test.py --net nets/run1.nnue --baseline-net nets/prev.nnue --games 200
python3 test.py --net nets/run1.nnue                       # vs. classical eval
```

Exits non-zero if steps 1-2 fail (a broken net); a losing Elo result in step
4 is a valid, correctly-reported outcome, not a pipeline failure.

**Why not the existing `chess_match` binary for the Elo match?**
`chess_match` (`src/match/match.cpp`) toggles Classical vs. NNUE per side but
has no per-side `EvalFile` path -- both sides share one globally loaded net,
so it can't compare two *different* trained `.nnue` files. Rather than modify
`match.cpp` (playing-strength code, out of scope here), `test.py` drives two
independent `chess` UCI processes itself (`uci_match.py`), reusing the same
12-opening book as `match_main.cpp` for comparable results. Its Elo/SPRT math
is a standard normal-approximation (the same style used by cutechess-cli/
fishtest), not a byte-for-byte port of `src/match/stats.cpp` -- treat match
results here as indicative, and re-confirm anything close to a shipping
decision with the built-in `chess_match --sprt` harness once you're ready to
compare against the classical evaluator specifically.

## Required hardware

| | Reference trainer (default) | Real Bullet (`--engine bullet`) |
|---|---|---|
| CPU | Any (single-thread NumPy) | Any (host process) |
| GPU | Not used | Strongly recommended (CUDA) -- Bullet is GPU-oriented and CPU training a 10,240x512 net at Bullet's normal scale is impractical |
| Rust toolchain | Not needed | Required (`cargo`) |
| Disk | A few MB per 100k samples (JSONL) + ~21MB per `.nnue`/checkpoint | Same, plus Bullet's own packed binary cache |

The reference trainer will run on essentially any machine that can run this
engine's build, including this sandbox (no GPU, no Rust, 2 CPU cores) -- that
constraint is exactly why it exists (see `docs/phaseA_nnue_bullet_audit.md`
for the full audit of what's and isn't available). It is not a substitute for
Bullet's training throughput at real scale; use it to validate the pipeline
and produce small/experimental nets, and switch to `--engine bullet` on a
GPU machine once you're training for strength.

## Expected runtime

Measured in this project's sandbox (2 CPU cores, no GPU) with the reference
trainer, as a concrete baseline -- scale roughly linearly with core count/
clock speed, and expect large speedups on a real workstation:

| Stage | Measured rate | Example |
|---|---|---|
| `generate.py` | ~2,500-3,700 samples/s at depth 3; ~650 samples/s at depth 6 (deeper search = slower labels, Phase A measurement) | 20 games/depth 3 -> 2,557 samples in 0.7s |
| `train.py` (reference) | ~400 samples/s/epoch, single-threaded NumPy | 1,900 samples -> ~5s/epoch; 200,000 samples -> roughly 8-9 min/epoch |
| `export.py` | Seconds, dominated by writing the ~21MB `.nnue` file | < 5s |
| `test.py` verify step | Seconds (8 fixed positions) | < 5s |
| `test.py` benchmark step | Depends on `--bench-depth`; default depth 12 is a few hundred ms per side | ~0.5-1s |
| `test.py` Elo match | ~0.2-0.3s/game at shallow depth (`--match-depth 2-3`); scale up with depth/game count | 6 games @ depth 2 -> ~1s |

A "real" training run (millions of self-play positions, 20-40 epochs) on the
reference trainer would take hours to days of single-core CPU time -- at that
scale, switch to real Bullet on a GPU box, where the same job is normally
minutes. The reference trainer's job is correctness and pipeline validation
at small scale, not production-scale training throughput.

## How to load a network

In any UCI-speaking GUI or from the command line:

```
setoption name EvalFile value C:\path\to\net.nnue
setoption name Use NNUE value true
```

`NNUE::load()` validates the file's feature/hidden-layer/output-bucket counts
against the compiled engine and refuses to load an incompatible file. To make
a trained net the default, `EvalFile`/`Use NNUE` must still be set explicitly
per session (or wired into your own launch config) -- this pipeline does not
change the engine's compiled-in default.

## Directory layout

```
tools/nnue_pipeline/
  nnue_format.py     shared NNUE math + .nnue binary I/O (no engine code touched)
  engine_paths.py     locate chess_train/chess/chess_match binaries
  generate.py          stage 1
  train.py             stage 2
  export.py            stage 3
  uci_match.py         Elo-match driver + Elo/SPRT math, used by test.py
  test.py              stage 4
  train_nnue.py        one-command orchestrator (generate->train->export->test)
  train_nnue           thin shell wrapper for train_nnue.py
  data/                default dataset output + manifest.jsonl
  checkpoints/         default train.py checkpoint output
  nets/                suggested export.py output location
  runs/<run_id>/       train_nnue.py's per-run artifacts (data, checkpoints, net, reports)
```

## Known limitations / honest caveats

* The reference trainer is plain NumPy (float64 Adam state, no fused
  kernels, no GPU) -- correct, not fast. It exists because this environment
  has no Rust toolchain or GPU; see `docs/phaseA_nnue_bullet_audit.md`.
* `--engine bullet` shells out to a Bullet crate written in an earlier phase
  of this project but **has not been run against real Bullet** in any
  environment available so far (no Rust/GPU here) -- verify its first real
  output with `test.py`'s verify step before trusting it, same as any new
  net.
* `test.py`'s Elo/SPRT math is a standard approximation, not a port of
  `src/match/stats.cpp`; for a shipping decision, cross-check with
  `chess_match --sprt` (classical vs. nnue toggle) once you're comparing
  against the classical evaluator.
* Without `python-chess` installed, `test.py`'s Elo match falls back to a
  fixed ply cap with no checkmate/stalemate/draw detection -- install it for
  meaningful match results.
* Small-sample Elo/SPRT results (e.g. a handful of games, as in the smoke
  tests used to validate this pipeline) can report extreme values (e.g. a
  shutout produces a large negative Elo with zero margin) -- this is a
  statistical artifact of small samples, not a pipeline bug. Use realistic
  `--games` counts (100+) before trusting a verdict.
