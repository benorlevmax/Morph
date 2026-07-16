# Training Server

`training_server/` is the backend that turns generated positions into
progressively stronger NNUE networks, automatically:

```
import positions -> clean/dedupe/validate -> batch -> train ->
export -> engine benchmark -> Elo test -> accept/reject -> experiments/net_XXX/
```

It does not touch engine strength code. It only drives the existing,
unmodified `chess` UCI binary (benchmark, verify, self-play match) as a
subprocess and trains a network file consumed by `src/nnue/`, which itself
is not changed.

It's the third piece of the data/training stack, built on top of the other
two rather than duplicating them:

* `tools/nnue_pipeline/` — single-machine generate/train/export/test scripts
  (train, export, and test are reused here via subprocess).
* `distributed/` — multi-machine self-play position generation
  (`server/validation.py`'s FEN/eval/result validation and dedup hash are
  reused here, and `training_server` can import straight from a distributed
  run's SQLite database).
* `training_server/` (this doc) — dataset versioning, GPU/CPU training with
  checkpoint/resume, automatic evaluation, and experiment tracking on top of
  both.

## Directory layout

```
training_server/
  config.py              shared paths (datasets dir, experiments dir, etc.)
  pipeline.py             the one-command orchestrator (run this)
  experiment.py            experiments/net_XXX/ folder lifecycle
  dataset/
    import_data.py          import + clean + validate + dedupe -> dataset version
    batches.py               deterministic train/val split + batching
  training/
    train.py                  checkpoint/resume training (wraps nnue_pipeline/train.py)
    gpu.py                     GPU/cargo detection, engine auto-selection
  evaluation/
    evaluate.py                export -> benchmark -> Elo match -> accept/reject
  datasets/                 imported dataset versions live here (gitignore-able)

experiments/                 one folder per training run
  net_001/
    config.json               training config: dataset version, hyperparams, engine, date
    logs/                     pipeline.log, train.log, metrics.jsonl
    checkpoints/              latest.npz (resumable training checkpoint)
    network/                  net_001.nnue (exported, quantized)
    results.json              validation score, benchmark results, Elo match, verdict
  net_002/
    ...
```

Every run's `experiments/net_XXX/` folder is a complete, self-contained
record — `config.json` + `results.json` together cover everything the spec
requires: network file, training configuration, dataset version, date,
validation score, and benchmark results.

## GPU requirements

Two independent things enable the GPU-accelerated path (the real
[Bullet](https://github.com/jw1912/bullet) NNUE trainer):

* An NVIDIA GPU, detected via `nvidia-smi`.
* A Rust toolchain (`cargo`), needed to build/run Bullet.

If either is missing, `training_server` automatically falls back to a
bundled NumPy CPU reference trainer (`tools/nnue_pipeline/train.py --engine
reference`) that implements the same NNUE forward/backward pass, just without
GPU throughput. This is not a stub — it produces real, correctly trained
checkpoints, just slower. Check what will be used before starting:

```sh
python3 training_server/training/gpu.py
```

```
=== GPU / training-engine report ===
  GPU available: False  []
  cargo (Rust) available: False
  recommended engine: reference  (no GPU detected (nvidia-smi unavailable or reports no devices))
```

Force a specific engine with `--engine reference` or `--engine bullet` on
`pipeline.py` (bypasses auto-detection); `--engine bullet` requires
`--bullet-dir` pointing at a checked-out Bullet repo if it's not at the
default `tools/nnue_training/bullet_trainer/`.

**Caveat:** the sandbox this was built and tested in has neither a GPU nor
`cargo`, so only the `reference` engine path has been exercised end-to-end
here. The `bullet` path is implemented (subprocess shell-out to `cargo run
--release`, same checkpoint/resume/metrics contract as the reference
trainer) but untested on real hardware — verify it on a GPU machine before
relying on it for a real training run.

## Starting training

### 1. Get positions into a dataset version

From a distributed-generation database:

```sh
python3 training_server/dataset/import_data.py --use-default-distributed-db
```

From one or more local `nnue_pipeline` JSONL files, or both sources at once:

```sh
python3 training_server/dataset/import_data.py \
  --distributed-db distributed/database/distributed.sqlite3 \
  --jsonl tools/nnue_pipeline/data/positions.jsonl
```

This validates every position (legal FEN, `side_to_move` matches the FEN,
`eval_cp` in range, `result` in {0, 0.5, 1}, sane depth/nodes,
non-empty `engine_version`), drops duplicates by content hash
(`sha256(fen|eval|result|depth|engine_version)`, the same hash `distributed/`
uses), and writes `training_server/datasets/<version>/all.jsonl` +
`manifest.json`. The version id (`v_<timestamp>_<hash>`) is deterministic
from the cleaned content, so re-importing the same data reproduces the same
version.

You normally don't need to run this by hand — `pipeline.py` (below) does it
for you unless you pass `--dataset-version` to reuse an existing one.

### 2. Run the pipeline

```sh
python3 training_server/pipeline.py \
  --use-default-distributed-db \
  --engine-bin build/bin/Release/chess \
  --epochs 10 --batch-size 4096
```

This does everything: creates `experiments/net_XXX/`, imports/batches the
dataset (or reuses `--dataset-version <id>` if given), trains
(checkpointing every epoch to `checkpoints/latest.npz`), exports a quantized
`.nnue`, benchmarks it against the compiled engine, runs an Elo match against
a baseline, and writes the accept/reject verdict.

Common flags:

| Flag | Meaning |
|---|---|
| `--dataset-version v_...` | reuse an already-imported dataset instead of re-importing |
| `--distributed-db PATH` / `--use-default-distributed-db` / `--jsonl PATH...` | dataset sources |
| `--engine {auto,reference,bullet}` | training engine (default: auto-detect) |
| `--epochs N` / `--batch-size N` / `--lr F` | training hyperparameters |
| `--resume-checkpoint PATH` | resume an interrupted run from a saved `.npz` checkpoint |
| `--qa N` / `--qb N` | export quantization scales |
| `--engine-bin PATH` (required) | compiled `chess` UCI binary used for benchmark/verify/match |
| `--bench-depth N` / `--match-games N` / `--match-depth N` | evaluation stage tuning |
| `--reject-elo-threshold F` | reject if candidate Elo vs. baseline is below this (default -15.0) |
| `--baseline-net PATH` | override auto-selected baseline network |

### Resuming an interrupted job

Every epoch's checkpoint is written to `experiments/net_XXX/checkpoints/`.
To resume:

```sh
python3 training_server/pipeline.py --dataset-version v_20260715_... \
  --resume-checkpoint experiments/net_005/checkpoints/latest.npz \
  --engine-bin build/bin/Release/chess --epochs 10
```

This starts a **new** experiment folder (a resumed run is still its own
tracked experiment) but continues optimizing from the saved weights instead
of from scratch.

### Testing with a small dataset first

Before committing to a full run, sanity-check the whole chain on a tiny
slice of data and a fast, low-depth evaluation:

```sh
python3 training_server/pipeline.py \
  --jsonl /path/to/small_sample.jsonl \
  --engine-bin build/bin/Release/chess \
  --epochs 1 --batch-size 128 --val-fraction 0.05 \
  --bench-depth 6 --match-games 4 --match-depth 2 \
  --reject-elo-threshold -10000
```

(`--reject-elo-threshold -10000` effectively forces an accept regardless of
the tiny match's noisy Elo estimate, useful for confirming the pipeline runs
end-to-end without a real accept/reject judgment call on 4 games.) This
full chain — import 300 positions, batch, train 1 epoch, export, verify 8/8
against the compiled engine, benchmark, run a 4-game Elo match, and write
`experiments/net_XXX/` — has been run and verified successfully as part of
building this backend.

## Comparing networks

Every run's accept/reject decision already compares the new network against
a baseline:

* **Baseline selection**: the most recent experiment whose `results.json`
  has `"verdict": "accept"`. If none exists yet, the network is compared
  against the classical (non-NNUE) evaluator instead. Override with
  `--baseline-net PATH`.
* **Verification**: the exported `.nnue` must produce the exact same
  evaluation as the training checkpoint on a set of sanity positions
  (`test.py`'s `verify` step) — a mismatch is an automatic reject, since it
  means export/quantization broke something.
* **Benchmark**: `bench` node/NPS comparison between candidate and baseline
  at `--bench-depth`, recorded in `results.json`'s `benchmark_results`.
* **Elo match**: a self-play match (`--match-games` games at
  `--match-depth`) between candidate and baseline, decided with the same
  GSPRT statistics used by `tools/nnue_pipeline/uci_match.py`. Recorded in
  `results.json`'s `test_report.elo_match` (estimated Elo, confidence
  interval, SPRT status).
* **Verdict**: reject if verification fails, reject if Elo is below
  `--reject-elo-threshold`, otherwise accept.

To manually compare any two already-exported networks outside the
pipeline, use `tools/nnue_pipeline/test.py --net A.nnue --baseline-net
B.nnue` directly (this is exactly what `evaluate.py` calls).

To see the accept/reject and baseline lineage across all runs:

```sh
python3 -c "
from training_server.experiment import list_experiments
for e in list_experiments():
    r = e.get('results', {})
    print(e['id'], r.get('verdict'), 'baseline=' + str(r.get('baseline_experiment')))
"
```

## What's untested here

This was built and verified in a sandbox with no GPU and no Rust toolchain,
so:

* The `reference` (NumPy CPU) training path is fully exercised end-to-end
  (multiple runs, accept/reject/baseline-selection all confirmed working).
* The `bullet` (GPU) path is implemented against the same
  checkpoint/resume/metrics contract but has not run on real hardware —
  test it on a GPU machine before depending on it.
