# Training

## The automated improvement loop

```
current strongest network (an 'accepted' network artifact)
        |
        v
distributed DATA_GENERATION  (real self-play, many workers)
        |
        v
validated dataset  (server-side structural + plausibility checks + dedup)
        |
        v
distributed TRAIN_NETWORK  (a trainer-capable worker, real HalfKP trainer)
        |
        v
candidate network  (real trained HalfKP weights, loadable .nnue)
        |
        v
distributed ELO_MATCH  (candidate vs. current strongest, real games,
                         SPRT-gated promotion -- see below)
        |
        v
reject, or accept as the new strongest network -- repeat
```

Every arrow above is real, verified compute. This document used to record
an earlier limitation here (the pipeline stopping at an undeployable
`checkpoint` artifact); that limitation is resolved, and this section now
describes the current, real state.

## The trainer: tools/nnue_pipeline/, not src/train/

`platform/trainer/train_network.py` (the `TRAIN_NETWORK` task executor)
wraps `tools/nnue_pipeline/` (`train.py` + `export.py` + `nnue_format.py`),
a from-scratch implementation of the engine's actual production
architecture -- HalfKP, 10,240 features, 16 king buckets, a 512-wide
dual-perspective accumulator, clipped-ReLU, 8 output buckets (see
`src/nnue/nnue.h`). It trains, quantizes, and writes the exact same binary
`.nnue` format `src/nnue/nnue.cpp`'s `write_net()`/`load()` read. This was
verified end-to-end: a network trained through this path (1) loads
successfully in the compiled engine, (2) produces evaluations that exactly
match a pure-Python reference implementation on fixed test positions
(proving correct feature indexing / king-bucket mapping / output-bucket
selection / quantization), and (3) produces genuinely different
evaluations than the in-code material-baseline default. `train_network.py`
uploads its output labeled `kind='network'`, not `checkpoint` --
`ELO_MATCH` and promotion operate on it directly, no separate bridge step.

On a worker with a detected, trainable GPU (CUDA or ROCm -- see
`platform/worker/capabilities.py`) and a working Rust/cargo toolchain, the
same executor instead dispatches to `tools/nnue_training/bullet_trainer`
(real GPU training via `bullet_lib`), falling back to the CPU path above
on any failure.

`src/train/` (`RefTrainer`, `chess_train train`/`chess_train distill`) is
a **separate, self-contained correctness check**, not part of this
pipeline: it trains a flat MLP architecturally incompatible with HalfKP,
and its own `distill` command discards that net's learned weights and
emits the fixed material baseline regardless of training input (see its
source comment in `src/apps/train_main.cpp`). It exists to verify the
engine-side encode -> forward -> loss -> backprop -> checkpoint plumbing
without external dependencies, not to produce deployable networks. A
LibTorch-based version of the same idea (`trainer_torch.cpp`) existed
briefly and was removed for the same reason: never wired into the real
pipeline, and architecturally incompatible with HalfKP regardless.

## Dataset artifact format

`TRAIN_NETWORK`'s `dataset_artifact_id` must point at a `chess_train gen
--format dat` binary file (`Dataset::save()`'s own format) -- **not** the
`--format bullet` text format `DATA_GENERATION` tasks upload as individual
validated position records. Confirmed directly: pointing `chess_train
train` at a bullet-format file fails to load. Today, assembling a dataset
artifact for `TRAIN_NETWORK` means either running `chess_train gen
--format dat` directly and registering the result
(`POST /admin/artifacts`), or exporting previously-accepted positions from
the server's `positions` table into that binary format -- the latter isn't
automated yet.

## Seeding a baseline network

A fresh server has no "strongest network" until an operator seeds one --
`GET /artifacts/strongest-network` returns 404 until then, and `ELO_MATCH`
tasks need a `baseline_artifact_id` to compare against:

```bash
curl -X POST $SERVER/admin/artifacts -H "X-Admin-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"kind":"network","file_path":"/path/to/baseline.nnue","accepted":true}'
```

## Running an Elo match manually

```bash
curl -X POST $SERVER/admin/tasks/typed -H "X-Admin-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"ELO_MATCH","payload":{
        "candidate_artifact_id":"a_...","baseline_artifact_id":"a_...",
        "games":40,"match_depth":6}}'
```

Any worker (trainer-capable or not -- `ELO_MATCH` has no capability gate)
will pick this up, download and hash-verify both networks, play a real
paired-opening, color-reversed match via
`tools/nnue_pipeline/uci_match.py`, and upload the aggregated result.
Promotion now requires a passing SPRT verdict, not just a raw Elo point
estimate -- see `platform/server/auto_pipeline.py`'s
`maybe_promote_candidates()` and the "Automatic promotion" section below.
You can still promote manually at any time regardless of accumulated
sample size:

```bash
curl -X POST $SERVER/admin/artifacts/$CANDIDATE_ID/accept -H "X-Admin-Token: $TOKEN"
```

## Automatic promotion (SPRT-gated)

`auto_pipeline.py`'s `maybe_promote_candidates()` aggregates every
`match_results` row recorded against a candidate and runs the project's
existing SPRT implementation (`tools/nnue_pipeline/uci_match.py`'s
`sprt()` -- the same function `tools/nnue_pipeline/test.py` already used
for its own accept/reject verdicts) against `--sprt-elo0`/`--sprt-elo1`.
A candidate is promoted only when SPRT reaches an `H1` verdict (the data
statistically supports "this candidate is at least `--sprt-elo1` Elo
stronger"); an `H0` verdict rejects it outright ("statistically not
stronger"); a `continue` verdict means neither conclusion is reachable yet
with the games played so far, so the loop leaves it pending and queues
more `ELO_MATCH` games next cycle instead of promoting on partial
evidence. The raw Elo point estimate and its confidence margin are still
computed and logged for visibility, but never gate the decision by
themselves -- see `platform/server/test_promotion.py` for regression
tests proving a weaker candidate and a small, statistically-insignificant
sample are both correctly rejected.

## Local (non-distributed) training

Everything above has a local equivalent that predates and doesn't depend
on `platform/`: `training_server/pipeline.py` runs the full
import->clean->train->export->benchmark->Elo->accept/reject loop on one
machine, and `tools/nnue_pipeline/test.py` runs the same load/verify/
benchmark/Elo-match validation `ELO_MATCH` tasks use, standalone. See
`docs/SELF_IMPROVEMENT_LOOP.md` and `docs/NNUE_TRAINING.md` for that
path's own documentation.
