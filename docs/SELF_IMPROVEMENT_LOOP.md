# Self-Improvement Loop

`automation/pipeline_controller.py` closes the loop that every earlier piece
of this project's data/training stack was building toward:

```
generate data -> train NNUE -> test Elo -> keep improvements -> repeat
```

It is an orchestration layer, not a new implementation: every stage below
already existed, already worked, and was already independently verified
before this file was written. The controller's only job is to run those
stages in sequence, decide whether each result is an improvement, and keep
`models/` and `results/` an honest record of what happened.

```
                 automation/pipeline_controller.py
                              |
        +---------------------+---------------------+
        |                     |                      |
   collect data          training_server/          models_registry.py
 (distributed DB    -->    pipeline.py       -->   + results index
  count / local             |
  generate.py top-up)       | import -> batch -> train (GPU/CPU) ->
                             | export -> benchmark -> SPRT Elo match ->
                             | accept/reject  (already built, untouched)
                             v
                     experiments/net_XXX/
              (full record: config, logs, checkpoints,
                     network, results.json)
```

Nothing in this loop touches `src/search`, `src/eval`, or `src/nnue`'s
inference code. The controller and every stage it calls only drive the
existing, unmodified `chess` / `chess_train` binaries as subprocesses.

## What already existed vs. what this adds

| Stage | Implementation | Status |
|---|---|---|
| Data generation (local) | `tools/nnue_pipeline/generate.py` | existing, reused |
| Data generation (networked) | `distributed/server/` + `distributed/worker/` | existing, reused (read-only: counted, not managed) |
| Dataset import/dedup/validate/batch | `training_server/dataset/` | existing, reused |
| Training (GPU Bullet / CPU reference, checkpoint/resume) | `training_server/training/` | existing, reused |
| Export + verify + benchmark + SPRT Elo + accept/reject | `training_server/evaluation/evaluate.py` | existing, reused |
| Experiment tracking | `training_server/experiment.py`, `experiments/net_XXX/` | existing, reused |
| **Continuous loop, dataset-size gating, promote/discard bookkeeping, crash recovery, notifications** | `automation/` | **new, this document** |

## Scope: internal loop first, no public deployment

Per the explicit instruction to keep this internal-only for now, this
controller:

* Drives a **single machine's** training loop end to end.
* Does **not** spawn, manage, or expose `distributed/server/`'s FastAPI
  service to the internet -- that remains local/LAN test infrastructure
  exactly as documented in `docs/DISTRIBUTED_DATA_GENERATION.md`. The
  controller only reads how many positions have accumulated in its SQLite
  database; starting and operating the server and its workers (potentially
  on other machines) is still a separate, manual step.
* Has no web UI, API, or remote-control surface of any kind. Everything is
  a command-line script writing to local files.
* Optionally runs its own **local** self-play generation
  (`tools/nnue_pipeline/generate.py`, via `--auto-generate`) to top up data
  when the configured sources are short -- this is the one data-producing
  subprocess the controller owns the full lifecycle of, since it's
  contained to one machine and one process.

Building this reliably, in isolation, before layering a public-facing
service on top of it is the point of doing it in this order.

## Directory layout

```
automation/
  config.py               shared paths
  state.py                 crash-safe cycle state (state.json)
  notify.py                 pluggable notifications (log always, optional webhook)
  logging_setup.py           rotating file + console logging
  models_registry.py          models/{current,candidates,rejected}/ lifecycle
  pipeline_controller.py       the controller itself (entry point)
  logs/controller.log          rotating log (10MB x 5 backups)
  generated/auto_positions.jsonl   accumulating --auto-generate output

models/
  current/
    current.nnue            always a copy of the latest ACCEPTED network
    current.json             its metadata (dataset version, validation score,
                              benchmark, Elo match, promoted_at)
    net_XXX.nnue, net_XXX.json    versioned history of every accepted network
  candidates/                transient staging area -- a network sits here only
                              between being exported and being promoted/rejected;
                              empty in steady state, non-empty only if the
                              controller crashed mid-decision (see Failure recovery)
  rejected/
    net_XXX.nnue, net_XXX.json    every rejected network + why, kept for audit

results/
  benchmarks/net_XXX.json     bench nodes/nps comparison, per run
  elo_tests/net_XXX.json       Elo match + SPRT verdict + accept/reject, per run

experiments/net_XXX/          UNCHANGED -- still the one full, authoritative
                              record of every run (see docs/TRAINING_SERVER.md)
```

`models/` and `results/` are a convenience index on top of `experiments/`,
not a second source of truth: every field in them is copied from (or is a
copy of) something `training_server/pipeline.py` already wrote into
`experiments/net_XXX/`.

## Running one cycle

```sh
python3 automation/pipeline_controller.py --once \
  --jsonl data/positions.jsonl \
  --engine-bin build/bin/Release/chess.exe \
  --min-new-positions 2000 --epochs 5 --match-games 40
```

This: measures how many positions are available, trains only if there are
at least `--min-new-positions` new ones since the last training run,
invokes `training_server/pipeline.py` (import -> batch -> train -> export
-> benchmark -> Elo -> accept/reject), then places the resulting network
into `models/current/` or `models/rejected/` and indexes its results.

### Testing with a small dataset first

Exactly this was used to validate the whole loop before trusting it with
real data:

```sh
python3 automation/pipeline_controller.py --once \
  --jsonl /path/to/tiny_sample.jsonl --min-new-positions 100 \
  --engine-bin build/bin/Release/chess.exe \
  --epochs 1 --batch-size 128 --val-fraction 0.05 \
  --bench-depth 6 --match-games 4 --match-depth 2 --reject-elo-threshold -10000
```

One full cycle (300 positions, 1 epoch, 4-game match) completed in under a
minute and produced: an accepted network in `models/current/`, its full
record in `experiments/net_009/`, and matching entries in
`results/benchmarks/` and `results/elo_tests/`. A second run with an
unreachable `--reject-elo-threshold` (forcing a loss) was correctly routed
to `models/rejected/` with `models/current/` left untouched. Running the
controller again immediately afterward with unchanged data correctly
detected "0 new positions" and skipped training rather than retraining on
the same data.

## Running continuously

```sh
python3 automation/pipeline_controller.py --loop --interval-seconds 3600 \
  --use-default-distributed-db --auto-generate \
  --engine-bin build/bin/Release/chess.exe \
  --min-new-positions 20000 --epochs 10 --match-games 200
```

Each iteration: check for enough new data (topping up with local self-play
via `--auto-generate` if configured and short) -> train -> evaluate ->
promote/reject -> sleep `--interval-seconds` -> repeat. `Ctrl-C`/`SIGTERM`
stops the daemon after the current cycle finishes rather than killing it
mid-training. In practice this is started once (e.g. under `systemd`,
Windows Task Scheduler, or the `schedule` skill) and left running; the
distributed server/workers, if used, are started separately and just keep
depositing positions into the database this controller polls.

## How "keep improvements" is decided

The accept/reject decision itself is `training_server/evaluation/evaluate.py`'s
existing, already-tested policy (unchanged here): the candidate network
must (1) exactly match the pure-Python reference implementation on a fixed
set of positions when loaded into the real compiled engine (a correctness
gate, not a strength one), and (2) score at or above `--reject-elo-threshold`
in an automated match against the current baseline, with the match's Elo
estimate and GSPRT verdict (`sprt.verdict`: `H0`/`H1`/`continue`) both
recorded in `results/elo_tests/net_XXX.json`. The controller's only added
logic is *acting* on that verdict: move the network into `models/current/`
or `models/rejected/` and update the pointer/index files accordingly.
Baseline selection (which network a new candidate is compared against) is
also unchanged: the most recent experiment with an `accept` verdict, so
`models/current/` and the internal baseline lineage always agree.

## Logging

Every controller run writes to `automation/logs/controller.log` (rotating,
10MB x 5 backups) and to the console. This includes every line of
`training_server/pipeline.py`'s own output, streamed live rather than
captured only at the end, so a `tail -f automation/logs/controller.log`
shows real-time progress through data collection, training, export, and
evaluation.

## Failure recovery

Two independent layers:

* **Per-stage retries.** Data collection and the training pipeline call are
  each wrapped in a retry with linear backoff (`--max-retries`,
  `--retry-backoff-seconds`). A transient failure (e.g. a flaky subprocess)
  is retried automatically before the cycle is marked failed.
* **Crash recovery across restarts.** `automation/state.py` writes
  `automation/state.json` after every stage of every cycle, not just at the
  end, so it always reflects the true in-progress stage
  (`collecting`/`training`/`evaluating`/`promoting`/`idle`/`failed`). If the
  controller process itself is killed mid-cycle -- verified by deliberately
  killing a run between "network exported" and "verdict recorded" -- the
  next startup's `recover_incomplete_cycle()` detects the stale non-terminal
  status, finds the already-exported network staged in
  `models/candidates/`, re-reads the run's already-written
  `experiments/net_XXX/results.json` (training itself is never re-run), and
  finishes the promote/reject decision before starting any new cycle. This
  is idempotent: if a network was already fully placed into
  `models/current/` or `models/rejected/` before the crash, recovery is a
  no-op. If the crash happened before `results.json` even existed (training
  itself was interrupted), there's nothing to recover -- the cycle is
  marked `failed` and the next cycle starts a fresh experiment (use
  `training_server/pipeline.py --resume-checkpoint` manually if you want to
  continue that specific interrupted training run instead of starting
  over).
* If `--max-consecutive-failures` cycles fail in a row (default 3), the
  daemon halts entirely rather than looping forever on a broken
  configuration -- it logs and notifies, and requires a manual restart
  once the underlying issue (bad engine path, out of disk, etc.) is fixed.

## Notifications

`automation/notify.py` always logs every event
(`cycle_start`, `cycle_skipped`, `promoted`, `rejected`, `cycle_failed`,
`cycle_recovered`, `loop_halted`). If the environment variable
`AUTOMATION_WEBHOOK_URL` is set, the same event is also POSTed as JSON
(with a Slack/Discord-compatible `"text"` field) to that URL, best-effort --
a webhook failure is logged and never allowed to interrupt the loop. No
email/SMS integration is included; that's a small, contained addition to
`notify.py` once there's a real service and credentials to send through.

## What's verified vs. what isn't

Verified end-to-end in this environment (no GPU, no distributed workers
running): dataset-size gating and skip-when-insufficient-data, the full
collect -> train -> export -> evaluate -> promote chain producing an
accepted network in `models/current/`, the same chain producing a rejected
network in `models/rejected/` with `models/current/` correctly left
unchanged, and crash recovery correctly finishing an interrupted
promote/reject decision on restart without re-running training.

Not exercisable in this sandbox, same caveat as `docs/TRAINING_SERVER.md`:
the real GPU-accelerated Bullet training path, and a real multi-machine
`distributed/` deployment feeding this controller continuously. The
counting/gating logic for the distributed database is straightforward
(`SELECT COUNT(*) FROM positions`) and doesn't depend on anything
untestable, but running the full loop against a live multi-worker
deployment for days at a time has not been done here.
