# Running a worker

`platform/worker/` is the contributor-facing client. It registers with a
server, reports what your machine can do, and repeatedly requests and
executes real work — no simulated tasks, no placeholder compute.

## Requirements

- The compiled engine binary (`chess`, and `chess_train` if you want
  `DATA_GENERATION`/`TRAIN_NETWORK` tasks — get both from a
  [release](../../../releases) or build them yourself, see the root
  [README](../../README.md#build-the-engine)).
- Python 3.9+ and `pip install -r platform/requirements.txt`, unless
  you're using a packaged release that bundles everything.

## Getting an API key

The official community server is `http://64.181.243.154:8000` (plain HTTP
for now, no TLS yet -- don't reuse a password you care about elsewhere).
Sign up and mint a key in three calls:

```bash
curl -X POST http://64.181.243.154:8000/accounts/register \
  -H "Content-Type: application/json" \
  -d '{"username": "yourname", "password": "yourpassword"}'

SESSION=$(curl -s -X POST http://64.181.243.154:8000/accounts/login \
  -H "Content-Type: application/json" \
  -d '{"username": "yourname", "password": "yourpassword"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_token'])")

curl -X POST http://64.181.243.154:8000/accounts/api-key/regenerate \
  -H "Authorization: Bearer $SESSION"
```

The last call prints your key (`cek_...`) once -- it can't be retrieved
again, only regenerated (which invalidates the old one). See
[SERVER.md](SERVER.md) if you're running your own server instead of
joining the official one -- `CHESS_PLATFORM_REGISTRATION_SECRET` is a
simpler shared-secret alternative some private deployments use instead.

## Running

```bash
python3 platform_worker.py \
  --server http://64.181.243.154:8000 \
  --engine-bin /path/to/chess \
  --api-key cek_...                      # or --registration-secret <secret>
```

First run registers the worker and saves credentials to
`--state-file` (default `worker_state.json` next to the script) — later
runs reuse them automatically; you don't need to pass `--api-key` again
unless you delete that file or pass `--force-register`.

## Every CLI flag

| Flag | Default | Meaning |
|---|---|---|
| `--server` | required | server base URL |
| `--engine-bin` | required | path to the compiled `chess` UCI binary |
| `--api-key` | none | per-account API key (recommended) |
| `--registration-secret` | none | legacy shared-secret registration, if the operator enabled it |
| `--state-file` | `worker_state.json` | where credentials are persisted |
| `--hostname` | local hostname | reported to the server, shown on `/workers` |
| `--threads` | `1` | concurrent self-play game instances for `SELF_PLAY` tasks |
| `--hash-mb` | `16` | engine `Hash` UCI option per game instance |
| `--poll-interval` | `5.0` | seconds between "no task available" retries |
| `--upload-batch-size` | `100` | positions per progress-reporting upload batch |
| `--max-plies` | `200` | per-game ply cap for `SELF_PLAY` |
| `--force-register` | off | re-register even with saved credentials |
| `--once` | off | process exactly one task then exit (default: loop forever) |
| `--max-cpu-percent` | none | soft cap on this process tree's CPU usage; the worker backs off between games when exceeded |
| `--max-memory-mb` | none | hard cap on RSS; the worker finishes its current batch, uploads, and exits cleanly when exceeded |
| `--resource-check-interval` | `10.0` | seconds between resource checks |
| `--trainer-capable` | off | opt in to `TRAIN_NETWORK` tasks (see below) |
| `--gpu-name-override` | none | cosmetic-only GPU name override for reporting |
| `--artifacts-cache-dir` | `artifacts_cache/` | local cache for downloaded, hash-verified artifacts |
| `--train-bin` | auto-detected next to `--engine-bin` | path to `chess_train`, needed for `DATA_GENERATION`/`TRAIN_NETWORK` |
| `--auto-update` / `--update-url` / `--update-check-interval` | off / none / `3600.0` | self-update from a release URL when the server advertises a newer version |

## What each task type does on your machine

The worker polls `GET /tasks/next-typed` and dispatches by type:

- **`SELF_PLAY`** — runs `--threads` concurrent games of the engine
  against itself, streaming positions to the server in batches as they're
  generated.
- **`DATA_GENERATION`** — runs the engine's native bulk self-play exporter
  (`chess_train gen`) once, then uploads the whole resulting batch.
  Faster than `SELF_PLAY` for large volumes since it's the engine's own
  optimized export path rather than one UCI `go`/`bestmove` round-trip at
  a time.
- **`ELO_MATCH`** — downloads and hash-verifies a candidate and a baseline
  network artifact, then plays a real paired-opening, color-reversed match
  between two engine processes (one per network) and uploads the
  aggregated win/loss/draw record.
- **`TRAIN_NETWORK`** — only ever offered if you passed `--trainer-capable`.
  Downloads and hash-verifies a dataset artifact, runs the real training
  pipeline (`tools/nnue_pipeline/train.py` + `export.py`, picking a real
  GPU backend via `capabilities.py` if one was detected, otherwise the CPU
  reference path), locally loads and verifies the exported `.nnue` against
  the compiled engine, and uploads it as a real `network` artifact with its
  real reported loss numbers. See [TRAINING.md](TRAINING.md) for the full
  pipeline detail.

If a task fails (a corrupted artifact download, an engine crash), the
worker logs it and moves on — the task's server-side lease simply expires
and gets reassigned to another worker, the same recovery path used for a
worker that disconnects or is killed outright.

## Resource limits

`--max-cpu-percent` and `--max-memory-mb` are cooperative, not OS-enforced
(no cgroups/job-object throttling — this needs to work unprivileged on
both Windows and Linux). A background thread samples the worker's own
process tree (via `psutil`) and sets a flag the work loop checks: over the
CPU cap, it sleeps briefly between games; over the memory cap, it finishes
its current batch, uploads, and exits cleanly rather than waiting to be
OOM-killed mid-upload.

## Artifact verification

Every artifact (a network, a dataset) is downloaded and its SHA-256 is
checked against what the server reports (`GET /artifacts/{id}`) *before*
it's loaded into the engine or trusted for anything. A mismatch raises
immediately and the task fails (see above) rather than silently using an
unverified file — see `platform/worker/artifacts.py`. Verified downloads
are cached locally by content hash under `--artifacts-cache-dir`, so
re-running a task that references the same artifact doesn't re-download
it, and a corrupted local cache file is detected and re-fetched
automatically.

## Standalone releases

A [Release](../../../releases) bundles everything needed to run a
worker with zero build step and zero Python install: the compiled
engine (`chess`, `chess_train`), the worker client itself frozen into
a single executable, and (as of the fix below) `train.py`/`export.py`
also frozen into their own standalone executables
(`nnue_train`/`nnue_export`) so `--trainer-capable`'s CPU training
path works out of the box too — a contributor should not need to
clone this repo, install a C++ toolchain, or install Python at all.
See `.github/workflows/release.yml` for exactly what each archive
contains and how it's built.

**Why this needed its own fix:** the frozen worker used to try running
`train.py`/`export.py` via `sys.executable` — correct when running
from source (a real Python interpreter), but under a PyInstaller-
frozen `worker.exe`, `sys.executable` points back at `worker.exe`
itself, not a real Python. That meant every packaged release's
`--trainer-capable` path was completely non-functional (it would fail
immediately with `worker.exe`'s own argument-parser error) until this
was fixed by freezing those two scripts the same way `worker.exe`
itself already is, and having the worker invoke them directly instead
of shelling out to a nonexistent interpreter.
