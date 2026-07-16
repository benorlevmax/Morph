# Distributed NNUE Data Generation

`distributed/` lets other computers contribute self-play training positions
to this engine's NNUE dataset over the network: a small FastAPI server hands
out "generate ~N positions" tasks, and worker processes (running the
compiled engine) pick them up, self-play, and upload labelled positions back.

This is **local/LAN testing infrastructure, not a public deployment** (see
[Security notes and limitations](#security-notes-and-limitations)). Nothing
here modifies engine strength code -- the server and workers only drive the
existing, unmodified `chess` UCI binary as a subprocess and store what it
produces.

## Architecture

```
distributed/
  server/       FastAPI app: task queue, worker registration, result upload, stats
  worker/       Self-contained worker: self-play via UCI, uploads results
  database/     SQLite schema + init helper
  tests/        test_local.py: one server + N workers, end to end
  requirements.txt
```

```
   admin -- POST /admin/tasks (X-Admin-Token) --> [server] --> tasks table (SQLite)
                                                      ^  |
                                    POST /register    |  | GET /tasks/next
                              (registration secret)   |  v
   worker 1 -----------------------------------------[server]<---- worker 2, worker 3, ...
       |  runs chess_train's engine binary via UCI self-play
       v
   POST /tasks/{id}/results (Bearer worker token) -- batched as it plays --> [server]
                                                                                 |
                                                                                 v
                                                                        positions table (SQLite)
```

Each worker is self-contained: it only needs `distributed/worker/`, a
compiled `chess` UCI binary, and Python (`requests`, `chess`). It does not
need the rest of this repository, so `distributed/worker/` can be copied
alone to a separate machine.

## Starting the server

```sh
cd distributed
pip install -r requirements.txt
python3 server/run_server.py --host 0.0.0.0 --port 8000
```

On first start (no `CHESS_DIST_REGISTRATION_SECRET`/`CHESS_DIST_ADMIN_TOKEN`
env vars set) the server generates and prints both secrets once:

```
==============================================================================
Distributed NNUE data-generation server
  database: distributed/database/distributed.sqlite3
  task lease: 1800s
  REGISTRATION SECRET (generated, share with workers): 3f9c...   <- give to workers
  ADMIN TOKEN (generated, keep private): a812...                 <- keep for yourself
==============================================================================
```

Set them explicitly for a stable, restart-safe setup:

```sh
export CHESS_DIST_REGISTRATION_SECRET=<pick-something>
export CHESS_DIST_ADMIN_TOKEN=<pick-something-else>
python3 server/run_server.py --host 0.0.0.0 --port 8000
```

Other environment variables (all optional): `CHESS_DIST_DB_PATH`,
`CHESS_DIST_TASK_LEASE_SECONDS` (how long a worker has to finish an assigned
task before it's reclaimed), `CHESS_DIST_DEFAULT_CHUNK_SIZE`.

### Creating a generation job (admin)

```sh
curl -X POST http://SERVER:8000/admin/tasks \
  -H "X-Admin-Token: $CHESS_DIST_ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"total_positions": 100000, "depth": 7, "randomplies": 6, "chunk_size": 2000}'
```

This splits the job into `ceil(total_positions / chunk_size)` independent
tasks so many workers can pull from the queue in parallel, and a crashed
worker only loses one chunk's progress. Check progress any time:

```sh
curl http://SERVER:8000/stats
curl http://SERVER:8000/workers
curl -H "X-Admin-Token: $CHESS_DIST_ADMIN_TOKEN" http://SERVER:8000/admin/tasks
```

## Running a worker

```sh
cd distributed
pip install -r requirements.txt
python3 worker/run_worker.py \
  --server http://SERVER:8000 \
  --engine-bin /path/to/build/bin/chess \
  --registration-secret <the secret the server printed/you set> \
  --threads 4
```

First run registers with the server and saves the worker's own bearer token
to `--state-file` (default `worker/worker_state.json`); every run after that
reuses it and `--registration-secret` is no longer needed:

```sh
python3 worker/run_worker.py --server http://SERVER:8000 --engine-bin /path/to/chess --threads 4
```

Key flags:

| Flag | Meaning |
|---|---|
| `--threads N` | run N self-play games concurrently (N engine subprocesses) |
| `--hash-mb N` | engine Hash size per game instance |
| `--poll-interval S` | seconds between "no task available" checks |
| `--upload-batch-size N` | upload every N generated positions (progress + resilience) |
| `--max-plies N` | cap on self-play game length |
| `--once` | process one task then exit (used by `tests/test_local.py`; omit for a long-running worker) |
| `--force-register` | discard saved credentials and register fresh |

A worker prints progress as it goes:

```
[05:12:08] got task t_c1781d6eafa6: target=30 depth=3 randomplies=3
[05:12:08] task t_c1781d6eafa6: 10/30 accepted so far
[05:12:08] task t_c1781d6eafa6: 20/30 accepted so far
[05:12:08] task t_c1781d6eafa6: 30/30 accepted so far
[05:12:08] task t_c1781d6eafa6: finished, 80/30 positions accepted
```

**Overshoot is expected, not a bug**: a self-play game's result (win/draw/
loss) is only known once the game ends, so a worker can't cleanly stop
mid-game without discarding that game's positions. With N concurrent
threads, up to N games already "in flight" when the target is reached will
still finish and get uploaded. Treat `target_positions` as an approximate
floor, not an exact count.

### Automatic reconnect

Every server call retries with exponential backoff (`client.py`) if the
server is unreachable -- a worker survives the server restarting, a Wi-Fi
drop, etc. without manual intervention; it just logs a retry message and
keeps going. If the worker's token is ever revoked (see
[Security](#security-notes-and-limitations)) it exits with a clear message
instead of retrying forever, since retrying can't fix an auth problem.

## Dataset flow

1. **Task creation** (admin): a target position count is split into chunked
   `tasks` rows (`pending`).
2. **Task assignment**: a worker calls `GET /tasks/next`; the server atomically
   picks the oldest pending task, marks it `assigned` to that worker with a
   lease (`CHESS_DIST_TASK_LEASE_SECONDS`, default 30 min). If a lease
   expires before the worker uploads results (crash, network loss), the next
   `GET /tasks/next` call from any worker reclaims it back to `pending`.
3. **Self-play generation** (worker, `worker/selfplay.py`): the worker drives
   its own `chess` UCI subprocess(es), one full game at a time, `--threads`
   games concurrently. Every searched position is captured with its **real**
   depth and node count from the engine's own `info depth ... nodes ...`
   output (not a constant label), and the game's eventual result is
   backfilled onto every position from that game once it ends.
4. **Upload** (`POST /tasks/{id}/results`): positions are uploaded in batches
   as they're generated (`--upload-batch-size`), not all at once at the end.
5. **Server-side validation** (`server/validation.py`) rejects, per position,
   before it's ever stored: an unparseable/illegal FEN, a `side_to_move` that
   doesn't match the FEN, `eval_cp` outside \[-32000, 32000\], `result` not in
   {0.0, 0.5, 1.0}, `depth`/`nodes` out of sane range, or a missing/empty
   `engine_version`. Each rejection comes back with a specific reason.
6. **Duplicate detection** (`server/db.py`): each accepted position is keyed
   by `sha256(fen|eval|result|depth|engine_version)`; a `UNIQUE` constraint on
   that hash means resubmitting the same data point (worker retry, or two
   workers overlapping near a lease expiry) is stored once and reported back
   as a duplicate, not an error.
7. **Task completion**: once a task's accepted-position count reaches its
   target, it's marked `completed` and stops being handed out.
8. **Stats**: `GET /stats` reports total positions, per-worker and
   per-engine-version breakdowns, and a result-value distribution -- useful
   for sanity-checking a run (e.g. a wildly skewed result distribution or one
   worker submitting a very different engine_version than the rest).

### Training data format

Every stored position has exactly the fields required by the pipeline spec:

| Field | Type | Notes |
|---|---|---|
| `fen` | string | position, validated as a legal FEN |
| `side_to_move` | `'w'`\|`'b'` | redundant with the FEN, kept for fast queries |
| `eval_cp` | int | White-relative centipawns (mate scores are folded into a large magnitude, see `worker/selfplay.py`'s `MATE_CP_BASE`) |
| `result` | float | White-relative game outcome: 1.0 / 0.5 / 0.0 |
| `depth` | int | the search depth **actually reached** for this position (not a constant label) |
| `nodes` | int | nodes searched to produce this position's eval |
| `engine_version` | string | the worker's `chess` binary's UCI `id name` string |

This is a superset of `tools/nnue_pipeline/generate.py`'s local (single-
machine) JSONL schema (adds real per-position `nodes`, which the bulk
`chess_train gen` exporter used there doesn't expose). Exporting the SQLite
`positions` table to that pipeline's JSONL format is a single query + one
`json.dumps` per row away (join on the fields both schemas share) if you want
to feed a distributed run's data into `tools/nnue_pipeline/train.py`.

## Security notes and limitations

Basics that are implemented:

* **Worker authentication tokens.** A worker never has a password; it
  presents a one-time `registration_secret` to `/register` and gets back a
  unique bearer token, stored server-side only as a SHA-256 hash. Every
  worker-facing endpoint requires `Authorization: Bearer <token>` and returns
  401 on anything invalid, missing, or belonging to a disabled worker.
* **Reject invalid submissions.** Every position is validated (see Dataset
  flow, step 5) before it touches the database; invalid records are counted
  and reported back with reasons, never silently dropped or silently stored.
* **Separate admin credential.** Task creation and full task listings need a
  distinct `X-Admin-Token`, never handed to workers.
* **Revocation.** `POST /admin/workers/{id}/disable` immediately invalidates
  a worker's token for all future requests.

What is explicitly **out of scope** for this phase (per the request -- "do
not add public deployment yet"):

* **No TLS.** Run this on a trusted LAN or over a VPN/SSH tunnel; bearer
  tokens and the registration secret travel in plaintext otherwise.
* **No rate limiting / DoS protection.** Fine for a handful of known workers,
  not for an open Internet-facing endpoint.
* **SQLite, single process.** `db.py` uses short-lived connections with WAL
  mode, which comfortably handles a handful to a few dozen concurrent
  workers polling every few seconds. A public, high-worker-count deployment
  would want Postgres and a proper task-queue system (Celery/RQ) instead --
  straightforward to swap in later since all DB access is isolated in
  `server/db.py`.
* **No HTTPS-only cookie/session model, no per-IP allowlisting, no captcha on
  `/register`.** Anyone with the registration secret can register a worker;
  treat that secret the way you'd treat a shared Wi-Fi password.

## Testing locally (one server, multiple workers)

```sh
cd distributed
python3 tests/test_local.py --engine-bin /path/to/build/bin/chess --workers 3 \
    --total-positions 90 --depth 3
```

This starts the server and 3 worker processes as subprocesses (all on
`127.0.0.1`), creates a chunked task for 90 positions, waits for all workers
to finish, then asserts: total stored positions meets the target, all 3
workers registered and contributed, and all tasks reached `completed`. Logs
for the server and each worker are kept in a temp directory printed at the
end for debugging. Example passing output:

```
[test] PASS: 200 positions from 3/3 contributing workers, {'completed': 3}
```

(200, not 90 -- see the overshoot note above; the assertion is `>=` the
target, not `==`, for exactly that reason.)
