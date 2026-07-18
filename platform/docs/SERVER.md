# Running a server

`platform/server/` is the task coordinator: it authenticates workers,
tracks their capabilities, queues and leases typed tasks, validates and
deduplicates everything workers upload, and tracks artifacts (datasets,
checkpoints, networks) and Elo match results. It's a FastAPI app backed by
SQLite.

## Quick start (local test)

```bash
cd platform/server
pip install -r ../requirements.txt
python3 run_server.py --host 0.0.0.0 --port 8000
```

On first startup with no `CHESS_PLATFORM_ADMIN_TOKEN` set, the server
generates one and prints it once — capture it, you'll need it for
`/admin/*` endpoints (creating tasks, seeding artifacts, disabling
workers). Set it explicitly for anything beyond a quick local test.

By default the server accepts worker registration only via per-account API
key (`POST /accounts/register` then `/accounts/api-key/regenerate`). To
also allow a simpler shared-secret registration (useful for a small
private deployment), set `CHESS_PLATFORM_REGISTRATION_SECRET`.

## Docker

```bash
cd platform/docker
CHESS_PLATFORM_ADMIN_TOKEN=$(openssl rand -hex 16) docker compose up -d
```

Unlike running the server standalone (`python3 run_server.py`), the compose
file requires `CHESS_PLATFORM_ADMIN_TOKEN` to be set explicitly rather than
letting the server generate one — see below.

The compose file builds from the repo root (the server imports sibling
code from `distributed/server/` via a relative path) and mounts a named
volume for the SQLite database and artifact store so they survive
container recreation. `/dashboard/*` reads entirely from that same live
database (see `platform/server/dashboard_data.py`) — there's nothing on
the host filesystem for it to read, so there's nothing to mount for it.

`docker compose up -d` starts two containers from the same image:

- `server` — the FastAPI coordinator described throughout this document.
- `auto-pipeline` — runs `auto_pipeline.py --loop` against `server` over
  HTTP (same admin token), which is what actually closes the loop end to
  end: exporting datasets, queuing training/Elo-match tasks, and promoting
  candidates once SPRT reaches a decisive verdict, with zero manual admin
  calls required. See `platform/server/auto_pipeline.py`'s module doc for
  the full cycle it drives, and the `AUTO_PIPELINE_*` environment
  variables in `docker-compose.yml` for the knobs it exposes (cycle
  interval, candidates per training round, Elo games per match). Omit or
  scale this service down if you'd rather drive the pipeline by hand via
  the `/admin/*` endpoints instead.

Neither container has a C++ toolchain and neither ever builds or runs the
engine itself — that only ever happens on the maintainer's machine or on
volunteer worker machines.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CHESS_PLATFORM_DB_PATH` | `platform/database/platform.sqlite3` | SQLite file path |
| `CHESS_PLATFORM_ARTIFACTS_DIR` | next to the DB, `artifacts/` | where uploaded artifact files are stored |
| `CHESS_PLATFORM_ADMIN_TOKEN` | random, printed at startup | bearer for `/admin/*` |
| `CHESS_PLATFORM_REGISTRATION_SECRET` | unset (disabled) | optional shared-secret worker registration |
| `CHESS_PLATFORM_TASK_LEASE_SECONDS` | `1800` | how long a worker has to complete an assigned task before it's reassigned |
| `CHESS_PLATFORM_DEFAULT_CHUNK_SIZE` | `500` | default `SELF_PLAY` batch size for `/admin/tasks` |
| `CHESS_PLATFORM_RATE_LIMIT_SUBMISSIONS_PER_MIN` | `30` | per-worker submission rate cap |
| `CHESS_PLATFORM_RATE_LIMIT_REGISTRATIONS_PER_HOUR` | `20` | per-IP registration/key-regen rate cap |
| `CHESS_PLATFORM_RATE_LIMIT_LOGIN_PER_15MIN` | `10` | per-IP login attempt cap |
| `CHESS_PLATFORM_MAX_ARTIFACT_BYTES` | `536870912` (512 MiB) | max single artifact upload size |
| `CHESS_PLATFORM_WORKER_VERSION` | `1.0.0` | advertised via `GET /version` for worker auto-update checks |
| `CHESS_PLATFORM_MAX_CONNECTED_WORKERS` | `40` | new `POST /register` calls get a 503 once this many workers are already connected -- see 'Capacity and alerts' below |

## API surface

**Accounts:** `POST /accounts/register`, `/accounts/login`, `/accounts/logout`,
`/accounts/api-key/regenerate`

**Worker registration + capability reporting:** `POST /register`,
`POST /workers/capabilities`

**Task polling (pick one, per worker):**
- `GET /tasks/next` — legacy, untyped, `SELF_PLAY`-only (distributed/-compatible wire format)
- `GET /tasks/next-typed` — capability-aware, returns any task type with a typed payload (what `platform/worker/` actually uses)

**Result submission:** `POST /tasks/{id}/results` (positions — `SELF_PLAY`/`DATA_GENERATION`),
`POST /tasks/{id}/match-result` (`ELO_MATCH`)

**Artifacts:** `GET /artifacts`, `GET /artifacts/{id}`,
`GET /artifacts/{id}/download`, `GET /artifacts/strongest-network`,
`POST /artifacts/upload` (multipart — datasets/checkpoints/candidate networks)

**Community (public, no auth):** `GET /stats`, `/workers`, `/leaderboard`,
`/dashboard/summary`, `/dashboard/elo-series`, `/version`, `/health`

**Admin (`X-Admin-Token` header):** `POST /admin/tasks` (legacy bulk
`SELF_PLAY`), `POST /admin/tasks/typed` (any task type),
`GET /admin/tasks`, `POST /admin/artifacts` (seed from a local file),
`POST /admin/artifacts/{id}/accept`, `POST /admin/workers/{id}/disable`,
`POST /admin/workers/{id}/enable`, `GET /admin/system-load` (see below)

Full request/response shapes: run the server and check `GET /docs`
(FastAPI's auto-generated interactive API docs).

## Capacity and alerts

A small deploy target (a single free-tier instance, say) can't take
unlimited concurrent workers. Two pieces work together to handle that
without requiring dedicated infrastructure-monitoring software:

**The cap itself.** Once `CHESS_PLATFORM_MAX_CONNECTED_WORKERS` workers
are simultaneously connected (not disabled, seen within the last 10
minutes), `POST /register` starts returning `503` to *new* registration
attempts with a message explaining the server is full. Already-registered
workers are never affected -- they keep polling and submitting normally;
only brand-new sign-ups are turned away while the server is at capacity.
Raise the env var if your hardware can take more; the default (`40`) is
deliberately conservative for a single-OCPU box.

**Seeing it coming.** `GET /admin/system-load` (admin token required)
returns a point-in-time snapshot: current connected-worker count against
the cap, pending task queue depth, and (Linux-only, parsed straight from
`/proc` -- no extra dependency) load average, memory, and disk usage of
the artifacts directory's filesystem. It's meant to be polled externally
-- e.g. a cron job, or a Cowork scheduled task -- rather than watched
live; the response itself doesn't track history, so an external poller
that wants a trend needs to keep its own.

```bash
curl -s $SERVER/admin/system-load -H "X-Admin-Token: $TOKEN"
```

```json
{
  "connected_workers": 12, "max_connected_workers": 40, "at_worker_capacity": false,
  "pending_tasks": 3, "cpu_count": 1,
  "load_average": {"1min": 0.31, "5min": 0.22, "15min": 0.18},
  "memory": {"total_mb": 981.2, "available_mb": 512.4, "used_percent": 47.8},
  "disk": {"total_gb": 45.0, "free_gb": 38.1, "used_percent": 15.3}
}
```

## Creating tasks

If `auto-pipeline` (see the Docker section above, or run
`auto_pipeline.py --loop` yourself) is running against this server, it
already does everything below on its own — exporting datasets, queuing
DATA_GENERATION/TRAIN_NETWORK/ELO_MATCH tasks, and promoting candidates —
on a timer. The commands below are for manual/one-off use: testing, a
deployment not running auto-pipeline, or forcing an out-of-cycle batch.

Queue a data-generation batch:

```bash
curl -X POST $SERVER/admin/tasks/typed -H "X-Admin-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"DATA_GENERATION","payload":{"games":100,"depth":8,"randomplies":6,"format":"bullet"}}'
```

Queue an Elo match once you have a candidate artifact (see
[TRAINING.md](TRAINING.md) for the full loop -- a TRAIN_NETWORK task
produces a real `network` artifact directly, `accepted=false` until
ELO_MATCH/promotion confirms it):

```bash
curl -X POST $SERVER/admin/tasks/typed -H "X-Admin-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"ELO_MATCH","payload":{"candidate_artifact_id":"a_...","baseline_artifact_id":"a_...","games":24,"match_depth":5}}'
```

## Architecture notes

`platform/server/database.py`'s `PlatformDatabase` subclasses
`distributed/server/db.py`'s `Database` rather than replacing it — every
method that already existed there (`register_worker`, `assign_next_task`,
`submit_positions`, `get_stats`, ...) is reused verbatim. `platform/`
extends it with accounts/sessions, capability-aware typed task assignment,
artifacts, and match results, via a second schema file
(`platform/database/schema_extra.sql`) plus idempotent `ALTER TABLE`
migrations run on every startup — `distributed/server/` and
`distributed/database/schema.sql` are never modified, so that simpler
LAN-only coordinator keeps working completely independently for trusted
local testing (see `docs/DISTRIBUTED_DATA_GENERATION.md`).

Rate limiting is in-process, in-memory (`platform/server/ratelimit.py`) —
correct for the reference single-process/single-SQLite-file deployment
this ships, but does not coordinate across multiple server processes.
Scaling to multiple uvicorn workers behind a load balancer would need a
shared store (Redis or a DB-backed counter) instead.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram and
[../../SECURITY.md]