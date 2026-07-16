# Architecture

## Contributor flow

```
GitHub repository
      |
      v
  README -> download a release (or clone + build)
      |
      v
  run worker  ---->  registers with server, reports capabilities
      |                        |
      |                        v
      |              GET /tasks/next-typed (capability-aware)
      |                        |
      v                        v
  worker receives a typed task: SELF_PLAY | DATA_GENERATION | ELO_MATCH | TRAIN_NETWORK
      |
      v
  worker downloads + sha256-verifies any needed artifacts
      |
      v
  worker executes REAL engine compute (subprocess: chess / chess_train)
      |
      v
  worker uploads results (positions / match result / trained network)
      |
      v
  server validates (structural checks, plausibility checks, dedup) and stores
      |
      v
  accepted data feeds the next stage of the improvement loop
```

## The four task types

| Type | Payload | Worker executor | What it produces |
|---|---|---|---|
| `SELF_PLAY` | `{target_positions, depth, randomplies}` | `platform/worker/platform_worker.py`'s `TaskRunner` (streaming, `--threads` concurrent games) | positions, uploaded incrementally |
| `DATA_GENERATION` | `{games, depth, randomplies, format}` | `platform/worker/data_generation.py` (wraps `chess_train gen`) | positions, uploaded as one batch |
| `ELO_MATCH` | `{candidate_artifact_id, baseline_artifact_id, games, match_depth, movetime_ms}` | `platform/worker/elo_match.py` (wraps `tools/nnue_pipeline/uci_match.py`) | a `match_results` row: wins/losses/draws |
| `TRAIN_NETWORK` | `{dataset_artifact_id, epochs, hidden}` | `platform/trainer/train_network.py` (wraps `tools/nnue_pipeline/train.py` + `export.py`) | a `network` artifact (real, loadable `.nnue`) + real loss metrics |

All four share the same underlying `tasks` table (`task_type` + JSON
`payload` columns, added via `platform/database/schema_extra.sql` and
idempotent migrations in `platform/server/database.py`) and the same
lease/reassignment mechanism `distributed/server/db.py` already had for
plain `SELF_PLAY` — `_reclaim_expired_leases` runs on every assignment
attempt and hands an expired lease's task back to the pending pool.

`TRAIN_NETWORK` is the only type gated by worker capability: a worker only
ever receives one if its last-reported `capabilities.trainer_capable` is
true (`assign_next_typed_task` in `platform/server/database.py`). Every
other type is assignable to any worker.

## Why `distributed/` is untouched

This project grew out of a simpler LAN-only self-play coordinator
(`distributed/`, documented in `docs/DISTRIBUTED_DATA_GENERATION.md`).
Rather than modify it, `platform/server/database.py`'s `PlatformDatabase`
**subclasses** `distributed/server/db.py`'s `Database` and reuses every
existing method verbatim (`register_worker`, `assign_next_task`,
`submit_positions`, `get_stats`, ...); `platform/` only adds what a public
deployment needs on top (accounts, capability-aware typed tasks,
artifacts, match results) via a second schema file
(`schema_extra.sql`) and additive, idempotent migrations. `distributed/`
keeps working completely independently for trusted local/LAN testing —
nothing in `platform/` can break it.

## Data model (additions over `distributed/database/schema.sql`)

- `tasks.task_type`, `tasks.payload` — added via migration; default
  `task_type='SELF_PLAY'` and `payload=NULL` so pre-existing rows and
  `distributed/`'s own `create_tasks_bulk()` path keep working unmodified.
- `workers.capabilities` — JSON: `cpu_cores`, `ram_mb`, `gpu_available`,
  `gpu_name`, `gpu_backends` (full multi-vendor detail — NVIDIA/CUDA,
  AMD/ROCm, Intel — each with `vendor`/`backend`/`name`/`trainable`/
  `detected_via`; see `platform/worker/capabilities.py`), `best_gpu_backend`
  (`'cuda'`/`'rocm'`/`None` — the one `platform/trainer/train_network.py`
  will actually try, since bullet_lib only has CUDA/ROCm compute backends),
  `trainer_capable`.
- `artifacts` — `id, kind (dataset|checkpoint|network), file_path, sha256,
  size_bytes, created_by_task_id, created_by_worker_id, accepted, metadata, created_at`.
  Content-addressed on disk (`platform/server/app.py`'s `upload_artifact`
  stores under `<artifacts_dir>/<kind>/<sha256>`, deduplicating identical
  uploads).
- `match_results` — `task_id, worker_id, candidate_artifact_id,
  baseline_artifact_id, wins, losses, draws, games, pgn_path, submitted_at`.
- `users`, `sessions`, `rejections` — accounts, login sessions, and an
  anti-cheat audit log (one row per individually-rejected record, for
  spotting a worker with a sustained pattern of bad submissions).

## Artifact lifecycle

```
operator seeds a baseline 'network' artifact (POST /admin/artifacts, accepted=true)
        |
        v
DATA_GENERATION produces training data -> positions table (not an artifact
        itself; a 'dataset' artifact is a chess_train-gen-format --dat file,
        assembled separately -- see TRAINING.md)
        |
        v
TRAIN_NETWORK consumes a dataset artifact, produces a 'network' artifact
        (accepted=false until ELO_MATCH/promotion confirms it -- see TRAINING.md)
        |
        v
ELO_MATCH consumes candidate + baseline artifacts, produces a match_results row
        |
        v
operator (or, once policy is defined, automation) reviews match_results and
        calls POST /admin/artifacts/{id}/accept to promote a candidate
        |
        v
GET /artifacts/strongest-network now returns the new