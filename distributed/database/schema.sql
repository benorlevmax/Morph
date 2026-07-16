-- schema.sql - Distributed NNUE data-generation database (SQLite).
--
-- One file, four tables. Kept intentionally simple (no migrations framework)
-- since this is for local/small-scale volunteer-style data generation, not a
-- production multi-tenant service (see docs/DISTRIBUTED_DATA_GENERATION.md).

PRAGMA foreign_keys = ON;

-- A registered worker (one row per worker install, not per process/run).
-- `token_hash` is SHA-256 of the worker's bearer token; the plaintext token
-- is only ever shown once, at registration time, and never stored.
CREATE TABLE IF NOT EXISTS workers (
    id                  TEXT PRIMARY KEY,      -- e.g. "w_<12 hex chars>"
    token_hash          TEXT NOT NULL UNIQUE,
    hostname            TEXT NOT NULL,
    engine_version      TEXT NOT NULL,
    threads             INTEGER NOT NULL DEFAULT 1,
    registered_at       TEXT NOT NULL,          -- ISO8601 UTC
    last_seen_at        TEXT NOT NULL,
    positions_generated INTEGER NOT NULL DEFAULT 0,
    submissions_count   INTEGER NOT NULL DEFAULT 0,
    disabled            INTEGER NOT NULL DEFAULT 0   -- 1 = revoked, reject all requests
);

-- A unit of work: "generate approximately `target_positions` positions".
-- Large requests are split into many rows of this table (see
-- server/tasks.py's chunking) so multiple workers can work in parallel and a
-- crashed worker only loses one chunk's progress, not the whole job.
CREATE TABLE IF NOT EXISTS tasks (
    id                  TEXT PRIMARY KEY,       -- e.g. "t_<12 hex chars>"
    status              TEXT NOT NULL DEFAULT 'pending',  -- pending|assigned|completed|failed
    target_positions    INTEGER NOT NULL,
    accepted_positions  INTEGER NOT NULL DEFAULT 0,
    depth               INTEGER NOT NULL,
    randomplies         INTEGER NOT NULL DEFAULT 6,
    batch_label         TEXT,                    -- groups tasks created by one bulk request
    assigned_worker_id  TEXT REFERENCES workers(id),
    assigned_at         TEXT,
    lease_expires_at    TEXT,
    completed_at        TEXT,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_lease ON tasks(status, lease_expires_at);

-- One accepted training position. `content_hash` is the dedup key: sha256 of
-- fen|eval|result|depth|engine_version, independent of which task/worker
-- produced it -- an identical data point submitted twice (worker retry after
-- a dropped connection, or two workers overlapping near a lease expiry) is
-- stored once.
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL REFERENCES tasks(id),
    worker_id       TEXT NOT NULL REFERENCES workers(id),
    fen             TEXT NOT NULL,
    side_to_move    TEXT NOT NULL,          -- 'w' | 'b' (redundant w/ FEN, stored for fast queries)
    eval_cp         INTEGER NOT NULL,       -- White-relative centipawns
    result          REAL NOT NULL,          -- White-relative: 1.0 / 0.5 / 0.0
    depth           INTEGER NOT NULL,
    nodes           INTEGER NOT NULL,
    engine_version  TEXT NOT NULL,
    content_hash    TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_positions_task ON positions(task_id);
CREATE INDEX IF NOT EXISTS idx_positions_worker ON positions(worker_id);

-- Audit trail: one row per upload call (a task is usually fed by several
-- incremental submissions as a worker generates positions in batches).
CREATE TABLE IF NOT EXISTS submissions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL REFERENCES tasks(id),
    worker_id           TEXT NOT NULL REFERENCES workers(id),
    positions_submitted INTEGER NOT NULL,
    positions_accepted  INTEGER NOT NULL,
    duplicates_skipped  INTEGER NOT NULL,
    rejected            INTEGER NOT NULL,
    rejected_reasons    TEXT,                -- JSON array, truncated, for debugging
    submitted_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_task ON submissions(task_id);
