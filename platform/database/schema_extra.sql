-- schema_extra.sql - Tables the public platform adds on top of
-- distributed/database/schema.sql's workers/tasks/positions/submissions
-- (which platform/server/db.py's PlatformDatabase creates first, unmodified,
-- by calling distributed/database/init_db.py's ensure_schema() before this
-- file is applied). Kept as a *second* file rather than edited into the
-- original schema.sql so distributed/'s own local/LAN-testing schema stays
-- exactly as documented in docs/DISTRIBUTED_DATA_GENERATION.md, untouched.

PRAGMA foreign_keys = ON;

-- A contributor account. Workers registered with an account's API key are
-- attributed to that account for the leaderboard/statistics page; workers
-- registered via the legacy shared secret (if enabled) have user_id = NULL
-- in the workers table and appear as "anonymous" on the leaderboard.
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,          -- 'u_<12 hex chars>'
    username        TEXT NOT NULL UNIQUE,
    email           TEXT UNIQUE,                -- optional (nullable), never shown publicly
    password_hash   TEXT NOT NULL,
    api_key_hash    TEXT UNIQUE,                 -- sha256 of the current API key; NULL until generated
    created_at      TEXT NOT NULL,
    disabled        INTEGER NOT NULL DEFAULT 0
);

-- Server-side login sessions (revocable, unlike a bare signed token -- see
-- accounts.py's note on why). A session token is a normal bearer token,
-- like a worker token, hashed the same way.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    created_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- Rejected-submission audit log, distinct from distributed's `submissions`
-- table (which already records accepted/duplicate/rejected *counts* per
-- upload). This table exists specifically for anti-cheat review: one row
-- per *individual* rejected record with enough context (which worker, which
-- check failed, a snippet of the offending data) for an admin to spot a
-- worker that's repeatedly submitting fabricated data, rather than just an
-- occasional honest validation failure. See anti_cheat.py.
CREATE TABLE IF NOT EXISTS rejections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id       TEXT NOT NULL,
    reason          TEXT NOT NULL,
    detail          TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rejections_worker ON rejections(worker_id, created_at);

-- A downloadable, hash-verified build product: a dataset (from a
-- DATA_GENERATION batch), a training checkpoint, or a candidate/accepted
-- NNUE network (from a TRAIN_NETWORK task). Workers fetch these by id over
-- the artifact-download endpoint and MUST verify sha256 before using or
-- executing anything derived from the file -- see docs/WORKER.md. This
-- table is the source of truth for "what network is currently strongest"
-- (kind='network', accepted=1, highest created_at) that the automated
-- improvement loop (server/routes around ELO_MATCH/TRAIN_NETWORK) reads.
CREATE TABLE IF NOT EXISTS artifacts (
    id                  TEXT PRIMARY KEY,          -- 'a_<12 hex chars>'
    kind                TEXT NOT NULL CHECK (kind IN ('dataset', 'checkpoint', 'network')),
    file_path           TEXT NOT NULL,
    sha256              TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL,
    created_by_task_id  TEXT,                       -- NULL for artifacts seeded manually (e.g. baseline net)
    created_by_worker_id TEXT,
    accepted            INTEGER NOT NULL DEFAULT 0,  -- 1 once an ELO_MATCH confirms a 'network' artifact beat the prior strongest
    metadata            TEXT,                        -- JSON: task-type-specific info (e.g. positions count, epochs, elo estimate)
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind, accepted, created_at);

-- Result of an ELO_MATCH task: candidate NNUE artifact vs. a baseline
-- artifact, paired-opening / color-reversed games (see
-- tools/nnue_pipeline/test.py's uci_match, which the worker-side ELO_MATCH
-- executor wraps). The server aggregates these to decide whether a
-- candidate artifact should be promoted (artifacts.accepted = 1).
CREATE TABLE IF NOT EXISTS match_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id                 TEXT NOT NULL,
    worker_id               TEXT NOT NULL,
    candidate_artifact_id   TEXT NOT NULL REFERENCES artifacts(id),
    baseline_artifact_id    TEXT NOT NULL REFERENCES artifacts(id),
    wins                    INTEGER NOT NULL,
    losses                  INTEGER NOT NULL,
    draws                   INTEGER NOT NULL,
    games                   INTEGER NOT NULL,
    pgn_path                TEXT,
    submitted_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_match_results_candidate ON match_results(candidate_artifact_id);
