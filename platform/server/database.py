#!/usr/bin/env python3
"""database.py - PlatformDatabase: everything distributed/server/db.py's Database
already does (worker registration/auth, task queue, position submission +
validation + dedup, stats) PLUS user accounts, sessions, API-key-based
worker registration, and a leaderboard. Subclasses rather than copies, so
every existing, already-tested method (register_worker, assign_next_task,
submit_positions, get_stats, ...) is reused verbatim -- this file only adds
what a public deployment needs on top.
"""
import hashlib
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'distributed', 'server'))
from db import Database as DistDatabase  # distributed/server/db.py, unmodified

import accounts

SCHEMA_EXTRA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'database', 'schema_extra.sql')

SESSION_LIFETIME_SECONDS = 30 * 24 * 3600   # 30 days


def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class PlatformDatabase(DistDatabase):
    def __init__(self, db_path):
        super().__init__(db_path)   # creates workers/tasks/positions/submissions (distributed's schema)
        self._apply_schema_extra()
        self._migrate_workers_user_id()
        self._migrate_tasks_task_type()
        self._migrate_tasks_payload()
        self._migrate_workers_capabilities()

    def _apply_schema_extra(self):
        conn = self._conn()
        try:
            with open(SCHEMA_EXTRA_PATH) as f:
                conn.executescript(f.read())
            conn.commit()
        finally:
            conn.close()

    def _migrate_workers_user_id(self):
        """Adds workers.user_id if it doesn't already exist -- SQLite has no
        'ADD COLUMN IF NOT EXISTS', so check PRAGMA table_info first. Safe to
        call on every startup."""
        conn = self._conn()
        try:
            cols = [r['name'] for r in conn.execute('PRAGMA table_info(workers)').fetchall()]
            if 'user_id' not in cols:
                conn.execute('ALTER TABLE workers ADD COLUMN user_id TEXT REFERENCES users(id)')
                conn.commit()
        finally:
            conn.close()

    def _migrate_tasks_task_type(self):
        """Adds tasks.task_type if missing, defaulting existing rows to
        SELF_PLAY (distributed/'s original, only task type) so pre-existing
        databases and distributed/'s own LAN-testing code path -- which
        creates rows via create_tasks_bulk() and knows nothing about typed
        tasks -- keep working unmodified."""
        conn = self._conn()
        try:
            cols = [r['name'] for r in conn.execute('PRAGMA table_info(tasks)').fetchall()]
            if 'task_type' not in cols:
                conn.execute(
                    "ALTER TABLE tasks ADD COLUMN task_type TEXT NOT NULL DEFAULT 'SELF_PLAY'")
                conn.commit()
        finally:
            conn.close()

    def _migrate_tasks_payload(self):
        """Adds tasks.payload (JSON, type-specific parameters) if missing.
        NULL/absent for plain SELF_PLAY rows, which already carry their
        params in the dedicated target_positions/depth/randomplies columns."""
        conn = self._conn()
        try:
            cols = [r['name'] for r in conn.execute('PRAGMA table_info(tasks)').fetchall()]
            if 'payload' not in cols:
                conn.execute('ALTER TABLE tasks ADD COLUMN payload TEXT')
                conn.commit()
        finally:
            conn.close()

    def _migrate_workers_capabilities(self):
        """Adds workers.capabilities (JSON: cpu_cores, ram_mb, gpu_available,
        gpu_name, trainer_capable) if missing. NULL until a worker reports
        itself via /workers/capabilities (or the register call), in which
        case it's treated as capability-unknown and only eligible for
        task types that don't require a declared capability (i.e. not
        TRAIN_NETWORK, which requires trainer_capable=1)."""
        conn = self._conn()
        try:
            cols = [r['name'] for r in conn.execute('PRAGMA table_info(workers)').fetchall()]
            if 'capabilities' not in cols:
                conn.execute('ALTER TABLE workers ADD COLUMN capabilities TEXT')
                conn.commit()
        finally:
            conn.close()

    # -- users ----------------------------------------------------------------
    def create_user(self, username, email, password):
        user_id = 'u_' + uuid.uuid4().hex[:12]
        pw_hash = accounts.hash_password(password)
        ts = now_iso()
        conn = self._conn()
        try:
            conn.execute(
                'INSERT INTO users (id, username, email, password_hash, created_at) '
                'VALUES (?,?,?,?,?)',
                (user_id, username, email or None, pw_hash, ts))
            conn.commit()
        finally:
            conn.close()
        return user_id

    def authenticate_user(self, username, password):
        conn = self._conn()
        try:
            row = conn.execute('SELECT * FROM users WHERE username = ? AND disabled = 0',
                               (username,)).fetchone()
        finally:
            conn.close()
        if row is None or not accounts.verify_password(password, row['password_hash']):
            return None
        return dict(row)

    def get_user_by_api_key(self, api_key):
        if not api_key:
            return None
        key_hash = accounts.hash_api_key(api_key)
        conn = self._conn()
        try:
            row = conn.execute('SELECT * FROM users WHERE api_key_hash = ? AND disabled = 0',
                               (key_hash,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def regenerate_api_key(self, user_id):
        key, key_hash = accounts.generate_api_key()
        conn = self._conn()
        try:
            conn.execute('UPDATE users SET api_key_hash = ? WHERE id = ?', (key_hash, user_id))
            conn.commit()
        finally:
            conn.close()
        return key

    def create_session(self, user_id):
        token, token_hash = accounts.generate_session_token()
        ts = now_iso()
        expires = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                 time.gmtime(time.time() + SESSION_LIFETIME_SECONDS))
        conn = self._conn()
        try:
            conn.execute('INSERT INTO sessions (token_hash, user_id, created_at, expires_at) '
                        'VALUES (?,?,?,?)', (token_hash, user_id, ts, expires))
            conn.commit()
        finally:
            conn.close()
        return token

    def get_user_by_session(self, token):
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        conn = self._conn()
        try:
            row = conn.execute(
                'SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id '
                'WHERE s.token_hash = ? AND s.expires_at > ? AND u.disabled = 0',
                (token_hash, now_iso())).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def revoke_session(self, token):
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        conn = self._conn()
        try:
            conn.execute('DELETE FROM sessions WHERE token_hash = ?', (token_hash,))
            conn.commit()
        finally:
            conn.close()

    # -- worker registration linked to an account ------------------------------
    def register_worker_for_user(self, user_id, hostname, engine_version, threads):
        worker_id, token = self.register_worker(hostname, engine_version, threads)
        conn = self._conn()
        try:
            conn.execute('UPDATE workers SET user_id = ? WHERE id = ?', (user_id, worker_id))
            conn.commit()
        finally:
            conn.close()
        return worker_id, token

    # -- leaderboard / community stats -----------------------------------------
    def get_leaderboard(self, limit=50):
        conn = self._conn()
        try:
            rows = conn.execute(
                'SELECT u.username, '
                '       COALESCE(SUM(w.positions_generated), 0) AS positions_generated, '
                '       COUNT(w.id) AS workers_count, '
                '       MAX(w.last_seen_at) AS last_active_at '
                'FROM users u LEFT JOIN workers w ON w.user_id = u.id '
                'WHERE u.disabled = 0 '
                'GROUP BY u.id '
                'ORDER BY positions_generated DESC '
                'LIMIT ?', (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_anonymous_positions(self):
        """Positions generated by workers with no linked account (legacy
        shared-secret registration), so the community stats page's totals
        still add up even when not every contributor has an account."""
        conn = self._conn()
        try:
            row = conn.execute(
                'SELECT COALESCE(SUM(positions_generated), 0) c FROM workers '
                'WHERE user_id IS NULL').fetchone()
            return row['c']
        finally:
            conn.close()

    # -- anti-cheat audit log ---------------------------------------------------
    def log_rejection(self, worker_id, reason, detail=''):
        conn = self._conn()
        try:
            conn.execute(
                'INSERT INTO rejections (worker_id, reason, detail, created_at) VALUES (?,?,?,?)',
                (worker_id, reason, (detail or '')[:500], now_iso()))
            conn.commit()
        finally:
            conn.close()

    def get_worker_rejection_count(self, worker_id, since_iso):
        conn = self._conn()
        try:
            row = conn.execute(
                'SELECT COUNT(*) c FROM rejections WHERE worker_id = ? AND created_at > ?',
                (worker_id, since_iso)).fetchone()
            return row['c']
        finally:
            conn.close()

    # -- worker capabilities ----------------------------------------------------
    def set_worker_capabilities(self, worker_id, capabilities: dict):
        """capabilities: {'cpu_cores': int, 'ram_mb': int, 'gpu_available': bool,
        'gpu_name': str|None, 'gpu_backends': [{'vendor','backend','name',
        'trainable','detected_via'}, ...], 'best_gpu_backend': str|None,
        'trainer_capable': bool} -- see platform/worker/capabilities.py's
        detect_capabilities() for exactly how gpu_backends/best_gpu_backend
        are populated (multi-vendor: NVIDIA/CUDA, AMD/ROCm, Intel, stored
        as-is here with no schema-specific handling since this column is
        just opaque JSON). Called on registration and on every heartbeat,
        so a worker's declared resource limits (it may opt to advertise
        less than its true hardware) stay current."""
        conn = self._conn()
        try:
            conn.execute('UPDATE workers SET capabilities = ? WHERE id = ?',
                         (json.dumps(capabilities), worker_id))
            conn.commit()
        finally:
            conn.close()

    def get_worker_capabilities(self, worker_id):
        conn = self._conn()
        try:
            row = conn.execute('SELECT capabilities FROM workers WHERE id = ?',
                               (worker_id,)).fetchone()
            if row is None or not row['capabilities']:
                return None
            return json.loads(row['capabilities'])
        finally:
            conn.close()

    # -- typed tasks (SELF_PLAY / DATA_GENERATION / ELO_MATCH / TRAIN_NETWORK) --
    #
    # These live alongside (not instead of) distributed/server/db.py's
    # create_tasks_bulk()/assign_next_task(), which remain the untouched
    # path for plain LAN SELF_PLAY use per docs/DISTRIBUTED_DATA_GENERATION.md.
    # A row created by either path lands in the same `tasks` table; task_type
    # defaults to 'SELF_PLAY' so both paths can coexist in one queue.
    VALID_TASK_TYPES = ('SELF_PLAY', 'DATA_GENERATION', 'ELO_MATCH', 'TRAIN_NETWORK')

    def create_typed_task(self, task_type, payload: dict, batch_label=None,
                           target_positions=0, depth=0, randomplies=0):
        """Creates a single typed task row. `payload` is task-type-specific,
        e.g.:
          DATA_GENERATION: {games, depth, randomplies, format}
          ELO_MATCH:       {candidate_artifact_id, baseline_artifact_id,
                             games, match_depth, movetime_ms}
          TRAIN_NETWORK:   {dataset_artifact_id, epochs, hidden}
        target_positions/depth/randomplies are kept as real columns (not just
        payload keys) only for SELF_PLAY/DATA_GENERATION, since distributed/'s
        existing dashboard and get_stats() queries already read them directly;
        other task types leave them at 0 and put everything in payload."""
        if task_type not in self.VALID_TASK_TYPES:
            raise ValueError(f'unknown task_type {task_type!r}')
        batch_label = batch_label or (task_type.lower() + '_' + uuid.uuid4().hex[:8])
        task_id = 't_' + uuid.uuid4().hex[:12]
        ts = now_iso()
        conn = self._conn()
        try:
            conn.execute(
                'INSERT INTO tasks (id, status, target_positions, depth, randomplies, '
                'batch_label, created_at, task_type, payload) VALUES (?,?,?,?,?,?,?,?,?)',
                (task_id, 'pending', target_positions, depth, randomplies, batch_label, ts,
                 task_type, json.dumps(payload)))
            conn.commit()
        finally:
            conn.close()
        return task_id

    def assign_next_typed_task(self, worker_id, lease_seconds, capabilities=None):
        """Capability-aware variant of distributed/server/db.py's
        assign_next_task(): a TRAIN_NETWORK task is only handed to a worker
        whose most recently reported capabilities have trainer_capable=1 --
        every other task type is assignable to any worker, same as before.
        `capabilities` is the worker's live self-report from this request
        (if the worker sends one); falls back to its last-persisted
        capabilities row if omitted, so a worker that only reports
        capabilities at registration time is still handled correctly on
        later poll requests that don't repeat them."""
        if capabilities is None:
            capabilities = self.get_worker_capabilities(worker_id)
        trainer_capable = bool(capabilities and capabilities.get('trainer_capable'))

        conn = self._conn()
        try:
            self._reclaim_expired_leases(conn)
            if trainer_capable:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE status='pending' ORDER BY created_at LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE status='pending' AND task_type != 'TRAIN_NETWORK' "
                    "ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                conn.commit()
                return None
            task_id = row['id']
            lease_expires = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                           time.gmtime(time.time() + lease_seconds))
            conn.execute(
                "UPDATE tasks SET status='assigned', assigned_worker_id=?, assigned_at=?, "
                "lease_expires_at=? WHERE id=?",
                (worker_id, now_iso(), lease_expires, task_id))
            conn.commit()
            task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            task = dict(task)
            task['payload'] = json.loads(task['payload']) if task.get('payload') else {}
            if task['task_type'] == 'SELF_PLAY' and not task['payload']:
                # Rows created via the legacy create_tasks_bulk() (distributed/'s
                # own path, or any pre-existing database) carry their params in
                # the dedicated target_positions/depth/randomplies columns, not
                # payload -- merge them in here so /tasks/next-typed is fully
                # self-describing for EVERY task_type, and a worker only ever
                # needs to read task['payload'], never fall back to top-level
                # columns depending on how the task happened to be created.
                task['payload'] = {
                    'target_positions': task['target_positions'],
                    'depth': task['depth'],
                    'randomplies': task['randomplies'],
                }
            return task
        finally:
            conn.close()

    # -- artifacts (datasets / checkpoints / candidate & accepted networks) -----
    def create_artifact(self, kind, file_path, sha256, size_bytes, created_by_task_id=None,
                         created_by_worker_id=None, metadata=None, accepted=False):
        if kind not in ('dataset', 'checkpoint', 'network'):
            raise ValueError(f'unknown artifact kind {kind!r}')
        artifact_id = 'a_' + uuid.uuid4().hex[:12]
        ts = now_iso()
        conn = self._conn()
        try:
            conn.execute(
                'INSERT INTO artifacts (id, kind, file_path, sha256, size_bytes, '
                'created_by_task_id, created_by_worker_id, accepted, metadata, created_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                (artifact_id, kind, file_path, sha256, size_bytes, created_by_task_id,
                 created_by_worker_id, int(bool(accepted)),
                 json.dumps(metadata) if metadata is not None else None, ts))
            conn.commit()
        finally:
            conn.close()
        return artifact_id

    def get_artifact(self, artifact_id):
        conn = self._conn()
        try:
            row = conn.execute('SELECT * FROM artifacts WHERE id = ?', (artifact_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            if d.get('metadata'):
                d['metadata'] = json.loads(d['metadata'])
            return d
        finally:
            conn.close()

    def list_artifacts(self, kind=None, accepted_only=False):
        conn = self._conn()
        try:
            q = 'SELECT * FROM artifacts'
            clauses, params = [], []
            if kind:
                clauses.append('kind = ?')
                params.append(kind)
            if accepted_only:
                clauses.append('accepted = 1')
            if clauses:
                q += ' WHERE ' + ' AND '.join(clauses)
            q += ' ORDER BY created_at DESC'
            rows = conn.execute(q, params).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                if d.get('metadata'):
                    d['metadata'] = json.loads(d['metadata'])
                out.append(d)
            return out
        finally:
            conn.close()

    def get_strongest_network(self):
        """The current 'strongest network' the automated improvement loop
        should treat as baseline: the most recently accepted 'network'
        artifact. Returns None if no network has been accepted yet (a fresh
        deployment must be seeded with an initial baseline artifact -- see
        docs/TRAINING.md -- before ELO_MATCH tasks have anything to compare
        candidates against)."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE kind='network' AND accepted=1 "
                "ORDER BY created_at DESC LIMIT 1").fetchone()
            if row is None:
                return None
            d = dict(row)
            if d.get('metadata'):
                d['metadata'] = json.loads(d['metadata'])
            return d
        finally:
            conn.close()

    def mark_artifact_accepted(self, artifact_id):
        conn = self._conn()
        try:
            conn.execute('UPDATE artifacts SET accepted = 1 WHERE id = ?', (artifact_id,))
            conn.commit()
        finally:
            conn.close()

    def complete_task_for_worker(self, task_id, worker_id):
        """Marks task_id completed, but only if it's currently assigned to
        worker_id (or unassigned) and not already completed -- same
        ownership guard as submit_match_result, generalized for any task
        type whose 'done' signal is an artifact upload rather than a
        positions/match-result submission (currently: TRAIN_NETWORK,
        uploading its trained checkpoint/candidate network -- see
        app.py's upload_artifact). Returns True if this call performed the
        completion, False if the task didn't exist, was already completed,
        or is assigned to a different worker (caller should treat False as
        "not an error, just not actionable by you")."""
        conn = self._conn()
        try:
            task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if task is None:
                return False
            if task['status'] == 'completed':
                return False
            if task['assigned_worker_id'] not in (None, worker_id):
                return False
            conn.execute(
                "UPDATE tasks SET status='completed', completed_at=? WHERE id=?",
                (now_iso(), task_id))
            conn.commit()
            return True
        finally:
            conn.close()

    # -- ELO_MATCH results --------------------------------------------------------
    def submit_match_result(self, task_id, worker_id, candidate_artifact_id,
                             baseline_artifact_id, wins, losses, draws, pgn_path=None):
        """Records one worker's ELO_MATCH outcome and marks the task
        completed (each ELO_MATCH task is a single self-contained match
        assignment, unlike SELF_PLAY/DATA_GENERATION tasks which accumulate
        partial submissions toward a target -- there is exactly one
        match_results row per completed ELO_MATCH task)."""
        games = wins + losses + draws
        ts = now_iso()
        conn = self._conn()
        try:
            task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if task is None:
                raise KeyError(f'unknown task_id {task_id!r}')
            conn.execute(
                'INSERT INTO match_results (task_id, worker_id, candidate_artifact_id, '
                'baseline_artifact_id, wins, losses, draws, games, pgn_path, submitted_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                (task_id, worker_id, candidate_artifact_id, baseline_artifact_id,
                 wins, losses, draws, games, pgn_path, ts))
            conn.execute(
                "UPDATE tasks SET status='completed', completed_at=? "
                "WHERE id=? AND status != 'completed'", (ts, task_id))
            conn.commit()
        finally:
            conn.close()
        return {'games': games, 'wins': wins, 'losses': losses, 'draws': draws}

    def get_match_results_for_artifact(self, candidate_artifact_id):
        conn = self._conn()
        try:
            rows = conn.execute(
                'SELECT * FROM match_results WHERE candidate_artifact_id = ? '
                'ORDER BY submitted_at', (candidate_artifact_id,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- automated pipeline support (Phase 2) ------------------------------------
    def get_max_position_id(self):
        """Highest `positions.id` currently in the database, or 0 if empty.
        Used as the natural dataset-export watermark (see
        export_positions_range) -- no separate watermark table needed since
        every exported dataset artifact records the max id it included in
        its own metadata (see app.py's /admin/pipeline/export-dataset)."""
        conn = self._conn()
        try:
            row = conn.execute('SELECT MAX(id) AS m FROM positions').fetchone()
            return int(row['m']) if row and row['m'] is not None else 0
        finally:
            conn.close()

    def export_positions_range(self, min_id_exclusive, limit):
        """Returns (rows, max_id_included) for positions with id >
        min_id_exclusive, oldest-first, capped at `limit`. Each row has the
        fen/eval_cp/result fields tools/nnue_pipeline's JSONL format needs
        (see database.py module doc / train_network.py), plus the optional
        score_swing/best_move_changes search-instability signal (NULL for
        rows submitted before that migration or by a source that doesn't
        report it -- see db.py's _migrate_positions_instability) so
        train.py's load_jsonl_datasets() can prioritize quality/difficulty
        during any later truncation, without this export step itself
        dropping anything. Deliberately does NOT filter by worker trust here
        -- see anti_cheat.py and get_worker_rejection_count for where
        untrustworthy submissions are already kept out of the `positions`
        table in the first place (rejected at submit_positions() time, never
        inserted at all)."""
        conn = self._conn()
        try:
            rows = conn.execute(
                'SELECT id, fen, eval_cp, result, score_swing, best_move_changes '
                'FROM positions WHERE id > ? ORDER BY id LIMIT ?',
                (min_id_exclusive, limit)).fetchall()
            rows = [dict(r) for r in rows]
            max_id = rows[-1]['id'] if rows else min_id_exclusive
            return rows, max_id
        finally:
            conn.close()

    def delete_positions_up_to(self, max_id):
        """Deletes every row in `positions` with id <= max_id. Only ever
        safe to call with a max_id that is <= the max_position_id of an
        already-created 'dataset' artifact (see export_positions_range /
        app.py's /admin/pipeline/export-dataset): once positions have been
        exported into a dataset artifact file, that file is the permanent
        record of their content (fen/eval_cp/result/etc.) -- the live rows
        in this table become redundant disk usage, not a second copy of
        data that's needed anywhere else. See
        /admin/pipeline/prune-positions (app.py), which is the only caller
        and picks max_id conservatively (an older, already-exported
        watermark, keeping a configurable number of the most recent
        datasets' raw rows as a safety margin -- never a position that
        hasn't been exported into a dataset artifact yet).

        Returns the number of rows actually deleted."""
        conn = self._conn()
        try:
            cur = conn.execute('DELETE FROM positions WHERE id <= ?', (max_id,))
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def count_tasks_by_type_status(self, task_type, status='pending'):
        conn = self._conn()
        try:
            row = conn.execute(
                'SELECT COUNT(*) c FROM tasks WHERE task_type = ? AND status = ?',
                (task_type, status)).fetchone()
            return int(row['c'])
        finally:
            conn.close()

    def get_task_counts_by_type(self):
        """{task_type: {status: count}} for every (task_type, status) pair
        that currently has at least one row -- one query, used by the
        dashboard (see dashboard_data.py) instead of calling
        count_tasks_by_type_status() once per type/status combination."""
        conn = self._conn()
        try:
            rows = conn.execute(
                'SELECT task_type, status, COUNT(*) c FROM tasks '
                'GROUP BY task_type, status').fetchall()
        finally:
            conn.close()
        out = {}
        for r in rows:
            out.setdefault(r['task_type'], {})[r['status']] = r['c']
        return out

    def get_worker_capability_counts(self, recent_seconds=600):
        """Live counts of CPU-only vs. GPU-capable workers, split out from
        the flat 'active_workers' (== not disabled) figure get_stats()
        already returns. 'connected' means an authenticated request
        (registration or a task poll, both of which touch last_seen_at)
        was seen within `recent_seconds` -- distinct from 'active' (merely
        not disabled), since a worker can be enabled but offline for days.
        capabilities is opaque JSON (see set_worker_capabilities); a worker
        that has never reported it counts as capability-unknown rather than
        being guessed into either bucket."""
        conn = self._conn()
        try:
            rows = conn.execute(
                'SELECT capabilities, last_seen_at, disabled FROM workers').fetchall()
        finally:
            conn.close()
        cutoff = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - recent_seconds))
        connected = cpu_workers = gpu_workers = capability_unknown = 0
        for r in rows:
            if r['disabled']:
                continue
            if r['last_seen_at'] and r['last_seen_at'] >= cutoff:
                connected += 1
            caps = json.loads(r['capabilities']) if r['capabilities'] else None
            if caps is None:
                capability_unknown += 1
            elif caps.get('gpu_available'):
                gpu_workers += 1
            else:
                cpu_workers += 1
        return {
            'connected_workers': connected,
            'cpu_workers': cpu_workers,
            'gpu_workers': gpu_workers,
            'capability_unknown': capability_unknown,
        }

    def get_active_contributor_count(self, recent_seconds=600):
        """Distinct *contributors* seen recently, as opposed to
        get_worker_capability_counts()'s 'connected_workers' (which counts
        worker installs, not people/accounts -- one contributor can run
        several workers, e.g. a desktop and a laptop, and get_leaderboard()
        already dedupes by users.id for exactly that reason). A registered
        account with at least one recently-seen, non-disabled worker counts
        once no matter how many workers it runs; a worker with no linked
        account (legacy shared-secret registration, see workers.user_id)
        counts individually since there's no account identity to dedupe by.
        Never fabricated: both halves are plain COUNT/COUNT DISTINCT over
        the real workers/users tables, same recency cutoff convention as
        get_worker_capability_counts()."""
        conn = self._conn()
        try:
            rows = conn.execute(
                'SELECT user_id, last_seen_at, disabled FROM workers').fetchall()
        finally:
            conn.close()
        cutoff = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - recent_seconds))
        active_user_ids = set()
        anonymous_workers = 0
        for r in rows:
            if r['disabled']:
                continue
            if not r['last_seen_at'] or r['last_seen_at'] < cutoff:
                continue
            if r['user_id']:
                active_user_ids.add(r['user_id'])
            else:
                anonymous_workers += 1
        return {
            'total': len(active_user_ids) + anonymous_workers,
            'registered_accounts': len(active_user_ids),
            'anonymous_workers': anonymous_workers,
        }

    def get_contribution_by_hardware(self):
        """Cumulative work actually performed, split by CPU vs. GPU, as
        opposed to get_worker_capability_counts()'s currently-connected
        snapshot. Two real, additive signals (nothing simulated):
          - positions_generated: workers.positions_generated is a running
            total maintained by submit_positions()/register_worker(), summed
            per hardware bucket using each worker's current reported
            capabilities (see set_worker_capabilities) -- the only caveat is
            that a worker's bucket reflects its *current* reported hardware,
            not necessarily what it had at the time of every past
            submission, since capabilities aren't historized per-submission.
          - completed TRAIN_NETWORK tasks: joined against the assigned
            worker's current capabilities the same way. Data generation
            (SELF_PLAY/DATA_GENERATION) is the CPU-bound task family and
            TRAIN_NETWORK is the GPU-preferring one in this architecture
            (see platform/trainer/train_network.py's backend selection), so
            these two counters are the most direct 'CPU contribution' /
            'GPU contribution' figures available without inventing new
            per-submission hardware tracking."""
        conn = self._conn()
        try:
            worker_rows = conn.execute(
                'SELECT capabilities, positions_generated FROM workers').fetchall()
            train_rows = conn.execute(
                "SELECT w.capabilities AS capabilities, COUNT(*) AS c "
                "FROM tasks t JOIN workers w ON w.id = t.assigned_worker_id "
                "WHERE t.task_type = 'TRAIN_NETWORK' AND t.status = 'completed' "
                "GROUP BY w.id").fetchall()
        finally:
            conn.close()

        def bucket_of(caps_json):
            caps = json.loads(caps_json) if caps_json else None
            if caps is None:
                return 'unknown'
            return 'gpu' if caps.get('gpu_available') else 'cpu'

        positions = {'cpu': 0, 'gpu': 0, 'unknown': 0}
        for r in worker_rows:
            positions[bucket_of(r['capabilities'])] += r['positions_generated'] or 0

        train_jobs = {'cpu': 0, 'gpu': 0, 'unknown': 0}
        for r in train_rows:
            train_jobs[bucket_of(r['capabilities'])] += r['c']

        return {
            'positions_generated': positions,
            'completed_training_jobs': train_jobs,
        }
