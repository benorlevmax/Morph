#!/usr/bin/env python3
"""db.py - SQLite access layer for the distributed data-generation server.

All queries live here so app.py (the FastAPI route layer) stays thin. Uses a
short-lived connection per call (SQLite + WAL mode handles concurrent readers
fine at this scale; see docs/DISTRIBUTED_DATA_GENERATION.md for the "when
would you outgrow this" note).
"""
import hashlib
import json
import secrets
import sqlite3
import time
import uuid

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation import validate_position, content_hash

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
from init_db import ensure_schema


def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        ensure_schema(db_path)
        self._migrate_positions_instability()

    def _migrate_positions_instability(self):
        """Adds positions.score_swing / positions.best_move_changes if
        missing -- SQLite has no 'ADD COLUMN IF NOT EXISTS', so check
        PRAGMA table_info first (same idempotent pattern platform/server/
        database.py already uses for its own migrations). Both columns are
        NULL for every position submitted before this migration, and for
        any worker/engine build that doesn't report the signal -- NULL means
        "not recorded", not "zero instability", and callers must treat it
        that way (e.g. dataset-export prioritization skips NULL rows rather
        than treating them as maximally stable). Purely additive: no existing
        row or column is touched."""
        conn = self._conn()
        try:
            cols = [r['name'] for r in conn.execute('PRAGMA table_info(positions)').fetchall()]
            if 'score_swing' not in cols:
                conn.execute('ALTER TABLE positions ADD COLUMN score_swing INTEGER')
            if 'best_move_changes' not in cols:
                conn.execute('ALTER TABLE positions ADD COLUMN best_move_changes INTEGER')
            conn.commit()
        finally:
            conn.close()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    # -- workers ------------------------------------------------------------
    def register_worker(self, hostname, engine_version, threads):
        worker_id = 'w_' + uuid.uuid4().hex[:12]
        token = secrets.token_hex(24)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        ts = now_iso()
        conn = self._conn()
        try:
            conn.execute(
                'INSERT INTO workers (id, token_hash, hostname, engine_version, threads, '
                'registered_at, last_seen_at) VALUES (?,?,?,?,?,?,?)',
                (worker_id, token_hash, hostname, engine_version, threads, ts, ts))
            conn.commit()
        finally:
            conn.close()
        return worker_id, token

    def authenticate_worker(self, token):
        """Returns the worker row (as a dict) or None. Updates last_seen_at
        on success (this doubles as the worker's heartbeat)."""
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        conn = self._conn()
        try:
            row = conn.execute(
                'SELECT * FROM workers WHERE token_hash = ? AND disabled = 0',
                (token_hash,)).fetchone()
            if row is None:
                return None
            conn.execute('UPDATE workers SET last_seen_at = ? WHERE id = ?',
                         (now_iso(), row['id']))
            conn.commit()
            return dict(row)
        finally:
            conn.close()

    def list_workers(self):
        conn = self._conn()
        try:
            rows = conn.execute(
                'SELECT id, hostname, engine_version, threads, registered_at, last_seen_at, '
                'positions_generated, submissions_count, disabled FROM workers '
                'ORDER BY last_seen_at DESC').fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def disable_worker(self, worker_id):
        conn = self._conn()
        try:
            conn.execute('UPDATE workers SET disabled = 1 WHERE id = ?', (worker_id,))
            conn.commit()
        finally:
            conn.close()

    # -- tasks ----------------------------------------------------------------
    def create_tasks_bulk(self, total_positions, chunk_size, depth, randomplies, batch_label=None):
        batch_label = batch_label or ('batch_' + uuid.uuid4().hex[:8])
        n_tasks = max(1, -(-total_positions // chunk_size))  # ceil div
        ts = now_iso()
        task_ids = []
        conn = self._conn()
        try:
            remaining = total_positions
            for _ in range(n_tasks):
                target = min(chunk_size, remaining)
                remaining -= target
                task_id = 't_' + uuid.uuid4().hex[:12]
                conn.execute(
                    'INSERT INTO tasks (id, status, target_positions, depth, randomplies, '
                    'batch_label, created_at) VALUES (?,?,?,?,?,?,?)',
                    (task_id, 'pending', target, depth, randomplies, batch_label, ts))
                task_ids.append(task_id)
            conn.commit()
        finally:
            conn.close()
        return task_ids, batch_label

    def _reclaim_expired_leases(self, conn):
        conn.execute(
            "UPDATE tasks SET status='pending', assigned_worker_id=NULL, assigned_at=NULL, "
            "lease_expires_at=NULL WHERE status='assigned' AND lease_expires_at < ?",
            (now_iso(),))

    def assign_next_task(self, worker_id, lease_seconds):
        conn = self._conn()
        try:
            self._reclaim_expired_leases(conn)
            row = conn.execute(
                "SELECT id FROM tasks WHERE status='pending' ORDER BY created_at LIMIT 1").fetchone()
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
            return dict(task)
        finally:
            conn.close()

    def get_task(self, task_id):
        conn = self._conn()
        try:
            row = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_tasks(self, status=None):
        conn = self._conn()
        try:
            if status:
                rows = conn.execute('SELECT * FROM tasks WHERE status=? ORDER BY created_at',
                                     (status,)).fetchall()
            else:
                rows = conn.execute('SELECT * FROM tasks ORDER BY created_at').fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- results ----------------------------------------------------------------
    def submit_positions(self, task_id, worker_id, records):
        """Validates + dedups + inserts `records` (list of dicts) against
        `task_id`, attributed to `worker_id`. Returns a summary dict."""
        conn = self._conn()
        accepted = 0
        duplicates = 0
        rejected = 0
        rejected_reasons = []
        try:
            task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if task is None:
                raise KeyError(f'unknown task_id {task_id!r}')

            ts = now_iso()
            for rec in records:
                reason = validate_position(rec)
                if reason is not None:
                    rejected += 1
                    if len(rejected_reasons) < 20:
                        rejected_reasons.append(reason)
                    continue
                chash = content_hash(rec)
                # Optional search-instability signal (see search.h's
                # SearchResult::scoreSwing/bestMoveChanges) -- absent for
                # records from a worker/executor that doesn't report it
                # (e.g. platform/worker/selfplay.py's per-game loop today),
                # in which case both columns store NULL ("not recorded").
                # Never fabricated, never required.
                score_swing = rec.get('score_swing')
                best_move_changes = rec.get('best_move_changes')
                try:
                    conn.execute(
                        'INSERT INTO positions (task_id, worker_id, fen, side_to_move, eval_cp, '
                        'result, depth, nodes, engine_version, content_hash, created_at, '
                        'score_swing, best_move_changes) '
                        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (task_id, worker_id, rec['fen'], rec['side_to_move'], int(rec['eval_cp']),
                         float(rec['result']), int(rec['depth']), int(rec['nodes']),
                         rec['engine_version'], chash, ts,
                         int(score_swing) if score_swing is not None else None,
                         int(best_move_changes) if best_move_changes is not None else None))
                    accepted += 1
                except sqlite3.IntegrityError:
                    duplicates += 1  # UNIQUE(content_hash) collision

            conn.execute(
                'UPDATE tasks SET accepted_positions = accepted_positions + ? WHERE id = ?',
                (accepted, task_id))
            new_total = conn.execute(
                'SELECT accepted_positions, target_positions FROM tasks WHERE id=?',
                (task_id,)).fetchone()
            if new_total['accepted_positions'] >= new_total['target_positions']:
                conn.execute(
                    "UPDATE tasks SET status='completed', completed_at=? "
                    "WHERE id=? AND status != 'completed'", (ts, task_id))

            conn.execute(
                'UPDATE workers SET positions_generated = positions_generated + ?, '
                'submissions_count = submissions_count + 1 WHERE id = ?',
                (accepted, worker_id))

            conn.execute(
                'INSERT INTO submissions (task_id, worker_id, positions_submitted, '
                'positions_accepted, duplicates_skipped, rejected, rejected_reasons, submitted_at) '
                'VALUES (?,?,?,?,?,?,?,?)',
                (task_id, worker_id, len(records), accepted, duplicates, rejected,
                 json.dumps(rejected_reasons), ts))

            conn.commit()
        finally:
            conn.close()

        return {'accepted': accepted, 'duplicates': duplicates, 'rejected': rejected,
                'rejected_reasons': rejected_reasons}

    # -- stats ----------------------------------------------------------------
    def get_stats(self):
        conn = self._conn()
        try:
            total_positions = conn.execute('SELECT COUNT(*) c FROM positions').fetchone()['c']
            total_workers = conn.execute('SELECT COUNT(*) c FROM workers').fetchone()['c']
            active_workers = conn.execute(
                "SELECT COUNT(*) c FROM workers WHERE disabled = 0").fetchone()['c']
            tasks_by_status = {r['status']: r['c'] for r in conn.execute(
                'SELECT status, COUNT(*) c FROM tasks GROUP BY status').fetchall()}
            by_worker = [dict(r) for r in conn.execute(
                'SELECT id, hostname, positions_generated, submissions_count, last_seen_at '
                'FROM workers ORDER BY positions_generated DESC').fetchall()]
            by_engine_version = [dict(r) for r in conn.execute(
                'SELECT engine_version, COUNT(*) positions FROM positions '
                'GROUP BY engine_version ORDER BY positions DESC').fetchall()]
            result_dist = [dict(r) for r in conn.execute(
                'SELECT result, COUNT(*) c FROM positions GROUP BY result').fetchall()]
            return {
                'total_positions': total_positions,
                'total_workers': total_workers,
                'active_workers': active_workers,
                'tasks_by_status': tasks_by_status,
                'positions_by_worker': by_worker,
                'positions_by_engine_version': by_engine_version,
                'result_distribution': result_dist,
            }
        finally:
            conn.close()
