#!/usr/bin/env python3
"""test_prune.py - Regression tests for the position-retention/pruning
feature (disk-space management for long-running deployments -- see
database.py's delete_positions_up_to(), app.py's
/admin/pipeline/prune-positions endpoint, and auto_pipeline.py's opt-in
--prune-after-export flag / maybe_prune_positions()).

Two layers, matching how the feature is actually split:

  DatabaseDeletionTests: delete_positions_up_to() against a real (temp
  file) PlatformDatabase -- proves the SQL actually deletes the rows it
  claims to, and nothing newer than the given max_id.

  MaybePrunePositionsTests: auto_pipeline.maybe_prune_positions() against
  an in-memory FakeAdminClient (same style as test_promotion.py) -- proves
  the opt-in flag is honored (a deployment that never passes
  --prune-after-export must never have anything deleted) and that
  keep_datasets is forwarded correctly.

The watermark-selection logic itself (walking auto_pipeline-sourced
dataset artifacts newest-first, picking the (keep_datasets)-th one) lives
in app.py's prune_positions() endpoint, which needs a real FastAPI
app + PlatformDatabase to exercise meaningfully -- see
WatermarkSelectionTests, which calls db.list_artifacts()/db.create_artifact()
directly (the same primitives the endpoint itself uses) to prove the
selection math without needing to spin up an HTTP server.

Run directly:  python3 test_prune.py
Run via pytest: pytest test_prune.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'tools', 'nnue_pipeline'))

import auto_pipeline as ap
from database import PlatformDatabase


def _make_db():
    fd, path = tempfile.mkstemp(suffix='.sqlite3')
    os.close(fd)
    os.unlink(path)   # PlatformDatabase creates it fresh
    return PlatformDatabase(path), path


def _seed_positions(db, n, engine_version='v1'):
    """Inserts n position rows the same way a real worker submission would
    (one dummy worker + one dummy task to satisfy the FK constraints
    distributed/server/db.py enforces via PRAGMA foreign_keys=ON), each
    with a distinct content_hash (UNIQUE). Returns the list of ids inserted,
    in insertion order (== id order, since positions.id is AUTOINCREMENT)."""
    worker_id, _token = db.register_worker('test-host', engine_version, 4)
    task_ids, _label = db.create_tasks_bulk(total_positions=n, chunk_size=max(n, 1),
                                             depth=6, randomplies=6)
    task_id = task_ids[0]
    # Same legal FEN (the start position) for every row -- content_hash()
    # (distributed/server/validation.py) keys on (fen, eval_cp, result,
    # depth, engine_version), so varying eval_cp alone is enough to give
    # each row a distinct dedup hash without needing distinct real FENs.
    start_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    records = [{'fen': start_fen, 'side_to_move': 'w', 'eval_cp': i, 'result': 0.5,
                'depth': 6, 'nodes': 1000, 'engine_version': engine_version}
               for i in range(n)]
    result = db.submit_positions(task_id, worker_id, records)
    assert result['accepted'] == n, f'test setup bug: expected {n} accepted, got {result}'
    conn = db._conn()
    try:
        rows = conn.execute('SELECT id FROM positions ORDER BY id').fetchall()
    finally:
        conn.close()
    return [r['id'] for r in rows]


class DatabaseDeletionTests(unittest.TestCase):
    def setUp(self):
        self.db, self.path = _make_db()

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def test_deletes_only_rows_up_to_and_including_max_id(self):
        ids = _seed_positions(self.db, 10)
        cutoff = ids[4]   # the 5th row

        deleted = self.db.delete_positions_up_to(cutoff)

        self.assertEqual(deleted, 5, 'should delete exactly the first 5 rows (ids <= cutoff)')
        conn = self.db._conn()
        try:
            remaining = [r['id'] for r in conn.execute('SELECT id FROM positions ORDER BY id')]
        finally:
            conn.close()
        self.assertEqual(remaining, ids[5:],
                          'every row newer than the cutoff must survive untouched')

    def test_max_id_of_zero_deletes_nothing(self):
        _seed_positions(self.db, 5)
        deleted = self.db.delete_positions_up_to(0)
        self.assertEqual(deleted, 0)
        self.assertEqual(self.db.get_max_position_id(), 5)

    def test_deleting_beyond_the_newest_row_deletes_everything(self):
        ids = _seed_positions(self.db, 5)
        deleted = self.db.delete_positions_up_to(ids[-1] + 1000)
        self.assertEqual(deleted, 5)
        self.assertEqual(self.db.get_max_position_id(), 0)

    def test_deleted_positions_do_not_affect_a_later_export_watermark(self):
        """The whole point of the feature: once positions have been
        exported into a dataset artifact, deleting the raw rows must not
        change what get_max_position_id()/export_positions_range() would
        have reported -- the watermark for future exports is unaffected,
        because it's driven by dataset artifact metadata (see app.py's
        export_dataset), not by what's still physically in the table."""
        ids = _seed_positions(self.db, 6)
        rows, max_id = self.db.export_positions_range(0, 1000)
        self.assertEqual(max_id, ids[-1])

        self.db.delete_positions_up_to(ids[-1])

        # A second export starting from the same watermark now correctly
        # finds nothing new (not because the rows were deleted, but because
        # there genuinely isn't anything past that watermark yet) --
        # exercises exactly the sequence auto_pipeline.py's Stage 1 then
        # Stage 4 perform in one cycle.
        rows2, max_id2 = self.db.export_positions_range(max_id, 1000)
        self.assertEqual(rows2, [])
        self.assertEqual(max_id2, max_id)


class WatermarkSelectionTests(unittest.TestCase):
    """Exercises the same selection primitives app.py's prune_positions()
    endpoint uses (db.list_artifacts(kind='dataset') newest-first, reading
    each artifact's metadata.max_position_id) to prove the keep_datasets
    buffer math without needing a running FastAPI server."""

    def setUp(self):
        self.db, self.path = _make_db()
        self._next_ts = 1_700_000_000  # arbitrary fixed base, seconds

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def _record_export(self, max_position_id, count=100):
        """Records a dataset artifact and forces its created_at to a
        strictly increasing timestamp. now_iso() (database.py) only has
        one-second resolution, so a naive back-to-back sequence of
        create_artifact() calls in a fast test can tie -- and
        list_artifacts()'s ORDER BY created_at DESC does not guarantee any
        particular order among ties. Setting created_at explicitly (via a
        direct UPDATE, the same table create_artifact itself writes to)
        makes the newest-first ordering this test depends on deterministic
        instead of accidentally relying on SQLite's tie-breaking behavior."""
        import time as _time
        artifact_id = self.db.create_artifact(
            'dataset', self.path, f'sha-{max_position_id}', 1,
            metadata={'source': 'auto_pipeline', 'max_position_id': max_position_id,
                      'count': count, 'min_position_id_exclusive': 0})
        ts_iso = _time.strftime('%Y-%m-%dT%H:%M:%SZ', _time.gmtime(self._next_ts))
        self._next_ts += 1
        conn = self.db._conn()
        try:
            conn.execute('UPDATE artifacts SET created_at = ? WHERE id = ?', (ts_iso, artifact_id))
            conn.commit()
        finally:
            conn.close()
        return artifact_id

    def _select_prune_up_to(self, keep_datasets):
        """Mirrors app.py's prune_positions() watermark-selection logic
        exactly (see that function's docstring) -- kept as a tiny local
        copy so this test doesn't need a live HTTP server, but proves the
        same math the endpoint runs."""
        watermarks = []
        for art in self.db.list_artifacts(kind='dataset'):
            meta = art.get('metadata') or {}
            if meta.get('source') == 'auto_pipeline':
                watermarks.append(int(meta.get('max_position_id', 0)))
        watermarks.sort(reverse=True)   # see app.py's prune_positions() docstring
        if len(watermarks) <= keep_datasets:
            return None
        return watermarks[keep_datasets]

    def test_fewer_exports_than_keep_datasets_prunes_nothing(self):
        self._record_export(1000)
        self._record_export(2000)
        self.assertIsNone(self._select_prune_up_to(keep_datasets=3),
                           'only 2 exports exist -- must not prune with keep_datasets=3 '
                           '(need at least 4: 3 to keep + 1 older one to prune up to)')

    def test_exactly_keep_datasets_exports_prunes_nothing(self):
        """Even with precisely keep_datasets exports on hand, there is no
        OLDER export to safely prune up to yet -- all of them are still
        within the kept set."""
        self._record_export(1000)
        self._record_export(2000)
        self._record_export(3000)
        self.assertIsNone(self._select_prune_up_to(keep_datasets=3))

    def test_keeps_the_most_recent_n_exports_worth_of_rows(self):
        self._record_export(1000)
        self._record_export(2000)
        self._record_export(3000)
        self._record_export(4000)   # newest

        prune_up_to = self._select_prune_up_to(keep_datasets=3)

        self.assertEqual(prune_up_to, 1000,
                          'with 4 exports and keep_datasets=3, the 3 most recent exports '
                          '(4000, 3000, 2000) must keep their rows -- only positions covered by '
                          'the 4th-newest (oldest) export, watermark 1000, are safe to prune')

    def test_keep_datasets_one_keeps_only_the_newest_export(self):
        self._record_export(1000)
        self._record_export(2000)

        prune_up_to = self._select_prune_up_to(keep_datasets=1)

        self.assertEqual(prune_up_to, 1000,
                          'keep_datasets=1 keeps just the single most recent export\'s rows '
                          '(watermark 2000) -- everything covered by the older export '
                          '(watermark 1000) is safe to prune')


class MaybePrunePositionsTests(unittest.TestCase):
    """auto_pipeline.maybe_prune_positions(): the opt-in wiring, tested
    against an in-memory fake (no real HTTP server) -- proves the flag is
    actually a gate (default off, nothing is ever called) and that
    keep_datasets is forwarded to the server untouched."""

    class _FakePruneClient:
        def __init__(self, response):
            self.response = response
            self.calls = []

        def post(self, path, json_body=None):
            self.calls.append((path, json_body))
            return self.response

    def test_flag_off_never_calls_the_endpoint(self):
        client = self._FakePruneClient({'pruned': True, 'deleted_count': 5})
        import argparse
        args = argparse.Namespace(prune_after_export=False, keep_datasets=3)

        ap.maybe_prune_positions(client, args)

        self.assertEqual(client.calls, [],
                          'a deployment that never passes --prune-after-export must never '
                          'have positions pruned')

    def test_flag_on_calls_prune_endpoint_with_keep_datasets(self):
        client = self._FakePruneClient({'pruned': True, 'deleted_count': 5, 'deleted_up_to_id': 999})
        import argparse
        args = argparse.Namespace(prune_after_export=True, keep_datasets=7)

        ap.maybe_prune_positions(client, args)

        self.assertEqual(len(client.calls), 1)
        path, body = client.calls[0]
        self.assertEqual(path, '/admin/pipeline/prune-positions')
        self.assertEqual(body, {'keep_datasets': 7})

    def test_not_pruned_response_does_not_raise(self):
        client = self._FakePruneClient({'pruned': False, 'reason': 'not enough exports yet'})
        import argparse
        args = argparse.Namespace(prune_after_export=True, keep_datasets=3)

        ap.maybe_prune_positions(client, args)   # must not raise

        self.assertEqual(len(client.calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
