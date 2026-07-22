#!/usr/bin/env python3
"""test_dataset_accumulation.py - Regression tests for the dataset-
accumulation fix (see auto_pipeline.py's --min-new-positions history
comment, and NNUE_TRAINING_PIPELINE_AUDIT.md).

What broke: --min-new-positions defaulted to 2000, four orders of magnitude
below --max-dataset-positions (200,000). /admin/pipeline/export-dataset
(app.py) exports and lets a TRAIN_NETWORK cycle fire the MOMENT that many
new positions are available past the last watermark -- it does not wait to
accumulate further toward max_positions. With min_new_positions=2000, the
auto-loop kept firing training cycles on whatever small slice (confirmed
live: 48,907 positions, not the intended ~200,000) had trickled in since
the last cycle. Root cause of a ~5.2M-parameter NNUE trained on roughly
1,150 total gradient updates, which then lost 100/100 games against the
classical evaluator.

Three layers:

  ArgumentDefaultsTests: proves the CLI defaults actually changed. If
  --min-new-positions is ever reverted back toward a tiny number relative
  to --max-dataset-positions, this fails loudly instead of silently
  reintroducing the same incident.

  RequestSchemaDefaultsTests: same idea for ExportDatasetRequest's own
  Pydantic defaults (schemas.py) -- these apply if the endpoint is ever
  called without explicit values, so they need to be fixed too, independent
  of auto_pipeline.py always passing explicit ones.

  ExportThresholdLogicTests: a real, isolated PlatformDatabase (same
  pattern as test_prune.py's _make_db()/_seed_positions() -- deliberately
  NOT the shared app_module.db singleton: importing the live `app` module
  initializes its Settings() from os.environ at that exact moment, and
  whichever test file in this directory happens to import it FIRST wins
  that snapshot for the rest of the pytest process -- confirmed empirically
  while writing this file: pairing an app-importing version of this test
  with test_system_load.py caused ITS CHESS_PLATFORM_MAX_CONNECTED_WORKERS
  env var to lose the race and its capacity tests to fail, even though
  neither file's logic was wrong in isolation. Testing db.export_positions_range()
  directly (the actual query app.py's export_dataset() runs) plus the same
  one-line threshold comparison it makes gets equivalent real coverage of
  the accumulation-gate logic without that shared-singleton hazard.

Run directly:  python3 test_dataset_accumulation.py
Run via pytest: pytest test_dataset_accumulation.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SERVER_DIR)
sys.path.insert(0, os.path.join(_SERVER_DIR, '..', '..', 'tools', 'nnue_pipeline'))

import auto_pipeline as ap  # noqa: E402
import schemas  # noqa: E402 -- lightweight (pydantic only), safe to import: does NOT
                 # construct app.py's Settings()/FastAPI app, so it can't race with
                 # another test file over shared env-var-derived global state.
from database import PlatformDatabase  # noqa: E402


def _make_db():
    fd, path = tempfile.mkstemp(suffix='.sqlite3')
    os.close(fd)
    os.unlink(path)   # PlatformDatabase creates it fresh
    return PlatformDatabase(path), path


def _seed_positions(db, n, engine_version='v1'):
    """Same real submit_positions() path a live worker submission goes
    through (validation + content_hash dedup unchanged), matching
    test_prune.py's helper of the same name."""
    worker_id, _token = db.register_worker(f'test-host-{engine_version}', engine_version, 4)
    task_ids, _label = db.create_tasks_bulk(total_positions=n, chunk_size=max(n, 1),
                                             depth=6, randomplies=6)
    task_id = task_ids[0]
    start_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    records = [{'fen': start_fen, 'side_to_move': 'w', 'eval_cp': i, 'result': 0.5,
                'depth': 6, 'nodes': 1000, 'engine_version': engine_version}
               for i in range(n)]
    result = db.submit_positions(task_id, worker_id, records)
    assert result['accepted'] == n, f'test setup bug: expected {n} accepted, got {result}'


class ArgumentDefaultsTests(unittest.TestCase):
    """Locks down the fixed CLI defaults so a future edit can't silently
    reintroduce the incident (e.g. someone "cleaning up" the CLI and
    dropping --min-new-positions back to a small round number)."""

    def setUp(self):
        # parse_args() reads sys.argv directly and requires --server/--admin-token,
        # so patch argv rather than trying to pass an explicit list.
        with mock.patch.object(sys, 'argv', ['auto_pipeline.py', '--server', 'http://test.invalid',
                                             '--admin-token', 'test-token']):
            self.args = ap.parse_args()

    def test_min_new_positions_matches_max_dataset_positions_by_default(self):
        self.assertEqual(self.args.min_new_positions, self.args.max_dataset_positions,
                          '--min-new-positions must default to the SAME value as '
                          '--max-dataset-positions, so a cycle only fires once a full-sized '
                          'batch is actually available (the old default of 2000 vs. 200,000 '
                          'is exactly what caused the ~48,907-position production incident)')

    def test_min_new_positions_default_is_not_the_old_tiny_value(self):
        self.assertNotEqual(self.args.min_new_positions, 2000,
                             'the old, undersized default must not come back')
        self.assertGreaterEqual(self.args.min_new_positions, 100_000)

    def test_train_epochs_raised_above_six(self):
        self.assertGreater(self.args.train_epochs, 6,
                            '6 epochs was (undeliberately) tuned against the old, much '
                            'smaller ~2000-49000-position exports; a real ~200k-position '
                            'batch needs more epochs to not be under-used')


class RequestSchemaDefaultsTests(unittest.TestCase):
    """ExportDatasetRequest's own Pydantic defaults apply any time the
    endpoint is called without explicit values -- independent of
    auto_pipeline.py always passing them explicitly, so these need to be
    fixed too."""

    def test_min_new_positions_schema_default_matches_max_positions_default(self):
        req = schemas.ExportDatasetRequest()
        self.assertEqual(req.min_new_positions, req.max_positions)
        self.assertNotEqual(req.min_new_positions, 2000)


class ExportThresholdLogicTests(unittest.TestCase):
    """Exercises the real query app.py's export_dataset() runs
    (db.export_positions_range()) plus the same threshold comparison it
    makes (`len(rows) < min_new_positions`), against a real isolated
    PlatformDatabase and the real submit_positions() ingestion path -- the
    actual mechanism behind the production incident, not a mock."""

    def setUp(self):
        self.db, self.path = _make_db()

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def test_below_threshold_is_correctly_identified_as_not_enough(self):
        """Direct regression test for the production incident: exactly the
        scenario that happened live (new positions available, but fewer
        than the intended full-batch size) must be recognized as
        insufficient. 300-of-1000 here stands in for the real
        48,907-of-200,000 incident, scaled down for test speed."""
        _seed_positions(self.db, 300)
        rows, _max_id = self.db.export_positions_range(0, 1000)
        # This is exactly app.py's export_dataset() gate: `if len(rows) < req.min_new_positions`.
        self.assertLess(len(rows), 1000)
        would_create = len(rows) >= 1000
        self.assertFalse(would_create,
                          'must refuse to export/train on a slice smaller than min_new_positions, '
                          'exactly like the real 48,907-vs-200,000 incident')

    def test_at_or_above_threshold_is_correctly_identified_as_enough(self):
        _seed_positions(self.db, 1200)
        rows, _max_id = self.db.export_positions_range(0, 1000)
        self.assertEqual(len(rows), 1000, 'must cap at max_positions even though more were available')
        would_create = len(rows) >= 1000
        self.assertTrue(would_create)

    def test_exact_boundary_counts_as_enough(self):
        """>= , not > -- reaching min_new_positions exactly must count as
        enough (matches app.py's `len(rows) < req.min_new_positions` check,
        which only rejects strictly-fewer)."""
        _seed_positions(self.db, 1000)
        rows, _max_id = self.db.export_positions_range(0, 1000)
        self.assertEqual(len(rows), 1000)
        would_create = len(rows) >= 1000
        self.assertTrue(would_create)

    def test_realistic_scale_default_config_would_have_rejected_the_real_incident(self):
        """Ties the fix directly to the real numbers from the incident:
        with the FIXED default (min_new_positions == max_positions ==
        200,000), an export finding only 48,907 new positions -- the
        actual figure from the live production run -- must be rejected.
        Uses a scaled-down stand-in (489 of a 2,000 threshold, same ratio)
        so the test runs fast; the boundary arithmetic is identical."""
        scale = 100  # 48,907/100 ~= 489, 200,000/100 = 2,000
        _seed_positions(self.db, 48_907 // scale)
        min_new_positions = 200_000 // scale
        rows, _max_id = self.db.export_positions_range(0, 200_000 // scale)
        self.assertLess(len(rows), min_new_positions,
                         'the real incident\'s 48,907-vs-200,000 shortfall, scaled down, must '
                         'still be correctly recognized as insufficient under the fixed default')


if __name__ == '__main__':
    unittest.main()
