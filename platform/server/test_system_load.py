#!/usr/bin/env python3
"""test_system_load.py - Regression tests for the worker-capacity safety
valve (POST /register turning away new registrations once the server is
at settings.max_connected_workers) and the GET /admin/system-load
monitoring endpoint that reports the same numbers plus load/memory/disk.

Two layers, matching how the feature is actually split:

  SystemLoadSnapshotTests: system_load.build_system_load_snapshot() as a
  pure function (no db, no HTTP) -- proves the at_worker_capacity
  threshold math and that it doesn't blow up when the real /proc reads
  return None (e.g. running on a non-Linux dev machine).

  CapacityEnforcementTests: a real FastAPI app + PlatformDatabase (via
  TestClient) -- proves POST /register actually returns 503 once
  connected workers hit the cap, that GET /admin/system-load requires
  the admin token, and that the numbers it reports match what /register
  is actually enforcing. The env vars that configure this (admin token,
  registration secret, worker cap, a throwaway db path) are set ONCE at
  module import time, matching how a real deployment actually works
  (Settings() is a module-level singleton created exactly once per
  process, not per-request) -- setUp() clears the workers table directly
  for per-test isolation instead of re-importing app.py fresh each time.
  IMPORTANT: only ever `sys.path.insert(0, <this dir>)` before importing
  app -- inserting distributed/server ahead of it too resolves the
  *other* project's same-named app.py instead (see app.py's own
  docstring on this exact footgun, which bit db.py/models.py earlier in
  the project, and bit this test file's own first draft too).

Run directly:  python3 test_system_load.py
Run via pytest: pytest test_system_load.py
"""
import os
import sys
import tempfile
import unittest

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SERVER_DIR)

import system_load

_TEST_DB_DIR = tempfile.mkdtemp()
os.environ.setdefault('CHESS_PLATFORM_DB_PATH', os.path.join(_TEST_DB_DIR, 'platform.sqlite3'))
os.environ.setdefault('CHESS_PLATFORM_ADMIN_TOKEN', 'test-admin-token')
os.environ.setdefault('CHESS_PLATFORM_MAX_CONNECTED_WORKERS', '2')
os.environ.setdefault('CHESS_PLATFORM_REGISTRATION_SECRET', 'test-secret')

import app as app_module  # noqa: E402 -- must come after the sys.path/env setup above
from fastapi.testclient import TestClient  # noqa: E402


class SystemLoadSnapshotTests(unittest.TestCase):
    def test_below_capacity(self):
        snap = system_load.build_system_load_snapshot(
            connected_workers=5, max_connected_workers=40,
            pending_tasks=3, artifacts_dir=tempfile.gettempdir())
        self.assertEqual(snap['connected_workers'], 5)
        self.assertEqual(snap['max_connected_workers'], 40)
        self.assertFalse(snap['at_worker_capacity'])
        self.assertEqual(snap['pending_tasks'], 3)

    def test_at_capacity_when_equal(self):
        """>= , not > -- reaching the cap exactly counts as full (matches
        app.py's POST /register check, which must reject the request
        that would make it (cap+1)th, i.e. the one arriving when the
        count already equals the cap)."""
        snap = system_load.build_system_load_snapshot(
            connected_workers=40, max_connected_workers=40,
            pending_tasks=0, artifacts_dir=tempfile.gettempdir())
        self.assertTrue(snap['at_worker_capacity'])

    def test_over_capacity(self):
        snap = system_load.build_system_load_snapshot(
            connected_workers=41, max_connected_workers=40,
            pending_tasks=0, artifacts_dir=tempfile.gettempdir())
        self.assertTrue(snap['at_worker_capacity'])

    def test_zero_workers_never_at_capacity_with_positive_cap(self):
        snap = system_load.build_system_load_snapshot(
            connected_workers=0, max_connected_workers=40,
            pending_tasks=0, artifacts_dir=tempfile.gettempdir())
        self.assertFalse(snap['at_worker_capacity'])

    def test_disk_usage_reports_a_real_directory(self):
        result = system_load.get_disk_usage(tempfile.gettempdir())
        self.assertIsNotNone(result)
        self.assertIn('total_gb', result)
        self.assertIn('free_gb', result)
        self.assertIn('used_percent', result)
        self.assertGreater(result['total_gb'], 0)

    def test_disk_usage_returns_none_for_nonexistent_path(self):
        result = system_load.get_disk_usage('/this/path/does/not/exist/at/all')
        self.assertIsNone(result)

    def test_meminfo_parses_on_linux_or_returns_none(self):
        result = system_load.read_proc_meminfo()
        if result is not None:
            self.assertIn('total_mb', result)
            self.assertGreater(result['total_mb'], 0)

    def test_load_average_does_not_raise(self):
        result = system_load.get_load_average()
        if result is not None:
            self.assertIn('1min', result)
            self.assertIn('5min', result)
            self.assertIn('15min', result)


class CapacityEnforcementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    def setUp(self):
        # Per-test isolation without re-importing app.py (see module
        # docstring): both /register's cap check and GET
        # /admin/system-load count rows in the `workers` table, so a
        # clean slate here is all each test actually needs.
        conn = app_module.db._conn()
        try:
            conn.execute('DELETE FROM workers')
            conn.commit()
        finally:
            conn.close()

    def _register(self, hostname):
        return self.client.post('/register', json={
            'hostname': hostname, 'engine_version': '1.0', 'threads': 1,
            'registration_secret': 'test-secret',
        })

    def test_registration_succeeds_below_capacity(self):
        r = self._register('worker-a')
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn('worker_token', r.json())

    def test_registration_rejected_once_at_capacity(self):
        # cap is 2 (see module-level env setup)
        self.assertEqual(self._register('worker-a').status_code, 200)
        self.assertEqual(self._register('worker-b').status_code, 200)
        r = self._register('worker-c')
        self.assertEqual(r.status_code, 503)
        self.assertIn('capacity', r.json()['detail'])

    def test_system_load_requires_admin_token(self):
        r = self.client.get('/admin/system-load')
        self.assertEqual(r.status_code, 401)

    def test_system_load_reflects_connected_worker_count(self):
        self._register('worker-a')
        r = self.client.get('/admin/system-load',
                            headers={'X-Admin-Token': 'test-admin-token'})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body['connected_workers'], 1)
        self.assertEqual(body['max_connected_workers'], 2)
        self.assertFalse(body['at_worker_capacity'])

    def test_system_load_at_capacity_matches_register_rejection(self):
        self._register('worker-a')
        self._register('worker-b')
        r = self.client.get('/admin/system-load',
                            headers={'X-Admin-Token': 'test-admin-token'})
        self.assertTrue(r.json()['at_worker_capacity'])
        self.assertEqual(self._register('worker-c').status_code, 503)


if __name__ == '__main__':
    unittest.main()
