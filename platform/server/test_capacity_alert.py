#!/usr/bin/env python3
"""test_capacity_alert.py - Regression tests for the ntfy.sh push-notification
capacity alerting feature (auto_pipeline.py's evaluate_capacity_alert(),
CapacityAlertState, and maybe_alert_on_capacity()).

This exists because polling the server from an *external* monitor turned
out to be unreliable in practice (some monitoring sandboxes can't reach an
arbitrary server IP/port at all -- a network restriction on the monitor's
side, not the server's). The fix was to push alerts OUT from the
already-running, same-box auto_pipeline.py loop instead of polling IN.

Three layers, matching the feature's own split:

  CapacityAlertEvaluationTests: evaluate_capacity_alert() against plain
  dict snapshots (the same shape GET /admin/system-load returns) -- pure
  function, no server or network involved.

  CapacityAlertStateTests: CapacityAlertState.observe() across a sequence
  of cycles -- proves immediate-notify-on-new-problem, periodic reminders
  while a problem persists, and a one-time resolved notice, without
  needing real time.sleep() or a real loop.

  MaybeAlertOnCapacityTests: auto_pipeline.maybe_alert_on_capacity() against
  an in-memory FakeAdminClient (same style as test_promotion.py /
  test_prune.py) with send_ntfy_notification monkeypatched out (never makes
  a real HTTP call to ntfy.sh in tests) -- proves the opt-in flag is
  honored and that a send actually happens end-to-end when it should.

Run directly:  python3 test_capacity_alert.py
Run via pytest: pytest test_capacity_alert.py
"""
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auto_pipeline as ap


def _snapshot(**overrides):
    """A fully-healthy GET /admin/system-load response by default; tests
    override just the field(s) they're pushing over a threshold."""
    base = {
        'connected_workers': 5,
        'max_connected_workers': 40,
        'at_worker_capacity': False,
        'pending_tasks': 3,
        'cpu_count': 2,
        'load_average': {'1min': 0.5, '5min': 0.4, '15min': 0.3},
        'memory': {'total_mb': 2000.0, 'available_mb': 1000.0, 'used_percent': 50.0},
        'disk': {'total_gb': 50.0, 'free_gb': 40.0, 'used_percent': 20.0},
    }
    base.update(overrides)
    return base


class CapacityAlertEvaluationTests(unittest.TestCase):
    def test_healthy_snapshot_returns_none(self):
        self.assertIsNone(ap.evaluate_capacity_alert(_snapshot()))

    def test_at_worker_capacity_flagged(self):
        msg = ap.evaluate_capacity_alert(_snapshot(
            at_worker_capacity=True, connected_workers=40))
        self.assertIsNotNone(msg)
        self.assertIn('worker capacity', msg)
        self.assertIn('40/40', msg)

    def test_pending_tasks_below_threshold_not_flagged(self):
        self.assertIsNone(ap.evaluate_capacity_alert(_snapshot(pending_tasks=99)))

    def test_pending_tasks_at_threshold_flagged(self):
        msg = ap.evaluate_capacity_alert(_snapshot(pending_tasks=100))
        self.assertIsNotNone(msg)
        self.assertIn('backlog', msg)
        self.assertIn('100 pending', msg)

    def test_memory_below_threshold_not_flagged(self):
        self.assertIsNone(ap.evaluate_capacity_alert(
            _snapshot(memory={'total_mb': 2000.0, 'available_mb': 400.0, 'used_percent': 84.9})))

    def test_memory_at_threshold_flagged(self):
        msg = ap.evaluate_capacity_alert(
            _snapshot(memory={'total_mb': 2000.0, 'available_mb': 300.0, 'used_percent': 85.0}))
        self.assertIsNotNone(msg)
        self.assertIn('memory at 85.0%', msg)

    def test_disk_at_threshold_flagged(self):
        msg = ap.evaluate_capacity_alert(
            _snapshot(disk={'total_gb': 50.0, 'free_gb': 7.0, 'used_percent': 86.0}))
        self.assertIsNotNone(msg)
        self.assertIn('disk at 86.0%', msg)

    def test_load_average_below_threshold_not_flagged(self):
        # cpu_count=2 -> threshold is 2.4
        self.assertIsNone(ap.evaluate_capacity_alert(
            _snapshot(cpu_count=2, load_average={'1min': 2.3, '5min': 2.0, '15min': 1.5})))

    def test_load_average_at_threshold_flagged(self):
        msg = ap.evaluate_capacity_alert(
            _snapshot(cpu_count=2, load_average={'1min': 2.4, '5min': 2.0, '15min': 1.5}))
        self.assertIsNotNone(msg)
        self.assertIn('load average 2.4', msg)

    def test_missing_optional_fields_dont_crash(self):
        # A snapshot with memory/disk/load_average as None (matches
        # system_load.py's own degrade-to-None behavior on non-Linux or
        # inaccessible paths) must never raise.
        msg = ap.evaluate_capacity_alert(_snapshot(memory=None, disk=None,
                                                     load_average=None, cpu_count=None))
        self.assertIsNone(msg)

    def test_multiple_problems_all_joined_in_one_message(self):
        msg = ap.evaluate_capacity_alert(_snapshot(
            at_worker_capacity=True, connected_workers=40, pending_tasks=150))
        self.assertIsNotNone(msg)
        self.assertIn('worker capacity', msg)
        self.assertIn('backlog', msg)


class CapacityAlertStateTests(unittest.TestCase):
    def test_healthy_never_notifies(self):
        state = ap.CapacityAlertState(reminder_cycles=3)
        for _ in range(10):
            self.assertIsNone(state.observe(None))

    def test_new_problem_notifies_immediately(self):
        state = ap.CapacityAlertState(reminder_cycles=3)
        result = state.observe('disk at 90.0%')
        self.assertEqual(result, 'disk at 90.0%')

    def test_persisting_problem_suppressed_until_reminder_cycle(self):
        state = ap.CapacityAlertState(reminder_cycles=3)
        self.assertIsNotNone(state.observe('disk at 90.0%'))  # cycle 0: immediate
        self.assertIsNone(state.observe('disk at 90.0%'))     # cycle 1: suppressed
        self.assertIsNone(state.observe('disk at 90.0%'))     # cycle 2: suppressed
        self.assertIsNotNone(state.observe('disk at 90.0%'))  # cycle 3: reminder due

    def test_resolved_sends_one_time_notice(self):
        state = ap.CapacityAlertState(reminder_cycles=3)
        state.observe('disk at 90.0%')
        result = state.observe(None)
        self.assertIsNotNone(result)
        self.assertIn('back to normal', result)
        # And it doesn't repeat on subsequent healthy cycles.
        self.assertIsNone(state.observe(None))

    def test_new_problem_after_resolution_notifies_immediately_again(self):
        state = ap.CapacityAlertState(reminder_cycles=3)
        state.observe('disk at 90.0%')
        state.observe(None)  # resolved notice
        result = state.observe('memory at 90.0%')
        self.assertIsNotNone(result)
        self.assertIn('memory', result)

    def test_changing_problem_while_still_alerting_does_not_reset_reminder_clock(self):
        # A different problem message while already alerting is still just
        # "still alerting" -- shouldn't re-notify until the reminder cycle,
        # same as the original problem persisting unchanged.
        state = ap.CapacityAlertState(reminder_cycles=3)
        state.observe('disk at 90.0%')          # cycle 0: immediate
        result = state.observe('memory at 90.0%')  # cycle 1: still within reminder window
        self.assertIsNone(result)


class FakeAdminClient:
    """Stands in for auto_pipeline.AdminClient: same .get() surface
    maybe_alert_on_capacity() calls, backed by plain Python state."""

    def __init__(self, system_load=None, raise_error=False):
        self.system_load = system_load
        self.raise_error = raise_error
        self.get_calls = []

    def get(self, path, **params):
        self.get_calls.append(path)
        if self.raise_error:
            raise ap.ApiError('GET /admin/system-load: HTTP 500: boom')
        if path == '/admin/system-load':
            return self.system_load
        raise AssertionError(f'FakeAdminClient: unexpected GET {path}')


class MaybeAlertOnCapacityTests(unittest.TestCase):
    def setUp(self):
        self._sent = []
        self._real_send = ap.send_ntfy_notification

        def fake_send(topic, message, priority='default', tags=None):
            self._sent.append((topic, message, priority, tags))

        ap.send_ntfy_notification = fake_send

    def tearDown(self):
        ap.send_ntfy_notification = self._real_send

    def test_disabled_when_no_topic_configured(self):
        client = FakeAdminClient(system_load=_snapshot(at_worker_capacity=True,
                                                          connected_workers=40))
        args = argparse.Namespace(ntfy_topic='', ntfy_reminder_cycles=12)
        state = ap.CapacityAlertState(args.ntfy_reminder_cycles)
        ap.maybe_alert_on_capacity(client, args, state)
        self.assertEqual(client.get_calls, [])  # never even checks when disabled
        self.assertEqual(self._sent, [])

    def test_healthy_snapshot_sends_nothing(self):
        client = FakeAdminClient(system_load=_snapshot())
        args = argparse.Namespace(ntfy_topic='test-topic', ntfy_reminder_cycles=12)
        state = ap.CapacityAlertState(args.ntfy_reminder_cycles)
        ap.maybe_alert_on_capacity(client, args, state)
        self.assertEqual(self._sent, [])

    def test_problem_snapshot_sends_via_ntfy(self):
        client = FakeAdminClient(system_load=_snapshot(at_worker_capacity=True,
                                                          connected_workers=40))
        args = argparse.Namespace(ntfy_topic='test-topic', ntfy_reminder_cycles=12)
        state = ap.CapacityAlertState(args.ntfy_reminder_cycles)
        ap.maybe_alert_on_capacity(client, args, state)
        self.assertEqual(len(self._sent), 1)
        topic, message, priority, tags = self._sent[0]
        self.assertEqual(topic, 'test-topic')
        self.assertIn('worker capacity', message)
        self.assertEqual(tags, 'warning')

    def test_unreachable_server_logged_not_raised(self):
        client = FakeAdminClient(raise_error=True)
        args = argparse.Namespace(ntfy_topic='test-topic', ntfy_reminder_cycles=12)
        state = ap.CapacityAlertState(args.ntfy_reminder_cycles)
        try:
            ap.maybe_alert_on_capacity(client, args, state)  # must not raise
        except ap.ApiError:
            self.fail('maybe_alert_on_capacity() must catch ApiError, not propagate it')
        self.assertEqual(self._sent, [])


if __name__ == '__main__':
    unittest.main()
