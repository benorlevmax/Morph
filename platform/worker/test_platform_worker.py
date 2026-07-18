#!/usr/bin/env python3
"""test_platform_worker.py - Regression test for main()'s task-dispatch loop
resilience: a failure inside _dispatch_task() must be logged and moved past
(the task's server-side lease simply expires and gets reassigned -- see
WORKER.md's "If a task fails ... the worker logs it and moves on"), never
allowed to crash the whole worker process.

Before this fix, only DataGenerationError/EloMatchError/TrainNetworkError
were caught around _dispatch_task() -- a ServerUnavailable raised deep
inside DATA_GENERATION/ELO_MATCH/TRAIN_NETWORK's own submit_results() call
(unlike SELF_PLAY's TaskRunner._flush, which already caught it locally)
escaped uncaught all the way out of main(), killing the process with an
unhandled-exception traceback. A real contributor hit exactly this after a
slow/overloaded server response caused nine retries to all read-timeout.

Mocks every external dependency (registration, HTTP client, resource
monitor, capability detection, update check) so this exercises the real
main() loop logic with zero real network/subprocess/file I/O.

Run directly:  python3 test_platform_worker.py
Run via pytest: pytest test_platform_worker.py
"""
import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import platform_worker as pw
from platform_client import ServerUnavailable
from data_generation import DataGenerationError


def _fake_args(**overrides):
    ns = types.SimpleNamespace(
        server='http://fake-server', engine_bin='/fake/chess', api_key='cek_fake',
        registration_secret=None, state_file='/fake/worker_state.json',
        hostname='test-host', threads=1, hash_mb=16, poll_interval=0.0,
        upload_batch_size=100, max_plies=200, force_register=False, once=True,
        max_cpu_percent=None, max_memory_mb=None, resource_check_interval=10.0,
        trainer_capable=False, gpu_name_override=None, artifacts_cache_dir='/fake/cache',
        train_bin=None, auto_update=False, update_url=None, update_check_interval=3600.0,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class DispatchResilienceTests(unittest.TestCase):
    def setUp(self):
        # Common patches every test needs: registration, HTTP client,
        # resource monitor, capability reporting, and the update checker
        # (which otherwise fires on the very first loop iteration since
        # last_update_check starts at 0.0).
        self.patchers = [
            patch.object(pw, 'ensure_registered',
                         return_value={'worker_id': 'w1', 'worker_token': 'tok',
                                        'engine_version': 'Morph 0.5'}),
            patch.object(pw, 'PlatformClient'),
            patch.object(pw, 'ResourceMonitor'),
            patch.object(pw, '_report_capabilities', return_value={}),
            patch.object(pw, '_maybe_check_for_update', side_effect=lambda c, a, t: t),
        ]
        self.mocks = {p.attribute: p.start() for p in self.patchers}
        for p in self.patchers:
            self.addCleanup(p.stop)

        self.client = MagicMock()
        self.mocks['PlatformClient'].return_value = self.client

        monitor = MagicMock()
        monitor.should_exit.is_set.return_value = False
        monitor.should_backoff.is_set.return_value = False
        self.mocks['ResourceMonitor'].return_value = monitor

    def test_server_unavailable_during_dispatch_does_not_crash(self):
        task = {'task_id': 't1', 'task_type': 'DATA_GENERATION', 'payload': {}}
        self.client.next_typed_task.return_value = task
        args = _fake_args(once=True)

        with patch.object(pw, '_dispatch_task', side_effect=ServerUnavailable('boom')), \
             patch.object(pw, 'parse_args', return_value=args):
            try:
                pw.main()  # must return normally, not raise
            except ServerUnavailable:
                self.fail('ServerUnavailable escaped main() instead of being caught')

    def test_unexpected_exception_during_dispatch_does_not_crash(self):
        # Anything not in the four named exception types must still be
        # logged and moved past, per WORKER.md's documented contract --
        # this is the broader except Exception catch-all.
        task = {'task_id': 't2', 'task_type': 'SELF_PLAY', 'payload': {}}
        self.client.next_typed_task.return_value = task
        args = _fake_args(once=True)

        with patch.object(pw, '_dispatch_task', side_effect=ValueError('unexpected')), \
             patch.object(pw, 'parse_args', return_value=args):
            try:
                pw.main()
            except ValueError:
                self.fail('unexpected exception escaped main() instead of being caught')

    def test_named_task_errors_still_caught(self):
        # Confirms the fix didn't regress the existing named-exception path.
        task = {'task_id': 't3', 'task_type': 'DATA_GENERATION', 'payload': {}}
        self.client.next_typed_task.return_value = task
        args = _fake_args(once=True)

        with patch.object(pw, '_dispatch_task',
                          side_effect=DataGenerationError('gen failed')), \
             patch.object(pw, 'parse_args', return_value=args):
            try:
                pw.main()
            except DataGenerationError:
                self.fail('DataGenerationError escaped main() (regression)')

    def test_successful_dispatch_still_exits_after_once(self):
        task = {'task_id': 't4', 'task_type': 'SELF_PLAY', 'payload': {}}
        self.client.next_typed_task.return_value = task
        args = _fake_args(once=True)

        with patch.object(pw, '_dispatch_task', return_value=None) as mock_dispatch, \
             patch.object(pw, 'parse_args', return_value=args):
            pw.main()
        mock_dispatch.assert_called_once()

    def test_no_task_available_polls_and_does_not_crash(self):
        # next_typed_task() returning None (nothing queued) must not be
        # confused with a failure -- covers the same loop for completeness.
        self.client.next_typed_task.return_value = None
        args = _fake_args(once=True, poll_interval=0.0)

        call_count = {'n': 0}

        def _next_task():
            call_count['n'] += 1
            if call_count['n'] >= 2:
                # stop the test from looping forever waiting for --once,
                # which only checks after a real dispatch -- simulate a
                # task appearing on the second poll.
                return {'task_id': 't5', 'task_type': 'SELF_PLAY', 'payload': {}}
            return None

        self.client.next_typed_task.side_effect = _next_task

        with patch.object(pw, '_dispatch_task', return_value=None), \
             patch.object(pw, 'parse_args', return_value=args):
            pw.main()
        self.assertGreaterEqual(call_count['n'], 2)


if __name__ == '__main__':
    unittest.main()
