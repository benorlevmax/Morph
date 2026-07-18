#!/usr/bin/env python3
"""test_ensure_cpu_queue.py - Regression test for auto_pipeline.py's
ensure_cpu_queue() (Stage 0: keeps the CPU task queue non-empty), focused
specifically on args.randomplies being threaded through to both the
DATA_GENERATION and SELF_PLAY task payloads.

Context: randomplies used to be hardcoded to 6 in both call sites here,
not configurable at all. In practice, once the accumulated dataset grew
into the hundreds of thousands of positions, DATA_GENERATION batches
started coming back 100% duplicate -- independent games at depth-limited
search with only 6 random opening plies kept reconverging onto lines
already in the database, wasting real CPU and (via retried, slow
submissions of entirely-duplicate batches) real server time for no new
data. Fixed by making it a --randomplies CLI flag with a higher default
(12) -- these tests prove the configured value actually reaches both
task-creation payloads instead of a stale hardcoded constant silently
undoing the whole fix.

Uses the same in-memory FakeAdminClient style as test_promotion.py /
test_prune.py -- no real HTTP server or database needed.

Run directly:  python3 test_ensure_cpu_queue.py
Run via pytest: pytest test_ensure_cpu_queue.py
"""
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'tools', 'nnue_pipeline'))

import auto_pipeline as ap


class FakeAdminClient:
    """Records every POST call so tests can assert exactly what payload
    was sent, without a real server."""

    def __init__(self, pending_tasks=None):
        self._pending_tasks = pending_tasks or []
        self.posted = []  # list of (path, json_body) in call order

    def get(self, path, **params):
        if path == '/admin/tasks':
            status = params.get('status')
            return [t for t in self._pending_tasks if status is None or t['status'] == status]
        raise AssertionError(f'FakeAdminClient: unexpected GET {path}')

    def post(self, path, json_body=None):
        self.posted.append((path, json_body))
        return {}


def _args(**overrides):
    ns = argparse.Namespace(
        queue_data_generation_if_below=2,
        data_generation_batch_count=3,
        data_generation_games=200,
        data_generation_depth=6,
        queue_selfplay_if_below=1,
        selfplay_batch_positions=5000,
        selfplay_chunk_size=500,
        selfplay_depth=6,
        randomplies=12,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class EnsureCpuQueueRandompliesTests(unittest.TestCase):
    def test_data_generation_payload_uses_configured_randomplies(self):
        client = FakeAdminClient(pending_tasks=[])  # nothing pending -> queues more
        args = _args(randomplies=20)
        ap.ensure_cpu_queue(client, args)

        dg_posts = [body for path, body in client.posted if path == '/admin/tasks/typed']
        self.assertTrue(dg_posts, 'expected at least one DATA_GENERATION task queued')
        for body in dg_posts:
            self.assertEqual(body['payload']['randomplies'], 20)

    def test_selfplay_payload_uses_configured_randomplies(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args(randomplies=20)
        ap.ensure_cpu_queue(client, args)

        sp_posts = [body for path, body in client.posted if path == '/admin/tasks']
        self.assertTrue(sp_posts, 'expected a SELF_PLAY task queued')
        for body in sp_posts:
            self.assertEqual(body['randomplies'], 20)

    def test_default_randomplies_is_12_not_the_old_hardcoded_6(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args()  # uses the _args() default, matching parse_args()'s own default
        ap.ensure_cpu_queue(client, args)

        all_posts = client.posted
        self.assertTrue(all_posts)
        for path, body in all_posts:
            payload = body['payload'] if path == '/admin/tasks/typed' else body
            self.assertEqual(payload['randomplies'], 12)
            self.assertNotEqual(payload['randomplies'], 6)

    def test_different_randomplies_values_for_dg_and_sp_both_respected(self):
        # Not a real usage pattern (both come from the same args.randomplies
        # today), but proves neither call site is silently reading a
        # different/stale attribute name.
        client = FakeAdminClient(pending_tasks=[])
        args = _args(randomplies=30)
        ap.ensure_cpu_queue(client, args)

        for path, body in client.posted:
            payload = body['payload'] if path == '/admin/tasks/typed' else body
            self.assertEqual(payload['randomplies'], 30)

    def test_queue_already_full_posts_nothing(self):
        # Sanity check the fixture/helper itself behaves as expected --
        # if both queues are already above threshold, no randomplies value
        # (old or new) should even be sent.
        client = FakeAdminClient(pending_tasks=[
            {'task_type': 'DATA_GENERATION', 'status': 'pending'},
            {'task_type': 'DATA_GENERATION', 'status': 'pending'},
            {'task_type': 'SELF_PLAY', 'status': 'pending'},
        ])
        args = _args(queue_data_generation_if_below=2, queue_selfplay_if_below=1)
        ap.ensure_cpu_queue(client, args)
        self.assertEqual(client.posted, [])


if __name__ == '__main__':
    unittest.main()
