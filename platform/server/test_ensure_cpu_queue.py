#!/usr/bin/env python3
"""test_ensure_cpu_queue.py - Regression test for auto_pipeline.py's
ensure_cpu_queue() (Stage 0: keeps the CPU task queue non-empty), covering
two related fixes to the DATA_GENERATION and SELF_PLAY task payloads:

1. args.randomplies being threaded through instead of a stale hardcoded 6.

2. args.random_move_prob -- randomplies alone (a fixed opening-only random
   prefix) stops preventing duplicate positions once the dataset is large:
   independent games' openings increasingly transpose into an
   already-explored position mid-game, and depth-limited search is
   deterministic from there on, so every move after that shared position is
   then byte-identical between the two games regardless of how different
   their openings were. random_move_prob keeps a small chance of injecting
   a random move at every ply *after* the opening too, not just during it.
   See src/train/selfplay.h's SelfPlayConfig::randomMoveProb for the full
   story (this was observed live: DATA_GENERATION batches still coming back
   100% duplicate at randomplies=12 once the dataset reached ~500K
   positions).

Also covers a real change to *how* SELF_PLAY tasks are created: they used
to go through the legacy /admin/tasks bulk endpoint (distributed/server/
models.py's CreateTasksRequest, a fixed-field Pydantic model that silently
drops any extra key -- passing random_move_prob to it would have been a
no-op). They now go through /admin/tasks/typed instead, with chunking
replicated manually here so a big batch still splits into multiple
independently-assignable tasks the same way the bulk endpoint used to do
server-side.

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
        return {'task_id': 't_fake', 'task_type': (json_body or {}).get('task_type')}


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
        random_move_prob=0.03,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _typed_posts(client, task_type):
    return [body for path, body in client.posted
            if path == '/admin/tasks/typed' and body.get('task_type') == task_type]


class EnsureCpuQueueRandompliesTests(unittest.TestCase):
    def test_data_generation_payload_uses_configured_randomplies(self):
        client = FakeAdminClient(pending_tasks=[])  # nothing pending -> queues more
        args = _args(randomplies=20)
        ap.ensure_cpu_queue(client, args)

        dg_posts = _typed_posts(client, 'DATA_GENERATION')
        self.assertTrue(dg_posts, 'expected at least one DATA_GENERATION task queued')
        for body in dg_posts:
            self.assertEqual(body['payload']['randomplies'], 20)

    def test_selfplay_payload_uses_configured_randomplies(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args(randomplies=20)
        ap.ensure_cpu_queue(client, args)

        sp_posts = _typed_posts(client, 'SELF_PLAY')
        self.assertTrue(sp_posts, 'expected at least one SELF_PLAY task queued')
        for body in sp_posts:
            self.assertEqual(body['payload']['randomplies'], 20)

    def test_default_randomplies_is_12_not_the_old_hardcoded_6(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args()  # uses the _args() default, matching parse_args()'s own default
        ap.ensure_cpu_queue(client, args)

        all_posts = client.posted
        self.assertTrue(all_posts)
        for path, body in all_posts:
            self.assertEqual(body['payload']['randomplies'], 12)
            self.assertNotEqual(body['payload']['randomplies'], 6)

    def test_different_randomplies_values_for_dg_and_sp_both_respected(self):
        # Not a real usage pattern (both come from the same args.randomplies
        # today), but proves neither call site is silently reading a
        # different/stale attribute name.
        client = FakeAdminClient(pending_tasks=[])
        args = _args(randomplies=30)
        ap.ensure_cpu_queue(client, args)

        for path, body in client.posted:
            self.assertEqual(body['payload']['randomplies'], 30)

    def test_queue_already_full_posts_nothing(self):
        # Sanity check the fixture/helper itself behaves as expected --
        # if both queues are already above threshold, nothing should be
        # posted at all.
        client = FakeAdminClient(pending_tasks=[
            {'task_type': 'DATA_GENERATION', 'status': 'pending'},
            {'task_type': 'DATA_GENERATION', 'status': 'pending'},
            {'task_type': 'SELF_PLAY', 'status': 'pending'},
        ])
        args = _args(queue_data_generation_if_below=2, queue_selfplay_if_below=1)
        ap.ensure_cpu_queue(client, args)
        self.assertEqual(client.posted, [])


class EnsureCpuQueueRandomMoveProbTests(unittest.TestCase):
    """Coverage for random_move_prob specifically -- the newer, deeper fix
    on top of randomplies (see module docstring)."""

    def test_data_generation_payload_carries_random_move_prob(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args(random_move_prob=0.07)
        ap.ensure_cpu_queue(client, args)

        dg_posts = _typed_posts(client, 'DATA_GENERATION')
        self.assertTrue(dg_posts)
        for body in dg_posts:
            self.assertEqual(body['payload']['random_move_prob'], 0.07)

    def test_selfplay_payload_carries_random_move_prob(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args(random_move_prob=0.07)
        ap.ensure_cpu_queue(client, args)

        sp_posts = _typed_posts(client, 'SELF_PLAY')
        self.assertTrue(sp_posts)
        for body in sp_posts:
            self.assertEqual(body['payload']['random_move_prob'], 0.07)

    def test_default_random_move_prob_is_nonzero(self):
        # Confirms the CLI default (0.03, set in parse_args()) is what
        # actually reaches the payload when nothing overrides it -- a
        # regression here would silently disable the fix for anyone running
        # with defaults, exactly the kind of thing that already happened
        # once with randomplies being hardcoded to 6.
        client = FakeAdminClient(pending_tasks=[])
        args = _args()
        ap.ensure_cpu_queue(client, args)

        for path, body in client.posted:
            self.assertEqual(body['payload']['random_move_prob'], 0.03)

    def test_zero_random_move_prob_reproduces_old_behavior(self):
        # 0.0 is documented as "disables this, old opening-only-randomness
        # behavior" -- prove it actually threads through as exactly 0.0, not
        # silently coerced to the default or dropped.
        client = FakeAdminClient(pending_tasks=[])
        args = _args(random_move_prob=0.0)
        ap.ensure_cpu_queue(client, args)

        for path, body in client.posted:
            self.assertEqual(body['payload']['random_move_prob'], 0.0)


class EnsureCpuQueueSelfPlayTypedTaskTests(unittest.TestCase):
    """SELF_PLAY switched from the legacy /admin/tasks bulk endpoint to
    /admin/tasks/typed (see module docstring for why) -- these tests cover
    that the switch preserves the important behaviors of the old path:
    chunking a big batch into multiple tasks, and using the
    'target_positions' key name the worker actually reads
    (self.payload['target_positions'] in platform_worker.py's TaskRunner)."""

    def test_selfplay_uses_typed_endpoint_not_legacy_bulk(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args()
        ap.ensure_cpu_queue(client, args)

        legacy_bulk_posts = [body for path, body in client.posted if path == '/admin/tasks']
        self.assertEqual(legacy_bulk_posts, [],
                          'SELF_PLAY should no longer use the legacy /admin/tasks bulk '
                          'endpoint, which silently drops random_move_prob')

    def test_selfplay_chunks_a_large_batch_into_multiple_tasks(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args(selfplay_batch_positions=5000, selfplay_chunk_size=500)
        ap.ensure_cpu_queue(client, args)

        sp_posts = _typed_posts(client, 'SELF_PLAY')
        self.assertEqual(len(sp_posts), 10, 'expected 5000/500 = 10 chunked tasks')
        total = sum(body['payload']['target_positions'] for body in sp_posts)
        self.assertEqual(total, 5000, 'chunk sizes should sum to the full requested batch')
        for body in sp_posts:
            self.assertLessEqual(body['payload']['target_positions'], 500)

    def test_selfplay_uneven_chunking_covers_the_remainder(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args(selfplay_batch_positions=1200, selfplay_chunk_size=500)
        ap.ensure_cpu_queue(client, args)

        sp_posts = _typed_posts(client, 'SELF_PLAY')
        sizes = [body['payload']['target_positions'] for body in sp_posts]
        self.assertEqual(sizes, [500, 500, 200])

    def test_selfplay_chunks_share_one_batch_label(self):
        client = FakeAdminClient(pending_tasks=[])
        args = _args(selfplay_batch_positions=1500, selfplay_chunk_size=500)
        ap.ensure_cpu_queue(client, args)

        sp_posts_bodies = [body for path, body in client.posted
                            if path == '/admin/tasks/typed' and body.get('task_type') == 'SELF_PLAY']
        labels = {body['batch_label'] for body in sp_posts_bodies}
        self.assertEqual(len(labels), 1, 'all chunks of one queueing pass should share a label')


if __name__ == '__main__':
    unittest.main()
