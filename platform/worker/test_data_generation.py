#!/usr/bin/env python3
"""test_data_generation.py - Regression test for run_data_generation()'s
chunked upload (see its docstring for the full story): a DATA_GENERATION
batch used to upload every generated record (often 20,000+ from a single
`chess_train gen --games 200` run) in ONE /tasks/{id}/results POST.
PlatformClient's HTTP timeout is a fixed 30s, and a submission that large
can take the server longer than that just to run every content-hash
INSERT -- observed live: the worker saw a ReadTimeout, burned through
several minutes of retries, and the retry that finally got a response back
in time reported every record as a 'duplicate' (because the server had, in
fact, already committed the earlier timed-out attempt's insert in its
background thread). If that retry sequence itself ran out of attempts, the
whole batch's positions were genuinely lost, not just delayed.

The fix chunks the upload into --upload-batch-size-sized pieces (same knob
platform_worker.py's SELF_PLAY TaskRunner already uses for exactly this
reason) and, mirroring TaskRunner._flush, confines a ServerUnavailable
failure to the chunk it happened on rather than the whole batch.

Uses a FakeClient (no real HTTP) and monkeypatches subprocess.run +
_find_train_binary so this runs fast and deterministically -- these tests
are about run_data_generation()'s own chunking/aggregation control flow,
not about chess_train's real output or the real network.
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_generation as dg
from platform_client import ServerUnavailable


class FakeClient:
    """Records every submit_results() call (task_id, positions, done) so
    tests can assert on chunk boundaries and the done flag, without any
    real HTTP. raise_on_chunk is a set of zero-based call indices that
    should raise ServerUnavailable instead of returning a response --
    simulates a chunk that exhausted all of PlatformClient's own retries."""

    def __init__(self, responses=None, raise_on_chunk=None):
        self.calls = []
        self._responses = responses
        self._raise_on_chunk = raise_on_chunk or set()

    def submit_results(self, task_id, positions, done=False):
        idx = len(self.calls)
        self.calls.append({'task_id': task_id, 'positions': list(positions), 'done': done})
        if idx in self._raise_on_chunk:
            raise ServerUnavailable(f'simulated failure on chunk {idx}')
        if self._responses is not None:
            return self._responses[idx]
        return {'accepted': len(positions), 'duplicates': 0, 'rejected': 0}


def _args(upload_batch_size=100):
    return types.SimpleNamespace(upload_batch_size=upload_batch_size, train_bin=None)


def _gen_output_lines(n):
    # Minimal well-formed bullet-ext lines -- see _parse_line(): needs a
    # legal-looking FEN (side-to-move field is all _parse_line checks),
    # an int eval, and a float result. Content doesn't need to be a real
    # game; this test never touches move generation.
    lines = [f'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | {i} | 0.5 | 10 | 0'
              for i in range(n)]
    return '\n'.join(lines) + '\n'


def _run(n_positions, upload_batch_size=100, responses=None, raise_on_chunk=None):
    client = FakeClient(responses=responses, raise_on_chunk=raise_on_chunk)
    args = _args(upload_batch_size=upload_batch_size)
    task = {'task_id': 't_test', 'payload': {'games': 1, 'depth': 4, 'randomplies': 4}}
    gen_output = _gen_output_lines(n_positions)

    def fake_run(cmd, **kwargs):
        out_path = cmd[cmd.index('--out') + 1]
        with open(out_path, 'w') as f:
            f.write(gen_output)
        return types.SimpleNamespace(returncode=0, stdout='generated\n', stderr='')

    with mock.patch.object(dg, '_find_train_binary', return_value='chess_train'), \
         mock.patch.object(dg.subprocess, 'run', side_effect=fake_run):
        resp = dg.run_data_generation(task, client, 'chess.exe', 'test-engine', args,
                                       log=lambda *a, **k: None)
    return client, resp


class DataGenerationChunkedUploadTests(unittest.TestCase):
    def test_uploads_in_chunks_not_one_giant_batch(self):
        client, _ = _run(n_positions=250, upload_batch_size=100)
        sizes = [len(c['positions']) for c in client.calls]
        self.assertEqual(sizes, [100, 100, 50])

    def test_only_the_last_chunk_is_marked_done(self):
        client, _ = _run(n_positions=250, upload_batch_size=100)
        self.assertEqual([c['done'] for c in client.calls], [False, False, True])

    def test_small_batch_is_a_single_request_marked_done(self):
        client, _ = _run(n_positions=30, upload_batch_size=100)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]['positions']), 30)
        self.assertTrue(client.calls[0]['done'])

    def test_exact_multiple_of_batch_size_does_not_add_an_empty_chunk(self):
        client, _ = _run(n_positions=200, upload_batch_size=100)
        sizes = [len(c['positions']) for c in client.calls]
        self.assertEqual(sizes, [100, 100])
        self.assertEqual([c['done'] for c in client.calls], [False, True])

    def test_totals_aggregate_across_chunks(self):
        responses = [
            {'accepted': 90, 'duplicates': 10, 'rejected': 0},
            {'accepted': 80, 'duplicates': 15, 'rejected': 5},
            {'accepted': 50, 'duplicates': 0, 'rejected': 0},
        ]
        client, resp = _run(n_positions=250, upload_batch_size=100, responses=responses)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(resp['accepted'], 220)
        self.assertEqual(resp['duplicates'], 25)
        self.assertEqual(resp['rejected'], 5)

    def test_one_chunk_failing_does_not_lose_the_whole_batch(self):
        # Chunk index 1 (the second of three) exhausts all of
        # PlatformClient's retries -- run_data_generation must keep going
        # and still upload chunk 2, not abort the whole task.
        client, resp = _run(n_positions=250, upload_batch_size=100, raise_on_chunk={1})
        self.assertEqual(len(client.calls), 3)  # all three chunks were attempted
        # Only chunks 0 and 2 actually contributed (100 + 50), chunk 1's
        # 100 positions were lost -- but the task didn't blow up, and what
        # DID land is reflected in the totals.
        self.assertEqual(resp['accepted'], 150)

    def test_every_chunk_failing_returns_none_without_raising(self):
        client, resp = _run(n_positions=150, upload_batch_size=100, raise_on_chunk={0, 1})
        self.assertEqual(len(client.calls), 2)
        self.assertIsNone(resp)

    def test_batch_size_defaults_to_100_if_args_lack_the_attribute(self):
        # A caller (or an older/simplified args object) that doesn't set
        # upload_batch_size at all must still get sane chunking, not a
        # crash or an unbounded single giant request.
        client = FakeClient()
        args = types.SimpleNamespace(train_bin=None)  # no upload_batch_size attribute
        task = {'task_id': 't_test', 'payload': {'games': 1, 'depth': 4, 'randomplies': 4}}
        gen_output = _gen_output_lines(150)

        def fake_run(cmd, **kwargs):
            out_path = cmd[cmd.index('--out') + 1]
            with open(out_path, 'w') as f:
                f.write(gen_output)
            return types.SimpleNamespace(returncode=0, stdout='generated\n', stderr='')

        with mock.patch.object(dg, '_find_train_binary', return_value='chess_train'), \
             mock.patch.object(dg.subprocess, 'run', side_effect=fake_run):
            dg.run_data_generation(task, client, 'chess.exe', 'test-engine', args,
                                    log=lambda *a, **k: None)
        sizes = [len(c['positions']) for c in client.calls]
        self.assertEqual(sizes, [100, 50])


if __name__ == '__main__':
    unittest.main()
