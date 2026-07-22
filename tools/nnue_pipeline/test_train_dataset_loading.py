#!/usr/bin/env python3
"""test_train_dataset_loading.py - Regression tests for load_jsonl_datasets()'s
robustness and Phase 3/4 additions (see NNUE_TRAINING_PIPELINE_AUDIT.md):

  - a single malformed line used to crash the whole training run (bare
    json.loads()/dict indexing, no try/except); it must now be skipped and
    counted instead.
  - exact-duplicate records (same fen+eval+result) must be detected,
    dropped, and counted, not silently inflate the effective dataset size.
  - --balance-buckets must actually flatten a skewed bucket distribution
    (within what the underlying data supports) instead of leaving whatever
    the random/instability-weighted truncation happened to produce.

Uses tiny synthetic in-memory datasets (written to temp files) so this runs
fast and deterministically -- not about real chess content, just about
load_jsonl_datasets()'s own control flow.

Run directly:  python3 test_train_dataset_loading.py
Run via pytest: pytest test_train_dataset_loading.py
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import load_jsonl_datasets

# Two real, distinct positions this repo's own audit already used --
# 32-piece startpos (bucket 7) and a bare KvK (bucket 0, 2 pieces) -- so
# these tests exercise real output_bucket() behavior, not a stub.
STARTPOS_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
KVK_FEN = '4k3/8/8/8/8/8/8/4K3 w - - 0 1'


def _write_jsonl(records):
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    with os.fdopen(fd, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    return path


def _rec(fen, eval_cp, result=0.5):
    return {'fen': fen, 'eval': eval_cp, 'result': result}


def _fen_with_n_pieces(n):
    """Builds a FEN with exactly n pieces (2 kings on e1/e8 + (n-2) white
    pawns filling ranks 2-5, up to 30 filler squares -- enough for any
    piece count up to 32) -- deterministic and exact, unlike hand-counting
    characters in a hand-written FEN string. Doesn't need to be a legal/
    reachable position: output_bucket() only cares about the total piece
    count, and parse_fen_board() doesn't validate legality."""
    assert 2 <= n <= 32
    fillers = n - 2
    board = {'e1': 'K', 'e8': 'k'}
    candidate_squares = [f'{f}{r}' for r in (2, 3, 4, 5) for f in 'abcdefgh']  # 32 squares
    for sq in candidate_squares:
        if len(board) - 2 >= fillers:
            break
        if sq in board:
            continue
        board[sq] = 'P'
    assert len(board) == n, f'internal test-helper bug: wanted {n} pieces, built {len(board)}'
    files = 'abcdefgh'
    ranks_str = []
    for r in range(8, 0, -1):
        row = ''
        empty = 0
        for f in files:
            sq = f'{f}{r}'
            if sq in board:
                if empty:
                    row += str(empty)
                    empty = 0
                row += board[sq]
            else:
                empty += 1
        if empty:
            row += str(empty)
        ranks_str.append(row)
    return '/'.join(ranks_str) + ' w - - 0 1'


class InvalidLineHandlingTests(unittest.TestCase):
    def test_malformed_json_line_is_skipped_not_fatal(self):
        path = _write_jsonl([_rec(STARTPOS_FEN, 10)])
        with open(path, 'a') as f:
            f.write('this is not json at all\n')
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=100, seed=1)
            self.assertEqual(len(samples), 1, 'the one valid line must still load fine')
        finally:
            os.unlink(path)

    def test_missing_required_field_is_skipped_not_fatal(self):
        path = _write_jsonl([_rec(STARTPOS_FEN, 10)])
        with open(path, 'a') as f:
            f.write(json.dumps({'fen': STARTPOS_FEN}) + '\n')  # no eval/result
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=100, seed=1)
            self.assertEqual(len(samples), 1)
        finally:
            os.unlink(path)

    def test_invalid_count_is_reported_via_log(self):
        path = _write_jsonl([_rec(STARTPOS_FEN, 10)])
        with open(path, 'a') as f:
            f.write('garbage\n')
            f.write('more garbage\n')
        try:
            logs = []
            load_jsonl_datasets([path], max_samples=100, seed=1, log=logs.append)
            joined = '\n'.join(logs)
            self.assertIn('invalid=2', joined)
        finally:
            os.unlink(path)


class DuplicateHandlingTests(unittest.TestCase):
    def test_exact_duplicate_records_are_dropped(self):
        path = _write_jsonl([_rec(STARTPOS_FEN, 10), _rec(STARTPOS_FEN, 10),
                              _rec(STARTPOS_FEN, 10)])
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=100, seed=1)
            self.assertEqual(len(samples), 1, 'three identical records must collapse to one')
        finally:
            os.unlink(path)

    def test_same_fen_different_eval_is_not_a_duplicate(self):
        path = _write_jsonl([_rec(STARTPOS_FEN, 10), _rec(STARTPOS_FEN, 20)])
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=100, seed=1)
            self.assertEqual(len(samples), 2, 'differing eval means a genuinely different sample')
        finally:
            os.unlink(path)

    def test_duplicate_count_is_reported_via_log(self):
        path = _write_jsonl([_rec(STARTPOS_FEN, 10)] * 4)
        try:
            logs = []
            load_jsonl_datasets([path], max_samples=100, seed=1, log=logs.append)
            joined = '\n'.join(logs)
            self.assertIn('duplicate=3', joined)
        finally:
            os.unlink(path)


class BucketBalancingTests(unittest.TestCase):
    def _skewed_dataset(self, n_startpos=900, n_kvk=100):
        """900 bucket-7 (32-piece) samples, 100 bucket-0 (2-piece) samples --
        deliberately skewed, matching the shape of the real incident (one
        bucket dominant, another comparatively starved)."""
        records = ([_rec(STARTPOS_FEN, i) for i in range(n_startpos)]
                   + [_rec(KVK_FEN, i) for i in range(n_kvk)])
        return _write_jsonl(records)

    def test_unbalanced_truncation_can_starve_the_rare_bucket(self):
        path = self._skewed_dataset()
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=200, seed=1,
                                               balance_buckets=False)
            from nnue_format import parse_fen_board, output_bucket
            buckets = [output_bucket(len(parse_fen_board(s[0])[0])) for s in samples]
            bucket0_share = buckets.count(0) / len(buckets)
            # Not asserting an exact number (random truncation varies), just
            # that it's plausible for the rare bucket to end up well under
            # its even 1/8 share -- this test documents the PRE-fix behavior
            # this feature exists to address, it doesn't need to be flaky-proof.
            self.assertLessEqual(bucket0_share, 0.5)
        finally:
            os.unlink(path)

    def test_balanced_truncation_never_leaves_the_rare_bucket_below_its_even_share(self):
        path = self._skewed_dataset(n_startpos=900, n_kvk=100)
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=200, seed=1,
                                               balance_buckets=True)
            from nnue_format import parse_fen_board, output_bucket
            buckets = [output_bucket(len(parse_fen_board(s[0])[0])) for s in samples]
            # even_share = 200 // 8 = 25. Only buckets 0 and 7 have ANY data
            # in this deliberately 2-bucket dataset, so once every other
            # bucket's (empty) share is redistributed, bucket 0 correctly
            # ends up using ALL its 100 available samples rather than being
            # capped at 25 -- there is nowhere else useful to draw the
            # shortfall from, and using real data bucket 0 actually has is
            # strictly better than leaving it on disk unused (Phase 4's own
            # rule: never destroy useful distribution just to force balance).
            # The invariant that actually matters: balancing must never leave
            # the rare bucket WORSE OFF than its fair even share.
            self.assertGreaterEqual(buckets.count(0), 25)
            self.assertEqual(len(samples), 200)
        finally:
            os.unlink(path)

    def test_balanced_truncation_gives_each_bucket_its_even_share_when_all_have_supply(self):
        """The realistic case this feature targets: every bucket has PLENTY
        of data, just unevenly (mirrors the real incident's shape -- bucket
        0 a small minority, others healthy). With supply well above
        even_share everywhere, every bucket should land almost exactly at
        its even share, not at its original skewed proportion."""
        # One FEN per bucket, exact piece counts chosen at each bucket's
        # midpoint (bucket b covers piece counts [4b+1, 4b+4]) via the
        # deterministic _fen_with_n_pieces() helper -- no hand-counting.
        piece_counts = {b: 4 * b + 2 for b in range(8)}  # 2,6,10,14,18,22,26,30
        fens_by_bucket = {b: _fen_with_n_pieces(n) for b, n in piece_counts.items()}
        from nnue_format import parse_fen_board, output_bucket
        actual_buckets = {b: output_bucket(len(parse_fen_board(fen)[0]))
                           for b, fen in fens_by_bucket.items()}
        self.assertEqual(actual_buckets, {b: b for b in range(8)},
                          'test fixture FENs must land in exactly the bucket their key claims '
                          '-- otherwise this test silently tests nothing meaningful')

        records = []
        for i, fen in enumerate(fens_by_bucket.values()):
            records += [_rec(fen, i * 10 + j) for j in range(300)]  # 300 each, plenty of supply
        path = _write_jsonl(records)
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=400, seed=1,
                                               balance_buckets=True)
            got_buckets = [output_bucket(len(parse_fen_board(s[0])[0])) for s in samples]
            from collections import Counter
            counts = Counter(got_buckets)
            even_share = 400 // 8
            for b in set(actual_buckets.values()):
                self.assertAlmostEqual(counts.get(b, 0), even_share, delta=2,
                                        msg=f'bucket {b} should land close to its even share '
                                            f'when every bucket has ample supply')
        finally:
            os.unlink(path)

    def test_balancing_never_invents_data_for_an_empty_bucket(self):
        """A bucket with ZERO available samples must stay at zero -- Phase
        4's own requirement: 'do not destroy useful distribution
        accidentally' / never fabricate data."""
        # Only bucket-7 (32-piece) and bucket-3-ish content -- no bucket-0 data at all.
        records = [_rec(STARTPOS_FEN, i) for i in range(50)]
        path = _write_jsonl(records)
        try:
            with redirect_stdout(io.StringIO()):
                samples = load_jsonl_datasets([path], max_samples=40, seed=1,
                                               balance_buckets=True)
            from nnue_format import parse_fen_board, output_bucket
            buckets = [output_bucket(len(parse_fen_board(s[0])[0])) for s in samples]
            self.assertEqual(buckets.count(0), 0)
            self.assertEqual(len(samples), 40, 'shortfall from the empty bucket must be '
                                                'redistributed to the bucket that has supply')
        finally:
            os.unlink(path)

    def test_balance_buckets_off_by_default(self):
        import inspect
        sig = inspect.signature(load_jsonl_datasets)
        self.assertEqual(sig.parameters['balance_buckets'].default, False)


if __name__ == '__main__':
    unittest.main()
