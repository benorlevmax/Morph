#!/usr/bin/env python3
"""test_selfplay_random_move_prob.py - Regression test for selfplay.py's
random_move_prob (see play_selfplay_game's docstring / src/train/
selfplay.h's SelfPlayConfig::randomMoveProb for the full story): a small
per-ply chance, at every ply AFTER the fixed --randomplies opening prefix,
of playing a uniformly random legal move instead of calling into the
engine. This is the fix for randomplies-alone no longer preventing
duplicate positions once the server's dataset is large (games that
transpose into an already-explored position mid-game then produce
identical, already-collected continuations forever after, since search is
deterministic).

Uses a fake engine stub (no real UCI subprocess) so this runs fast and
deterministically -- these tests are about play_selfplay_game()'s own
control flow (does it call engine.search() or not, does it record a
sample or not), not about the real engine's search quality.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selfplay import play_selfplay_game


class FakeEngine:
    """Always returns a fixed, always-legal-from-the-start-position move
    sequence isn't realistic long-term, but play_selfplay_game() only calls
    .search() when it's NOT injecting a random move, so we just need
    .search() to be callable and countable -- the real move legality is
    handled by python-chess inside play_selfplay_game() itself for the
    random-move branch, and by whatever .search() returns for the
    non-random branch. Returns a fixed legal opening reply (e2e4 the first
    call from the start position, but since games run past the opening
    with random plies already applied, we instead just track call count
    and return a plausible pseudo-legal-enough move by asking the board
    itself for a legal one -- simplest correct approach: this fake ignores
    engine strength entirely and always plays whatever move python-chess's
    own move ordering would call first, so the game remains legal to
    completion."""

    def __init__(self):
        self.search_calls = 0
        self.engine_version = 'fake-1.0'

    def search(self, fen, moves):
        self.search_calls += 1
        import chess as _pychess
        board = _pychess.Board(fen)
        for mv in moves:
            board.push_uci(mv)
        legal = list(board.legal_moves)
        if not legal:
            return None, 0, 1, 0
        # Deterministic "first legal move" -- fine for this test, which only
        # cares about call counts and record counts, not move quality.
        return legal[0].uci(), 0, 1, 0

    def new_game(self):
        pass


class RandomMoveProbTests(unittest.TestCase):
    def test_zero_prob_never_injects_random_moves(self):
        import random
        engine = FakeEngine()
        rng = random.Random(42)
        records = play_selfplay_game(engine, randomplies=0, max_plies=20, rng=rng,
                                      random_move_prob=0.0)
        # Every one of the 20 plies should have gone through engine.search()
        # -- random_move_prob=0.0 must reproduce the exact old behavior.
        self.assertEqual(engine.search_calls, len(records))
        self.assertGreater(engine.search_calls, 0)

    def test_prob_1_never_calls_search_after_opening(self):
        import random
        engine = FakeEngine()
        rng = random.Random(42)
        records = play_selfplay_game(engine, randomplies=0, max_plies=20, rng=rng,
                                      random_move_prob=1.0)
        # Every post-opening ply is forced random -- engine.search() should
        # never be called, and nothing should be recorded (random-move
        # positions are never real training samples).
        self.assertEqual(engine.search_calls, 0)
        self.assertEqual(records, [])

    def test_intermediate_prob_reduces_but_does_not_eliminate_search_calls(self):
        import random
        engine = FakeEngine()
        rng = random.Random(7)
        records = play_selfplay_game(engine, randomplies=0, max_plies=60, rng=rng,
                                      random_move_prob=0.5)
        # With a fair coin per ply over up to 60 plies, expect a genuine mix
        # -- not all 60, not 0. This proves the probability is actually
        # being rolled per-ply, not just checked once.
        self.assertGreater(engine.search_calls, 0)
        self.assertLess(engine.search_calls, 60)
        # Every recorded sample corresponds 1:1 with a real search call
        # (random-move plies are never recorded).
        self.assertEqual(len(records), engine.search_calls)

    def test_random_moves_still_advance_a_legal_game(self):
        # Sanity check: injecting random moves throughout (not just the
        # opening) shouldn't produce an illegal position or crash -- the
        # game should still terminate normally (hits max_plies or a real
        # game-over) with a well-formed result backfilled onto every record.
        import random
        engine = FakeEngine()
        rng = random.Random(99)
        records = play_selfplay_game(engine, randomplies=4, max_plies=40, rng=rng,
                                      random_move_prob=0.2)
        for r in records:
            self.assertIn(r['result'], (0.0, 0.5, 1.0))
            self.assertIn(r['side_to_move'], ('w', 'b'))

    def test_default_random_move_prob_is_zero_backward_compatible(self):
        # play_selfplay_game()'s own default (not auto_pipeline.py's CLI
        # default) must stay 0.0 -- any caller that doesn't pass
        # random_move_prob at all (e.g. an older platform_worker.py, or a
        # direct test/script) must get the exact old behavior.
        import inspect
        sig = inspect.signature(play_selfplay_game)
        self.assertEqual(sig.parameters['random_move_prob'].default, 0.0)


if __name__ == '__main__':
    unittest.main()
