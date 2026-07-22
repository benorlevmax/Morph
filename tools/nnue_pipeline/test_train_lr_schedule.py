#!/usr/bin/env python3
"""test_train_lr_schedule.py - Regression tests for lr_for_epoch() (Phase 5,
NNUE_TRAINING_PIPELINE_AUDIT.md): linear LR decay added because
--train-epochs was raised from 6 to 20 (see auto_pipeline.py's history
comment) -- a flat LR held for a longer run risks late-training
noise/instability instead of settling.

Run directly:  python3 test_train_lr_schedule.py
Run via pytest: pytest test_train_lr_schedule.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import lr_for_epoch


class LrScheduleTests(unittest.TestCase):
    def test_first_epoch_is_peak_lr(self):
        self.assertAlmostEqual(lr_for_epoch(0, total_epochs=20, lr=0.01, lr_final_fraction=0.1), 0.01)

    def test_last_epoch_is_final_fraction_of_peak(self):
        self.assertAlmostEqual(
            lr_for_epoch(19, total_epochs=20, lr=0.01, lr_final_fraction=0.1), 0.001)

    def test_midpoint_is_between_peak_and_final(self):
        lr = lr_for_epoch(9, total_epochs=19, lr=0.01, lr_final_fraction=0.1)  # progress ~0.5
        self.assertGreater(lr, 0.001)
        self.assertLess(lr, 0.01)

    def test_monotonically_non_increasing_across_a_full_run(self):
        total = 20
        lrs = [lr_for_epoch(e, total, 0.01, 0.1) for e in range(total)]
        for a, b in zip(lrs, lrs[1:]):
            self.assertGreaterEqual(a, b)

    def test_final_fraction_1_reproduces_flat_lr(self):
        """1.0 must disable decay entirely -- the old, pre-this-feature behavior."""
        for e in range(10):
            self.assertAlmostEqual(lr_for_epoch(e, total_epochs=10, lr=0.01, lr_final_fraction=1.0),
                                    0.01)

    def test_single_total_epoch_never_divides_by_zero(self):
        # total_epochs=1 would divide by (total_epochs - 1) == 0 without a guard.
        self.assertAlmostEqual(lr_for_epoch(0, total_epochs=1, lr=0.01, lr_final_fraction=0.1), 0.01)

    def test_epoch_beyond_total_epochs_clamps_to_final_not_negative_or_beyond(self):
        """A resumed run that runs a couple epochs past its originally
        planned --total-epochs must clamp at the final LR, not extrapolate
        past it (which could go negative for some fraction/lr combinations
        in an unclamped linear formula)."""
        lr_at_end = lr_for_epoch(19, total_epochs=20, lr=0.01, lr_final_fraction=0.1)
        lr_past_end = lr_for_epoch(25, total_epochs=20, lr=0.01, lr_final_fraction=0.1)
        self.assertAlmostEqual(lr_at_end, lr_past_end)
        self.assertGreater(lr_past_end, 0)

    def test_resumed_multi_invocation_run_matches_a_single_invocation_run(self):
        """The whole point of --total-epochs existing separately from
        --epochs: a run split across several --resume calls (small --epochs
        each, same --total-epochs every time) must produce the EXACT same
        per-epoch LR sequence as running it in one shot -- the schedule
        must not reset to peak LR on every resume."""
        total = 12
        single_shot = [lr_for_epoch(e, total, 0.01, 0.1) for e in range(total)]
        # Simulate 3 resumed calls of 4 epochs each, same total_epochs each time.
        resumed = []
        for start in (0, 4, 8):
            resumed += [lr_for_epoch(e, total, 0.01, 0.1) for e in range(start, start + 4)]
        self.assertEqual(single_shot, resumed)


if __name__ == '__main__':
    unittest.main()
