#!/usr/bin/env python3
"""state.py - Crash-safe persistence for the controller's loop state.

Written after every stage of every cycle (not just at the end), so a
controller that's killed mid-cycle (power loss, Ctrl-C, OOM) can be
restarted and: (a) know which cycle/stage it was on, (b) not silently lose
the fact that a training run happened, (c) know how many new positions have
already been "spent" on training (the dataset watermark), so it doesn't
either retrain on the exact same data immediately or lose track of new data
that arrived during a crash.

This does NOT try to resume a half-finished training subprocess itself --
training_server/training/train.py's own --resume-checkpoint mechanism is the
resume primitive for that. This file just remembers enough for the
controller to make sane decisions on its next iteration.

The state file is small, plain JSON, and safe to inspect/edit by hand.
Writes are atomic (write to a .tmp file, then os.replace) so a crash mid-write
never leaves a half-written, corrupt state.json behind.
"""
import json
import os
import time

import config

_DEFAULT = {
    'cycle_count': 0,
    'status': 'idle',            # idle | collecting | training | evaluating | promoting | failed
    'last_cycle_started_at': None,
    'last_cycle_finished_at': None,
    'last_experiment_id': None,
    'last_verdict': None,
    'last_error': None,
    'consecutive_failures': 0,
    'total_promoted': 0,
    'total_rejected': 0,
    'dataset_watermark': 0,       # position count already used for training
}


def load():
    if not os.path.isfile(config.STATE_FILE):
        return dict(_DEFAULT)
    try:
        with open(config.STATE_FILE) as f:
            state = json.load(f)
        merged = dict(_DEFAULT)
        merged.update(state)
        return merged
    except (json.JSONDecodeError, OSError):
        # A corrupt state file should never crash the loop -- start fresh
        # but keep the corrupt file around (renamed) for post-mortem.
        try:
            os.replace(config.STATE_FILE, config.STATE_FILE + '.corrupt')
        except OSError:
            pass
        return dict(_DEFAULT)


def save(state):
    os.makedirs(os.path.dirname(config.STATE_FILE), exist_ok=True)
    tmp_path = config.STATE_FILE + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, config.STATE_FILE)  # atomic on POSIX and Windows


def update(**kwargs):
    state = load()
    state.update(kwargs)
    save(state)
    return state


def touch_cycle_start():
    return update(status='collecting',
                  last_cycle_started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
