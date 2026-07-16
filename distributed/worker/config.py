#!/usr/bin/env python3
"""config.py - Worker CLI args + persisted registration state.

On first run, a worker registers with the server (using a shared
registration secret provided out-of-band by whoever runs the server) and
gets back a worker_id + worker_token. That's saved to --state-file so
subsequent runs (including after a crash/reboot) reuse the same identity
without re-registering or needing the registration secret again.
"""
import argparse
import json
import os
import platform


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Distributed NNUE data-generation worker')
    ap.add_argument('--server', required=True, help='server base URL, e.g. http://192.168.1.10:8000')
    ap.add_argument('--engine-bin', required=True, help='path to the compiled chess UCI binary')
    ap.add_argument('--registration-secret', default=None,
                     help='required on first run only (not needed once --state-file exists)')
    ap.add_argument('--state-file', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'worker_state.json'))
    ap.add_argument('--hostname', default=platform.node() or 'unknown-host')
    ap.add_argument('--threads', type=int, default=1,
                     help='number of self-play games to run concurrently')
    ap.add_argument('--hash-mb', type=int, default=16, help='engine Hash option, per game instance')
    ap.add_argument('--poll-interval', type=float, default=5.0,
                     help='seconds between "no task available" retries')
    ap.add_argument('--upload-batch-size', type=int, default=100,
                     help='upload positions in batches of this size (progress reporting)')
    ap.add_argument('--max-plies', type=int, default=200)
    ap.add_argument('--force-register', action='store_true',
                     help='re-register even if --state-file already has credentials')
    ap.add_argument('--once', action='store_true',
                     help='process at most one task then exit (default: loop forever)')
    return ap.parse_args(argv)


def load_state(state_file):
    if os.path.isfile(state_file):
        with open(state_file) as f:
            return json.load(f)
    return None


def save_state(state_file, state):
    os.makedirs(os.path.dirname(os.path.abspath(state_file)) or '.', exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
