#!/usr/bin/env python3
"""platform_config.py - Worker CLI args + persisted registration state for
the public community-compute worker client.

On first run, a worker registers with the server -- either with a per-user
--api-key (recommended: get one from the server's /accounts endpoints, see
platform/docs/WORKER.md) or, if the operator enabled it, a shared
--registration-secret. Either way the server hands back a worker_id +
worker_token, saved to --state-file so subsequent runs (including after a
crash/reboot) reuse the same identity without re-registering or needing the
credential again.

Named platform_config.py (not config.py) for the same reason as
platform_client.py: distributed/worker/ has its own config.py, and this
project has already hit the same-basename sys.path shadowing bug three
times under platform/server/ (db.py/models.py/config.py). This directory is
also meant to be run completely standalone, so it does not put
distributed/worker/ on sys.path at all.
"""
import argparse
import json
import os
import platform


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Morph Community Compute worker client')
    ap.add_argument('--server', required=True,
                     help='platform server base URL, e.g. https://compute.example.org')
    ap.add_argument('--engine-bin', required=True, help='path to the compiled chess UCI binary')
    ap.add_argument('--api-key', default=None,
                     help='per-account API key from POST /accounts/api-key/regenerate '
                          '(recommended -- see platform/docs/WORKER.md)')
    ap.add_argument('--registration-secret', default=None,
                     help='legacy shared-secret registration, only if the server operator '
                          'enabled CHESS_PLATFORM_REGISTRATION_SECRET; prefer --api-key')
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

    # Resource limits (resource_limits.py) -- opt-in caps so a volunteer's
    # machine stays usable for its owner while contributing spare capacity.
    ap.add_argument('--max-cpu-percent', type=float, default=None,
                     help='soft cap on this process tree\'s CPU usage, 0-100 * cpu_count; '
                          'if exceeded the worker sleeps briefly between games to back off '
                          '(default: no cap)')
    ap.add_argument('--max-memory-mb', type=float, default=None,
                     help='hard cap on this process tree\'s RSS; if exceeded the worker '
                          'finishes its current batch, uploads, and exits cleanly rather than '
                          'risking an OOM kill (default: no cap)')
    ap.add_argument('--resource-check-interval', type=float, default=10.0,
                     help='seconds between resource checks')

    # Capability reporting (capabilities.py) -- detected automatically
    # (CPU cores, RAM, GPU presence) and reported to the server so
    # capability-aware task assignment can route work appropriately. See
    # capabilities.py's module docstring for why trainer_capable is an
    # explicit opt-in rather than automatic GPU detection.
    ap.add_argument('--trainer-capable', action='store_true',
                     help='opt this worker in to TRAIN_NETWORK tasks (runs the real NNUE '
                          'training pipeline -- see platform/trainer/ and docs/TRAINING.md; '
                          'more resource-intensive and long-running than self-play/data-gen '
                          'tasks, works via the CPU reference trainer even without a GPU)')
    ap.add_argument('--gpu-name-override', default=None,
                     help='report this GPU name instead of what nvidia-smi/torch.cuda detect '
                          '(cosmetic only -- for operator visibility, does not affect task '
                          'eligibility)')
    ap.add_argument('--artifacts-cache-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'artifacts_cache'),
        help='local cache directory for downloaded, hash-verified artifacts (datasets, '
             'checkpoints, NNUE networks -- see artifacts.py)')
    ap.add_argument('--train-bin', default=None,
                     help='path to the chess_train binary, for DATA_GENERATION tasks '
                          '(default: look for chess_train/chess_train.exe next to '
                          '--engine-bin, then on PATH)')

    # Auto-update (updater.py)
    ap.add_argument('--auto-update', action='store_true',
                     help='on startup and periodically, check the server\'s advertised worker '
                          'version and self-update from --update-url if newer (default: check '
                          'and warn only, never modify files unless this flag is given)')
    ap.add_argument('--update-url', default=None,
                     help='base URL to fetch worker-<version>.tar.gz from, e.g. a '
                          'GitHub Releases download URL (see .github/workflows/release.yml); '
                          'required for --auto-update to actually download anything')
    ap.add_argument('--update-check-interval', type=float, default=3600.0,
                     help='seconds between update checks while the worker is running')

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
