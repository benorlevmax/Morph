#!/usr/bin/env python3
"""run_worker.py - Entrypoint.

    python3 run_worker.py --server http://SERVER:8000 \
        --engine-bin /path/to/chess --registration-secret <secret> --threads 4

See docs/DISTRIBUTED_DATA_GENERATION.md for the full walkthrough.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import chess  # noqa: F401
except ImportError:
    sys.exit("python-chess is required for the worker ('pip install chess'). "
              "It's used as the independent board/legality tracker around the engine's "
              "own moves so every generated position can be reported with its correct FEN.")

import requests  # noqa: F401  -- fail fast with a clear message, not a traceback mid-run

from worker import main

if __name__ == '__main__':
    main()
