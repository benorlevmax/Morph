#!/usr/bin/env python3
"""run_platform_worker.py - Entrypoint for the community-compute worker.

    python3 run_platform_worker.py --server https://compute.example.org \\
        --engine-bin /path/to/chess_engine --api-key cek_xxxxxxxx --threads 4

See platform/docs/WORKER.md and platform/scripts/ for full setup
instructions, including how to get an API key and set up as a background
service on Windows/Linux.
"""
import sys

try:
    import chess  # noqa: F401
except ImportError:
    sys.exit('missing dependency: python-chess. Run: pip install -r requirements.txt')
try:
    import requests  # noqa: F401
except ImportError:
    sys.exit('missing dependency: requests. Run: pip install -r requirements.txt')
try:
    import psutil  # noqa: F401
except ImportError:
    sys.exit('missing dependency: psutil. Run: pip install -r requirements.txt')

from platform_worker import main

if __name__ == '__main__':
    main()
