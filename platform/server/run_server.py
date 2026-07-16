#!/usr/bin/env python3
"""run_server.py - Entrypoint for the platform server.

    python3 run_server.py --host 0.0.0.0 --port 8000

See platform/docs/SERVER.md for the full setup walkthrough, and
platform/docker/ for a containerized deployment.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--workers', type=int, default=1,
                     help='uvicorn worker processes -- keep at 1 unless the rate limiter is '
                          'moved to a shared store first, see ratelimit.py')
    args = ap.parse_args()

    import uvicorn
    uvicorn.run('app:app', host=args.host, port=args.port, workers=args.workers)


if __name__ == '__main__':
    main()
