#!/usr/bin/env python3
"""run_server.py - Entrypoint: `python3 run_server.py [--host 0.0.0.0] [--port 8000]`

Local/LAN use only -- see docs/DISTRIBUTED_DATA_GENERATION.md's "not for
public deployment yet" note (no TLS, no rate limiting, single-process SQLite).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--reload', action='store_true', help='dev only: auto-reload on code change')
    args = ap.parse_args()

    import uvicorn
    uvicorn.run('app:app', host=args.host, port=args.port, reload=args.reload,
                app_dir=os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    main()
