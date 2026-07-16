#!/usr/bin/env python3
"""init_db.py - Create (or verify) the SQLite database for the distributed
NNUE data-generation system from schema.sql.

Usage:
    python3 init_db.py [--db path/to/distributed.sqlite3] [--force]

--force drops and recreates all tables (DESTROYS existing data) -- only for
local testing/dev, never used by the server itself at startup (the server
just runs this same schema idempotently via ensure_schema()).
"""
import argparse
import os
import sqlite3
import sys

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.sql')
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'distributed.sqlite3')


def ensure_schema(db_path, force=False):
    """Idempotently create all tables. Safe to call on every server startup."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        if force:
            conn.executescript("""
                DROP TABLE IF EXISTS submissions;
                DROP TABLE IF EXISTS positions;
                DROP TABLE IF EXISTS tasks;
                DROP TABLE IF EXISTS workers;
            """)
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DEFAULT_DB_PATH)
    ap.add_argument('--force', action='store_true', help='drop and recreate all tables')
    args = ap.parse_args()
    ensure_schema(args.db, force=args.force)
    print(f'database ready: {args.db}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
