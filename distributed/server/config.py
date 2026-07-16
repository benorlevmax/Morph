#!/usr/bin/env python3
"""config.py - Server configuration, sourced from environment variables with
local-testing-friendly defaults. No secrets are hardcoded: if you don't set
CHESS_DIST_REGISTRATION_SECRET / CHESS_DIST_ADMIN_TOKEN, random ones are
generated at startup and printed once so you can copy them into workers/admin
calls for that run.
"""
import os
import secrets

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'database', 'distributed.sqlite3')


class Settings:
    def __init__(self):
        self.db_path = os.environ.get('CHESS_DIST_DB_PATH', DEFAULT_DB_PATH)

        # Shared secret workers must present once, at /register, to be issued
        # their own per-worker bearer token. Rotate by restarting the server
        # with a new CHESS_DIST_REGISTRATION_SECRET; existing workers keep
        # working (their already-issued tokens aren't affected).
        self.registration_secret = os.environ.get(
            'CHESS_DIST_REGISTRATION_SECRET') or secrets.token_hex(16)
        self._registration_secret_was_generated = (
            'CHESS_DIST_REGISTRATION_SECRET' not in os.environ)

        # Separate, higher-privilege token for creating tasks / reading full
        # stats / disabling workers. Never handed to workers.
        self.admin_token = os.environ.get('CHESS_DIST_ADMIN_TOKEN') or secrets.token_hex(16)
        self._admin_token_was_generated = 'CHESS_DIST_ADMIN_TOKEN' not in os.environ

        # How long a worker has to submit results for an assigned task before
        # the server reclaims it and offers it to someone else. Generous by
        # default since self-play at real depth can be slow.
        self.task_lease_seconds = int(os.environ.get('CHESS_DIST_TASK_LEASE_SECONDS', '1800'))

        # Default chunk size when an admin requests a large bulk generation
        # job -- keeps any single task small enough that one worker dropping
        # out doesn't lose much work.
        self.default_chunk_size = int(os.environ.get('CHESS_DIST_DEFAULT_CHUNK_SIZE', '500'))

    def print_startup_banner(self):
        print('=' * 78)
        print('Distributed NNUE data-generation server')
        print(f'  database: {self.db_path}')
        print(f'  task lease: {self.task_lease_seconds}s')
        if self._registration_secret_was_generated:
            print(f'  REGISTRATION SECRET (generated, share with workers): '
                  f'{self.registration_secret}')
        else:
            print('  registration secret: set via CHESS_DIST_REGISTRATION_SECRET')
        if self._admin_token_was_generated:
            print(f'  ADMIN TOKEN (generated, keep private): {self.admin_token}')
        else:
            print('  admin token: set via CHESS_DIST_ADMIN_TOKEN')
        print('=' * 78)


settings = Settings()
