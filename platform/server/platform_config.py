#!/usr/bin/env python3
"""platform_config.py - Settings for the public platform server.

Named platform_config.py rather than config.py deliberately: app.py puts
distributed/server on sys.path (to reuse its db.py/auth.py/validation.py),
and distributed/server also has its own config.py. A same-named module here
would silently lose that import race depending on sys.path order -- this
exact bug already bit db.py (-> database.py) and models.py (-> schemas.py)
earlier in this project; config.py is the same class of collision, fixed
the same way: distinct basenames instead of relying on import order.

This is the "production" evolution of distributed/server/ (see that
module's own config.py, which stays completely unmodified and keeps
working for local/LAN trusted testing per docs/DISTRIBUTED_DATA_GENERATION.md).
The two differences that matter for a public deployment:

  * a real database file, separate from the LAN one (CHESS_PLATFORM_DB_PATH),
    so pointing a public server at this code never touches or mixes with
    someone's existing local test data.
  * per-user accounts/API keys instead of (or alongside) one shared
    registration secret -- see accounts.py.

No secrets are hardcoded; anything unset is either generated once at
startup (printed so a first-time operator can capture it) or, for
per-request session signing, generated fresh at process start (meaning
existing sessions are invalidated on restart -- documented, not hidden;
see accounts.py's note on why this is an acceptable tradeoff for a
volunteer-compute server rather than a persistent-login consumer service).
"""
import os
import secrets

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'database', 'platform.sqlite3')


class Settings:
    def __init__(self):
        self.db_path = os.environ.get('CHESS_PLATFORM_DB_PATH', DEFAULT_DB_PATH)

        # Legacy shared-secret registration, kept as an OPTIONAL alternative
        # to per-user API keys (e.g. for a private/invite-only deployment
        # that doesn't want public signups at all). Unset by default -- if
        # empty, only account-linked (API-key) worker registration works.
        self.registration_secret = os.environ.get('CHESS_PLATFORM_REGISTRATION_SECRET', '')

        self.admin_token = os.environ.get('CHESS_PLATFORM_ADMIN_TOKEN') or secrets.token_hex(16)
        self._admin_token_was_generated = 'CHESS_PLATFORM_ADMIN_TOKEN' not in os.environ

        self.task_lease_seconds = int(os.environ.get('CHESS_PLATFORM_TASK_LEASE_SECONDS', '1800'))
        self.default_chunk_size = int(os.environ.get('CHESS_PLATFORM_DEFAULT_CHUNK_SIZE', '500'))

        # Rate limiting (see ratelimit.py). Conservative defaults sized for
        # "one worker process submitting every ~15s per docs" plus headroom,
        # not for a browser hammering the API.
        self.rate_limit_submissions_per_minute = int(
            os.environ.get('CHESS_PLATFORM_RATE_LIMIT_SUBMISSIONS_PER_MIN', '30'))
        self.rate_limit_registrations_per_hour = int(
            os.environ.get('CHESS_PLATFORM_RATE_LIMIT_REGISTRATIONS_PER_HOUR', '20'))
        self.rate_limit_login_attempts_per_15min = int(
            os.environ.get('CHESS_PLATFORM_RATE_LIMIT_LOGIN_PER_15MIN', '10'))

        # Latest worker version manifest, for the auto-update check
        # (see platform/worker/updater.py). A real deployment points this at
        # a URL the operator controls (e.g. a GitHub raw file or this same
        # server's own /version endpoint); documented, not faked.
        self.latest_worker_version = os.environ.get('CHESS_PLATFORM_WORKER_VERSION', '1.0.0')

        # Where artifact files (datasets, checkpoints, candidate/accepted
        # NNUE networks -- see database.py's `artifacts` table) are actually
        # stored on disk. Server-side content-addressed by sha256 so two
        # uploads of identical bytes collapse to one file. This is the
        # server's own read-write artifact store, written to by
        # TRAIN_NETWORK/DATA_GENERATION artifact uploads -- the dashboard
        # and every other live figure this server reports come from the
        # database itself (see dashboard_data.py), never from files on disk.
        # NOTE: this must derive from self.db_path (already resolved above,
        # honoring CHESS_PLATFORM_DB_PATH), not the module-level
        # DEFAULT_DB_PATH constant -- using the constant here meant the
        # artifacts dir silently ignored CHESS_PLATFORM_DB_PATH and always
        # fell back to a path next to the *source tree's* default database
        # location. In the Docker image that's /app/platform/database/,
        # which is root-owned and read-only to the non-root `platform`
        # user the container actually runs as -- every deployment setting
        # CHESS_PLATFORM_DB_PATH (which includes the shipped Dockerfile,
        # pointing it at the writable /data volume) hit a PermissionError
        # on startup as soon as it tried to create this directory.
        self.artifacts_dir = os.environ.get(
            'CHESS_PLATFORM_ARTIFACTS_DIR',
            os.path.join(os.path.dirname(self.db_path), 'artifacts'))
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Hard cap on a single artifact upload (candidate NNUE nets and
        # datasets are both bounded in practice; this just stops a
        # malfunctioning or malicious worker from filling the disk).
        self.max_artifact_upload_bytes = int(
            os.environ.get('CHESS_PLATFORM_MAX_ARTIFACT_BYTES', str(512 * 1024 * 1024)))  # 512 MiB

        # Caps how many *currently connected* workers (see
        # database.py's count_connected_workers -- not disabled, seen
        # within the last 10 minutes) POST /register will accept before
        # turning new registrations away with a 503. This is a load
        # safety valve, not a community-size limit: an already-registered
        # worker that goes quiet for a while doesn't count against the
        # cap, and no existing worker is ever kicked -- only *new*
        # registrations are throttled while the server is full. Default
        # of 40 is deliberately conservative for a small single-OCPU
        # deploy target (see GET /admin/system-load for the live number
        # this is being compared against); raise it freely on bigger
        # hardware.
        self.max_connected_workers = int(
            os.environ.get('CHESS_PLATFORM_MAX_CONNECTED_WORKERS', '40'))

    def print_startup_banner(self):
        print('=' * 78)
        print('Morph Community Compute Platform - server')
        print(f'  database: {self.db_path}')
        print(f'  task lease: {self.task_lease_seconds}s')
        print(f'  rate limits: {self.rate_limit_submissions_per_minute}/min submissions, '
              f'{self.rate_limit_registrations_per_hour}/hr registrations, '
              f'{self.rate_limit_login_attempts_per_15min}/15min logins')
        if self.registration_secret:
            print('  legacy shared registration secret: ENABLED (set via '
                  'CHESS_PLATFORM_REGISTRATION_SECRET)')
        else:
            print('  legacy shared registration secret: disabled -- account/API-key '
                  'registration only')
        if self._admin_token_was_generated:
            print(f'  ADMIN TOKEN (generated, keep private): {self.admin_token}')
        else:
            print('  admin token: set via CHESS_PLATFORM_ADMIN_TOKEN')
        print('=' * 78)


settings = Settings()
