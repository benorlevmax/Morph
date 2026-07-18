#!/usr/bin/env python3
"""platform_client.py - HTTP client for the public community-compute
server (platform/server/app.py), with retry/backoff so a worker survives
server restarts and network blips without manual intervention.

Named platform_client.py (not client.py) and kept self-contained rather
than importing distributed/worker/client.py, for two reasons: (1) this
directory is meant to be independently downloadable -- a contributor should
not need the rest of this repo, just platform/worker/ plus a
compiled engine binary; (2) distributed/worker/ already has its own
client.py, and this project has been bitten three times (db.py, models.py,
config.py under platform/server/) by same-basename modules silently
shadowing each other when both directories end up on sys.path -- using a
distinct name here sidesteps that whole class of bug rather than relying on
import order.

Talks to the same /tasks/next and /tasks/{id}/results wire format as
distributed/server (unchanged), plus this server's account-linked
/register (api_key) and /version (for updater.py).
"""
import time

import requests


class ServerUnavailable(Exception):
    pass


class PlatformClient:
    def __init__(self, base_url, token=None, max_retries=8, backoff_base=2.0, backoff_cap=60.0,
                 timeout=30):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.timeout = timeout

    def _headers(self):
        h = {'Content-Type': 'application/json'}
        if self.token:
            h['Authorization'] = f'Bearer {self.token}'
        return h

    def _request(self, method, path, json_body=None, retry=True, log=print):
        url = f'{self.base_url}{path}'
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.request(method, url, json=json_body, headers=self._headers(),
                                         timeout=self.timeout)
                if resp.status_code == 401:
                    raise PermissionError(f'{method} {path}: 401 {resp.text}')
                if resp.status_code == 204:
                    return None
                if resp.status_code == 409:
                    raise PermissionError(f'{method} {path}: 409 {resp.text}')
                if resp.status_code == 429:
                    # Rate-limited (see platform/server/ratelimit.py) --
                    # this is expected, routine behavior for a healthy
                    # worker (e.g. multiple --threads submitting batches
                    # close together), not a fatal error, so it goes
                    # through the same retry/backoff path as a connection
                    # blip rather than crashing the whole worker process.
                    # Honors a Retry-After header if the server sends one,
                    # otherwise falls back to the same exponential backoff
                    # used below (the limiter's window is short -- default
                    # 60s -- so this reliably clears within max_retries).
                    if not retry or attempt > self.max_retries:
                        raise RuntimeError(
                            f'{method} {path}: 429 rate limited after {attempt} attempts: '
                            f'{resp.text}')
                    retry_after = resp.headers.get('Retry-After')
                    try:
                        delay = float(retry_after) if retry_after else None
                    except ValueError:
                        delay = None
                    if delay is None:
                        delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap)
                    log(f'[client] {method} {path} rate limited (429); '
                        f'retrying in {delay:.0f}s (attempt {attempt}/{self.max_retries})')
                    time.sleep(delay)
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f'{method} {path}: HTTP {resp.status_code}: {resp.text}')
                if not resp.content:
                    return None
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                if not retry or attempt > self.max_retries:
                    raise ServerUnavailable(
                        f'{method} {path}: server unreachable after {attempt} attempts: {e}')
                delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap)
                log(f'[client] {method} {path} failed ({e.__class__.__name__}); '
                    f'retrying in {delay:.0f}s (attempt {attempt}/{self.max_retries})')
                time.sleep(delay)

    def register(self, hostname, engine_version, threads, api_key=None, registration_secret=None):
        body = {'hostname': hostname, 'engine_version': engine_version, 'threads': threads}
        if api_key:
            body['api_key'] = api_key
        if registration_secret:
            body['registration_secret'] = registration_secret
        return self._request('POST', '/register', body)

    def next_task(self):
        return self._request('GET', '/tasks/next')

    def submit_results(self, task_id, positions, done=False):
        return self._request('POST', f'/tasks/{task_id}/results', {
            'positions': positions, 'done': done,
        })

    # -- typed tasks (SELF_PLAY / DATA_GENERATION / ELO_MATCH / TRAIN_NETWORK) --
    def report_capabilities(self, capabilities: dict):
        return self._request('POST', '/workers/capabilities', capabilities)

    def next_typed_task(self):
        """Capability-aware polling -- see GET /tasks/next-typed. Returns
        {'task_id', 'task_type', 'payload'} or None if nothing is available
        for this worker right now."""
        return self._request('GET', '/tasks/next-typed')

    def submit_match_result(self, task_id, candidate_artifact_id, baseline_artifact_id,
                             wins, losses, draws, pgn_base64=None):
        return self._request('POST', f'/tasks/{task_id}/match-result', {
            'candidate_artifact_id': candidate_artifact_id,
            'baseline_artifact_id': baseline_artifact_id,
            'wins': wins, 'losses': losses, 'draws': draws,
            'pgn_base64': pgn_base64,
        })

    # -- artifacts (datasets / checkpoints / candidate & accepted networks) -----
    def get_artifact(self, artifact_id):
        return self._request('GET', f'/artifacts/{artifact_id}')

    def get_strongest_network(self):
        """Returns None (not an error) if no network has been accepted yet
        on this deployment -- see /artifacts/strongest-network's 404
        meaning "not seeded", which callers here treat as absence rather
        than a hard failure."""
        try:
            return self._request('GET', '/artifacts/strongest-network', retry=False)
        except RuntimeError as e:
            if '404' in str(e):
                return None
            raise

    def download_artifact(self, artifact_id, dest_path):
        """Streams the artifact's raw bytes to dest_path. Unlike
        _request(), this is not a JSON call -- callers (see artifacts.py's
        fetch_artifact) are responsible for verifying the downloaded file's
        sha256 against get_artifact()'s reported hash before trusting it."""
        url = f'{self.base_url}/artifacts/{artifact_id}/download'
        attempt = 0
        while True:
            attempt += 1
            try:
                with requests.get(url, headers=self._headers(), timeout=self.timeout,
                                   stream=True) as resp:
                    if resp.status_code == 401:
                        raise PermissionError(f'GET {url}: 401 {resp.text}')
                    if resp.status_code >= 400:
                        raise RuntimeError(f'GET {url}: HTTP {resp.status_code}: {resp.text}')
                    with open(dest_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                return dest_path
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt > self.max_retries:
                    raise ServerUnavailable(
                        f'GET {url}: server unreachable after {attempt} attempts: {e}')
                delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap)
                time.sleep(delay)

    def upload_artifact(self, kind, file_path, task_id=None, metadata=None):
        """Multipart upload of a produced artifact (candidate .nnue net,
        generated dataset, or training checkpoint). The server computes
        sha256/size itself from the received bytes -- see
        platform/server/app.py's upload_artifact -- so nothing here needs
        to pre-hash the file for integrity purposes (only for the caller's
        own logging/verification if it wants to)."""
        import json as _json
        url = f'{self.base_url}/artifacts/upload'
        data = {'kind': kind}
        if task_id:
            data['task_id'] = task_id
        if metadata is not None:
            data['metadata_json'] = _json.dumps(metadata)
        headers = {}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        with open(file_path, 'rb') as f:
            resp = requests.post(url, data=data, files={'file': f}, headers=headers,
                                  timeout=max(self.timeout, 120))
        if resp.status_code == 401:
            raise PermissionError(f'POST {url}: 401 {resp.text}')
        if resp.status_code >= 400:
            raise RuntimeError(f'POST {url}: HTTP {resp.status_code}: {resp.text}')
        return resp.json()

    def health(self):
        return self._request('GET', '/health', retry=False)

    def server_version(self):
        """Used by updater.py -- returns the {'worker_client_version': ...}
        manifest the server advertises, or None if unreachable (auto-update
        checks are best-effort and must never block the work loop)."""
        try:
            return self._request('GET', '/version', retry=False)
        except Exception:
            return None
