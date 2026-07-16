#!/usr/bin/env python3
"""client.py - HTTP client for the distributed data-generation server, with
retry/backoff so a worker survives server restarts and network blips without
manual intervention ("automatic reconnect" requirement).
"""
import time

import requests


class ServerUnavailable(Exception):
    pass


class Client:
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

    def register(self, registration_secret, hostname, engine_version, threads):
        return self._request('POST', '/register', {
            'registration_secret': registration_secret, 'hostname': hostname,
            'engine_version': engine_version, 'threads': threads,
        })

    def next_task(self):
        return self._request('GET', '/tasks/next')

    def submit_results(self, task_id, positions, done=False):
        return self._request('POST', f'/tasks/{task_id}/results', {
            'positions': positions, 'done': done,
        })

    def health(self):
        return self._request('GET', '/health', retry=False)
