#!/usr/bin/env python3
"""test_platform_client.py - Regression test for PlatformClient._request()'s
handling of HTTP 429 (rate limited) responses.

Context: platform/server/app.py's submission endpoints (POST
/tasks/{id}/results, /match-result, /artifacts/upload) all rate-limit per
worker (see ratelimit.py, default 30/min) and return a plain 429 when a
worker exceeds it -- an expected, routine condition for any worker running
with --threads > 1 or a fast upload cadence, not a sign of anything actually
wrong. _request() already has retry-with-backoff for connection
errors/timeouts (so a worker survives server restarts and network blips);
this proves 429 goes through that same resilience path instead of being
treated as fatal and crashing the whole worker process, which is what it
did before this fix.

Mocks requests.request and time.sleep directly (no requests-mock dependency
-- platform/worker/ is deliberately dependency-light, see requirements.txt)
so these tests run in milliseconds with no real network or real waiting.

Run directly:  python3 test_platform_client.py
Run via pytest: pytest test_platform_client.py
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import platform_client as pc


def _response(status_code, text='', json_body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = headers or {}
    if json_body is not None:
        resp.content = b'placeholder'
        resp.json.return_value = json_body
    else:
        resp.content = text.encode('utf-8') if text else b''
    return resp


class RateLimitRetryTests(unittest.TestCase):
    def setUp(self):
        self.client = pc.PlatformClient('http://fake-server', token='t', max_retries=3,
                                         backoff_base=1.0, backoff_cap=10.0)
        self.sleep_calls = []

    def _patched_sleep(self, seconds):
        self.sleep_calls.append(seconds)

    def test_429_retries_then_succeeds(self):
        responses = [
            _response(429, text='rate limited'),
            _response(429, text='rate limited'),
            _response(200, json_body={'ok': True}),
        ]
        with patch('platform_client.requests.request', side_effect=responses) as mock_req, \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            result = self.client._request('POST', '/tasks/t1/results', {'positions': []},
                                            log=lambda msg: None)
        self.assertEqual(result, {'ok': True})
        self.assertEqual(mock_req.call_count, 3)
        self.assertEqual(len(self.sleep_calls), 2)  # slept before the two retries, not the success

    def test_429_honors_retry_after_header(self):
        responses = [
            _response(429, text='rate limited', headers={'Retry-After': '7'}),
            _response(200, json_body={'ok': True}),
        ]
        with patch('platform_client.requests.request', side_effect=responses), \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            result = self.client._request('POST', '/tasks/t1/results', {}, log=lambda msg: None)
        self.assertEqual(result, {'ok': True})
        self.assertEqual(self.sleep_calls, [7.0])

    def test_429_falls_back_to_backoff_without_retry_after_header(self):
        responses = [_response(429, text='rate limited'), _response(200, json_body={'ok': True})]
        with patch('platform_client.requests.request', side_effect=responses), \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            self.client._request('POST', '/tasks/t1/results', {}, log=lambda msg: None)
        # backoff_base=1.0 -> first retry delay is 1.0 * 2**(1-1) = 1.0
        self.assertEqual(self.sleep_calls, [1.0])

    def test_429_malformed_retry_after_falls_back_to_backoff(self):
        responses = [
            _response(429, text='rate limited', headers={'Retry-After': 'not-a-number'}),
            _response(200, json_body={'ok': True}),
        ]
        with patch('platform_client.requests.request', side_effect=responses), \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            self.client._request('POST', '/tasks/t1/results', {}, log=lambda msg: None)
        self.assertEqual(self.sleep_calls, [1.0])

    def test_429_gives_up_after_max_retries_raises_runtime_error(self):
        # max_retries=3 -> attempts 1,2,3 all 429, attempt 4 exceeds max_retries -> raises
        responses = [_response(429, text='still limited')] * 4
        with patch('platform_client.requests.request', side_effect=responses), \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            with self.assertRaises(RuntimeError) as ctx:
                self.client._request('POST', '/tasks/t1/results', {}, log=lambda msg: None)
        self.assertIn('429', str(ctx.exception))
        self.assertIn('rate limited', str(ctx.exception))

    def test_429_with_retry_false_raises_immediately_no_sleep(self):
        responses = [_response(429, text='rate limited')]
        with patch('platform_client.requests.request', side_effect=responses), \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            with self.assertRaises(RuntimeError):
                self.client._request('GET', '/health', retry=False, log=lambda msg: None)
        self.assertEqual(self.sleep_calls, [])

    def test_non_429_error_codes_still_raise_immediately(self):
        # A plain 500 should NOT go through the 429 retry path.
        responses = [_response(500, text='server error')]
        with patch('platform_client.requests.request', side_effect=responses), \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            with self.assertRaises(RuntimeError) as ctx:
                self.client._request('POST', '/tasks/t1/results', {}, log=lambda msg: None)
        self.assertIn('500', str(ctx.exception))
        self.assertEqual(self.sleep_calls, [])

    def test_401_still_raises_permission_error_not_retried(self):
        responses = [_response(401, text='invalid API key')]
        with patch('platform_client.requests.request', side_effect=responses), \
             patch('platform_client.time.sleep', side_effect=self._patched_sleep):
            with self.assertRaises(PermissionError):
                self.client._request('POST', '/register', {}, log=lambda msg: None)
        self.assertEqual(self.sleep_calls, [])


if __name__ == '__main__':
    unittest.main()
