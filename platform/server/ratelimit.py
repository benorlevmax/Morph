#!/usr/bin/env python3
"""ratelimit.py - Simple in-memory sliding-window rate limiter.

Scope note: this is a single-process, in-memory limiter (a plain dict
behind a lock). That's the honest, correct choice for the reference
deployment this platform ships (one `uvicorn` process, one SQLite file --
see platform/docs/SERVER.md's deployment notes) and is trivial to reason
about and test. It does NOT coordinate across multiple server processes or
machines; a horizontally-scaled deployment (multiple uvicorn workers behind
a load balancer) would need a shared store (Redis, or a DB-backed counter)
instead. That's a real, documented limitation, not a hidden one -- flagged
again in platform/docs/SERVER.md's "Future improvements".
"""
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._hits = defaultdict(deque)   # identity -> deque of unix timestamps

    def check(self, identity, max_events, window_seconds):
        """Returns True if `identity` is allowed one more event right now
        (and records it), False if it's over `max_events` within the last
        `window_seconds`."""
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            q = self._hits[identity]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= max_events:
                return False
            q.append(now)
            return True

    def remaining(self, identity, max_events, window_seconds):
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            q = self._hits[identity]
            while q and q[0] < cutoff:
                q.popleft()
            return max(0, max_events - len(q))

    def reset(self, identity=None):
        """Clears rate-limit history. With no argument, clears everything
        (used by tests); with an identity, clears just that one."""
        with self._lock:
            if identity is None:
                self._hits.clear()
            else:
                self._hits.pop(identity, None)


# One process-wide limiter, with separate named buckets per endpoint class
# (a submission burst shouldn't count against a user's login attempts).
limiter = RateLimiter()
