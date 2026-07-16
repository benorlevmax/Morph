#!/usr/bin/env python3
"""notify.py - Pluggable notifications for the self-improvement loop.

Always logs (that's the one guaranteed channel, and log lines are how
tests/CI verify a notification fired without needing real network access).
Optionally also posts a JSON payload to a webhook URL (compatible with
Slack/Discord/generic incoming-webhooks, which all accept a top-level
"text" field) if AUTOMATION_WEBHOOK_URL is set in the environment.

A notification failure (webhook unreachable, DNS failure, etc.) is caught
and logged, never allowed to break the training loop it's reporting on.

No email/SMS backend is implemented here -- wiring one in is a small,
well-contained addition to send() once there's a real service/credentials
to send through; faking one that silently does nothing would be worse than
just not having it.
"""
import json
import logging
import os
import time
import urllib.request
import urllib.error

logger = logging.getLogger('automation.notify')

WEBHOOK_URL_ENV = 'AUTOMATION_WEBHOOK_URL'


def notify(event, message, **fields):
    """event: short machine-readable tag, e.g. 'cycle_start', 'promoted',
    'rejected', 'stage_failed', 'loop_halted'. message: human-readable
    summary. fields: arbitrary extra JSON-serializable context (experiment
    id, elo, dataset size, etc.), logged and forwarded to the webhook."""
    payload = {
        'event': event, 'message': message,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        **fields,
    }
    logger.info('[notify] %s: %s %s', event, message, fields or '')

    url = os.environ.get(WEBHOOK_URL_ENV)
    if not url:
        return
    body = json.dumps({'text': f'[{event}] {message}', **payload}).encode()
    req = urllib.request.Request(url, data=body, method='POST',
                                  headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except (urllib.error.URLError, OSError) as e:
        logger.warning('[notify] webhook delivery failed (%s): %s', url, e)
