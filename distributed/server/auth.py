#!/usr/bin/env python3
"""auth.py - FastAPI auth dependencies.

Two independent credentials:
  * worker bearer token  - issued once at /register, required on every
    worker-facing endpoint (task fetch, result upload). Invalid/missing ->
    401. A disabled worker's token stops working immediately (checked in
    Database.authenticate_worker via `disabled = 0`).
  * admin token           - a separate, higher-privilege secret (never handed
    to workers) required for task creation and full stats/worker management.
    Constant-time compared to resist timing attacks on the comparison itself.
"""
import hmac

from fastapi import Header, HTTPException

_db = None
_settings = None


def configure(db, settings):
    global _db, _settings
    _db = db
    _settings = settings


def require_worker(authorization: str = Header(default=None)):
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail='missing bearer token')
    worker = _db.authenticate_worker(token)
    if worker is None:
        raise HTTPException(status_code=401, detail='invalid or disabled worker token')
    return worker


def require_admin(x_admin_token: str = Header(default=None)):
    if not x_admin_token or not hmac.compare_digest(x_admin_token, _settings.admin_token):
        raise HTTPException(status_code=401, detail='missing or invalid admin token')
    return True


def _bearer_token(authorization_header):
    if not authorization_header:
        return None
    parts = authorization_header.split(' ', 1)
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    return parts[1].strip()
