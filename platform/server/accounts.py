#!/usr/bin/env python3
"""accounts.py - Password hashing and API key helpers.

Deliberately stdlib-only (hashlib.pbkdf2_hmac): this repo's stated principle
throughout (see docs/*.md) is no proprietary/unnecessary dependencies, and
PBKDF2-HMAC-SHA256 at a modern iteration count is a perfectly reasonable,
audited, standard choice for a volunteer-compute platform's login (not a
bank) -- bcrypt/argon2 would be marginally stronger but pull in a native
dependency for a fairly small real-world gain here. Documented so a future
maintainer can make an informed call about upgrading it, not left implicit.
"""
import hashlib
import hmac
import secrets

PBKDF2_ITERATIONS = 310_000   # OWASP's 2023 minimum recommendation for PBKDF2-HMAC-SHA256
_ALGO_TAG = 'pbkdf2_sha256'


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt),
                                  PBKDF2_ITERATIONS).hex()
    return f'{_ALGO_TAG}${PBKDF2_ITERATIONS}${salt}${digest}'


def verify_password(password, stored_hash):
    try:
        algo, iterations, salt, digest = stored_hash.split('$')
        if algo != _ALGO_TAG:
            return False
        iterations = int(iterations)
    except (ValueError, AttributeError):
        return False
    check = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt),
                                 iterations).hex()
    return hmac.compare_digest(check, digest)


def generate_api_key():
    """Returns (plaintext_key, sha256_hash_hex). The plaintext is shown to
    the user exactly once (at generation time) and never stored -- only its
    hash is, exactly the same pattern distributed/server/db.py already uses
    for worker bearer tokens."""
    key = 'cek_' + secrets.token_hex(24)   # 'cek_' = Chess Engine Key, so it's recognizable in logs
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


def hash_api_key(key):
    return hashlib.sha256(key.encode()).hexdigest()


def generate_session_token():
    """Opaque bearer session token issued at login. Sessions are stored
    server-side (see db.py's sessions table) so they can be individually
    revoked (logout / admin action) -- a signed-JWT-without-a-server-side
    store would not support real revocation, which matters more here than
    avoiding one extra DB lookup per request."""
    token = secrets.token_hex(24)
    return token, hashlib.sha256(token.encode()).hexdigest()
