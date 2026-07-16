#!/usr/bin/env python3
"""artifacts.py - Worker-side artifact fetch: download a dataset/checkpoint/
NNUE network the server has approved, verify its sha256 against what the
server's metadata says BEFORE trusting or executing anything derived from
the file, and cache it locally by content hash so re-running a task that
references the same artifact doesn't re-download it.

This is the one piece of the worker that touches files a stranger's server
told it to fetch, so the hash check here is load-bearing, not decorative:
platform/server/database.py's create_artifact/create_artifact-via-upload
always records the sha256 the SERVER itself computed from the bytes it
received (never a client-supplied claim -- see app.py's upload_artifact),
so a mismatch here means the transfer was corrupted or tampered with in
transit, not that the server's bookkeeping might be wrong.
"""
import hashlib
import os


class ArtifactVerificationError(Exception):
    pass


def _sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fetch_artifact(client, artifact_id, cache_dir, log=print):
    """Downloads (or reuses a cached copy of) the artifact identified by
    artifact_id. Returns the verified local file path. Raises
    ArtifactVerificationError if the downloaded bytes don't match the
    server-reported sha256 -- callers must treat that as fatal for the
    current task (do not execute/use the file), not something to retry
    blindly forever.
    """
    meta = client.get_artifact(artifact_id)
    expected_sha256 = meta['sha256']

    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, f'{artifact_id}-{expected_sha256}')
    if os.path.isfile(cached_path):
        if _sha256_file(cached_path) == expected_sha256:
            log(f'[artifacts] {artifact_id}: using cached copy at {cached_path}')
            return cached_path
        log(f'[artifacts] {artifact_id}: cached copy at {cached_path} failed hash check '
            f'(local file corrupted or stale) -- re-downloading')
        os.remove(cached_path)

    tmp_path = cached_path + '.part'
    log(f'[artifacts] {artifact_id}: downloading ({meta["kind"]}, {meta["size_bytes"]} bytes, '
        f'expected sha256 {expected_sha256[:12]}...)')
    client.download_artifact(artifact_id, tmp_path)

    actual_sha256 = _sha256_file(tmp_path)
    if actual_sha256 != expected_sha256:
        os.remove(tmp_path)
        raise ArtifactVerificationError(
            f'artifact {artifact_id!r}: sha256 mismatch after download -- expected '
            f'{expected_sha256}, got {actual_sha256}. Refusing to use this file (possible '
            f'corrupted transfer or tampering). Not retrying automatically.')

    os.replace(tmp_path, cached_path)
    log(f'[artifacts] {artifact_id}: verified OK -> {cached_path}')
    return cached_path
