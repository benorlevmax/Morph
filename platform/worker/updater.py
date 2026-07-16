#!/usr/bin/env python3
"""updater.py - Automatic update check + optional self-update for the
worker client.

Check (always, cheap, no side effects): compare this install's VERSION
file against the server's GET /version manifest
(worker_client_version, see platform/server/app.py). A mismatch is only
ever logged unless --auto-update was explicitly passed -- a worker must
never silently start replacing its own files just because it noticed a
version skew.

Self-update (only with --auto-update and --update-url): downloads
worker-<version>.tar.gz from --update-url (intended to be a GitHub
Releases download URL produced by .github/workflows/release.yml), verifies
it's a well-formed tar archive, extracts it to a temp directory, then
atomically-ish swaps it in (rename the old install dir aside, rename the
new one into place) and re-execs the running process via os.execv so the
new code takes effect immediately without the operator needing to notice
and restart it manually.

This is real, working code (tested against a local HTTP server in this
repo's test suite -- see the update flow exercised during task #14's
end-to-end pass) but has only ever been exercised against a
locally-served tarball, not a real GitHub Release; the URL format is
designed to match what release.yml publishes, documented in
platform/docs/WORKER.md.
"""
import os
import shutil
import tarfile
import tempfile

import requests


def check_for_update(client, local_version):
    """Returns the server-advertised version string if it differs from
    local_version, else None. Never raises -- update checks are best-effort
    and must not interrupt the work loop (client.server_version() already
    swallows connection errors)."""
    manifest = client.server_version()
    if not manifest:
        return None
    remote_version = manifest.get('worker_client_version')
    if remote_version and remote_version != local_version:
        return remote_version
    return None


def read_local_version(install_dir):
    version_file = os.path.join(install_dir, 'VERSION')
    try:
        with open(version_file) as f:
            return f.read().strip()
    except OSError:
        return '0.0.0'


def perform_self_update(update_url, install_dir, target_version, log=print, timeout=60):
    """Downloads worker-{target_version}.tar.gz from update_url,
    extracts it, and swaps it in for install_dir. Returns True on success
    (caller should os.execv to restart into the new code), False on any
    failure (leaves install_dir completely untouched on failure -- either
    the whole swap happens or none of it does)."""
    archive_url = f"{update_url.rstrip('/')}/worker-{target_version}.tar.gz"
    log(f'[updater] downloading {archive_url}')

    tmp_dir = tempfile.mkdtemp(prefix='chess_worker_update_')
    archive_path = os.path.join(tmp_dir, 'update.tar.gz')
    try:
        resp = requests.get(archive_url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            log(f'[updater] download failed: HTTP {resp.status_code}')
            return False
        with open(archive_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        extract_dir = os.path.join(tmp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                _safe_extract(tar, extract_dir)
        except tarfile.TarError as e:
            log(f'[updater] not a valid tar.gz archive: {e}')
            return False

        # Archives may or may not have a single top-level directory --
        # normalize to "the directory that contains VERSION".
        new_root = _find_version_dir(extract_dir)
        if new_root is None:
            log('[updater] downloaded archive has no VERSION file at any level -- refusing '
                'to install (this would silently break auto-update forever)')
            return False

        # Preserve worker_state.json (registration credentials) across the
        # update -- the whole point is the worker keeps its identity.
        state_file = os.path.join(install_dir, 'worker_state.json')
        preserved_state = None
        if os.path.isfile(state_file):
            with open(state_file, 'rb') as f:
                preserved_state = f.read()

        backup_dir = install_dir.rstrip('/\\') + '.bak'
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
        os.rename(install_dir, backup_dir)
        try:
            shutil.copytree(new_root, install_dir)
            if preserved_state is not None:
                with open(os.path.join(install_dir, 'worker_state.json'), 'wb') as f:
                    f.write(preserved_state)
        except Exception as e:
            log(f'[updater] install failed ({e}), rolling back')
            if os.path.isdir(install_dir):
                shutil.rmtree(install_dir)
            os.rename(backup_dir, install_dir)
            return False

        shutil.rmtree(backup_dir, ignore_errors=True)
        log(f'[updater] updated {install_dir} to version {target_version}')
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _safe_extract(tar, extract_dir):
    """Guards against path-traversal ('../../etc/passwd') entries in a
    downloaded archive before extracting anything."""
    extract_dir_abs = os.path.abspath(extract_dir)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(extract_dir, member.name))
        if not member_path.startswith(extract_dir_abs + os.sep) and member_path != extract_dir_abs:
            raise tarfile.TarError(f'unsafe path in archive: {member.name!r}')
    tar.extractall(extract_dir)


def _find_version_dir(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if 'VERSION' in filenames:
            return dirpath
    return None
