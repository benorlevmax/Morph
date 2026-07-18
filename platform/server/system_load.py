#!/usr/bin/env python3
"""system_load.py - Lightweight, dependency-free system load snapshot for
GET /admin/system-load (see app.py). Deliberately avoids psutil or any
other extra package: every dependency is extra weight to install/pull on
a memory-constrained deploy target (see platform/docs/SERVER.md's
'Capacity and alerts' section for why this exists at all -- a single
Oracle Cloud Always Free instance running this server hit real OOM
trouble during initial setup, which is what prompted building this).

Reads /proc directly (Linux-only) and os.getloadavg() (POSIX-only) --
both are guarded so this degrades to partial/None fields rather than
raising if ever run somewhere else (e.g. a contributor's own Windows dev
machine running the server standalone for local testing).

Every function here is pure (no db/network access, easy to unit-test in
isolation) except read_proc_meminfo/get_load_average/get_disk_usage,
which read real OS state -- build_system_load_snapshot is the one that
combines everything, and takes its worker/task counts as plain arguments
rather than a db handle, so it's testable without a real database too.
"""
import os
import shutil


def read_proc_meminfo():
    """Parses /proc/meminfo for MemTotal/MemAvailable. MemAvailable
    (not MemFree) is the kernel's own estimate of how much could actually
    be allocated to a new process without swapping -- the right number
    for 'are we close to OOM', not just literally-unused pages (which
    undercounts memory that's reclaimable page cache).

    Returns None if /proc/meminfo doesn't exist (non-Linux) or is
    unparseable, rather than raising -- this is a monitoring endpoint,
    not a critical path, and a partial response beats a 500."""
    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                key, _, rest = line.partition(':')
                parts = rest.strip().split()
                if not parts:
                    continue
                info[key] = int(parts[0])  # value is in kB
        total_kb = info.get('MemTotal')
        available_kb = info.get('MemAvailable')
        if not total_kb:
            return None
        used_percent = None
        if available_kb is not None:
            used_percent = round(100 * (total_kb - available_kb) / total_kb, 1)
        return {
            'total_mb': round(total_kb / 1024, 1),
            'available_mb': round(available_kb / 1024, 1) if available_kb is not None else None,
            'used_percent': used_percent,
        }
    except (OSError, ValueError):
        return None


def get_load_average():
    """os.getloadavg() -- POSIX only (returns 1/5/15-minute load
    averages). None on platforms without it instead of raising."""
    try:
        load1, load5, load15 = os.getloadavg()
        return {'1min': round(load1, 2), '5min': round(load5, 2), '15min': round(load15, 2)}
    except (OSError, AttributeError):
        return None


def get_disk_usage(path):
    """shutil.disk_usage on the artifacts directory's filesystem (the
    thing actually at risk of filling up on this box -- see
    prune_positions in app.py, built for the same underlying concern:
    a small free-tier disk filling with community-contributed data)."""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            'total_gb': round(total / (1024 ** 3), 2),
            'free_gb': round(free / (1024 ** 3), 2),
            'used_percent': round(100 * used / total, 1) if total else None,
        }
    except OSError:
        return None


def build_system_load_snapshot(connected_workers, max_connected_workers,
                                pending_tasks, artifacts_dir):
    """Combines everything GET /admin/system-load returns. Takes plain
    values (not a db handle) for connected_workers/pending_tasks so this
    is unit-testable without a real database -- see test_system_load.py."""
    return {
        'connected_workers': connected_workers,
        'max_connected_workers': max_connected_workers,
        'at_worker_capacity': connected_workers >= max_connected_workers,
        'pending_tasks': pending_tasks,
        'cpu_count': os.cpu_count(),
        'load_average': get_load_average(),
        'memory': read_proc_meminfo(),
        'disk': get_disk_usage(artifacts_dir),
    }
