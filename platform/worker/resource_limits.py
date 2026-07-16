#!/usr/bin/env python3
"""resource_limits.py - Optional CPU/memory caps for the worker client, so
contributing spare capacity does not make a volunteer's machine unusable
for its owner.

Deliberately soft/cooperative rather than using OS-level cgroups/job
objects: this worker is meant to run on arbitrary contributor machines
(Windows and Linux, unprivileged), where cgroups aren't available on
Windows at all and job-object CPU throttling on Windows requires extra
ctypes plumbing. Two simple, portable mechanisms instead:

  * --max-cpu-percent: a background thread samples the worker's own process
    tree's CPU usage (psutil) and sets a `should_backoff` Event when over
    the cap; platform_worker.py's main loop sleeps briefly when this is set,
    which throttles the actual measured usage without touching engine
    subprocess priorities.
  * --max-memory-mb: same sampling, but exceeding this sets `should_exit` --
    platform_worker.py treats this like a normal shutdown request (finish
    the current batch, upload, exit 0) rather than waiting for the OS to
    OOM-kill the process mid-upload.

Both are opt-in (None = no cap, the default) and best-effort: psutil is a
required dependency (see requirements.txt) so sampling itself won't fail,
but a monitor thread that dies for some other reason must never take the
work loop down with it -- see the try/except wrapping run().
"""
import threading
import time


def _iter_process_tree(proc):
    """proc plus all live descendants (self-play engine subprocesses)."""
    procs = [proc]
    try:
        procs.extend(proc.children(recursive=True))
    except Exception:
        pass
    return procs


class ResourceMonitor:
    def __init__(self, max_cpu_percent=None, max_memory_mb=None, check_interval=10.0, log=print):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.log = log
        self.should_backoff = threading.Event()
        self.should_exit = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self.max_cpu_percent is None and self.max_memory_mb is None:
            return   # nothing to monitor -- don't even spin up the thread
        import psutil
        self._psutil = psutil
        self._self_proc = psutil.Process()
        # First call always returns 0.0 (psutil needs a baseline interval);
        # prime it now so the first real reading in run() is meaningful.
        for p in _iter_process_tree(self._self_proc):
            try:
                p.cpu_percent(None)
            except Exception:
                pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.check_interval)
            try:
                self._check_once()
            except Exception as e:
                # A sampling error (e.g. a child process exited mid-scan)
                # must not kill the monitor thread permanently -- just skip
                # this round.
                self.log(f'[resource_limits] check failed (non-fatal): {e}')

    def _check_once(self):
        procs = _iter_process_tree(self._self_proc)

        if self.max_cpu_percent is not None:
            total_cpu = 0.0
            for p in procs:
                try:
                    total_cpu += p.cpu_percent(None)
                except Exception:
                    pass
            if total_cpu > self.max_cpu_percent:
                if not self.should_backoff.is_set():
                    self.log(f'[resource_limits] CPU {total_cpu:.0f}% > cap '
                             f'{self.max_cpu_percent:.0f}% -- backing off')
                self.should_backoff.set()
            else:
                self.should_backoff.clear()

        if self.max_memory_mb is not None:
            total_mb = 0.0
            for p in procs:
                try:
                    total_mb += p.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass
            if total_mb > self.max_memory_mb and not self.should_exit.is_set():
                self.log(f'[resource_limits] memory {total_mb:.0f}MB > cap '
                         f'{self.max_memory_mb:.0f}MB -- requesting clean shutdown')
                self.should_exit.set()
