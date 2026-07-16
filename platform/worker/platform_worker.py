#!/usr/bin/env python3
"""platform_worker.py - Main worker loop for the public community-compute
platform: register (once), report capabilities, then repeatedly fetch a
typed task and dispatch to the right executor:

    SELF_PLAY        -> TaskRunner (below): --threads concurrent self-play
                         engine instances, streaming batched uploads.
    DATA_GENERATION  -> data_generation.run_data_generation: wraps the
                         engine's native chess_train gen bulk exporter.
    ELO_MATCH        -> elo_match.run_elo_match: candidate vs. baseline
                         NNUE match via tools/nnue_pipeline/uci_match.py.
    TRAIN_NETWORK    -> platform/trainer/train_network.run_train_network:
                         real HalfKP training run via tools/nnue_pipeline
                         (train.py + export.py), local load+verify against
                         the compiled engine, uploads a genuine 'network'
                         artifact (see that module's docstring for the
                         history of why this used to upload an unusable
                         'checkpoint' instead). Only ever offered to
                         workers that opted in with --trainer-capable.

Adapted from distributed/worker/worker.py (SELF_PLAY's TaskRunner keeps the
same shape and upload-batching/resilience logic, already tested there) with
additions for a public, volunteer-run, multi-task-type deployment:
    * registers via --api-key (or legacy --registration-secret)
    * reports CPU/RAM/GPU/trainer capabilities (capabilities.py)
    * polls the capability-aware GET /tasks/next-typed instead of the
      untyped GET /tasks/next
    * an optional ResourceMonitor (resource_limits.py) that backs off or
      exits cleanly under CPU/memory caps
    * an optional periodic update check / self-update (updater.py)

Flow: connects to server -> registers -> reports capabilities -> receives a
typed task -> executes it for real -> uploads evidence/results -> repeat.
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from platform_client import PlatformClient, ServerUnavailable
from platform_config import parse_args, load_state, save_state
from resource_limits import ResourceMonitor
from selfplay import SelfPlayEngine, play_selfplay_game
from updater import check_for_update, perform_self_update, read_local_version
from capabilities import detect_capabilities
from data_generation import run_data_generation, DataGenerationError
from elo_match import run_elo_match, EloMatchError

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trainer'))
from train_network import run_train_network, TrainNetworkError  # noqa: E402

INSTALL_DIR = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def ensure_registered(args):
    state = None if args.force_register else load_state(args.state_file)
    if state and state.get('worker_id') and state.get('worker_token'):
        log(f"using saved credentials for worker {state['worker_id']} ({args.state_file})")
        return state

    if not args.api_key and not args.registration_secret:
        sys.exit(
            f'no saved credentials at {args.state_file} and neither --api-key nor '
            f'--registration-secret was given -- see platform/docs/WORKER.md for how to '
            f'get an API key from the server.')

    client = PlatformClient(args.server)
    engine_version = _probe_engine_version(args.engine_bin)
    log(f'registering with {args.server} as {args.hostname!r} (engine {engine_version!r})')
    resp = client.register(args.hostname, engine_version, args.threads,
                            api_key=args.api_key, registration_secret=args.registration_secret)
    state = {'server': args.server, 'worker_id': resp['worker_id'],
              'worker_token': resp['worker_token'], 'hostname': args.hostname,
              'engine_version': engine_version, 'linked_account': resp.get('linked_account')}
    save_state(args.state_file, state)
    acct = f" (linked to account {resp['linked_account']!r})" if resp.get('linked_account') else ''
    log(f"registered as {resp['worker_id']}{acct}; credentials saved to {args.state_file}")
    return state


def _probe_engine_version(engine_bin):
    eng = SelfPlayEngine(engine_bin, depth=1)
    v = eng.engine_version
    eng.close()
    return v


class TaskRunner:
    """Runs --threads concurrent self-play generators against one SELF_PLAY
    task, uploading in batches until the task's target is met (as reported
    back by the server after each upload) or the server reassigns the task
    elsewhere. Also respects a ResourceMonitor: backs off between games if
    over the CPU cap, and stops early (uploading what it has) if over the
    memory cap.

    `task` is the typed-task shape from GET /tasks/next-typed:
    {'task_id', 'task_type', 'payload': {'target_positions', 'depth',
    'randomplies'}} -- see database.py's assign_next_typed_task, which
    merges those fields into payload for every SELF_PLAY task regardless of
    whether it was created via the typed or legacy bulk path."""

    def __init__(self, client, engine_bin, task, args, monitor=None):
        self.client = client
        self.engine_bin = engine_bin
        self.task = task
        self.payload = task['payload']
        self.args = args
        self.monitor = monitor
        self.record_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.generated_count = 0
        self.lock = threading.Lock()

    def _worker_thread(self, thread_idx):
        rng_seed = hash((self.task['task_id'], thread_idx, time.time())) & 0xFFFFFFFF
        import random
        rng = random.Random(rng_seed)
        engine = SelfPlayEngine(self.engine_bin, depth=self.payload['depth'],
                                 hash_mb=self.args.hash_mb)
        try:
            while not self.stop_event.is_set():
                if self.monitor and self.monitor.should_backoff.is_set():
                    time.sleep(2.0)
                    continue
                if self.monitor and self.monitor.should_exit.is_set():
                    break
                records = play_selfplay_game(
                    engine, randomplies=self.payload['randomplies'],
                    max_plies=self.args.max_plies, rng=rng)
                for r in records:
                    self.record_queue.put(r)
                with self.lock:
                    self.generated_count += len(records)
        finally:
            engine.close()

    def run(self):
        threads = [threading.Thread(target=self._worker_thread, args=(i,), daemon=True)
                   for i in range(self.args.threads)]
        for t in threads:
            t.start()

        task_id = self.task['task_id']
        target = self.payload['target_positions']
        total_accepted = 0
        last_progress = time.time()
        batch = []

        try:
            while not self.stop_event.is_set():
                if self.monitor and self.monitor.should_exit.is_set():
                    log('resource cap exceeded -- finishing current batch and stopping')
                    self.stop_event.set()
                    break
                try:
                    rec = self.record_queue.get(timeout=1.0)
                    batch.append(rec)
                except queue.Empty:
                    pass

                should_flush = len(batch) >= self.args.upload_batch_size
                should_report = time.time() - last_progress > 15
                if batch and (should_flush or should_report):
                    total_accepted += self._flush(task_id, batch)
                    batch = []
                    last_progress = time.time()
                    log(f'task {task_id}: {total_accepted}/{target} accepted so far')
                    if total_accepted >= target:
                        self.stop_event.set()
        finally:
            self.stop_event.set()
            for t in threads:
                t.join(timeout=30)
            # Drain anything left in the queue and do a final, `done=True` upload.
            while not self.record_queue.empty():
                batch.append(self.record_queue.get_nowait())
            total_accepted += self._flush(task_id, batch, done=True)

        log(f'task {task_id}: finished, {total_accepted}/{target} positions accepted')
        return total_accepted

    def _flush(self, task_id, batch, done=False):
        if not batch and not done:
            return 0
        try:
            resp = self.client.submit_results(task_id, batch, done=done)
        except ServerUnavailable as e:
            log(f'WARNING: failed to upload {len(batch)} positions after retries: {e} '
                f'-- {len(batch)} positions lost for this batch')
            return 0
        if resp is None:
            return 0
        if resp.get('rejected', 0):
            log(f"  ({resp['rejected']} rejected: {resp.get('rejected_reasons', [])[:3]})")
        if resp.get('task_status') == 'completed':
            self.stop_event.set()
        return resp.get('accepted', 0)


def _maybe_check_for_update(client, args, last_check):
    if time.time() - last_check < args.update_check_interval:
        return last_check
    local_version = read_local_version(INSTALL_DIR)
    remote_version = check_for_update(client, local_version)
    if remote_version:
        if args.auto_update and args.update_url:
            log(f'new worker version available ({local_version} -> {remote_version}), updating')
            if perform_self_update(args.update_url, INSTALL_DIR, remote_version, log=log):
                log('update installed -- restarting into new version')
                os.execv(sys.executable, [sys.executable] + sys.argv)
                # execv never returns on success
        else:
            log(f'worker version {local_version} is out of date (server advertises '
                f'{remote_version}); pass --auto-update --update-url <url> to self-update, '
                f'or update manually')
    return time.time()


def _report_capabilities(client, args):
    """Detects and reports this machine's CPU/RAM/GPU capabilities so the
    server's capability-aware task assignment (assign_next_typed_task) can
    decide whether this worker is eligible for TRAIN_NETWORK tasks. Called
    once at startup; best-effort -- a failure here must not stop the worker
    from doing SELF_PLAY/DATA_GENERATION/ELO_MATCH work, which don't depend
    on capability reporting at all."""
    caps = detect_capabilities(trainer_capable=args.trainer_capable,
                                gpu_name_override=args.gpu_name_override)
    try:
        client.report_capabilities(caps)
        gpu_note = f", GPU: {caps['gpu_name']}" if caps['gpu_available'] else ", no GPU detected"
        trainer_note = ' (trainer-capable)' if caps['trainer_capable'] else ''
        log(f"capabilities reported: {caps['cpu_cores']} cores, {caps['ram_mb']}MB RAM"
            f"{gpu_note}{trainer_note}")
    except Exception as e:
        log(f'WARNING: failed to report capabilities (non-fatal, continuing): {e}')
    return caps


def _dispatch_task(task, client, args, engine_version, monitor):
    """Routes one typed task to its executor. Returns normally on success;
    raises on a hard failure (the task's server-side lease will simply
    expire and get reassigned -- see database.py's _reclaim_expired_leases
    -- which is the same recovery path used for a crashed/disconnected
    worker, not a special case here)."""
    task_type = task['task_type']
    payload = task['payload']

    if task_type == 'SELF_PLAY':
        log(f"got SELF_PLAY task {task['task_id']}: target={payload['target_positions']} "
            f"depth={payload['depth']} randomplies={payload['randomplies']}")
        TaskRunner(client, args.engine_bin, task, args, monitor=monitor).run()

    elif task_type == 'DATA_GENERATION':
        log(f"got DATA_GENERATION task {task['task_id']}: {payload}")
        run_data_generation(task, client, args.engine_bin, engine_version, args, log=log)

    elif task_type == 'ELO_MATCH':
        log(f"got ELO_MATCH task {task['task_id']}: {payload}")
        run_elo_match(task, client, args.engine_bin, args, log=log)

    elif task_type == 'TRAIN_NETWORK':
        # Only ever offered to workers that reported trainer_capable=true
        # (see assign_next_typed_task). Uploads a real, deployable 'network'
        # artifact -- see platform/trainer/train_network.py's module
        # docstring for the full pipeline (tools/nnue_pipeline/train.py +
        # export.py, local load+verify against the compiled engine).
        log(f"got TRAIN_NETWORK task {task['task_id']}: {payload}")
        run_train_network(task, client, args.engine_bin, args, log=log)

    else:
        log(f"WARNING: server assigned unknown task_type {task_type!r} for task "
            f"{task['task_id']!r} -- skipping (lease will expire and it will be reassigned)")


def main():
    args = parse_args()
    state = ensure_registered(args)
    client = PlatformClient(args.server, token=state['worker_token'])
    _report_capabilities(client, args)

    monitor = ResourceMonitor(max_cpu_percent=args.max_cpu_percent,
                               max_memory_mb=args.max_memory_mb,
                               check_interval=args.resource_check_interval, log=log)
    monitor.start()

    log(f"worker {state['worker_id']} online, server={args.server}, threads={args.threads}")
    if args.max_cpu_percent or args.max_memory_mb:
        log(f'resource limits: cpu<={args.max_cpu_percent or "unlimited"}%, '
            f'memory<={args.max_memory_mb or "unlimited"}MB')

    last_update_check = 0.0
    tasks_done = 0
    try:
        while True:
            last_update_check = _maybe_check_for_update(client, args, last_update_check)

            if monitor.should_exit.is_set():
                log('resource cap exceeded before starting a new task -- exiting cleanly')
                return

            try:
                task = client.next_typed_task()
            except ServerUnavailable as e:
                log(f'server unreachable, giving up this attempt: {e}')
                time.sleep(args.poll_interval)
                continue
            except PermissionError as e:
                sys.exit(f'authentication failed ({e}) -- worker token may have been revoked '
                         f'or the account disabled; delete {args.state_file} and re-register.')

            if task is None:
                time.sleep(args.poll_interval)
                continue

            try:
                _dispatch_task(task, client, args, state['engine_version'], monitor)
            except (DataGenerationError, EloMatchError, TrainNetworkError) as e:
                log(f"task {task['task_id']} ({task['task_type']}) failed: {e} -- "
                    f"its lease will expire and it will be reassigned")
            tasks_done += 1

            if args.once:
                log(f'--once given, exiting after {tasks_done} task(s)')
                return
    finally:
        monitor.stop()


if __name__ == '__main__':
    # Without this guard, `python3 platform_worker.py ...` -- exactly what
    # platform/docs/WORKER.md documents for a from-source run, and exactly
    # the entry script .github/workflows/release.yml's PyInstaller freeze
    # targets for the distributed worker.exe/worker binary -- imports every
    # module, defines main(), and then exits 0 having done nothing: no
    # registration, no capability report, no task poll, zero output. Caught
    # by the final release audit's fresh-contributor simulation (a real
    # `python3 platform_worker.py --once` run against a real local server
    # produced no worker registration and left the queued task pending
    # forever), not by the CI smoke test, because `./worker --help` also
    # exits 0 with no output either way -- a passing smoke test that never
    # actually reached argparse. `run_platform_worker.py` already has this
    # guard and works correctly; this makes platform_worker.py itself work
    # the same way instead of silently depending on which of the two ever
    # gets invoked.
    main()
