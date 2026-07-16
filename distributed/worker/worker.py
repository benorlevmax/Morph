#!/usr/bin/env python3
"""worker.py - Main worker loop: register (once), then repeatedly fetch a
task, generate positions via self-play across --threads concurrent engine
instances, upload in batches (progress reporting + resilience against losing
a big batch to one failed upload), and move on to the next task.

Flow per the spec:
    connects to server -> receives task -> runs engine self-play ->
    creates training positions -> uploads results
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import Client, ServerUnavailable
from config import parse_args, load_state, save_state
from selfplay import SelfPlayEngine, play_selfplay_game


def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def ensure_registered(args):
    state = None if args.force_register else load_state(args.state_file)
    if state and state.get('worker_id') and state.get('worker_token'):
        log(f"using saved credentials for worker {state['worker_id']} ({args.state_file})")
        return state

    if not args.registration_secret:
        sys.exit(
            f'no saved credentials at {args.state_file} and --registration-secret was not '
            f'given -- pass the secret printed by the server at startup for first-time setup.')

    client = Client(args.server)
    engine_version = _probe_engine_version(args.engine_bin)
    log(f'registering with {args.server} as {args.hostname!r} (engine {engine_version!r})')
    resp = client.register(args.registration_secret, args.hostname, engine_version, args.threads)
    state = {'server': args.server, 'worker_id': resp['worker_id'],
              'worker_token': resp['worker_token'], 'hostname': args.hostname,
              'engine_version': engine_version}
    save_state(args.state_file, state)
    log(f"registered as {resp['worker_id']}; credentials saved to {args.state_file}")
    return state


def _probe_engine_version(engine_bin):
    eng = SelfPlayEngine(engine_bin, depth=1)
    v = eng.engine_version
    eng.close()
    return v


class TaskRunner:
    """Runs --threads concurrent self-play generators against one task,
    uploading in batches until the task's target is met (as reported back by
    the server after each upload) or the server reassigns the task elsewhere."""

    def __init__(self, client, engine_bin, task, args):
        self.client = client
        self.engine_bin = engine_bin
        self.task = task
        self.args = args
        self.record_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.generated_count = 0
        self.lock = threading.Lock()

    def _worker_thread(self, thread_idx):
        rng_seed = hash((self.task['task_id'], thread_idx, time.time())) & 0xFFFFFFFF
        import random
        rng = random.Random(rng_seed)
        engine = SelfPlayEngine(self.engine_bin, depth=self.task['depth'],
                                 hash_mb=self.args.hash_mb)
        try:
            while not self.stop_event.is_set():
                records = play_selfplay_game(
                    engine, randomplies=self.task['randomplies'],
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
        target = self.task['target_positions']
        total_accepted = 0
        last_progress = time.time()
        batch = []

        try:
            while not self.stop_event.is_set():
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


def main():
    args = parse_args()
    state = ensure_registered(args)
    client = Client(args.server, token=state['worker_token'])

    log(f"worker {state['worker_id']} online, server={args.server}, threads={args.threads}")

    tasks_done = 0
    while True:
        try:
            task = client.next_task()
        except ServerUnavailable as e:
            log(f'server unreachable, giving up this attempt: {e}')
            time.sleep(args.poll_interval)
            continue
        except PermissionError as e:
            sys.exit(f'authentication failed ({e}) -- worker token may have been revoked; '
                     f'delete {args.state_file} and re-register with a valid secret.')

        if task is None:
            time.sleep(args.poll_interval)
            continue

        log(f"got task {task['task_id']}: target={task['target_positions']} "
            f"depth={task['depth']} randomplies={task['randomplies']}")
        runner = TaskRunner(client, args.engine_bin, task, args)
        runner.run()
        tasks_done += 1

        if args.once:
            log(f'--once given, exiting after {tasks_done} task(s)')
            return


if __name__ == '__main__':
    main()
