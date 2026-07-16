#!/usr/bin/env python3
"""test_local.py - Local integration test: one server + multiple workers.

Spins up the FastAPI server and N worker processes as subprocesses (all on
localhost, no public deployment), creates a bulk task, waits for the workers
to fill it, then asserts the server's stats make sense and shuts everything
down cleanly. This is the "test locally with one server, multiple workers"
requirement, as a re-runnable script rather than a one-off manual session.

Usage:
    python3 test_local.py --engine-bin /path/to/build/bin/chess [--workers 3]
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

DIST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(DIST_DIR, 'server')
WORKER_DIR = os.path.join(DIST_DIR, 'worker')


def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url, body, headers=None, timeout=10):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method='POST',
                                  headers={'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_server(base_url, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_get(f'{base_url}/health')
            return True
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine-bin', required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--port', type=int, default=8199)
    ap.add_argument('--total-positions', type=int, default=90)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--threads-per-worker', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=180)
    args = ap.parse_args()

    if not os.path.isfile(args.engine_bin):
        sys.exit(f'--engine-bin {args.engine_bin!r} does not exist')

    workdir = tempfile.mkdtemp(prefix='chess_dist_test_')
    db_path = os.path.join(workdir, 'test.sqlite3')
    reg_secret = 'local-test-secret'
    admin_token = 'local-test-admin'
    base_url = f'http://127.0.0.1:{args.port}'

    env = dict(os.environ)
    env['CHESS_DIST_DB_PATH'] = db_path
    env['CHESS_DIST_REGISTRATION_SECRET'] = reg_secret
    env['CHESS_DIST_ADMIN_TOKEN'] = admin_token

    procs = []

    def cleanup():
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()

    try:
        print(f'[test] work dir: {workdir}')
        print(f'[test] starting server on {base_url} ...')
        server_log = open(os.path.join(workdir, 'server.log'), 'w')
        server = subprocess.Popen(
            [sys.executable, os.path.join(SERVER_DIR, 'run_server.py'), '--port', str(args.port)],
            cwd=SERVER_DIR, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        procs.append(server)

        if not wait_for_server(base_url):
            print('[test] FAIL: server did not become healthy in time')
            print(open(os.path.join(workdir, 'server.log')).read())
            sys.exit(1)
        print('[test] server healthy')

        print(f'[test] creating task: {args.total_positions} positions, depth {args.depth}')
        created = http_post(f'{base_url}/admin/tasks', {
            'total_positions': args.total_positions, 'depth': args.depth, 'randomplies': 4,
            'chunk_size': max(1, args.total_positions // args.workers),
        }, headers={'X-Admin-Token': admin_token})
        print(f'[test] created {len(created["task_ids"])} task(s): {created["task_ids"]}')

        print(f'[test] starting {args.workers} worker(s) ...')
        for i in range(args.workers):
            state_file = os.path.join(workdir, f'worker{i}_state.json')
            log_file = open(os.path.join(workdir, f'worker{i}.log'), 'w')
            w = subprocess.Popen([
                sys.executable, os.path.join(WORKER_DIR, 'run_worker.py'),
                '--server', base_url, '--engine-bin', args.engine_bin,
                '--registration-secret', reg_secret, '--hostname', f'test-worker-{i}',
                '--threads', str(args.threads_per_worker), '--upload-batch-size', '10',
                '--max-plies', '20', '--poll-interval', '1', '--state-file', state_file,
                '--once',
            ], env=env, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append(w)
        worker_procs = procs[1:]

        print('[test] waiting for workers to finish their task(s) ...')
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            if all(p.poll() is not None for p in worker_procs):
                break
            time.sleep(1)
        else:
            print('[test] FAIL: timed out waiting for workers to finish')
            sys.exit(1)

        failed = [i for i, p in enumerate(worker_procs) if p.returncode != 0]
        if failed:
            print(f'[test] FAIL: worker(s) {failed} exited non-zero; see logs in {workdir}')
            for i in failed:
                print(f'--- worker{i}.log ---')
                print(open(os.path.join(workdir, f'worker{i}.log')).read())
            sys.exit(1)

        stats = http_get(f'{base_url}/stats')
        workers_list = http_get(f'{base_url}/workers')

        print('[test] --- final stats ---')
        print(json.dumps(stats, indent=2))

        ok = True
        if stats['total_positions'] < args.total_positions:
            print(f'[test] FAIL: total_positions {stats["total_positions"]} < requested '
                  f'{args.total_positions}')
            ok = False
        if stats['total_workers'] != args.workers:
            print(f'[test] FAIL: expected {args.workers} registered workers, '
                  f'got {stats["total_workers"]}')
            ok = False
        if stats['tasks_by_status'].get('completed', 0) < len(created['task_ids']):
            print(f'[test] FAIL: not all tasks completed: {stats["tasks_by_status"]}')
            ok = False
        contributing_workers = sum(1 for w in workers_list if w['positions_generated'] > 0)
        if contributing_workers < 1:
            print('[test] FAIL: no worker shows any positions_generated')
            ok = False

        if ok:
            print(f'[test] PASS: {stats["total_positions"]} positions from '
                  f'{contributing_workers}/{args.workers} contributing workers, '
                  f'{stats["tasks_by_status"]}')
        else:
            sys.exit(1)
    finally:
        cleanup()
        print(f'[test] logs kept at {workdir}')


if __name__ == '__main__':
    main()
