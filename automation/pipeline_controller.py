#!/usr/bin/env python3
"""pipeline_controller.py - The continuous self-improvement loop:

    generate data -> train NNUE -> test Elo -> keep improvements -> repeat

This is an orchestration layer on top of already-verified pieces -- it does
not reimplement any of them:

  * data generation:  tools/nnue_pipeline/generate.py (local self-play) and/or
                       distributed/ (networked worker positions, counted from
                       its SQLite database)
  * training:          training_server/pipeline.py, which itself chains
                       dataset import/dedup/validate -> batch -> train
                       (GPU Bullet or CPU reference, checkpoint/resume) ->
                       export -> benchmark -> SPRT Elo match -> accept/reject
  * bookkeeping:       automation/models_registry.py (models/current,
                       candidates, rejected) and automation/state.py
                       (crash-safe cycle state)

Nothing here touches src/search, src/eval, or src/nnue's inference code --
the controller only shells out to the existing, unmodified `chess` /
`chess_train` binaries and to the already-tested Python pipeline stages.

Scope note (explicit, per the request to keep this internal-only for now):
this drives a SINGLE machine's training loop. distributed/ workers are
long-running networked services meant to be started once and left running
independently (see docs/DISTRIBUTED_DATA_GENERATION.md) -- this controller
does not spawn or manage them, it only reads how many positions they've
produced so far from their database. Local top-up generation
(--auto-generate) is the one data-producing subprocess this controller
does own the lifecycle of, since that's fully containable on one machine.
No public-facing website or API is part of this; see
docs/SELF_IMPROVEMENT_LOOP.md.

Usage (single cycle, for testing on a small dataset):
    python3 automation/pipeline_controller.py --once \
        --engine-bin build/bin/Release/chess.exe \
        --auto-generate --generate-games 50 --generate-depth 4 \
        --min-new-positions 200 --epochs 1 --match-games 8

Usage (continuous daemon):
    python3 automation/pipeline_controller.py --loop --interval-seconds 3600 \
        --engine-bin build/bin/Release/chess.exe --use-default-distributed-db \
        --auto-generate --min-new-positions 20000 --epochs 10
"""
import argparse
import glob
import logging
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import state as state_mod
import models_registry
from notify import notify
from logging_setup import setup_logging

log = logging.getLogger('automation.controller')

EXPERIMENT_ID_RE = re.compile(r'^net_\d+$')

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    log.warning('received signal %s -- will stop after the current cycle finishes', signum)
    _stop_requested = True


# --------------------------------------------------------------------------
# Stage 1: data collection / monitoring
# --------------------------------------------------------------------------

def count_distributed_positions(db_paths):
    total = 0
    for db_path in db_paths:
        if not os.path.isfile(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            try:
                total += conn.execute('SELECT COUNT(*) FROM positions').fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error as e:
            log.warning('could not read %s: %s', db_path, e)
    return total


def count_jsonl_positions(paths):
    total = 0
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            total += sum(1 for line in f if line.strip())
    return total


def resolve_distributed_dbs(args):
    dbs = list(args.distributed_db)
    if args.use_default_distributed_db and os.path.isfile(config.DEFAULT_DISTRIBUTED_DB):
        dbs.append(config.DEFAULT_DISTRIBUTED_DB)
    return dbs


def resolve_jsonl_files(args):
    files = []
    for pattern in args.jsonl:
        matches = glob.glob(pattern)
        files.extend(matches if matches else [pattern])
    if args.auto_generate and os.path.isfile(config.AUTO_GENERATED_JSONL):
        files.append(config.AUTO_GENERATED_JSONL)
    return files


def measure_available_positions(args):
    dbs = resolve_distributed_dbs(args)
    jsonl_files = resolve_jsonl_files(args)
    total = count_distributed_positions(dbs) + count_jsonl_positions(jsonl_files)
    return total


def run_local_generation(args, games):
    """Top up automation/generated/auto_positions.jsonl with `games` more
    self-play games via the existing, unmodified generate.py (which itself
    just drives the compiled, unmodified engine binary). Returns the number
    of positions written."""
    os.makedirs(config.GENERATED_DIR, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(config.NNUE_PIPELINE_DIR, 'generate.py'),
        '--games', str(games), '--depth', str(args.generate_depth),
        '--randomplies', str(args.generate_randomplies),
        '--out', config.AUTO_GENERATED_JSONL, '--append',
    ]
    if args.bin_dir:
        cmd += ['--bin-dir', args.bin_dir]
    log.info('[generate] $ %s', ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        log.info('[generate] %s', line)
    if proc.returncode != 0:
        log.error('[generate] failed (exit %s): %s', proc.returncode, proc.stderr[-2000:])
        raise RuntimeError(f'local generation failed (exit {proc.returncode})')
    return proc.stdout


def collect_data(args, state):
    """Ensure enough NEW data exists (beyond state['dataset_watermark'])
    before training. Returns the total available position count. Raises
    RuntimeError if data stays insufficient after --auto-generate is
    exhausted (or isn't enabled) so the caller can skip this cycle."""
    total = measure_available_positions(args)
    new_positions = total - state['dataset_watermark']
    log.info('[collect] %d positions available, %d new since last training (need %d)',
              total, new_positions, args.min_new_positions)

    if not args.auto_generate:
        return total

    rounds = 0
    while new_positions < args.min_new_positions and rounds < args.max_generate_rounds:
        rounds += 1
        log.info('[collect] short by %d positions -- running local generation round %d/%d',
                  args.min_new_positions - new_positions, rounds, args.max_generate_rounds)
        run_local_generation(args, args.generate_games)
        total = measure_available_positions(args)
        new_positions = total - state['dataset_watermark']

    return total


# --------------------------------------------------------------------------
# Stage 2: training_server/pipeline.py invocation
# --------------------------------------------------------------------------

def build_pipeline_command(args):
    cmd = [sys.executable, os.path.join(config.TRAINING_SERVER_DIR, 'pipeline.py')]
    for db in args.distributed_db:
        cmd += ['--distributed-db', db]
    if args.use_default_distributed_db:
        cmd += ['--use-default-distributed-db']
    for jf in resolve_jsonl_files(args):
        cmd += ['--jsonl', jf]
    cmd += ['--val-fraction', str(args.val_fraction), '--batch-seed', str(args.batch_seed)]
    cmd += ['--engine', args.engine, '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size), '--lr', str(args.lr)]
    if args.bullet_dir:
        cmd += ['--bullet-dir', args.bullet_dir]
    cmd += ['--qa', str(args.qa), '--qb', str(args.qb)]
    cmd += ['--engine-bin', args.engine_bin]
    if args.bin_dir:
        cmd += ['--bin-dir', args.bin_dir]
    cmd += ['--bench-depth', str(args.bench_depth), '--match-games', str(args.match_games),
            '--match-depth', str(args.match_depth), '--elo0', str(args.elo0),
            '--elo1', str(args.elo1), '--reject-elo-threshold', str(args.reject_elo_threshold)]
    if args.baseline_net:
        cmd += ['--baseline-net', args.baseline_net]
    return cmd


def run_training_pipeline(args):
    """Runs training_server/pipeline.py as a subprocess, streaming its
    output into our own log, and returns the experiment_id it printed as
    its last line. Raises RuntimeError on any failure (non-zero exit,
    missing/garbled experiment id)."""
    cmd = build_pipeline_command(args)
    log.info('[train] $ %s', ' '.join(cmd))
    proc = subprocess.Popen(cmd, cwd=config.TRAINING_SERVER_DIR, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_lines = []
    for line in proc.stdout:
        line = line.rstrip('\n')
        log.info('[pipeline] %s', line)
        if line.strip():
            last_lines.append(line.strip())
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f'training_server/pipeline.py failed (exit {proc.returncode})')
    if not last_lines or not EXPERIMENT_ID_RE.match(last_lines[-1]):
        raise RuntimeError(
            f'training_server/pipeline.py exited 0 but did not print a valid experiment id '
            f'(last output line: {last_lines[-1] if last_lines else "<empty>"})')
    return last_lines[-1]


def with_retries(fn, max_retries, backoff_seconds, description):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = backoff_seconds * attempt
            log.warning('[retry] %s failed (attempt %d/%d): %s -- retrying in %ds',
                        description, attempt, max_retries + 1, e, wait)
            time.sleep(wait)


# --------------------------------------------------------------------------
# Stage 3: reading results + promote/discard
# --------------------------------------------------------------------------

def load_experiment(experiment_id):
    import json
    exp_dir = os.path.join(config.EXPERIMENTS_DIR, experiment_id)
    with open(os.path.join(exp_dir, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(exp_dir, 'results.json')) as f:
        results = json.load(f)
    return exp_dir, cfg, results


def act_on_result(experiment_id, exp_dir, cfg, results):
    network_path = results.get('network_file')
    if not network_path or not os.path.isfile(network_path):
        raise RuntimeError(f'{experiment_id}: results.json has no valid network_file')

    candidate_path = models_registry.stage_candidate(experiment_id, network_path)
    models_registry.record_results_index(experiment_id, results)

    verdict = results.get('verdict')
    elo_match = (results.get('test_report') or {}).get('elo_match') or {}
    elo = elo_match.get('elo')
    sprt_verdict = (elo_match.get('sprt') or {}).get('verdict')

    if verdict == 'accept':
        dest = models_registry.promote(experiment_id, candidate_path, results, cfg)
        notify('promoted',
               f'{experiment_id} promoted to models/current/ (Elo {elo}, SPRT {sprt_verdict})',
               experiment_id=experiment_id, elo=elo, sprt_verdict=sprt_verdict,
               network_file=dest, verdict_reason=results.get('verdict_reason'))
        return 'accept', dest
    else:
        dest = models_registry.reject(experiment_id, candidate_path, results)
        notify('rejected',
               f'{experiment_id} rejected, moved to models/rejected/ (Elo {elo}, '
               f'SPRT {sprt_verdict}): {results.get("verdict_reason")}',
               experiment_id=experiment_id, elo=elo, sprt_verdict=sprt_verdict,
               network_file=dest, verdict_reason=results.get('verdict_reason'))
        return 'reject', dest


# --------------------------------------------------------------------------
# One full cycle
# --------------------------------------------------------------------------

def run_cycle(args):
    state = state_mod.touch_cycle_start()
    cycle_n = state['cycle_count'] + 1
    log.info('=== cycle %d starting ===', cycle_n)
    notify('cycle_start', f'cycle {cycle_n} starting')

    try:
        state = state_mod.update(status='collecting')
        total = with_retries(lambda: collect_data(args, state), args.max_retries,
                              args.retry_backoff_seconds, 'data collection')
        new_positions = total - state['dataset_watermark']
        if new_positions < args.min_new_positions:
            log.info('[collect] still only %d new positions (need %d) -- skipping training '
                     'this cycle', new_positions, args.min_new_positions)
            notify('cycle_skipped',
                   f'cycle {cycle_n}: insufficient new data ({new_positions}/'
                   f'{args.min_new_positions})', new_positions=new_positions)
            state_mod.update(status='idle', cycle_count=cycle_n,
                              last_cycle_finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                                    time.gmtime()),
                              consecutive_failures=0)
            return

        state = state_mod.update(status='training')
        experiment_id = with_retries(lambda: run_training_pipeline(args), args.max_retries,
                                      args.retry_backoff_seconds, 'training pipeline')
        log.info('[train] experiment %s finished', experiment_id)

        state = state_mod.update(status='evaluating', last_experiment_id=experiment_id)
        exp_dir, cfg, results = load_experiment(experiment_id)

        state = state_mod.update(status='promoting')
        verdict, dest = act_on_result(experiment_id, exp_dir, cfg, results)

        totals = {'total_promoted': state['total_promoted'] + (1 if verdict == 'accept' else 0),
                  'total_rejected': state['total_rejected'] + (1 if verdict == 'reject' else 0)}
        state_mod.update(status='idle', cycle_count=cycle_n, last_verdict=verdict,
                          last_error=None, consecutive_failures=0,
                          dataset_watermark=total,
                          last_cycle_finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                                time.gmtime()),
                          **totals)
        log.info('=== cycle %d done: %s (%s) ===', cycle_n, experiment_id, verdict)

    except Exception as e:
        log.exception('cycle %d failed', cycle_n)
        failures = state.get('consecutive_failures', 0) + 1
        state_mod.update(status='failed', cycle_count=cycle_n, last_error=str(e),
                          consecutive_failures=failures,
                          last_cycle_finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                                time.gmtime()))
        notify('cycle_failed', f'cycle {cycle_n} failed: {e}', consecutive_failures=failures)
        raise


# --------------------------------------------------------------------------
# Startup crash recovery
# --------------------------------------------------------------------------

def recover_incomplete_cycle():
    """If the controller was killed mid-cycle, state.json is left with
    status in {'training', 'evaluating', 'promoting'} instead of the
    terminal 'idle'/'failed' -- meaning the last experiment's candidate
    network may be sitting in models/candidates/ without ever having been
    promoted or rejected. Finish acting on it (idempotently) before
    starting any new cycle, rather than leaving it stranded and silently
    training a new experiment on top.

    This is safe to call every startup: if the experiment was already
    fully placed into models/current/ or models/rejected/, it's a no-op.
    """
    state = state_mod.load()
    experiment_id = state.get('last_experiment_id')
    if state.get('status') not in ('training', 'evaluating', 'promoting') or not experiment_id:
        return

    exp_dir = os.path.join(config.EXPERIMENTS_DIR, experiment_id)
    already_placed = (
        os.path.isfile(os.path.join(config.MODELS_CURRENT_DIR, f'{experiment_id}.nnue')) or
        os.path.isfile(os.path.join(config.MODELS_REJECTED_DIR, f'{experiment_id}.nnue')))
    if already_placed:
        log.info('[recover] %s was already fully placed -- clearing stale status', experiment_id)
        state_mod.update(status='idle')
        return

    results_path = os.path.join(exp_dir, 'results.json')
    config_path = os.path.join(exp_dir, 'config.json')
    if not (os.path.isdir(exp_dir) and os.path.isfile(results_path) and
            os.path.isfile(config_path)):
        log.warning('[recover] %s has no results.json/config.json yet -- training itself was '
                    'interrupted before a verdict existed; nothing to recover, next cycle starts '
                    'a fresh experiment', experiment_id)
        state_mod.update(status='failed',
                          last_error=f'{experiment_id} interrupted before results.json existed')
        return

    log.warning('[recover] resuming interrupted cycle for %s (was stuck at status=%s)',
                experiment_id, state['status'])
    notify('cycle_recovered',
           f'controller restarted mid-cycle; finishing {experiment_id} (was stuck at '
           f'status={state["status"]})', experiment_id=experiment_id)

    exp_dir, cfg, results = load_experiment(experiment_id)
    verdict, dest = act_on_result(experiment_id, exp_dir, cfg, results)
    state_mod.update(
        status='idle', last_verdict=verdict, last_error=None, consecutive_failures=0,
        total_promoted=state['total_promoted'] + (1 if verdict == 'accept' else 0),
        total_rejected=state['total_rejected'] + (1 if verdict == 'reject' else 0),
        last_cycle_finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log.info('[recover] %s finalized as %s', experiment_id, verdict)


# --------------------------------------------------------------------------
# CLI / main loop
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # loop control
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='run a single cycle and exit (default)')
    mode.add_argument('--loop', action='store_true', help='run continuously')
    ap.add_argument('--interval-seconds', type=int, default=3600,
                     help='sleep between cycles in --loop mode')
    ap.add_argument('--max-cycles', type=int, default=0, help='0 = unlimited (--loop mode)')
    ap.add_argument('--max-retries', type=int, default=2,
                     help='retries per stage before the cycle is marked failed')
    ap.add_argument('--retry-backoff-seconds', type=int, default=30)
    ap.add_argument('--max-consecutive-failures', type=int, default=3,
                     help='stop the daemon (not just skip a cycle) after this many cycles in a row fail')
    ap.add_argument('--verbose', action='store_true')

    # data collection
    ap.add_argument('--distributed-db', nargs='*', default=[])
    ap.add_argument('--use-default-distributed-db', action='store_true')
    ap.add_argument('--jsonl', nargs='*', default=[],
                     help='static JSONL dataset file(s)/glob(s) always included')
    ap.add_argument('--auto-generate', action='store_true',
                     help='run local self-play generation to top up data when short')
    ap.add_argument('--generate-games', type=int, default=200)
    ap.add_argument('--generate-depth', type=int, default=6)
    ap.add_argument('--generate-randomplies', type=int, default=6)
    ap.add_argument('--max-generate-rounds', type=int, default=5)
    ap.add_argument('--min-new-positions', type=int, default=2000)

    # training
    ap.add_argument('--engine', choices=['auto', 'reference', 'bullet'], default='auto')
    ap.add_argument('--bullet-dir', default=None)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--val-fraction', type=float, default=0.02)
    ap.add_argument('--batch-seed', type=int, default=1)

    # export
    ap.add_argument('--qa', type=int, default=256)
    ap.add_argument('--qb', type=int, default=256)

    # evaluation
    ap.add_argument('--engine-bin', required=True, help='path to the compiled chess UCI binary')
    ap.add_argument('--bin-dir', default=None)
    ap.add_argument('--bench-depth', type=int, default=12)
    ap.add_argument('--match-games', type=int, default=40)
    ap.add_argument('--match-depth', type=int, default=5)
    ap.add_argument('--elo0', type=float, default=0.0)
    ap.add_argument('--elo1', type=float, default=10.0)
    ap.add_argument('--reject-elo-threshold', type=float, default=-15.0)
    ap.add_argument('--baseline-net', default=None,
                     help='override auto-selected baseline (default: models/current/current.nnue '
                          'lineage via experiments/, handled inside training_server)')

    return ap.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()
    setup_logging(verbose=args.verbose)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        recover_incomplete_cycle()
    except Exception:
        log.exception('crash recovery itself failed -- continuing to a fresh cycle anyway')

    loop_mode = args.loop  # --once is the implicit default
    cycles_run = 0

    while True:
        state = state_mod.load()
        if state['consecutive_failures'] >= args.max_consecutive_failures:
            log.error('halting: %d consecutive cycle failures (>= --max-consecutive-failures=%d). '
                      'Fix the underlying issue and restart the controller.',
                      state['consecutive_failures'], args.max_consecutive_failures)
            notify('loop_halted',
                   f'controller halted after {state["consecutive_failures"]} consecutive failures',
                   last_error=state.get('last_error'))
            return 1

        try:
            run_cycle(args)
        except Exception:
            pass  # already logged + notified inside run_cycle; keep the daemon alive

        cycles_run += 1
        if not loop_mode:
            break
        if args.max_cycles and cycles_run >= args.max_cycles:
            log.info('reached --max-cycles=%d, stopping', args.max_cycles)
            break
        if _stop_requested:
            log.info('stop requested, exiting after this cycle')
            break

        log.info('sleeping %ds until next cycle', args.interval_seconds)
        for _ in range(args.interval_seconds):
            if _stop_requested:
                break
            time.sleep(1)
        if _stop_requested:
            break

    return 0


if __name__ == '__main__':
    sys.exit(main())
