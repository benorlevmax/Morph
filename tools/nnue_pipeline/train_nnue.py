#!/usr/bin/env python3
"""train_nnue.py - The one-command NNUE pipeline:

    generate data -> train network -> export network -> benchmark -> Elo test

Each stage is a normal subprocess call to generate.py / train.py / export.py /
test.py in this directory -- this script does not reimplement their logic, it
just sequences them into one run with a shared run directory for provenance,
and stops at the first stage that fails (a broken checkpoint should not
silently produce a "tested" .nnue).

All engine binaries used (chess_train, chess) must already be built -- see
DEVELOPMENT.md's Build section. This script never invokes the build system and
never touches src/.

Usage (defaults are deliberately small so a first run finishes in minutes,
not hours -- see docs/NNUE_TRAINING.md for realistic full-scale settings):

    python3 train_nnue.py
    python3 train_nnue.py --games 2000 --epochs 10 --max-samples 500000 --match-games 100

Resuming an interrupted run's training stage only:
    python3 train_nnue.py --resume runs/20260101_120000_abcdef/checkpoints/latest.npz \
        --skip-generate --data runs/20260101_120000_abcdef/data.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PIPELINE_DIR, 'runs')


def run_stage(name, argv):
    print(f'\n{"=" * 78}\n[train_nnue] STAGE: {name}\n  $ {" ".join(argv)}\n{"=" * 78}',
          flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable] + argv)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f'\n[train_nnue] STAGE FAILED: {name} (exit {proc.returncode}, {elapsed:.1f}s)')
        sys.exit(proc.returncode)
    print(f'[train_nnue] stage OK: {name} ({elapsed:.1f}s)')
    return elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # generate
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--data', default=None, help='reuse an existing JSONL dataset (implies --skip-generate)')
    ap.add_argument('--games', type=int, default=200, help='self-play games to generate')
    ap.add_argument('--gen-depth', type=int, default=6, help='search depth used to label positions')
    ap.add_argument('--randomplies', type=int, default=6)
    # train
    ap.add_argument('--resume', default=None, help='resume training from a checkpoint .npz')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-samples', type=int, default=200_000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--engine', choices=['reference', 'bullet'], default='reference')
    # export
    ap.add_argument('--qa', type=int, default=256)
    ap.add_argument('--qb', type=int, default=256)
    # benchmark / elo test
    ap.add_argument('--baseline-net', default=None, help='compare against this .nnue instead of classical eval')
    ap.add_argument('--bench-depth', type=int, default=12)
    ap.add_argument('--match-games', type=int, default=24)
    ap.add_argument('--match-depth', type=int, default=5)
    ap.add_argument('--elo0', type=float, default=0.0)
    ap.add_argument('--elo1', type=float, default=10.0)
    ap.add_argument('--skip-match', action='store_true')
    # shared
    ap.add_argument('--bin-dir', default=None, help='engine build bin/ directory')
    ap.add_argument('--run-id', default=None)
    args = ap.parse_args()

    run_id = args.run_id or time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    data_path = args.data or os.path.join(run_dir, 'data.jsonl')
    net_path = os.path.join(run_dir, 'net.nnue')
    report_path = os.path.join(run_dir, 'test_report.json')

    print(f'[train_nnue] run_id={run_id}')
    print(f'[train_nnue] run directory: {run_dir}')

    stage_times = {}
    bin_dir_args = ['--bin-dir', args.bin_dir] if args.bin_dir else []

    # --- Stage 1: generate ---
    if args.data or args.skip_generate:
        if not os.path.isfile(data_path):
            sys.exit(f'[train_nnue] --skip-generate/--data given but {data_path} does not exist')
        print(f'[train_nnue] skipping generate; using existing dataset {data_path}')
    else:
        stage_times['generate'] = run_stage('generate', [
            os.path.join(PIPELINE_DIR, 'generate.py'),
            '--games', str(args.games), '--depth', str(args.gen_depth),
            '--randomplies', str(args.randomplies), '--out', data_path,
        ] + bin_dir_args)

    # --- Stage 2: train ---
    train_argv = [
        os.path.join(PIPELINE_DIR, 'train.py'),
        '--data', data_path, '--out', ckpt_dir,
        '--epochs', str(args.epochs), '--max-samples', str(args.max_samples),
        '--batch-size', str(args.batch_size), '--lr', str(args.lr),
        '--engine', args.engine,
    ]
    if args.resume:
        train_argv += ['--resume', args.resume]
    stage_times['train'] = run_stage('train', train_argv)

    latest_ckpt = os.path.join(ckpt_dir, 'latest.npz')
    if not os.path.isfile(latest_ckpt):
        sys.exit(f'[train_nnue] expected checkpoint {latest_ckpt} was not produced')

    # --- Stage 3: export ---
    stage_times['export'] = run_stage('export', [
        os.path.join(PIPELINE_DIR, 'export.py'),
        '--checkpoint', latest_ckpt, '--out', net_path,
        '--qa', str(args.qa), '--qb', str(args.qb),
    ])

    # --- Stage 4: benchmark + Elo test ---
    test_argv = [
        os.path.join(PIPELINE_DIR, 'test.py'),
        '--net', net_path, '--bench-depth', str(args.bench_depth),
        '--games', str(args.match_games), '--match-depth', str(args.match_depth),
        '--elo0', str(args.elo0), '--elo1', str(args.elo1),
        '--report', report_path,
    ] + bin_dir_args
    if args.baseline_net:
        test_argv += ['--baseline-net', args.baseline_net]
    if args.skip_match:
        test_argv += ['--skip-match']
    stage_times['test'] = run_stage('benchmark + elo test', test_argv)

    summary = {
        'run_id': run_id, 'run_dir': os.path.abspath(run_dir),
        'data': os.path.abspath(data_path), 'checkpoint': os.path.abspath(latest_ckpt),
        'net': os.path.abspath(net_path), 'report': os.path.abspath(report_path),
        'stage_times_s': stage_times, 'total_s': sum(stage_times.values()),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    with open(os.path.join(run_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"=" * 78}\n[train_nnue] PIPELINE COMPLETE\n{"=" * 78}')
    print(f'  net:    {net_path}')
    print(f'  report: {report_path}')
    print(f'  total time: {summary["total_s"]:.1f}s')
    print(f'\nLoad it in the engine with:')
    print(f'  setoption name EvalFile value {net_path}')
    print(f'  setoption name Use NNUE value true')
    return 0


if __name__ == '__main__':
    sys.exit(main())
