#!/usr/bin/env python3
"""pipeline.py - The one-command training backend:

    import dataset -> clean/validate/dedupe -> batch -> train ->
    export -> benchmark -> Elo test -> accept/reject -> experiments/net_XXX/

Every stage is implemented in its own module (dataset/, training/,
evaluation/, experiment.py) and reused here, not duplicated. See
docs/TRAINING_SERVER.md for the full walkthrough.

Usage:
    python3 pipeline.py --use-default-distributed-db --engine-bin /path/to/chess \
        --epochs 10 --batch-size 4096

Reuse an already-imported dataset version instead of re-importing:
    python3 pipeline.py --dataset-version v_20260716_ab12cd34 --engine-bin /path/to/chess
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset.import_data import import_and_clean, write_dataset, dataset_version_id
from dataset.batches import make_batches
from training.train import run_training
from evaluation.evaluate import evaluate_experiment
from experiment import create_experiment, save_config


def make_logger(exp_dir):
    log_path = os.path.join(exp_dir, 'logs', 'pipeline.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, 'a')

    def log(msg=''):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        f.write(line + '\n')
        f.flush()
    return log


def get_or_create_dataset(args, log):
    if args.dataset_version:
        ds_dir = os.path.join(config.DATASETS_DIR, args.dataset_version)
        if not os.path.isdir(ds_dir):
            sys.exit(f'--dataset-version {args.dataset_version!r} not found under {config.DATASETS_DIR}')
        log(f'reusing existing dataset version {args.dataset_version}')
        return args.dataset_version, ds_dir

    distributed_dbs = list(args.distributed_db)
    if args.use_default_distributed_db and os.path.isfile(config.DEFAULT_DISTRIBUTED_DB):
        distributed_dbs.append(config.DEFAULT_DISTRIBUTED_DB)
    if not distributed_dbs and not args.jsonl:
        sys.exit('no dataset source given: pass --dataset-version, --distributed-db, --jsonl, '
                 'or --use-default-distributed-db')

    cleaned, stats = import_and_clean(distributed_dbs, args.jsonl, log=log)
    if not cleaned:
        sys.exit('no valid positions after cleaning -- nothing to train on')
    version = dataset_version_id(cleaned)
    ds_dir = os.path.join(config.DATASETS_DIR, version)
    write_dataset(cleaned, stats, ds_dir)
    log(f'dataset version: {version} ({len(cleaned)} positions)')
    return version, ds_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # dataset
    ap.add_argument('--dataset-version', default=None, help='reuse an existing dataset version')
    ap.add_argument('--distributed-db', nargs='*', default=[])
    ap.add_argument('--jsonl', nargs='*', default=[])
    ap.add_argument('--use-default-distributed-db', action='store_true')
    ap.add_argument('--val-fraction', type=float, default=0.02)
    ap.add_argument('--batch-seed', type=int, default=1)
    # training
    ap.add_argument('--engine', choices=['auto', 'reference', 'bullet'], default='auto')
    ap.add_argument('--bullet-dir', default=None)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--resume-checkpoint', default=None)
    # export
    ap.add_argument('--qa', type=int, default=256)
    ap.add_argument('--qb', type=int, default=256)
    # evaluation
    ap.add_argument('--engine-bin', required=True, help='path to the compiled chess binary')
    ap.add_argument('--bin-dir', default=None)
    ap.add_argument('--bench-depth', type=int, default=12)
    ap.add_argument('--match-games', type=int, default=40)
    ap.add_argument('--match-depth', type=int, default=5)
    ap.add_argument('--elo0', type=float, default=0.0)
    ap.add_argument('--elo1', type=float, default=10.0)
    ap.add_argument('--reject-elo-threshold', type=float, default=-15.0,
                     help='reject if candidate Elo vs. baseline is below this')
    ap.add_argument('--baseline-net', default=None,
                     help='override auto-selected baseline (most recent accepted experiment)')
    args = ap.parse_args()

    experiment_id, exp_dir = create_experiment()
    log = make_logger(exp_dir)
    log(f'experiment {experiment_id} started -> {exp_dir}')

    version, ds_dir = get_or_create_dataset(args, log)

    if not os.path.isfile(os.path.join(ds_dir, 'train.jsonl')):
        batches_manifest = make_batches(version, args.batch_size, args.val_fraction, args.batch_seed)
        log(f'batches: {batches_manifest["n_train"]} train / {batches_manifest["n_val"]} val')
    else:
        log('reusing existing train.jsonl/val.jsonl split for this dataset version')

    hyperparams = {
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
        'qa': args.qa, 'qb': args.qb, 'val_fraction': args.val_fraction,
    }
    save_config(exp_dir, dataset_version=version, hyperparams=hyperparams, engine=args.engine,
                extra={'bench_depth': args.bench_depth, 'match_games': args.match_games,
                       'match_depth': args.match_depth})

    log('=== training ===')
    train_result = run_training(
        ds_dir, exp_dir, args.epochs, args.batch_size, args.lr, engine=args.engine,
        resume=args.resume_checkpoint, bullet_dir=args.bullet_dir, seed=args.batch_seed, log=log)
    log(f'training done: engine={train_result["engine"]} elapsed={train_result["elapsed_s"]:.1f}s '
        f'final_metrics={train_result["final_metrics"]}')

    log('=== evaluation (export -> benchmark -> Elo -> accept/reject) ===')
    eval_result = evaluate_experiment(
        train_result['checkpoint'], exp_dir, experiment_id, args.qa, args.qb, args.bin_dir,
        args.bench_depth, args.match_games, args.match_depth, args.elo0, args.elo1,
        args.reject_elo_threshold, baseline_net_override=args.baseline_net,
        training_metrics=train_result['final_metrics'], log=log)

    # Fold training-stage info into config.json's saved record too (dataset
    # version, date, and hyperparams were saved before training; this adds
    # what training actually did, e.g. which engine really ran).
    save_config(exp_dir, dataset_version=version, hyperparams=hyperparams, engine=args.engine,
                extra={'actual_engine': train_result['engine'],
                       'engine_reason': train_result['engine_reason'],
                       'gpu_info': train_result['gpu_info'],
                       'training_elapsed_s': train_result['elapsed_s']})

    log('')
    log(f'=== {experiment_id}: {eval_result["verdict"].upper()} ===')
    log(f'  {eval_result["verdict_reason"]}')
    log(f'  network:  {eval_result["network_file"]}')
    log(f'  config:   {os.path.join(exp_dir, "config.json")}')
    log(f'  results:  {os.path.join(exp_dir, "results.json")}')
    print(experiment_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
