#!/usr/bin/env python3
"""train.py - Training stage: run tools/nnue_pipeline/train.py against a
versioned dataset's train/val split, inside an experiment folder.

Deliberately a thin wrapper (subprocess call), not a re-implementation: the
NumPy reference trainer (CPU fallback) and the real-Bullet shell-out (GPU
acceleration) both already exist, are already checkpoint/resume-capable, and
were already verified end-to-end in tools/nnue_pipeline/. This module's job
is just: pick an engine (training_server/training/gpu.py), point the
existing trainer at a dataset version's train.jsonl/val.jsonl (produced by
training_server/dataset/batches.py), and put its checkpoints + metrics under
the current experiment's folder.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from gpu import recommend_engine, gpu_info, has_cargo


def run_training(dataset_dir, experiment_dir, epochs, batch_size, lr, engine='auto',
                 resume=None, bullet_dir=None, seed=1, log=print):
    train_file = os.path.join(dataset_dir, 'train.jsonl')
    val_manifest_path = os.path.join(dataset_dir, 'batches_manifest.json')
    if not os.path.isfile(train_file):
        raise FileNotFoundError(f'{train_file} not found -- run dataset/batches.py first')

    chosen_engine, reason = recommend_engine(engine)
    log(f'[train] engine={chosen_engine} ({reason})')

    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(experiment_dir, 'logs', 'train.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(val_manifest_path) as f:
        n_train = json.load(f)['n_train']

    cmd = [
        sys.executable, os.path.join(config.NNUE_PIPELINE_DIR, 'train.py'),
        '--data', train_file, '--out', ckpt_dir,
        '--epochs', str(epochs), '--batch-size', str(batch_size), '--lr', str(lr),
        '--max-samples', str(n_train), '--seed', str(seed), '--engine', chosen_engine,
    ]
    if resume:
        cmd += ['--resume', resume]
    if chosen_engine == 'bullet' and bullet_dir:
        cmd += ['--bullet-dir', bullet_dir]

    log(f'[train] $ {" ".join(cmd)}')
    t0 = time.time()
    with open(log_path, 'a') as logf:
        logf.write(f'\n=== run at {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} ===\n')
        logf.write(' '.join(cmd) + '\n')
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(f'training failed (exit {proc.returncode}); see {log_path}')

    latest_ckpt = os.path.join(ckpt_dir, 'latest.npz')
    if not os.path.isfile(latest_ckpt):
        raise RuntimeError(f'training exited 0 but {latest_ckpt} was not produced; see {log_path}')

    metrics_path = os.path.join(ckpt_dir, 'metrics.jsonl')
    final_metrics = None
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metric_lines = [json.loads(l) for l in f if l.strip()]
        if metric_lines:
            final_metrics = metric_lines[-1]

    return {
        'engine': chosen_engine, 'engine_reason': reason,
        'checkpoint': os.path.abspath(latest_ckpt), 'elapsed_s': elapsed,
        'epochs_requested': epochs, 'batch_size': batch_size, 'lr': lr,
        'final_metrics': final_metrics, 'log_file': os.path.abspath(log_path),
        'gpu_info': gpu_info(), 'cargo_available': has_cargo(),
    }
