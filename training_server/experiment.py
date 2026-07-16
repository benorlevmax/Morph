#!/usr/bin/env python3
"""experiment.py - experiments/net_XXX/ folder lifecycle.

Each experiment folder is the complete, self-contained record of one
training run:

    experiments/net_001/
        config.json      training configuration (hyperparams, dataset version, engine, date)
        logs/             training.log, per-epoch metrics.jsonl
        network/          net_001.nnue (the exported, quantized network)
        results.json      validation score, benchmark results, Elo match, accept/reject verdict

`config.json` and `results.json` together cover every field the spec
requires every run to save: network file (network/), training configuration
(config.json), dataset version (config.json's `dataset_version`), date
(config.json's `created_at`), validation score and benchmark results
(results.json).
"""
import json
import os
import re
import time

from config import EXPERIMENTS_DIR


def _existing_experiment_numbers():
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    nums = []
    for name in os.listdir(EXPERIMENTS_DIR):
        m = re.match(r'^net_(\d+)$', name)
        if m:
            nums.append(int(m.group(1)))
    return nums


def allocate_experiment_id():
    nums = _existing_experiment_numbers()
    next_n = (max(nums) + 1) if nums else 1
    return f'net_{next_n:03d}'


def create_experiment(experiment_id=None):
    experiment_id = experiment_id or allocate_experiment_id()
    exp_dir = os.path.join(EXPERIMENTS_DIR, experiment_id)
    if os.path.exists(exp_dir):
        raise FileExistsError(f'{exp_dir} already exists')
    os.makedirs(exp_dir)
    os.makedirs(os.path.join(exp_dir, 'logs'))
    os.makedirs(os.path.join(exp_dir, 'network'))
    os.makedirs(os.path.join(exp_dir, 'checkpoints'))
    return experiment_id, exp_dir


def save_config(exp_dir, dataset_version, hyperparams, engine, extra=None):
    cfg = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'dataset_version': dataset_version,
        'engine': engine,
        'hyperparams': hyperparams,
    }
    if extra:
        cfg.update(extra)
    path = os.path.join(exp_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    return path


def save_results(exp_dir, results):
    path = os.path.join(exp_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    return path


def list_experiments():
    out = []
    if not os.path.isdir(EXPERIMENTS_DIR):
        return out
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        if not re.match(r'^net_\d+$', name):
            continue
        exp_dir = os.path.join(EXPERIMENTS_DIR, name)
        entry = {'id': name, 'dir': exp_dir}
        cfg_path = os.path.join(exp_dir, 'config.json')
        res_path = os.path.join(exp_dir, 'results.json')
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                entry['config'] = json.load(f)
        if os.path.isfile(res_path):
            with open(res_path) as f:
                entry['results'] = json.load(f)
        out.append(entry)
    return out
