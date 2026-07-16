#!/usr/bin/env python3
"""batches.py - Dataset handling, stage 2: create training batches.

Takes a versioned dataset (training_server/datasets/<version>/all.jsonl,
produced by import_data.py) and materializes a deterministic train/val split
plus a batching plan, so that:
  (a) the validation set is fixed across resumed/repeated training runs on
      the same dataset version (comparing val_mse epoch-to-epoch, or between
      two experiments trained on the same data, is meaningful), and
  (b) "how this dataset will be batched" is an inspectable artifact
      (batches_manifest.json), not just an implicit detail of the training
      loop.

Usage:
    python3 batches.py --version v_20260716_ab12cd34 --batch-size 16384 --val-fraction 0.02
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config


def make_batches(version, batch_size, val_fraction, seed=1):
    ds_dir = os.path.join(config.DATASETS_DIR, version)
    all_path = os.path.join(ds_dir, 'all.jsonl')
    if not os.path.isfile(all_path):
        raise FileNotFoundError(f'{all_path} not found -- run import_data.py first')

    with open(all_path) as f:
        lines = [l for l in f if l.strip()]

    rng = random.Random(seed)
    order = list(range(len(lines)))
    rng.shuffle(order)

    n_val = max(1, int(len(lines) * val_fraction)) if len(lines) > 20 else max(1, len(lines) // 10)
    val_idx = set(order[:n_val])

    train_path = os.path.join(ds_dir, 'train.jsonl')
    val_path = os.path.join(ds_dir, 'val.jsonl')
    n_train = 0
    n_val_written = 0
    with open(train_path, 'w') as ft, open(val_path, 'w') as fv:
        for i in order:
            if i in val_idx:
                fv.write(lines[i])
                n_val_written += 1
            else:
                ft.write(lines[i])
                n_train += 1

    n_batches = -(-n_train // batch_size)  # ceil div
    batches_manifest = {
        'version': version, 'batch_size': batch_size, 'val_fraction': val_fraction, 'seed': seed,
        'n_train': n_train, 'n_val': n_val_written, 'n_batches_per_epoch': n_batches,
        'train_file': os.path.abspath(train_path), 'val_file': os.path.abspath(val_path),
    }
    with open(os.path.join(ds_dir, 'batches_manifest.json'), 'w') as f:
        json.dump(batches_manifest, f, indent=2)
    return batches_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--batch-size', type=int, default=16384)
    ap.add_argument('--val-fraction', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    manifest = make_batches(args.version, args.batch_size, args.val_fraction, args.seed)
    print(f'[batches] {manifest["n_train"]} train / {manifest["n_val"]} val, '
          f'{manifest["n_batches_per_epoch"]} batches/epoch @ batch_size={args.batch_size}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
