#!/usr/bin/env python3
"""train.py - Stage 2 of the NNUE training pipeline: train a network on
datasets produced by generate.py.

Two training engines are supported:

  --engine reference (default, always available)
      A small NumPy Adam trainer that trains the exact production shape
      (10,240 HalfKP features -> 512-wide dual-perspective accumulator ->
      8 output buckets, clipped-ReLU activation) directly against
      generate.py's JSONL datasets. This is NOT Bullet -- it is a
      correctness-first stand-in (plain NumPy, no CUDA) used because this
      environment has no Rust toolchain and no GPU (see
      docs/phaseA_nnue_bullet_audit.md). It produces a *correct* net,
      verified end-to-end against the C++ engine (see test.py), just not a
      maximally strong one at large scale.

  --engine bullet (opt-in, requires a real Bullet checkout + Rust + GPU)
      Shells out to `cargo run --release` in --bullet-dir (default:
      tools/nnue_training/bullet_trainer, the custom HalfKP SparseInputType
      crate written in Phase A). Requires `cargo` on PATH; raises a clear,
      actionable error otherwise rather than silently falling back, so a CI
      run can tell the difference between "used the real trainer" and "used
      the stand-in".

Checkpoints are saved as .npz (raw float32 weights + Adam optimizer state +
step/epoch counters), NOT as a ready-to-load .nnue file -- quantization and
the production binary layout are export.py's job (Stage 3). This split lets
you resume training without re-quantizing, and re-export the same checkpoint
at different --qa/--qb settings.

Usage:
    python3 train.py --data data/positions_abc123.jsonl --out checkpoints/run1 \
        --epochs 3 --max-samples 50000
    python3 train.py --resume checkpoints/run1/latest.npz --epochs 2   # continue
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnue_format import (
    NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS, WHITE, BLACK, KING,
    parse_fen_board, feature_index, output_bucket,
)

MAX_ACTIVE = 32          # pad active-feature lists (30 non-king pieces max)
PAD_IDX = NNUE_FEATURES  # extra dummy row, permanently zero, never updated
LAMBDA = 0.5              # eval/result blend, matches src/train/encoding.h win_prob_target()


# ---------------------------------------------------------------------------
# Dataset loading (generate.py's JSONL format)
# ---------------------------------------------------------------------------
def load_jsonl_datasets(paths, max_samples, seed=1):
    """Loads (fen, score_cp, wdl) samples from `paths`, shuffles, and (if
    max_samples truncates) prioritizes retention by search-instability
    signal (score_swing/best_move_changes, see search.h's SearchResult and
    platform/server/app.py's export_dataset(), which writes these two
    optional fields into the JSONL when the server has them recorded).

    Quality-aware truncation, not blind random truncation: when at least one
    loaded record carries a non-null instability signal, up to half of the
    kept budget is reserved for the highest-instability samples (positions
    where the engine's own search was least confident/stable -- tactically
    sharp, contested, or otherwise hard positions), and the remainder is
    filled by continued random sampling from what's left, so the kept set
    still spans the full difficulty distribution rather than collapsing to
    only sharp tactics (a network also needs quiet/simple positions to
    calibrate on). This never discards data outside the requested sample
    budget -- every position not selected for THIS training run remains on
    disk in the dataset file, untouched, and can be sampled by a later run;
    nothing is deleted or invalidated. If no record carries the signal (an
    older export, or a source that never reported it), behavior is
    byte-for-byte identical to the original: pure random shuffle + truncate."""
    samples = []
    has_signal = False
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Accept both this pipeline's own field name ('eval') and the
                # richer canonical schema training_server/ produces
                # ('eval_cp', plus extra fields like side_to_move/nodes/source
                # that we simply don't need here).
                eval_field = 'eval' if 'eval' in rec else 'eval_cp'
                score_swing = rec.get('score_swing')
                best_move_changes = rec.get('best_move_changes')
                if score_swing is not None or best_move_changes is not None:
                    has_signal = True
                samples.append((rec['fen'], int(rec[eval_field]), float(rec['result']),
                                 score_swing, best_move_changes))
    rng = random.Random(seed)
    rng.shuffle(samples)

    if max_samples and len(samples) > max_samples:
        if has_signal:
            def instability(s):
                sw, bmc = s[3], s[4]
                return (sw or 0) + (bmc or 0) * 30

            reserved = max_samples // 2   # up to half the budget for hardest positions
            # `samples` is already shuffled (rng.shuffle above), so sorting it
            # with a stable sort means ties (e.g. every hard/easy sample sharing
            # the same instability score) keep that shuffled relative order --
            # ranked[:reserved] is therefore a random subset of the highest-
            # instability samples, not a positionally-biased one.
            ranked = sorted(samples, key=instability, reverse=True)
            priority_keep = ranked[:reserved]
            priority_ids = {id(s) for s in priority_keep}
            remainder_pool = [s for s in samples if id(s) not in priority_ids]
            # Removing `priority_keep` preferentially strips the earliest-
            # occurring high-instability samples from `samples`'s shuffled
            # order (that's exactly what the stable sort above selected),
            # which would otherwise bias remainder_pool's front toward
            # low-instability samples. Re-shuffle before slicing so the fill
            # is a genuine random sample of what's left.
            rng.shuffle(remainder_pool)
            fill = remainder_pool[:max_samples - len(priority_keep)]
            samples = priority_keep + fill
            rng.shuffle(samples)   # re-shuffle so priority/fill aren't order-correlated
        else:
            samples = samples[:max_samples]

    # Strip the instability fields back off -- every downstream consumer
    # (build_arrays, encode_sample, ...) expects plain (fen, score_cp, wdl).
    return [(fen, score, wdl) for fen, score, wdl, _sw, _bmc in samples]


def encode_sample(fen, score_cp, wdl):
    board, stm = parse_fen_board(fen)
    wk = next(s for s, (c, pt) in board.items() if c == WHITE and pt == KING)
    bk = next(s for s, (c, pt) in board.items() if c == BLACK and pt == KING)
    king_of = {WHITE: wk, BLACK: bk}

    idx = {}
    for persp in (WHITE, BLACK):
        active = [feature_index(persp, king_of[persp], c, pt, s)
                  for s, (c, pt) in board.items() if pt != KING]
        active = active[:MAX_ACTIVE] + [PAD_IDX] * (MAX_ACTIVE - len(active))
        idx[persp] = active

    n_pieces = len(board)
    bucket = output_bucket(n_pieces)
    other = BLACK if stm == WHITE else WHITE

    eval_p = 1.0 / (1.0 + np.exp(-score_cp / 400.0))
    result_p = wdl
    target_white = LAMBDA * eval_p + (1 - LAMBDA) * result_p
    target_stm = target_white if stm == WHITE else (1.0 - target_white)

    return idx[stm], idx[other], bucket, target_stm


def build_arrays(samples):
    n = len(samples)
    own_idx = np.zeros((n, MAX_ACTIVE), dtype=np.int32)
    opp_idx = np.zeros((n, MAX_ACTIVE), dtype=np.int32)
    buckets = np.zeros(n, dtype=np.int32)
    targets = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(samples):
        own, opp, bucket, target = encode_sample(*s)
        own_idx[i] = own
        opp_idx[i] = opp
        buckets[i] = bucket
        targets[i] = target
    return own_idx, opp_idx, buckets, targets


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def new_params(seed):
    rng = np.random.default_rng(seed)
    ft_w = (rng.standard_normal((NNUE_FEATURES + 1, NNUE_HL)) * 0.01).astype(np.float32)
    ft_w[PAD_IDX] = 0.0
    ft_b = np.zeros(NNUE_HL, dtype=np.float32)
    out_w = (rng.standard_normal((NNUE_OUT_BUCKETS, 2 * NNUE_HL)) * 0.01).astype(np.float32)
    out_b = np.zeros(NNUE_OUT_BUCKETS, dtype=np.float32)
    state = {
        'ft_w_m': np.zeros_like(ft_w, dtype=np.float64), 'ft_w_v': np.zeros_like(ft_w, dtype=np.float64),
        'ft_b_m': np.zeros_like(ft_b, dtype=np.float64), 'ft_b_v': np.zeros_like(ft_b, dtype=np.float64),
        'out_w_m': np.zeros_like(out_w, dtype=np.float64), 'out_w_v': np.zeros_like(out_w, dtype=np.float64),
        'out_b_m': np.zeros_like(out_b, dtype=np.float64), 'out_b_v': np.zeros_like(out_b, dtype=np.float64),
    }
    return ft_w, ft_b, out_w, out_b, state, 0


def save_checkpoint(path, ft_w, ft_b, out_w, out_b, state, step, epoch, args_dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, ft_w=ft_w, ft_b=ft_b, out_w=out_w, out_b=out_b,
                         step=step, epoch=epoch, args_json=json.dumps(args_dict), **state)


def load_checkpoint(path):
    z = np.load(path, allow_pickle=False)
    ft_w, ft_b, out_w, out_b = z['ft_w'], z['ft_b'], z['out_w'], z['out_b']
    state = {k: z[k] for k in z.files if k.endswith('_m') or k.endswith('_v')}
    step = int(z['step'])
    epoch = int(z['epoch'])
    prev_args = json.loads(str(z['args_json'])) if 'args_json' in z.files else {}
    return ft_w, ft_b, out_w, out_b, state, step, epoch, prev_args


# ---------------------------------------------------------------------------
# Reference (NumPy) trainer
# ---------------------------------------------------------------------------
def train_reference(args):
    t0 = time.time()
    data_paths = args.data
    if args.data_dir:
        data_paths += sorted(glob.glob(os.path.join(args.data_dir, '*.jsonl')))
    if not data_paths:
        raise SystemExit('no --data files and no --data-dir with .jsonl files given')

    samples = load_jsonl_datasets(data_paths, args.max_samples, args.seed)
    if not samples:
        raise SystemExit(f'no samples loaded from {data_paths}')
    print(f'[train] loaded {len(samples)} samples from {len(data_paths)} file(s)')

    # Held-out validation split.
    n_val = max(1, int(len(samples) * args.val_fraction)) if len(samples) > 10 else 0
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    own_idx, opp_idx, buckets, targets = build_arrays(train_samples)
    if val_samples:
        v_own, v_opp, v_bk, v_tgt = build_arrays(val_samples)
    print(f'[train] encoded in {time.time() - t0:.1f}s  '
          f'(train={len(train_samples)}, val={len(val_samples)})')

    start_epoch = 0
    if args.resume:
        print(f'[train] resuming from {args.resume}')
        ft_w, ft_b, out_w, out_b, state, step, start_epoch, prev_args = load_checkpoint(args.resume)
        if ft_w.shape != (NNUE_FEATURES + 1, NNUE_HL):
            raise SystemExit(f'checkpoint shape mismatch: ft_w is {ft_w.shape}, '
                              f'expected {(NNUE_FEATURES + 1, NNUE_HL)} -- architecture changed?')
    else:
        ft_w, ft_b, out_w, out_b, state, step = new_params(args.seed)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    clip_hi = 32767.0 / args.qa_preview  # float-space preview of post-quant clipping range

    def adam_update(param, grad, key):
        nonlocal step
        m, v = state[key + '_m'], state[key + '_v']
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** step)
        vhat = v / (1 - beta2 ** step)
        param -= (args.lr * mhat / (np.sqrt(vhat) + eps)).astype(param.dtype)

    n = len(train_samples)
    order = np.arange(n)
    metrics_path = os.path.join(args.out, 'metrics.jsonl')
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def eval_loss(oi, pi_, bk, tgt):
        acc_own = ft_b + ft_w[oi].sum(axis=1)
        acc_opp = ft_b + ft_w[pi_].sum(axis=1)
        h_own = np.clip(acc_own, 0.0, clip_hi)
        h_opp = np.clip(acc_opp, 0.0, clip_hi)
        w_own = out_w[bk, :NNUE_HL]
        w_opp = out_w[bk, NNUE_HL:]
        b_out = out_b[bk]
        logit_cp = (h_own * w_own).sum(axis=1) + (h_opp * w_opp).sum(axis=1) + b_out
        pred = sigmoid(logit_cp / 400.0)
        return float(np.mean((pred - tgt) ** 2))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            step += 1
            bidx = order[start:start + args.batch_size]
            bs = len(bidx)
            oi, pi_, bk, tgt = own_idx[bidx], opp_idx[bidx], buckets[bidx], targets[bidx]

            acc_own = ft_b + ft_w[oi].sum(axis=1)
            acc_opp = ft_b + ft_w[pi_].sum(axis=1)
            h_own = np.clip(acc_own, 0.0, clip_hi)
            h_opp = np.clip(acc_opp, 0.0, clip_hi)

            w_own = out_w[bk, :NNUE_HL]
            w_opp = out_w[bk, NNUE_HL:]
            b_out = out_b[bk]

            logit_cp = (h_own * w_own).sum(axis=1) + (h_opp * w_opp).sum(axis=1) + b_out
            pred = sigmoid(logit_cp / 400.0)

            err = pred - tgt
            dlogit = err * pred * (1 - pred) / 400.0

            grad_w_own = dlogit[:, None] * h_own
            grad_w_opp = dlogit[:, None] * h_opp
            grad_b_out = dlogit

            grad_h_own = dlogit[:, None] * w_own
            grad_h_opp = dlogit[:, None] * w_opp
            grad_acc_own = grad_h_own * ((acc_own > 0.0) & (acc_own < clip_hi))
            grad_acc_opp = grad_h_opp * ((acc_opp > 0.0) & (acc_opp < clip_hi))

            grad_ft_w = np.zeros_like(ft_w)
            grad_ft_b = (grad_acc_own.sum(axis=0) + grad_acc_opp.sum(axis=0)) / bs
            np.add.at(grad_ft_w, oi, grad_acc_own[:, None, :] / bs * np.ones((1, MAX_ACTIVE, 1)))
            np.add.at(grad_ft_w, pi_, grad_acc_opp[:, None, :] / bs * np.ones((1, MAX_ACTIVE, 1)))
            grad_ft_w[PAD_IDX] = 0.0

            grad_out_w = np.zeros_like(out_w)
            grad_out_b = np.zeros_like(out_b)
            for bkt in range(NNUE_OUT_BUCKETS):
                m = (bk == bkt)
                if not m.any():
                    continue
                grad_out_w[bkt, :NNUE_HL] = grad_w_own[m].mean(axis=0) * m.sum() / bs
                grad_out_w[bkt, NNUE_HL:] = grad_w_opp[m].mean(axis=0) * m.sum() / bs
                grad_out_b[bkt] = grad_b_out[m].sum() / bs

            adam_update(ft_w, grad_ft_w, 'ft_w')
            adam_update(ft_b, grad_ft_b, 'ft_b')
            adam_update(out_w, grad_out_w, 'out_w')
            adam_update(out_b, grad_out_b, 'out_b')

            epoch_loss += float(np.sum((pred - tgt) ** 2))

        train_mse = epoch_loss / n
        val_mse = eval_loss(v_own, v_opp, v_bk, v_tgt) if val_samples else float('nan')
        elapsed = time.time() - t0
        print(f'[train] epoch {epoch + 1}  train_mse={train_mse:.5f}  '
              f'val_mse={val_mse:.5f}  elapsed={elapsed:.1f}s')

        with open(metrics_path, 'a') as mf:
            mf.write(json.dumps({
                'epoch': epoch + 1, 'step': step, 'train_mse': train_mse, 'val_mse': val_mse,
                'n_train': n, 'n_val': len(val_samples), 'elapsed_s': elapsed,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }) + '\n')

        ckpt_path = os.path.join(args.out, f'epoch{epoch + 1}.npz')
        save_checkpoint(ckpt_path, ft_w, ft_b, out_w, out_b, state, step, epoch + 1, vars(args))
        latest_path = os.path.join(args.out, 'latest.npz')
        shutil.copyfile(ckpt_path, latest_path)
        if args.save_every and (epoch + 1) % args.save_every != 0:
            os.remove(ckpt_path)  # keep only 'latest' between save_every checkpoints

    print(f'[train] done. latest checkpoint -> {os.path.join(args.out, "latest.npz")}')
    return os.path.join(args.out, 'latest.npz')


# ---------------------------------------------------------------------------
# Real Bullet trainer (opt-in, requires cargo + a GPU-capable machine)
# ---------------------------------------------------------------------------
def train_bullet(args):
    """Runs the real GPU trainer (tools/nnue_training/bullet_trainer) as a
    `cargo run --release` subprocess, passing --data/--out/--epochs straight
    through so this behaves like train_reference() from the caller's point
    of view (same CLI contract platform/trainer/train_network.py relies on).
    Requires the Rust toolchain; raises a clear, actionable error otherwise
    rather than silently falling back -- callers that want graceful
    CPU-fallback behavior (e.g. train_network.py) should check
    shutil.which('cargo') themselves BEFORE choosing --engine bullet, and
    fall back to --engine reference if it's absent, rather than relying on
    this function to do it silently."""
    if not args.data:
        raise SystemExit('--engine bullet requires --data <dataset file(s)> '
                          '(the Rust trainer takes one dataset path via --data)')
    if len(args.data) != 1:
        raise SystemExit('--engine bullet currently accepts exactly one --data file '
                          '(concatenate multiple datasets first if needed)')
    bullet_dir = args.bullet_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'nnue_training', 'bullet_trainer')
    bullet_dir = os.path.abspath(bullet_dir)
    if shutil.which('cargo') is None:
        raise SystemExit(
            "train.py --engine bullet requires the Rust toolchain ('cargo' not found on "
            "PATH). Install rustup, or use --engine reference (default) which needs only "
            "NumPy and runs on CPU. See docs/NNUE_TRAINING.md and "
            "docs/NNUE_TRAINING_BULLET.md for details.")
    if not os.path.isdir(bullet_dir):
        raise SystemExit(f"--bullet-dir {bullet_dir!r} does not exist. Clone/prepare a "
                          f"bullet trainer crate there first (see docs/NNUE_TRAINING_BULLET.md).")

    net_id = os.path.basename(args.out.rstrip('/\\')) or 'candidate'
    ckpt_out_dir = os.path.abspath(args.out)
    os.makedirs(ckpt_out_dir, exist_ok=True)

    # --gpu-backend picks which of bullet_lib's real compute backends to
    # build against (cuda=NVIDIA, rocm=AMD -- see
    # tools/nnue_training/bullet_trainer/Cargo.toml's [features] section
    # and its comment citing bullet_lib's own crates/gpu/Cargo.toml, which
    # is the authoritative source for what backends bullet actually has;
    # there is no Intel/SYCL/Level-Zero/DirectML backend to select here).
    # --cuda is kept as a deprecated alias for --gpu-backend cuda so any
    # existing caller/script that only knows about --cuda keeps working.
    gpu_backend = getattr(args, 'gpu_backend', None)
    if gpu_backend is None and getattr(args, 'cuda', False):
        gpu_backend = 'cuda'
    cmd = ['cargo', 'run', '--release']
    if gpu_backend:
        if gpu_backend not in ('cuda', 'rocm'):
            raise SystemExit(f"--gpu-backend must be 'cuda' or 'rocm' (got {gpu_backend!r}) -- "
                              f"bullet_lib has no other GPU compute backend")
        cmd += ['--features', gpu_backend]
    cmd += ['--', '--data', os.path.abspath(args.data[0]), '--out', ckpt_out_dir,
            '--net-id', net_id, '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size)]
    print(f'[train] engine=bullet: {" ".join(cmd)}  (cwd={bullet_dir})')
    print('[train] NOTE: the custom HalfKP SparseInputType/OutputBuckets in this crate are '
          'real implementations written against bullet_lib\'s actual API, but have never '
          'been through cargo check or a real GPU run in this environment (see '
          'main.rs\'s module doc) -- verify its output against nnue_format.py / test.py '
          'before trusting a trained network from this path.')
    subprocess.run(cmd, cwd=bullet_dir, check=True)

    quantised_path = os.path.join(ckpt_out_dir, net_id, 'quantised.bin')
    if not os.path.isfile(quantised_path):
        raise SystemExit(
            f'bullet run exited 0 but {quantised_path!r} was not produced -- check the '
            f'net_id/output_directory match between this call and main.rs\'s '
            f'TrainingSchedule/LocalSettings before trusting anything else about this run.')
    print(f'[train] OK -> {quantised_path}')
    print('[train] convert with: python3 export.py --bullet-quantised '
          f'{quantised_path} --out <net>.nnue --scale 128')
    return quantised_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='*', default=[], help='JSONL dataset file(s) from generate.py')
    ap.add_argument('--data-dir', default=None, help='directory of .jsonl files (all included)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'checkpoints', 'default'),
                     help='checkpoint output directory')
    ap.add_argument('--resume', default=None, help='resume from a checkpoint .npz (e.g. latest.npz)')
    ap.add_argument('--engine', choices=['reference', 'bullet'], default='reference')
    ap.add_argument('--bullet-dir', default=None, help='path to a Bullet trainer crate (--engine bullet)')
    ap.add_argument('--gpu-backend', choices=['cuda', 'rocm'], default=None,
                     help='build/run the bullet trainer against this GPU compute backend '
                          '(--engine bullet only). cuda requires an NVIDIA GPU + CUDA toolkit '
                          'at build time; rocm requires an AMD GPU + ROCm toolkit at build time. '
                          'bullet_lib has no Intel/SYCL/Level-Zero/DirectML backend, so an '
                          'Intel-only GPU cannot be selected here -- see '
                          'platform/trainer/train_network.py for the fallback-to-CPU behavior '
                          'when only an untrainable backend is detected.')
    ap.add_argument('--cuda', action='store_true',
                     help='deprecated alias for --gpu-backend cuda')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-samples', type=int, default=200_000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--val-fraction', type=float, default=0.05)
    ap.add_argument('--qa-preview', type=int, default=256,
                     help='preview quantization scale used only to size the training-time '
                          'clipping range; the real --qa/--qb are chosen at export.py time')
    ap.add_argument('--save-every', type=int, default=1, help='keep a numbered checkpoint every N epochs')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    if args.engine == 'bullet':
        train_bullet(args)
        return 0
    ckpt = train_reference(args)
    print(f'[train] OK -> {ckpt}')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
