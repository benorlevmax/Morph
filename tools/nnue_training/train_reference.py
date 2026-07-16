#!/usr/bin/env python3
"""train_reference.py - Small numpy trainer that trains the PRODUCTION NNUE
shape (10,240 HalfKP features -> 512-wide dual-perspective accumulator -> 8
output buckets) on data in Bullet's own text ingestion format
(`<FEN> | <score> | <wdl>`, produced by chess_train's `--format bullet` or
Dataset::save_bullet()).

THIS IS A STAND-IN FOR BULLET, NOT BULLET ITSELF.

The real training tool for this project is Bullet (jw1912/bullet, MIT,
Rust+CUDA) -- see bullet_trainer/ in this same directory for the actual Bullet
crate config, which should be run on a machine with internet access, a Rust
toolchain, and (ideally) a GPU. None of those three things are available in
the sandbox this Phase A smoke test was built in (outbound network from the
shell is blocked by an allowlist proxy, no `cargo`/`rustc` is installed, and
`nvidia-smi` is absent).

This script exists solely to PROVE the DATA -> FEATURES -> TRAIN -> QUANTIZE
-> .NNUE -> C++ INFERENCE chain end-to-end inside that constrained sandbox,
using the exact same feature indexing, accumulator shape, activation, output
bucketing, and binary file format as the production engine and as the real
Bullet config in bullet_trainer/. Its numerics (plain SGD/Adam in float64,
no CUDA, no fused kernels) are not meant to produce a strong network -- only
a CORRECT one, byte-for-byte loadable and verifiable in the real C++ engine.

Usage:
    python3 train_reference.py --data selfplay.bulletfmt.txt --out smoke_A.nnue \
        --epochs 3 --max-samples 6000 --qa 256 --qb 256
"""
import argparse
import random
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from reference_nnue import (
    NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS, WHITE, BLACK, KING,
    parse_fen_board, feature_index, output_bucket, RefNet,
)

MAX_ACTIVE = 32  # pad active-feature lists to this length (30 non-king pieces max)
PAD_IDX = NNUE_FEATURES  # extra dummy row, permanently zero, never updated


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

    # Training target: same win_prob blend as src/train/encoding.h
    # (lambda * sigmoid(eval/400) + (1-lambda) * result_prob), lambda=0.5.
    eval_p = 1.0 / (1.0 + np.exp(-score_cp / 400.0))
    result_p = wdl  # already 0.0/0.5/1.0, white-relative
    lam = 0.5
    target_white = lam * eval_p + (1 - lam) * result_p
    # Re-express white-relative target as stm-relative (what the net predicts:
    # win prob for the side whose accumulator is "own").
    target_stm = target_white if stm == WHITE else (1.0 - target_white)

    return idx[stm], idx[other], bucket, target_stm


def load_dataset(path, max_samples, seed=1):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fen_part, score_part, wdl_part = line.split('|')
            fen = fen_part.strip()
            score = int(score_part.strip())
            wdl = float(wdl_part.strip())
            samples.append((fen, score, wdl))
    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples:
        samples = samples[:max_samples]
    return samples


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-samples', type=int, default=6000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--qa', type=int, default=256, help='feature-transformer quantization factor')
    ap.add_argument('--qb', type=int, default=256, help='output-layer quantization factor')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    samples = load_dataset(args.data, args.max_samples, args.seed)
    print(f'loaded {len(samples)} samples from {args.data}')
    own_idx, opp_idx, buckets, targets = build_arrays(samples)
    print(f'encoded in {time.time() - t0:.1f}s')

    rng = np.random.default_rng(args.seed)
    # float32 params, +1 padding row (index NNUE_FEATURES) permanently zero.
    ft_w = (rng.standard_normal((NNUE_FEATURES + 1, NNUE_HL)) * 0.01).astype(np.float32)
    ft_w[PAD_IDX] = 0.0
    ft_b = np.zeros(NNUE_HL, dtype=np.float32)
    out_w = (rng.standard_normal((NNUE_OUT_BUCKETS, 2 * NNUE_HL)) * 0.01).astype(np.float32)
    out_b = np.zeros(NNUE_OUT_BUCKETS, dtype=np.float32)

    # Adam state.
    def adam_state(shape):
        return {'m': np.zeros(shape, dtype=np.float64), 'v': np.zeros(shape, dtype=np.float64)}
    state = {'ft_w': adam_state(ft_w.shape), 'ft_b': adam_state(ft_b.shape),
              'out_w': adam_state(out_w.shape), 'out_b': adam_state(out_b.shape)}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    n = len(samples)
    clip_hi = 32767.0 / args.qa  # float-space equivalent of the engine's CR_MAX after quantization by QA

    def adam_update(param, grad, key):
        nonlocal step
        s = state[key]
        s['m'] = beta1 * s['m'] + (1 - beta1) * grad
        s['v'] = beta2 * s['v'] + (1 - beta2) * (grad * grad)
        mhat = s['m'] / (1 - beta1 ** step)
        vhat = s['v'] / (1 - beta2 ** step)
        param -= (args.lr * mhat / (np.sqrt(vhat) + eps)).astype(param.dtype)

    order = np.arange(n)
    for epoch in range(args.epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            step += 1
            bidx = order[start:start + args.batch_size]
            bs = len(bidx)
            oi, pi_, bk, tgt = own_idx[bidx], opp_idx[bidx], buckets[bidx], targets[bidx]

            acc_own = ft_b + ft_w[oi].sum(axis=1)   # (bs, HL)
            acc_opp = ft_b + ft_w[pi_].sum(axis=1)  # (bs, HL)
            h_own = np.clip(acc_own, 0.0, clip_hi)
            h_opp = np.clip(acc_opp, 0.0, clip_hi)

            w_own = out_w[bk, :NNUE_HL]      # (bs, HL)
            w_opp = out_w[bk, NNUE_HL:]      # (bs, HL)
            b_out = out_b[bk]                # (bs,)

            # logit_cp is already float-domain centipawns (h_own/h_opp are clipped
            # to the float-space equivalent of the engine's post-quantization
            # CR_MAX, i.e. everything here mirrors what "raw int64 sum / scale"
            # will compute at inference once quantized by QA*QB).
            logit_cp = (h_own * w_own).sum(axis=1) + (h_opp * w_opp).sum(axis=1) + b_out
            pred = sigmoid(logit_cp / 400.0)

            err = pred - tgt                          # dL/dpred for MSE-on-prob
            dlogit = err * pred * (1 - pred) / 400.0   # chain through sigmoid(x/400)

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
        print(f'epoch {epoch + 1}/{args.epochs}  mse {epoch_loss / n:.5f}  '
              f'elapsed {time.time() - t0:.1f}s')

    # --- Quantize and export in the exact production .nnue binary format ---
    net = RefNet()
    net.scale = args.qa * args.qb
    net.ft_bias = [int(round(v * args.qa)) for v in ft_b]
    for f in range(NNUE_FEATURES):
        net.ft_weights[f] = [int(round(v * args.qa)) for v in ft_w[f]]
    for b in range(NNUE_OUT_BUCKETS):
        net.out_weights[b] = [int(round(v * args.qb)) for v in out_w[b]]
    net.out_bias = [int(round(v * args.qa * args.qb)) for v in out_b]

    # Overflow check (int16 weights/bias, int32 out_bias) -- fail loudly per
    # the user's explicit instruction, exactly like Bullet's own quantiser.
    def check_i16(vals, name):
        bad = [v for v in vals if not (-32768 <= v <= 32767)]
        if bad:
            raise SystemExit(f'QUANTIZATION OVERFLOW in {name}: {len(bad)} values out of int16 '
                              f'range (e.g. {bad[0]}). Reduce --qa/--qb or add weight clipping.')

    check_i16(net.ft_bias, 'ft_bias')
    for f in range(NNUE_FEATURES):
        check_i16(net.ft_weights[f], f'ft_weights[{f}]')
    for b in range(NNUE_OUT_BUCKETS):
        check_i16(net.out_weights[b], f'out_weights[{b}]')
    bad_bias = [v for v in net.out_bias if not (-2**31 <= v <= 2**31 - 1)]
    if bad_bias:
        raise SystemExit(f'QUANTIZATION OVERFLOW in out_bias: {bad_bias}')

    net.save(args.out)
    print(f'quantized and wrote {args.out}  (scale={net.scale}, qa={args.qa}, qb={args.qb})')
    print(f'total wall time {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
