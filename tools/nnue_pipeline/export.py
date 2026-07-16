#!/usr/bin/env python3
"""export.py - Stage 3 of the NNUE training pipeline: quantize a train.py
checkpoint (.npz, raw float32 weights) into the engine's production .nnue
binary format ("NNU2", see src/nnue/nnue.cpp write_net()/load()).

Two input kinds are supported:

  --checkpoint <path>.npz   (default) a checkpoint from train.py --engine reference
  --bullet-quantised <path> a real Bullet run's quantised.bin (already
                             quantized int16 weights in Bullet's own
                             SavedFormat order) -- see
                             docs/NNUE_TRAINING_BULLET.md section 5 for the
                             exact byte-layout mapping this depends on.

Quantization: float weights are scaled by --qa (feature-transformer) and
--qb (output layer) and rounded to int16. The on-disk `scale` field is set
to qa*qb so that `NNUE::evaluate()`'s integer pipeline (accumulate in i16,
clip to [0, CR_MAX], multiply-accumulate in i64, divide by `scale`) reproduces
the same centipawn values the float trainer optimized for. Every value is
checked against the int16 (weights/ftBias) or int32 (outBias) range and the
export FAILS LOUDLY on overflow rather than silently clamping, exactly like
train_reference.py's original quantizer -- a silently clamped weight is a
silent correctness bug in a trained network.

Usage:
    python3 export.py --checkpoint checkpoints/run1/latest.npz \
        --out nets/run1.nnue --qa 256 --qb 256
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnue_format import (
    RefNet, NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS, MAGIC, VERSION,
    check_i16, check_i32,
)


def export_from_checkpoint(ckpt_path, out_path, qa, qb):
    z = np.load(ckpt_path, allow_pickle=False)
    ft_w, ft_b, out_w, out_b = z['ft_w'], z['ft_b'], z['out_w'], z['out_b']
    if ft_w.shape != (NNUE_FEATURES + 1, NNUE_HL):
        raise SystemExit(f'{ckpt_path}: ft_w shape {ft_w.shape} != expected '
                          f'{(NNUE_FEATURES + 1, NNUE_HL)} (architecture mismatch)')

    net = RefNet()
    net.scale = qa * qb
    net.ft_bias = [int(round(v * qa)) for v in ft_b]
    for f in range(NNUE_FEATURES):
        net.ft_weights[f] = [int(round(v * qa)) for v in ft_w[f]]
    for b in range(NNUE_OUT_BUCKETS):
        net.out_weights[b] = [int(round(v * qb)) for v in out_w[b]]
    net.out_bias = [int(round(v * qa * qb)) for v in out_b]

    check_i16(net.ft_bias, 'ft_bias')
    for f in range(NNUE_FEATURES):
        check_i16(net.ft_weights[f], f'ft_weights[{f}]')
    for b in range(NNUE_OUT_BUCKETS):
        check_i16(net.out_weights[b], f'out_weights[{b}]')
    check_i32(net.out_bias, 'out_bias')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    net.save(out_path)

    epoch = int(z['epoch']) if 'epoch' in z.files else None
    step = int(z['step']) if 'step' in z.files else None
    meta = {'source': 'checkpoint', 'checkpoint': os.path.abspath(ckpt_path),
             'epoch': epoch, 'step': step, 'qa': qa, 'qb': qb, 'scale': net.scale}
    return meta


def export_from_bullet_quantised(bin_path, out_path, scale):
    """Convert a real Bullet quantised.bin (payload already int16/int32-quantized
    in our SavedFormat order, per bullet_trainer/src/main.rs) to a .nnue file
    by prepending our header. See docs/NNUE_TRAINING_BULLET.md section 5 --
    this path has not been exercised against a real Bullet run in this
    sandbox (no Rust/GPU available here); verify its output with test.py
    before trusting it."""
    ft_bias_bytes = NNUE_HL * 2
    ft_weights_bytes = NNUE_FEATURES * NNUE_HL * 2
    out_weights_bytes = NNUE_OUT_BUCKETS * 2 * NNUE_HL * 2
    out_bias_bytes = NNUE_OUT_BUCKETS * 4
    payload_bytes = ft_bias_bytes + ft_weights_bytes + out_weights_bytes + out_bias_bytes

    with open(bin_path, 'rb') as f:
        data = f.read()
    if len(data) < payload_bytes:
        raise SystemExit(
            f'{bin_path} is {len(data)} bytes, expected at least {payload_bytes}. '
            f'The SavedFormat order in the Bullet crate does not match this shape -- '
            f'do not trust this file until that is fixed (see '
            f'docs/NNUE_TRAINING_BULLET.md section 5).')
    payload = data[:payload_bytes]
    padding = data[payload_bytes:]
    if any(b != 0 for b in padding):
        print(f'WARNING: {len(padding)} trailing bytes are non-zero; expected pure '
              f'padding. Double check the SavedFormat order before trusting this net.',
              file=sys.stderr)

    header = struct.pack('<IIIII', MAGIC, VERSION, NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS)
    header += struct.pack('<i', scale)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(payload)

    return {'source': 'bullet_quantised', 'bullet_bin': os.path.abspath(bin_path), 'scale': scale}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None, help='train.py checkpoint .npz')
    ap.add_argument('--bullet-quantised', default=None,
                     help='real Bullet quantised.bin instead of a reference checkpoint')
    ap.add_argument('--out', required=True, help='output .nnue path')
    ap.add_argument('--qa', type=int, default=256, help='feature-transformer quantization factor')
    ap.add_argument('--qb', type=int, default=256, help='output-layer quantization factor')
    ap.add_argument('--scale', type=int, default=None,
                     help='override the on-disk scale field (bullet path only; default qa*qb)')
    args = ap.parse_args()

    if bool(args.checkpoint) == bool(args.bullet_quantised):
        raise SystemExit('pass exactly one of --checkpoint or --bullet-quantised')

    if args.checkpoint:
        meta = export_from_checkpoint(args.checkpoint, args.out, args.qa, args.qb)
    else:
        scale = args.scale if args.scale is not None else args.qa * args.qb
        meta = export_from_bullet_quantised(args.bullet_quantised, args.out, scale)

    # Round-trip sanity check: load what we just wrote and confirm the header
    # parses and a few evaluations run without throwing.
    net = RefNet.load(args.out)
    smoke_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    ]
    for fen in smoke_fens:
        val = net.evaluate_fen(fen)
        print(f'[export] sanity eval: {val:>6} cp   {fen}')

    meta['out'] = os.path.abspath(args.out)
    meta_path = args.out + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[export] wrote {args.out}  (scale={net.scale})')
    print(f'[export] wrote metadata -> {meta_path}')
    print('[export] OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
