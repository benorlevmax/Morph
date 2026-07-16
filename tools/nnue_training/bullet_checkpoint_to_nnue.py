#!/usr/bin/env python3
"""bullet_checkpoint_to_nnue.py - Convert a real Bullet training checkpoint's
`quantised.bin` into a .nnue file this engine's NNUE::load() can read.

Per docs/4-saved-networks.md, a Bullet checkpoint directory contains
`quantised.bin`: the SavedFormat-ordered, quantized fields, concatenated and
padded to a multiple of 64 bytes. bullet_trainer/src/main.rs's `.save_format`
list is ordered to exactly match src/nnue/nnue.cpp's write_net() layout:

    ftBias  (l0b): NNUE_HL int16               =   512 x 2 =   1,024 bytes
    ftWeights(l0w): NNUE_FEATURES*NNUE_HL int16 = 10240*512*2 = 10,485,760 bytes
    outWeights(l1w): NNUE_OUT_BUCKETS*2*NNUE_HL int16 = 8*1024*2 = 16,384 bytes
    outBias (l1b): NNUE_OUT_BUCKETS int32       =     8 x 4 =      32 bytes

So this script just needs to: strip Bullet's 64-byte padding, verify the
total size matches exactly, and prepend our own 24-byte header (magic,
version, NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS, scale) -- no reordering or
repacking of the payload itself is needed IF (and only if) the SavedFormat
order in bullet_trainer/src/main.rs is exactly as commented there.

**This has not been run against a real Bullet output in this sandbox** (no
Rust toolchain / no internet -- see the Phase A audit report). It has been
written to be exactly consistent with reference_nnue.py's RefNet.save(),
which HAS been verified byte-for-byte against the real C++ engine (see
Finding #2, docs/phaseA_nnue_bullet_audit.md). Before trusting a real Bullet
run's output, re-run the same reference-verification harness
(verify_against_engine.py) against a net produced by THIS script, not just
against train_reference.py's own output.

Usage:
    python3 bullet_checkpoint_to_nnue.py \
        checkpoints/chessengine_production_nnue-40/quantised.bin \
        production.nnue --scale 128
"""
import argparse
import struct
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from reference_nnue import MAGIC, VERSION, NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS

FT_BIAS_BYTES = NNUE_HL * 2
FT_WEIGHTS_BYTES = NNUE_FEATURES * NNUE_HL * 2
OUT_WEIGHTS_BYTES = NNUE_OUT_BUCKETS * 2 * NNUE_HL * 2
OUT_BIAS_BYTES = NNUE_OUT_BUCKETS * 4
PAYLOAD_BYTES = FT_BIAS_BYTES + FT_WEIGHTS_BYTES + OUT_WEIGHTS_BYTES + OUT_BIAS_BYTES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('quantised_bin')
    ap.add_argument('out_nnue')
    ap.add_argument('--scale', type=int, required=True,
                     help='ENGINE_STORED_SCALE from bullet_trainer/src/main.rs (== QB, '
                          'if QA was chosen equal to eval_scale per that file\'s derivation)')
    args = ap.parse_args()

    with open(args.quantised_bin, 'rb') as f:
        data = f.read()

    if len(data) < PAYLOAD_BYTES:
        raise SystemExit(
            f'quantised.bin is {len(data)} bytes, expected at least {PAYLOAD_BYTES} '
            f'({FT_BIAS_BYTES} ftBias + {FT_WEIGHTS_BYTES} ftWeights + '
            f'{OUT_WEIGHTS_BYTES} outWeights + {OUT_BIAS_BYTES} outBias). '
            f'This means the SavedFormat list in bullet_trainer/src/main.rs does not '
            f'actually match this shape -- STOP and re-check it against the real '
            f'trait/API (this file is marked UNVERIFIED, see its header comment) '
            f'before trusting anything downstream.')

    payload = data[:PAYLOAD_BYTES]
    padding = data[PAYLOAD_BYTES:]
    if any(b != 0 for b in padding):
        print(f'WARNING: {len(padding)} trailing bytes in quantised.bin are non-zero; '
              f'expected pure padding. Double-check PAYLOAD_BYTES math above against the '
              f'real SavedFormat order before proceeding.', file=sys.stderr)

    header = struct.pack('<IIIII', MAGIC, VERSION, NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS)
    header += struct.pack('<i', args.scale)

    with open(args.out_nnue, 'wb') as f:
        f.write(header)
        f.write(payload)

    print(f'wrote {args.out_nnue}: {len(header)}-byte header + {len(payload)}-byte payload '
          f'(scale={args.scale})')
    print('Now run tools/nnue_training/verify_against_engine.py against it before trusting it.')


if __name__ == '__main__':
    main()
