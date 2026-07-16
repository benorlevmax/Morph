#!/usr/bin/env python3
"""generate.py - Stage 1 of the NNUE training pipeline: generate labelled
training positions via engine self-play.

Wraps the existing `chess_train gen --format bullet` self-play generator
(src/apps/train_main.cpp -> src/train/selfplay.cpp), which already plays the
CURRENT, unmodified engine against itself at a fixed search depth/node budget
and labels each reached position with the engine's own search score and the
eventual game result. This script does not change how positions are searched
or generated -- it only invokes the existing binary and re-shapes its output.

Output: a JSONL dataset, one training sample per line, with the fields
required by the pipeline spec:
    fen             - position (FEN)
    result          - game outcome, White-relative, in {0.0, 0.5, 1.0}
    eval            - engine search score at generation time, White-relative cp
    depth           - search depth used to label this position
    engine_version  - 'id name' string of the chess_train/chess binary used
    generated_at    - ISO8601 UTC timestamp of this generation run
    run_id          - identifier for this generate.py invocation (for provenance)

Usage:
    python3 generate.py --games 500 --depth 6 --randomplies 6 \
        --out data/run1.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_paths import find_binary, engine_version

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(PIPELINE_DIR, 'data')


def parse_bullet_line(line):
    """'<FEN> | <score> | <wdl>' -> (fen, score_cp, wdl_float)."""
    fen_part, score_part, wdl_part = line.split('|')
    return fen_part.strip(), int(score_part.strip()), float(wdl_part.strip())


def run_selfplay(chess_train_bin, games, depth, nodes, randomplies, workdir, verbose=True):
    out_txt = os.path.join(workdir, 'selfplay.bulletfmt.txt')
    cmd = [chess_train_bin, 'gen', '--games', str(games), '--randomplies', str(randomplies),
           '--format', 'bullet', '--out', out_txt]
    if nodes > 0:
        cmd += ['--nodes', str(nodes)]
    else:
        cmd += ['--depth', str(depth)]

    if verbose:
        print(f'[generate] running: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
    elapsed = time.time() - t0
    if verbose:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f'chess_train gen failed (exit {proc.returncode}):\n{proc.stderr}')
    if not os.path.isfile(out_txt):
        raise RuntimeError(f'chess_train gen did not produce {out_txt}')
    return out_txt, elapsed


def convert_to_jsonl(bullet_txt_path, out_path, depth, version, run_id, append=False):
    n = 0
    mode = 'a' if append else 'w'
    generated_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with open(bullet_txt_path) as fin, open(out_path, mode) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fen, score_cp, wdl = parse_bullet_line(line)
            record = {
                'fen': fen,
                'result': wdl,
                'eval': score_cp,
                'depth': depth,
                'engine_version': version,
                'generated_at': generated_at,
                'run_id': run_id,
            }
            fout.write(json.dumps(record) + '\n')
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--games', type=int, default=200, help='self-play games to generate from')
    ap.add_argument('--depth', type=int, default=6, help='search depth used to label positions')
    ap.add_argument('--nodes', type=int, default=0,
                     help='use a node budget instead of fixed depth (0 = use --depth)')
    ap.add_argument('--randomplies', type=int, default=6,
                     help='random opening plies for game diversity')
    ap.add_argument('--out', default=None,
                     help='output JSONL path (default: data/positions_<run_id>.jsonl)')
    ap.add_argument('--append', action='store_true',
                     help='append to --out instead of overwriting')
    ap.add_argument('--bin-dir', default=None, help='directory containing chess_train(.exe)')
    ap.add_argument('--keep-raw', action='store_true',
                     help='keep the intermediate bullet-format .txt file')
    args = ap.parse_args()

    chess_train_bin = find_binary('chess_train', args.bin_dir)
    # chess_train is a gen/train/distill CLI, not a UCI loop -- query the
    # actual UCI binary (built alongside it) for the 'id name' version string.
    try:
        chess_bin = find_binary('chess', args.bin_dir)
        version = engine_version(chess_bin)
    except FileNotFoundError:
        version = 'unknown (chess UCI binary not found alongside chess_train)'
    run_id = uuid.uuid4().hex[:12]

    out_path = args.out or os.path.join(DEFAULT_OUT_DIR, f'positions_{run_id}.jsonl')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f'[generate] engine_version={version!r}  run_id={run_id}')
    print(f'[generate] games={args.games} depth={args.depth} nodes={args.nodes} '
          f'randomplies={args.randomplies}')

    with tempfile.TemporaryDirectory(prefix='nnue_gen_') as workdir:
        raw_txt, elapsed = run_selfplay(chess_train_bin, args.games, args.depth, args.nodes,
                                         args.randomplies, workdir)
        n = convert_to_jsonl(raw_txt, out_path, args.depth, version, run_id, append=args.append)
        if args.keep_raw:
            keep_path = out_path + '.raw.txt'
            with open(raw_txt) as f, open(keep_path, 'w') as g:
                g.write(f.read())
            print(f'[generate] kept raw bullet-format text -> {keep_path}')

    rate = n / elapsed if elapsed > 0 else float('nan')
    print(f'[generate] wrote {n} samples -> {out_path}  '
          f'({elapsed:.1f}s, {rate:.0f} samples/s)')

    # Manifest entry for provenance / reproducibility.
    manifest_path = os.path.join(DEFAULT_OUT_DIR, 'manifest.jsonl')
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    with open(manifest_path, 'a') as mf:
        mf.write(json.dumps({
            'run_id': run_id, 'out': os.path.abspath(out_path), 'samples': n,
            'games': args.games, 'depth': args.depth, 'nodes': args.nodes,
            'randomplies': args.randomplies, 'engine_version': version,
            'elapsed_s': elapsed,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }) + '\n')

    print(f'[generate] OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
