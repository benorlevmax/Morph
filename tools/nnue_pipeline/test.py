#!/usr/bin/env python3
"""test.py - Stage 4 of the NNUE training pipeline: validate a newly exported
.nnue file before it's trusted.

Four checks, in order (each gates the next -- no point running a slow Elo
match against a net that doesn't even load correctly):

  1. load        - RefNet.load() parses the file and the header matches the
                    engine's compiled architecture (feature/HL/bucket counts).
  2. verify       - the real compiled `chess` binary is driven via UCI with
                    `setoption name EvalFile value <net>` and asked to `eval`
                    a handful of fixed positions; each result must match the
                    pure-Python reference implementation (nnue_format.RefNet)
                    EXACTLY (integer centipawns, not "close enough"). This is
                    the same technique as tools/nnue_training/
                    verify_against_engine.py and catches feature-indexing /
                    quantization / byte-layout bugs before they reach a match.
  3. benchmark    - runs the engine's built-in `bench` command with the new
                    net loaded vs. with the baseline (a previous .nnue, or
                    the classical evaluator if none given), and reports
                    nodes/nps/time for both. This is a smoke/perf comparison,
                    not a strength measurement.
  4. elo match    - plays an automated match (candidate net vs. baseline)
                    via uci_match.py and reports the W/L/D record, an Elo
                    estimate with a 95% margin, and (optionally) a GSPRT
                    verdict, exactly analogous to what chess_match/SPRT does
                    for classical-vs-classical A/B testing.

Exits non-zero if step 1 or 2 fails (a broken net); steps 3/4 are reported
even on a "losing" Elo result -- that's a valid outcome to report, not a
pipeline failure.

Usage:
    python3 test.py --net nets/run1.nnue --baseline-net nets/prev.nnue --games 40
    python3 test.py --net nets/run1.nnue                      # vs classical eval
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnue_format import RefNet
from engine_paths import find_binary, run_uci
from uci_match import UCIEngine, play_match, elo_estimate, sprt, OPENING_BOOK

VERIFY_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",
    "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 2 7",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkb1r/pp1p1ppp/2p2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
]


def step_load(net_path):
    print(f'\n=== [1/4] load: {net_path} ===')
    net = RefNet.load(net_path)   # raises ValueError with a clear message on mismatch
    print(f'OK: parsed header (features={len(net.ft_weights)}, hl={len(net.ft_bias)}, '
          f'buckets={len(net.out_bias)}, scale={net.scale})')
    return net


def step_verify(chess_bin, net_path, net):
    print(f'\n=== [2/4] verify against engine: {net_path} ===')
    cmds = ['setoption name Use NNUE value true', f'setoption name EvalFile value {net_path}']
    for fen in VERIFY_FENS:
        cmds += [f'position fen {fen}', 'eval']
    cmds.append('quit')
    out = run_uci(chess_bin, cmds, timeout=60)
    engine_evals = [int(l.split()[1]) for l in out.splitlines()
                    if l.strip().startswith('eval ') and l.strip().endswith(' cp')]
    if len(engine_evals) != len(VERIFY_FENS):
        raise SystemExit(f'FAIL: expected {len(VERIFY_FENS)} eval lines from the engine, '
                          f'got {len(engine_evals)}. Engine stdout:\n{out}')
    python_evals = [net.evaluate_fen(fen) for fen in VERIFY_FENS]
    mismatches = []
    for fen, py, eng in zip(VERIFY_FENS, python_evals, engine_evals):
        status = 'OK' if py == eng else 'MISMATCH'
        print(f'  [{status}] python={py:>6}  engine={eng:>6}   {fen}')
        if py != eng:
            mismatches.append((fen, py, eng))
    if mismatches:
        raise SystemExit(f'FAIL: {len(mismatches)}/{len(VERIFY_FENS)} position(s) '
                          f'mismatched between the Python reference and the compiled engine. '
                          f'Do not trust {net_path} -- check feature indexing, king-bucket '
                          f'mapping, output bucket selection, or the quantization scale.')
    print(f'OK: all {len(VERIFY_FENS)} positions match exactly.')


def run_bench(chess_bin, net_path, depth=12):
    cmds = []
    if net_path:
        cmds += ['setoption name Use NNUE value true', f'setoption name EvalFile value {net_path}']
    else:
        cmds += ['setoption name Use NNUE value false']
    cmds += [f'bench {depth}', 'quit']
    out = run_uci(chess_bin, cmds, timeout=120)
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('bench:'):
            # "bench: <nodes> nodes <nps> nps <ms> ms depth <d>"
            parts = line.replace(':', '').split()
            try:
                nodes = int(parts[1])
                nps = int(parts[3])
                ms = int(parts[5])
            except (IndexError, ValueError):
                nodes = nps = ms = None
            return {'raw': line, 'nodes': nodes, 'nps': nps, 'ms': ms}
    return {'raw': None, 'nodes': None, 'nps': None, 'ms': None}


def step_benchmark(chess_bin, net_path, baseline_net_path, depth):
    print(f'\n=== [3/4] benchmark (depth {depth}) ===')
    candidate = run_bench(chess_bin, net_path, depth)
    baseline = run_bench(chess_bin, baseline_net_path, depth)  # None -> classical eval
    label = baseline_net_path or 'classical eval'
    print(f'  candidate ({net_path}): {candidate["raw"]}')
    print(f'  baseline  ({label}): {baseline["raw"]}')
    return {'candidate': candidate, 'baseline': baseline, 'baseline_label': label}


def step_elo_match(chess_bin, net_path, baseline_net_path, games, depth, movetime_ms,
                    elo0, elo1):
    label = baseline_net_path or 'classical eval'
    print(f'\n=== [4/4] Elo match: candidate ({net_path}) vs baseline ({label}), '
          f'{games} games, depth {depth} ===')
    t0 = time.time()
    engine_a = UCIEngine(chess_bin, net_path=net_path, use_nnue=True, depth=depth,
                         movetime_ms=movetime_ms)
    if baseline_net_path:
        engine_b = UCIEngine(chess_bin, net_path=baseline_net_path, use_nnue=True, depth=depth,
                             movetime_ms=movetime_ms)
    else:
        engine_b = UCIEngine(chess_bin, net_path=None, use_nnue=False, depth=depth,
                             movetime_ms=movetime_ms)
    try:
        a_wins, b_wins, draws = play_match(engine_a, engine_b, games)
    finally:
        engine_a.close()
        engine_b.close()
    elapsed = time.time() - t0

    elo, margin = elo_estimate(a_wins, b_wins, draws)
    sprt_result = sprt(a_wins, b_wins, draws, elo0, elo1)

    n = a_wins + b_wins + draws
    print(f'  result: +{a_wins} -{b_wins} ={draws}  ({n} games, {elapsed:.1f}s)')
    print(f'  Elo(candidate - baseline): {elo:+.1f} +/- {margin:.1f}')
    print(f'  SPRT[{elo0},{elo1}]: llr={sprt_result["llr"]:.3f} '
          f'({sprt_result["lower"]:.3f},{sprt_result["upper"]:.3f}) -> {sprt_result["verdict"]}')

    return {'wins': a_wins, 'losses': b_wins, 'draws': draws, 'games': n,
            'elapsed_s': elapsed, 'elo': elo, 'elo_margin': margin, 'sprt': sprt_result,
            'baseline_label': label}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net', required=True, help='candidate .nnue to test')
    ap.add_argument('--baseline-net', default=None,
                     help='previous .nnue to compare against (default: classical evaluator)')
    ap.add_argument('--bin-dir', default=None)
    ap.add_argument('--bench-depth', type=int, default=12)
    ap.add_argument('--games', type=int, default=24, help='Elo match games')
    ap.add_argument('--match-depth', type=int, default=5, help='search depth per move in the Elo match')
    ap.add_argument('--movetime', type=int, default=0, help='ms per move instead of --match-depth (0=off)')
    ap.add_argument('--elo0', type=float, default=0.0, help='SPRT H0 elo bound')
    ap.add_argument('--elo1', type=float, default=10.0, help='SPRT H1 elo bound')
    ap.add_argument('--skip-match', action='store_true', help='run only steps 1-3')
    ap.add_argument('--report', default=None, help='write a JSON report to this path')
    args = ap.parse_args()

    chess_bin = find_binary('chess', args.bin_dir)

    net = step_load(args.net)
    step_verify(chess_bin, args.net, net)
    bench = step_benchmark(chess_bin, args.net, args.baseline_net, args.bench_depth)

    match = None
    if not args.skip_match:
        movetime = args.movetime if args.movetime > 0 else None
        match = step_elo_match(chess_bin, args.net, args.baseline_net, args.games,
                                args.match_depth, movetime, args.elo0, args.elo1)

    report = {
        'net': os.path.abspath(args.net),
        'baseline_net': os.path.abspath(args.baseline_net) if args.baseline_net else None,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'verify': 'PASS',
        'benchmark': bench,
        'elo_match': match,
    }
    report_path = args.report or (args.net + '.test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[test] wrote report -> {report_path}')
    print('[test] OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
