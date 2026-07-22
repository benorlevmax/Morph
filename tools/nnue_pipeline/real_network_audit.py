#!/usr/bin/env python3
"""real_network_audit.py - Phases B/C/D of the NNUE forensic audit (see
../../NNUE_FORENSIC_AUDIT_REPORT.md and NNUE_TRAINING_PIPELINE_AUDIT.md), run
against the REAL trained .nnue file (and, for --check-bucket-coverage, the
REAL training dataset JSONL). This cannot run in a sandbox that doesn't have
those files, so it's meant to be run locally on the machine that has them.

Background for why --check-bucket-coverage exists: a controlled pilot
experiment (synthetic, perfectly-labeled data, run in the audit sandbox)
found that NNUE_OUT_BUCKETS=8 selects a bucket by piece count
(`output_bucket()` in nnue_format.py), and each bucket has its own
independently-trained output-layer weight slice. A bucket that gets few or
no training examples does NOT fail loudly -- it either stays at its
near-zero initialization (evaluates ~flat 0 regardless of material, seen
directly in the pilot for the low-piece-count bucket when the synthetic
dataset never reached low piece counts) or, if it gets a *little* data but
not enough, overfits into noisy, frequently WRONG-SIGN evaluations (also
reproduced directly in the pilot: a network trained on a small, evenly-
spread-across-all-8-buckets dataset gave "+11 cp" for a position where Black
is up a whole queen). Either failure mode is invisible in aggregate loss
curves (which average across all buckets) but would make the engine actively
misjudge whichever piece-count range is affected -- exactly the kind of thing
that could turn "weak" into "loses every game". --check-bucket-coverage
answers the one question that pilot couldn't: whether the REAL dataset's
buckets are actually balanced.

Three things this script does:

  1. --check-bucket-coverage <dataset.jsonl>: the cheapest, most decisive
     check to run first. Reads the real training dataset (same JSONL
     train.py consumes) and reports how many samples fall into each of the
     8 output buckets. A bucket with a tiny fraction of the total is a
     strong, direct explanation for bad play in that piece-count range.

  2. Bulk network health stats (Phase D): plays random legal games with
     python-chess, evaluates every position reached with the real net (via
     nnue_format.RefNet, the same pure-Python reference already verified
     byte-for-byte against the compiled engine), and reports, BOTH overall
     and broken down per output bucket:
       - mean/variance of the raw cp output
       - % of accumulator neurons sitting at the clip boundaries (0 or
         CR_MAX) -- high saturation would indicate a scaling problem
       - dead-neuron count -- neurons whose activation never leaves 0
         across the whole sample (never contributes to any evaluation)
     Plus raw weight-level statistics (mean/std/min/max/%zero/dead rows) for
     the feature-transformer and output-layer weight matrices themselves.

  3. Per-move game trace (Phase C): plays N full games, NNUE candidate vs.
     classical baseline, via the real compiled engine over UCI, logging
     FEN / side-to-move / classical eval / NNUE eval / bestmove / depth for
     every ply of every LOSS, so you get the exact per-move data the audit
     needs instead of just a final score.

Usage:
    python3 real_network_audit.py --check-bucket-coverage path\to\dataset.jsonl
    python3 real_network_audit.py --net path\to\network.nnue --bin-dir path\to\worker\folder

Requires: numpy is NOT needed here (bulk/weight stats use RefNet, pure
stdlib). python-chess (`pip install chess`) is required for legal move
generation (bulk stats and game traces only -- not needed for
--check-bucket-coverage).
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnue_format import RefNet, CR_MIN, CR_MAX, NNUE_HL, WHITE, BLACK  # noqa: E402
from engine_paths import find_binary, run_uci  # noqa: E402

try:
    import chess
except ImportError:
    sys.exit('This script needs python-chess: pip install chess')


def check_bucket_coverage(dataset_paths):
    """Reads real dataset JSONL file(s) (train.py's own format: one
    {"fen"/"eval"/"eval_cp","result",...} object per line) and reports how
    many samples fall into each of the 8 output buckets by piece count. This
    is the cheapest, most direct test of the output-bucket-starvation
    hypothesis: a bucket with a near-zero share of the data is a strong,
    concrete explanation for bad/wrong-sign play in that piece-count range,
    independent of anything else in the pipeline."""
    from nnue_format import parse_fen_board, output_bucket, NNUE_OUT_BUCKETS
    counts = [0] * NNUE_OUT_BUCKETS
    total = 0
    bad = 0
    for path in dataset_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    board, _ = parse_fen_board(rec['fen'])
                except Exception:
                    bad += 1
                    continue
                counts[output_bucket(len(board))] += 1
                total += 1
    print(f'\n=== Output-bucket coverage across {total} real training sample(s) '
          f'({bad} unparseable line(s) skipped) ===')
    print(f'{"bucket":>8} {"piece range":>14} {"count":>10} {"% of total":>12}')
    # output_bucket(n) = clamp((n-1)//4, 0, 7); invert to show each bucket's
    # piece-count range for readability.
    ranges = []
    for b in range(NNUE_OUT_BUCKETS):
        lo = b * 4 + 1
        hi = (b + 1) * 4 if b < NNUE_OUT_BUCKETS - 1 else 32
        ranges.append((lo, hi))
    starved = []
    for b in range(NNUE_OUT_BUCKETS):
        pct = 100.0 * counts[b] / total if total else 0.0
        lo, hi = ranges[b]
        print(f'{b:>8} {f"{lo}-{hi}":>14} {counts[b]:>10} {pct:>11.2f}%')
        if total and pct < (100.0 / NNUE_OUT_BUCKETS) / 4:  # < 1/4 of the even share
            starved.append(b)
    if starved:
        print(f'\nFLAG: bucket(s) {starved} have well under their even 1/{NNUE_OUT_BUCKETS} '
              f'share of the data -- these piece-count ranges are at real risk of the flat-zero '
              f'or overfit-noisy-sign failure modes reproduced in the pilot experiment. Expect '
              f'unreliable evaluations specifically in that piece-count range.')
    return counts


def weight_stats(net_path):
    """Raw weight-level health check (Phase 3's 'Weights' checklist): mean,
    std, min, max, % exact zero, and dead rows (a feature-transformer row --
    one HalfKP feature -- whose weight vector is all exact zero, meaning
    that feature has never been updated away from its... well, init isn't
    zero, so an all-zero row post-training would itself be a red flag) for
    both the feature-transformer and output-layer weight matrices."""
    from nnue_format import RefNet, NNUE_HL
    net = RefNet.load(net_path)
    print(f'\n=== Phase 3: raw weight statistics ({net_path}) ===')

    def stats(flat_vals, name):
        n = len(flat_vals)
        mean = sum(flat_vals) / n
        var = sum((v - mean) ** 2 for v in flat_vals) / n
        std = var ** 0.5
        zeros = sum(1 for v in flat_vals if v == 0)
        print(f'  {name}: n={n} mean={mean:.3f} std={std:.3f} '
              f'min={min(flat_vals)} max={max(flat_vals)} pct_zero={100.0*zeros/n:.2f}%')

    ft_flat = [v for row in net.ft_weights for v in row]
    stats(ft_flat, 'ft_weights (feature transformer, 10240 x 512)')
    stats(net.ft_bias, 'ft_bias (512)')
    out_flat = [v for row in net.out_weights for v in row]
    stats(out_flat, 'out_weights (8 buckets x 1024)')
    stats(net.out_bias, 'out_bias (8)')

    dead_rows = sum(1 for row in net.ft_weights if all(v == 0 for v in row))
    print(f'  dead ft_weights rows (entire feature never contributes, all-zero row): '
          f'{dead_rows} / {len(net.ft_weights)}')
    if dead_rows > len(net.ft_weights) * 0.05:
        print(f'  FLAG: >5% of features are completely dead -- consistent with those '
              f'king-bucket/piece/square combinations never appearing (or appearing too '
              f'rarely) in the training data.')


def random_legal_walk(max_plies, rng):
    """Yields FENs from a random-legal-move game (no engine involved --
    pure random walk, so this samples a much wider variety of positions
    than self-play would, including ones the net may never have trained
    on closely)."""
    board = chess.Board()
    fens = [board.fen()]
    for _ in range(max_plies):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))
        fens.append(board.fen())
    return fens


def bulk_stats(net_path, n_positions, plies_per_walk, seed):
    from nnue_format import output_bucket, NNUE_OUT_BUCKETS
    print(f'\n=== Phase D: bulk network health over ~{n_positions} positions ===')
    net = RefNet.load(net_path)
    rng = random.Random(seed)

    outputs = []
    outputs_by_bucket = {b: [] for b in range(NNUE_OUT_BUCKETS)}
    neuron_seen_nonzero = [False] * NNUE_HL  # own-perspective activation, either color
    neuron_saturated_hi = 0
    neuron_saturated_lo = 0
    neuron_total = 0

    collected = 0
    while collected < n_positions:
        for fen in random_legal_walk(plies_per_walk, rng):
            if collected >= n_positions:
                break
            try:
                board, stm = None, None
                from nnue_format import parse_fen_board
                board, stm = parse_fen_board(fen)
                if not any(pt == 6 and c == WHITE for _, (c, pt) in board.items()):
                    continue
                if not any(pt == 6 and c == BLACK for _, (c, pt) in board.items()):
                    continue
            except Exception:
                continue

            cp = net.evaluate_fen(fen)
            outputs.append(cp)
            outputs_by_bucket[output_bucket(len(board))].append(cp)

            wk = next(s for s, (c, pt) in board.items() if c == WHITE and pt == 6)
            bk = next(s for s, (c, pt) in board.items() if c == BLACK and pt == 6)
            king_of = {WHITE: wk, BLACK: bk}
            for persp in (WHITE, BLACK):
                acc = net.accumulator(board, persp, king_of[persp])
                for i, x in enumerate(acc):
                    neuron_total += 1
                    if x != 0:
                        neuron_seen_nonzero[i] = True
                    if x <= CR_MIN:
                        neuron_saturated_lo += 1
                    elif x >= CR_MAX:
                        neuron_saturated_hi += 1
            collected += 1

    dead = sum(1 for seen in neuron_seen_nonzero if not seen)
    mean_cp = statistics.mean(outputs)
    stdev_cp = statistics.pstdev(outputs)
    pct_pos = 100.0 * sum(1 for v in outputs if v > 0) / len(outputs)
    pct_neg = 100.0 * sum(1 for v in outputs if v < 0) / len(outputs)
    pct_zero = 100.0 * sum(1 for v in outputs if v == 0) / len(outputs)

    per_bucket = {}
    print(f'\n  per-output-bucket breakdown (this is where a flat-zero or noisy-sign '
          f'bucket -- see the pilot experiment in the audit report -- would show up):')
    print(f'  {"bucket":>8} {"n":>8} {"mean_cp":>10} {"stdev_cp":>10} {"pct_zero":>10}')
    for b in range(NNUE_OUT_BUCKETS):
        vals = outputs_by_bucket[b]
        if not vals:
            print(f'  {b:>8} {0:>8}   (no positions reached this bucket in this sample)')
            per_bucket[b] = None
            continue
        bmean = statistics.mean(vals)
        bstd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        bzero = 100.0 * sum(1 for v in vals if v == 0) / len(vals)
        print(f'  {b:>8} {len(vals):>8} {bmean:>10.1f} {bstd:>10.1f} {bzero:>9.1f}%')
        per_bucket[b] = {'n': len(vals), 'mean_cp': bmean, 'stdev_cp': bstd, 'pct_zero': bzero}
        if bstd < 15 or bzero > 50:
            print(f'    FLAG: bucket {b} looks flat/uninformative (stdev={bstd:.1f}, '
                  f'{bzero:.0f}% exactly zero) -- likely undertrained for this piece-count range.')

    result = {
        'n_positions': len(outputs),
        'mean_cp': mean_cp,
        'stdev_cp': stdev_cp,
        'min_cp': min(outputs),
        'max_cp': max(outputs),
        'pct_positive': pct_pos,
        'pct_negative': pct_neg,
        'pct_exactly_zero': pct_zero,
        'dead_neurons_of_512': dead,
        'pct_neurons_saturated_low': 100.0 * neuron_saturated_lo / neuron_total,
        'pct_neurons_saturated_high': 100.0 * neuron_saturated_hi / neuron_total,
        'per_output_bucket': per_bucket,
    }
    print()
    print(json.dumps({k: v for k, v in result.items() if k != 'per_output_bucket'}, indent=2))
    if dead > 50:
        print(f'FLAG: {dead}/512 neurons never activated across this sample -- '
              f'consistent with an undertrained network.')
    if stdev_cp < 20:
        print(f'FLAG: output stdev is very low ({stdev_cp:.1f} cp) -- the net is barely '
              f'distinguishing positions, consistent with near-random initialization '
              f'never having moved much during training.')
    return result


def losing_game_trace(chess_bin, net_path, n_games, depth, seed, out_json):
    print(f'\n=== Phase C: per-move trace for up to {n_games} game(s) ===')
    rng = random.Random(seed)
    games_logged = []

    for g in range(n_games):
        board = chess.Board()
        # Candidate (NNUE) plays White on even games, Black on odd, for balance.
        candidate_white = (g % 2 == 0)
        moves_log = []

        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200:
            fen = board.fen()
            candidate_to_move = (board.turn == chess.WHITE) == candidate_white

            cmds = ['setoption name Use NNUE value true', f'setoption name EvalFile value {net_path}',
                    f'position fen {fen}', 'eval']
            nnue_out = run_uci(chess_bin, cmds + ['quit'], timeout=30)
            nnue_eval = next((int(l.split()[1]) for l in nnue_out.splitlines()
                               if l.strip().startswith('eval ') and l.strip().endswith(' cp')), None)

            cmds2 = ['setoption name Use NNUE value false', f'position fen {fen}', 'eval']
            classical_out = run_uci(chess_bin, cmds2 + ['quit'], timeout=30)
            classical_eval = next((int(l.split()[1]) for l in classical_out.splitlines()
                                    if l.strip().startswith('eval ') and l.strip().endswith(' cp')), None)

            use_nnue = candidate_to_move
            search_cmds = [f'setoption name Use NNUE value {"true" if use_nnue else "false"}']
            if use_nnue:
                search_cmds.append(f'setoption name EvalFile value {net_path}')
            search_cmds += [f'position fen {fen}', f'go depth {depth}']
            search_out = run_uci(chess_bin, search_cmds + ['quit'], timeout=60)
            bestmove = None
            last_score = None
            for line in search_out.splitlines():
                line = line.strip()
                if line.startswith('bestmove'):
                    bestmove = line.split()[1]
                if line.startswith('info') and ' score cp ' in line:
                    last_score = int(line.split('score cp')[1].split()[0])

            moves_log.append({
                'ply': len(moves_log) + 1, 'fen': fen, 'side_to_move': 'w' if board.turn else 'b',
                'mover_is_candidate': candidate_to_move, 'classical_eval_cp': classical_eval,
                'nnue_eval_cp': nnue_eval, 'search_bestmove': bestmove,
                'search_score_cp': last_score, 'depth': depth,
            })

            if bestmove is None or bestmove not in [m.uci() for m in board.legal_moves]:
                break
            board.push_uci(bestmove)

        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else '*'
        candidate_lost = (result == '1-0' and not candidate_white) or \
                          (result == '0-1' and candidate_white)
        print(f'  game {g+1}: candidate={"White" if candidate_white else "Black"} '
              f'result={result} candidate_lost={candidate_lost} plies={len(moves_log)}')
        games_logged.append({
            'game': g + 1, 'candidate_color': 'white' if candidate_white else 'black',
            'result': result, 'candidate_lost': candidate_lost, 'moves': moves_log,
        })

    with open(out_json, 'w') as f:
        json.dump(games_logged, f, indent=2)
    print(f'\nWrote {len(games_logged)} game trace(s) -> {out_json}')
    n_losses = sum(1 for g in games_logged if g['candidate_lost'])
    print(f'{n_losses}/{len(games_logged)} games logged were candidate losses.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--net', default=None, help='path to the real trained .nnue file')
    ap.add_argument('--check-bucket-coverage', nargs='+', default=None, metavar='DATASET.jsonl',
                     help='run ONLY the output-bucket coverage check against these real training '
                          'dataset JSONL file(s), then exit -- no .nnue file or engine binary needed. '
                          'Run this first; it is the cheapest, most direct diagnostic.')
    ap.add_argument('--bin-dir', default=None, help='folder containing chess.exe (default: auto-detect)')
    ap.add_argument('--bulk-positions', type=int, default=3000)
    ap.add_argument('--bulk-plies-per-walk', type=int, default=60)
    ap.add_argument('--games', type=int, default=10)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--report', default='real_network_audit_report.json')
    ap.add_argument('--game-trace', default='real_network_game_traces.json')
    ap.add_argument('--skip-bulk', action='store_true')
    ap.add_argument('--skip-games', action='store_true')
    ap.add_argument('--skip-weights', action='store_true')
    args = ap.parse_args()

    if args.check_bucket_coverage:
        check_bucket_coverage(args.check_bucket_coverage)
        return

    if not args.net:
        raise SystemExit('--net is required unless using --check-bucket-coverage')

    if not args.skip_weights:
        weight_stats(args.net)

    bulk_result = None
    if not args.skip_bulk:
        bulk_result = bulk_stats(args.net, args.bulk_positions, args.bulk_plies_per_walk, args.seed)
        with open(args.report, 'w') as f:
            json.dump(bulk_result, f, indent=2)
        print(f'\nWrote bulk stats -> {args.report}')

    if not args.skip_games:
        chess_bin = find_binary('chess', args.bin_dir)
        losing_game_trace(chess_bin, args.net, args.games, args.depth, args.seed, args.game_trace)

    print('\n[real_network_audit] done. Send back real_network_audit_report.json and '
          'real_network_game_traces.json (or paste their contents) to fold into the forensic report.')


if __name__ == '__main__':
    main()
