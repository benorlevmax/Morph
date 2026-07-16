#!/usr/bin/env python3
"""regression_match.py - Binary-vs-binary SPRT regression gate.

Answers a different question than platform/server/auto_pipeline.py's
promotion loop (which SPRT-tests a *trained network* against the current
strongest network, both loaded into the SAME compiled `chess` binary). This
script instead SPRT-tests two *compiled engine binaries* against each other
directly -- e.g. `chess` built from `main` vs. `chess` built from a PR
branch, or yesterday's release vs. today's local build -- to catch a real
playing-strength regression from an engine/search/eval source change before
it merges. Neither existing match tool in this repo does that:

  - src/match/match.cpp (chess_match CLI) only ever runs two EngineConfigs
    of the SAME compiled process in-process (e.g. materialOnly vs full
    eval) -- it cannot load a second binary at all.
  - tools/nnue_pipeline/uci_match.py's own callers (platform/server/
    auto_pipeline.py, test.py) always point both UCIEngine instances at the
    same `binary`, varying only which .nnue EvalFile each one loads.

uci_match.py's UCIEngine already accepts an arbitrary `binary` path per
side, though, so no new match-driving code is needed here -- this script is
a thin CLI wrapper that points UCIEngine at two *different* binaries and
reuses uci_match.py's existing play_match()/sprt() (the same sprt() function
platform/server/auto_pipeline.py's promotion loop and test.py's own
accept/reject verdicts already use) unmodified.

Non-regression convention (matches common fishtest-style STC/LTC gates):
default --sprt-elo0 -5 --sprt-elo1 0 means "H1 = candidate is not measurably
weaker than baseline (consistent with 0 Elo change), H0 = candidate is at
least 5 Elo weaker (a real regression)". A 'continue' verdict (not enough
games to decide either way) is NOT treated as a failure -- only a decisive
H0 fails the gate; see --games to trade run time for statistical power.

Usage:
    python3 regression_match.py --baseline-bin ./chess_old --candidate-bin ./chess_new \
        --games 200 --depth 6

Exit code: 0 if the verdict is H1 or 'continue' (no proven regression),
1 if the verdict is H0 (a statistically decisive regression was detected).
"""
import argparse
import sys
import time

from uci_match import UCIEngine, play_match, sprt, elo_estimate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline-bin', required=True, help='path to the known-good chess UCI binary')
    ap.add_argument('--candidate-bin', required=True, help='path to the binary being tested')
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--depth', type=int, default=6, help='fixed search depth per move (ignored if --movetime-ms is set)')
    ap.add_argument('--movetime-ms', type=int, default=None, help='fixed movetime per move instead of a fixed depth')
    ap.add_argument('--hash-mb', type=int, default=16)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--sprt-elo0', type=float, default=-5.0,
                     help='H0 boundary: candidate this many Elo weaker than baseline counts as a regression')
    ap.add_argument('--sprt-elo1', type=float, default=0.0,
                     help='H1 boundary: candidate at least this many Elo -- 0 means "no worse"')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--beta', type=float, default=0.05)
    args = ap.parse_args()

    print(f'[regression_match] baseline={args.baseline_bin}  candidate={args.candidate_bin}')
    print(f'[regression_match] games={args.games} depth={args.depth} '
          f'movetime_ms={args.movetime_ms} sprt=[{args.sprt_elo0:+.1f}, {args.sprt_elo1:+.1f}] '
          f'alpha={args.alpha} beta={args.beta}')

    t0 = time.time()
    candidate = UCIEngine(args.candidate_bin, depth=args.depth, movetime_ms=args.movetime_ms,
                           hash_mb=args.hash_mb, threads=args.threads)
    baseline = UCIEngine(args.baseline_bin, depth=args.depth, movetime_ms=args.movetime_ms,
                          hash_mb=args.hash_mb, threads=args.threads)
    try:
        # play_match(engine_a, engine_b, games) -> (a_wins, b_wins, draws), colors
        # alternated across its opening book. engine_a = candidate here, so
        # "wins"/"losses" below are from the candidate's point of view --
        # exactly what sprt()/elo_estimate() expect (same convention
        # auto_pipeline.py's _sprt_verdict/maybe_promote_candidates use).
        cand_wins, base_wins, draws = play_match(candidate, baseline, args.games)
    finally:
        candidate.close()
        baseline.close()

    elapsed = time.time() - t0
    total = cand_wins + base_wins + draws
    elo, margin = elo_estimate(cand_wins, base_wins, draws)
    verdict = sprt(cand_wins, base_wins, draws, args.sprt_elo0, args.sprt_elo1,
                    alpha=args.alpha, beta=args.beta)

    print(f'[regression_match] {total} games in {elapsed:.1f}s: '
          f'candidate +{cand_wins} -{base_wins} ={draws}, Elo {elo:+.1f} +/- {margin:.1f}')
    print(f"[regression_match] SPRT llr={verdict['llr']:.2f} "
          f"(bounds [{verdict['lower']:.2f}, {verdict['upper']:.2f}]) -> {verdict['verdict']}")

    if verdict['verdict'] == 'H0':
        print(f"[regression_match] REGRESSION DETECTED: candidate is statistically at least "
              f"{-args.sprt_elo0:.1f} Elo weaker than baseline (alpha={args.alpha}, "
              f"beta={args.beta}) -- failing")
        return 1
    if verdict['verdict'] == 'H1':
        print('[regression_match] PASS: no statistically significant regression '
              '(candidate is consistent with 0 Elo change or better)')
    else:
        print(f"[regression_match] PASS (inconclusive): SPRT verdict still 'continue' after "
              f"{total} games -- not enough evidence of a regression either way. Run more "
              f"games (--games) for a more decisive verdict; this is NOT treated as a failure "
              f"since only a decisive H0 proves a regression.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
