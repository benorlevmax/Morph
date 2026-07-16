#!/usr/bin/env python3
"""verify_against_engine.py - Reference-verification harness (Phase A,
requirement: "For multiple known positions: extract production NNUE features,
run reference forward pass, load the exported .nnue in the C++ engine,
evaluate the identical position, compare outputs... Fail loudly on any
mismatch.")

Computes NNUE::evaluate()-equivalent centipawn scores for a list of FENs using
the pure-Python reference_nnue.py oracle, then drives the real compiled
`chess` UCI binary (with the same .nnue file loaded via `setoption name
EvalFile`) to get its independently-computed values for the same positions,
and asserts they match EXACTLY (these are integer centipawn scores computed
via bit-identical integer arithmetic on both sides -- any mismatch, even by
1, indicates a real bug in feature indexing, king bucketing, perspective
orientation, output bucketing, quantization, or the binary format, not
floating-point noise to be shrugged off).

Usage:
    python3 verify_against_engine.py <path-to-chess-binary> <path-to-.nnue> \
        [positions_file]

Exit code 0 and "ALL POSITIONS MATCH" iff every position matches exactly;
otherwise exits 1 and prints every mismatch.
"""
import subprocess
import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from reference_nnue import RefNet

DEFAULT_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",
    "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 2 7",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkb1r/pp1p1ppp/2p2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
    "r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
]


def run_engine_evals(engine_path, nnue_path, fens):
    cmds = ["setoption name Use NNUE value true",
            f"setoption name EvalFile value {nnue_path}"]
    for fen in fens:
        cmds.append(f"position fen {fen}")
        cmds.append("eval")
    cmds.append("quit")
    proc = subprocess.run([engine_path], input="\n".join(cmds) + "\n",
                           capture_output=True, text=True, timeout=60)
    evals = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("eval ") and line.endswith(" cp"):
            evals.append(int(line.split()[1]))
    if "nnue loaded" not in proc.stdout and "info string nnue loaded" not in proc.stdout:
        print("WARNING: did not see NNUE-loaded confirmation in engine stdout; "
              "results below may reflect the classical evaluator, not the net.",
              file=sys.stderr)
    return evals


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    engine_path, nnue_path = sys.argv[1], sys.argv[2]
    fens = DEFAULT_POSITIONS
    if len(sys.argv) > 3:
        with open(sys.argv[3]) as f:
            fens = [l.strip() for l in f if l.strip()]

    net = RefNet.load(nnue_path)
    python_evals = [net.evaluate_fen(fen) for fen in fens]
    engine_evals = run_engine_evals(engine_path, nnue_path, fens)

    if len(engine_evals) != len(fens):
        print(f"FAIL: expected {len(fens)} eval outputs from the engine, got "
              f"{len(engine_evals)}. Engine stdout parsing likely broken, or the "
              f"engine crashed/hung on one of these positions.")
        sys.exit(1)

    mismatches = []
    for fen, py, eng in zip(fens, python_evals, engine_evals):
        status = "OK" if py == eng else "MISMATCH"
        print(f"[{status}] python={py:>6}  engine={eng:>6}   {fen}")
        if py != eng:
            mismatches.append((fen, py, eng))

    if mismatches:
        print(f"\nFAIL: {len(mismatches)}/{len(fens)} position(s) mismatched. "
              f"Do not trust this .nnue file or the pipeline that produced it "
              f"until every mismatch is root-caused (check feature indexing, "
              f"king bucket mapping, perspective orientation, output bucket "
              f"selection, quantization scale, or the binary layout, in that "
              f"order of likelihood).")
        sys.exit(1)

    print(f"\nALL {len(fens)} POSITIONS MATCH EXACTLY.")


if __name__ == "__main__":
    main()
