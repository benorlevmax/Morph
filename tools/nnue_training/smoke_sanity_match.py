#!/usr/bin/env python3
"""smoke_sanity_match.py - Tiny (not statistically significant) sanity match:
the Phase-A smoke-trained NNUE net vs. the classical evaluator, both driven
by the SAME engine binary, to confirm the trained net doesn't crash, hang, or
play illegally under real search. This is NOT an Elo measurement -- it's a
functional smoke test, per Phase A's explicit "small Elo sanity match" ask
(a handful of games, not a SPRT run). Do not read the score as a strength
verdict; the net was trained on ~3k self-play positions for 2 epochs on CPU.
"""
import sys
import time
import chess
import chess.engine

ENGINE_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/build_baseline/bin/chess"
NNUE_PATH = sys.argv[2] if len(sys.argv) > 2 else "/tmp/phaseA_data/smoke_A.nnue"
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 6

OPENINGS = [
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
]

nnue_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
nnue_engine.configure({"Hash": 16, "Threads": 1, "Use NNUE": True, "EvalFile": NNUE_PATH})

classical_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
classical_engine.configure({"Hash": 16, "Threads": 1, "Use NNUE": False})

limit = chess.engine.Limit(time=0.2)
results = []
t0 = time.time()
for g in range(GAMES):
    opening = OPENINGS[g % len(OPENINGS)]
    nnue_is_white = (g % 2 == 0)
    board = chess.Board(opening)
    white = nnue_engine if nnue_is_white else classical_engine
    black = classical_engine if nnue_is_white else nnue_engine
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < 120:
        eng = white if board.turn == chess.WHITE else black
        r = eng.play(board, limit)
        if r.move is None:
            break
        board.push(r.move)
        ply += 1
    outcome = board.outcome(claim_draw=True)
    result_str = outcome.result() if outcome else "1/2-1/2 (ply cap)"
    if result_str.startswith("1-0"):
        winner = "nnue" if nnue_is_white else "classical"
    elif result_str.startswith("0-1"):
        winner = "classical" if nnue_is_white else "nnue"
    else:
        winner = "draw"
    results.append((g, nnue_is_white, result_str, ply, winner))
    print(f"game {g}: nnue={'white' if nnue_is_white else 'black'} "
          f"result={result_str} plies={ply} winner={winner}")

nnue_engine.quit()
classical_engine.quit()

wins = sum(1 for r in results if r[4] == "nnue")
losses = sum(1 for r in results if r[4] == "classical")
draws = sum(1 for r in results if r[4] == "draw")
print(f"\nsmoke_A NNUE vs classical: +{wins} -{losses} ={draws}  "
      f"({len(results)} games, {time.time() - t0:.1f}s)")
print("(functional smoke test only -- not statistically meaningful)")
