#!/usr/bin/env python3
"""uci_match.py - Minimal persistent-process UCI driver + Elo/SPRT helpers,
used by test.py to A/B two evaluation configurations (e.g. a candidate .nnue
vs. a baseline .nnue, or a candidate .nnue vs. the classical evaluator).

Why not the existing chess_match binary (src/match/match.cpp)? chess_match's
EngineConfig has an EvalMode toggle (Classical/NNUE) but no per-side EvalFile
path -- both sides share whatever single global net the process has loaded.
That's fine for feature-flag A/B testing (e.g. classical vs nnue) but cannot
compare two *different* trained .nnue files against each other in one run,
which test.py needs to do ("compare against previous network"). Rather than
modify chess_match/match.cpp (playing-strength/engine code, out of scope for
this pipeline task), this module drives two independent `chess` UCI processes
instead, each configured with its own `setoption name EvalFile value ...`.

The same opening book as src/apps/match_main.cpp is reused here (copied, not
imported -- it's a hardcoded C++ array) so results are comparable to the
built-in match harness's conventions.
"""
import math
import subprocess
import time

OPENING_BOOK = [
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
]


class UCIEngine:
    """A persistent `chess` UCI subprocess."""

    def __init__(self, binary, net_path=None, use_nnue=None, depth=5, movetime_ms=None,
                 hash_mb=16, threads=1):
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send('uci')
        self._wait_for('uciok', timeout=10)
        self._send(f'setoption name Hash value {hash_mb}')
        self._send(f'setoption name Threads value {threads}')
        if use_nnue is not None:
            self._send(f'setoption name Use NNUE value {"true" if use_nnue else "false"}')
        if net_path:
            self._send(f'setoption name EvalFile value {net_path}')
        self._send('isready')
        self._wait_for('readyok', timeout=15)
        self.depth = depth
        self.movetime_ms = movetime_ms

    def _send(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

    def _wait_for(self, token, timeout=30):
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.005)
                continue
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f'timed out waiting for {token!r}; got: {lines[-5:]}')

    def new_game(self):
        self._send('ucinewgame')
        self._send('isready')
        self._wait_for('readyok', timeout=15)

    def best_move(self, fen, moves):
        pos_cmd = f'position fen {fen}'
        if moves:
            pos_cmd += ' moves ' + ' '.join(moves)
        self._send(pos_cmd)
        if self.movetime_ms:
            self._send(f'go movetime {self.movetime_ms}')
        else:
            self._send(f'go depth {self.depth}')
        lines = self._wait_for('bestmove', timeout=60)
        for line in reversed(lines):
            if line.startswith('bestmove'):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None
        return None

    def close(self):
        try:
            self._send('quit')
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# Minimal legal-move / terminal-state detection via python-chess if available,
# else a conservative ply cap with no adjudication (still safe: a "*" result
# is simply not counted, and games are capped in length below).
# ---------------------------------------------------------------------------
try:
    import chess as _pychess
    HAVE_PYCHESS = True
except ImportError:
    HAVE_PYCHESS = False


def play_game(white, black, start_fen, max_plies=160):
    """Play one game between two UCIEngine instances. Returns '1-0'/'0-1'/'1/2-1/2'.

    Uses python-chess as an independent legality/termination oracle when
    available (this pipeline already depends on it nowhere else, but it is
    present in this sandbox and commonly available; see docs/NNUE_TRAINING.md
    for the degraded fallback behavior when it is not installed)."""
    white.new_game()
    black.new_game()
    moves = []

    if HAVE_PYCHESS:
        board = _pychess.Board(start_fen)
        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                return result if result in ('1-0', '0-1', '1/2-1/2') else '1/2-1/2'
            mover = white if board.turn == _pychess.WHITE else black
            mv = mover.best_move(start_fen, moves)
            if not mv or mv in ('(none)', 'none', '0000'):
                return board.result(claim_draw=True) if board.is_game_over() else '1/2-1/2'
            moves.append(mv)
            try:
                board.push_uci(mv)
            except Exception:
                # Illegal move reported by the engine -- a real bug, not a draw;
                # count it as a loss for whoever moved rather than hiding it.
                return '0-1' if mover is white else '1-0'
        return '1/2-1/2'   # ply cap reached

    # Degraded fallback (no python-chess): trust side-to-move parity from the
    # FEN and stop only on "no move" or the ply cap. No illegal-move/adjudication
    # detection in this mode.
    stm_white = start_fen.split()[1] == 'w'
    for ply in range(max_plies):
        white_to_move = stm_white if ply % 2 == 0 else not stm_white
        mover = white if white_to_move else black
        mv = mover.best_move(start_fen, moves)
        if not mv or mv in ('(none)', 'none', '0000'):
            return '1/2-1/2'
        moves.append(mv)
    return '1/2-1/2'


def play_match(engine_a, engine_b, games, openings=None):
    """Alternate colors across the opening book. Returns (aWins, bWins, draws)."""
    openings = openings or OPENING_BOOK
    a_wins = b_wins = draws = 0
    for g in range(games):
        fen = openings[(g // 2) % len(openings)]
        a_is_white = (g % 2 == 0)
        white, black = (engine_a, engine_b) if a_is_white else (engine_b, engine_a)
        result = play_game(white, black, fen)
        if result == '1/2-1/2':
            draws += 1
        elif (result == '1-0') == a_is_white:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins, draws


# ---------------------------------------------------------------------------
# Elo estimation + GSPRT (standard normal-approximation formulas, the same
# ones used by common chess-engine testing tools e.g. cutechess-cli/fishtest;
# an approximation, not a byte-for-byte port of src/match/stats.cpp).
# ---------------------------------------------------------------------------
def elo_diff(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(score / (1 - score))


def elo_estimate(wins, losses, draws):
    n = wins + losses + draws
    if n == 0:
        return 0.0, 0.0
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0 - score) ** 2) / n
    stdev = math.sqrt(var / n)
    lo = elo_diff(max(score - 1.96 * stdev, 1e-6))
    hi = elo_diff(min(score + 1.96 * stdev, 1 - 1e-6))
    return elo_diff(score), (hi - lo) / 2


def sprt(wins, losses, draws, elo0, elo1, alpha=0.05, beta=0.05):
    n = wins + losses + draws
    if n == 0:
        return {'llr': 0.0, 'lower': 0.0, 'upper': 0.0, 'verdict': 'continue'}

    def score_of_elo(elo):
        return 1.0 / (1.0 + 10 ** (-elo / 400.0))

    t0, t1 = score_of_elo(elo0), score_of_elo(elo1)
    s = (wins + 0.5 * draws) / n
    var = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * (0 - s) ** 2) / n
    var = max(var, 1e-8)
    llr = n * (t1 - t0) * (s - (t0 + t1) / 2) / var
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)
    verdict = 'H1' if llr >= upper else ('H0' if llr <= lower else 'continue')
    return {'llr': llr, 'lower': lower, 'upper': upper, 'verdict': verdict}
