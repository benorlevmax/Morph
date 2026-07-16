#!/usr/bin/env python3
"""selfplay.py - UCI-driven self-play position generation for a community
worker.

Vendored copy of distributed/worker/selfplay.py, byte-for-byte identical in
behavior. Duplicated (not imported via a relative path into distributed/)
on purpose: platform/worker/ is meant to be downloadable and runnable
on its own (see platform/docs/WORKER.md and platform/scripts/) on a
contributor's machine that has never cloned this whole repo -- just this
directory plus a compiled engine binary. If you're modifying self-play
logic, keep this in sync with distributed/worker/selfplay.py or fold both
into a shared package; they're intentionally decoupled for now rather than
sharing a relative import that would break the "just download the worker"
story.

Drives the engine one move at a time over the UCI protocol so it can
capture the REAL per-position search depth and node count the platform's
position schema requires, not just a constant depth label.

Each generated record's `eval_cp` is normalized to be White-relative (the
engine reports scores from the side-to-move's perspective per the UCI spec;
this flips sign when it's Black to move), matching the convention already
used by src/train/dataset.h's Sample struct.

Requires python-chess (pip install chess) -- used as the independent
legality/termination/FEN oracle around the engine's own moves, and there is
no meaningful degraded mode without it (a worker cannot report the FEN a
score belongs to without tracking the board itself). run_platform_worker.py
checks for it at startup and fails with a clear message if it's missing.
"""
import random
import re
import subprocess
import time

import chess as _pychess

MATE_CP_BASE = 30000   # forced-mate scores are encoded as +/-(MATE_CP_BASE - plies_to_mate)

_INFO_RE = re.compile(
    r'info .*?\bdepth (\d+)\b.*?\bscore (cp|mate) (-?\d+)\b.*?\bnodes (\d+)\b')


class SelfPlayEngine:
    """A persistent `chess` UCI subprocess used to generate one worker's
    share of a task via self-play (one engine plays both colors)."""

    def __init__(self, binary, depth=6, movetime_ms=None, hash_mb=16, threads=1):
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.depth = depth
        self.movetime_ms = movetime_ms
        self._send('uci')
        self._wait_for('uciok', timeout=10)
        self._send(f'setoption name Hash value {hash_mb}')
        self._send(f'setoption name Threads value {threads}')
        self._send('isready')
        self._wait_for('readyok', timeout=15)
        self.engine_version = self._query_version()

    def _query_version(self):
        self._send('uci')
        lines = self._wait_for('uciok', timeout=10)
        for line in lines:
            if line.startswith('id name '):
                return line[len('id name '):].strip()
        return 'unknown'

    def _send(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

    def _wait_for(self, token, timeout=60):
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.005)
                continue
            lines.append(line.strip())
            if token in line:
                return lines
        raise TimeoutError(f'timed out waiting for {token!r}; got: {lines[-5:]}')

    def new_game(self):
        self._send('ucinewgame')
        self._send('isready')
        self._wait_for('readyok', timeout=15)

    def search(self, fen, moves):
        """Search the position, return (bestmove_uci, eval_white_relative_cp,
        actual_depth, nodes) -- eval/depth/nodes come from the last 'info'
        line seen before 'bestmove', i.e. the deepest completed iteration."""
        pos_cmd = f'position fen {fen}'
        if moves:
            pos_cmd += ' moves ' + ' '.join(moves)
        self._send(pos_cmd)
        if self.movetime_ms:
            self._send(f'go movetime {self.movetime_ms}')
        else:
            self._send(f'go depth {self.depth}')
        lines = self._wait_for('bestmove', timeout=120)

        best_depth, best_nodes, best_score_cp, is_mate = self.depth, 0, 0, False
        for line in lines:
            m = _INFO_RE.search(line)
            if m:
                best_depth = int(m.group(1))
                is_mate = (m.group(2) == 'mate')
                best_score_cp = int(m.group(3))
                best_nodes = int(m.group(4))

        board = _pychess.Board(fen)
        for mv in moves:
            board.push_uci(mv)
        stm_white = board.turn == _pychess.WHITE

        if is_mate:
            mate_plies = best_score_cp
            magnitude = max(MATE_CP_BASE - abs(mate_plies), 1)
            score_stm = magnitude if mate_plies > 0 else -magnitude
        else:
            score_stm = best_score_cp
        eval_white = score_stm if stm_white else -score_stm

        bestmove = None
        for line in reversed(lines):
            if line.startswith('bestmove'):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
                break
        return bestmove, eval_white, best_depth, best_nodes

    def close(self):
        try:
            self._send('quit')
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _random_opening(start_fen, plies, rng):
    """Play `plies` uniformly-random legal moves from `start_fen` for game
    diversity (mirrors src/train/selfplay.cpp's --randomplies)."""
    if plies <= 0:
        return []
    board = _pychess.Board(start_fen)
    moves = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        moves.append(mv.uci())
        board.push(mv)
    return moves


def play_selfplay_game(engine, start_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                        randomplies=6, max_plies=200, rng=None):
    """Play one full self-play game (engine vs itself) and return a list of
    position records: {fen, side_to_move, eval_cp, depth, nodes,
    engine_version}, plus `result` (White-relative game outcome) backfilled
    onto every record once the game ends."""
    rng = rng or random.Random()
    engine.new_game()

    opening_moves = _random_opening(start_fen, randomplies, rng)
    moves = list(opening_moves)
    records = []

    board = _pychess.Board(start_fen)
    for uci_mv in opening_moves:
        board.push_uci(uci_mv)

    for _ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        stm = 'w' if board.turn == _pychess.WHITE else 'b'

        bestmove, eval_white, depth, nodes = engine.search(start_fen, moves)
        if not bestmove or bestmove in ('(none)', 'none', '0000'):
            break

        records.append({
            'fen': board.fen(),
            'side_to_move': stm,
            'eval_cp': eval_white,
            'depth': depth,
            'nodes': nodes,
            'engine_version': engine.engine_version,
        })

        moves.append(bestmove)
        try:
            board.push_uci(bestmove)
        except Exception:
            break

    result = 0.5
    if board.is_game_over(claim_draw=True):
        outcome = board.result(claim_draw=True)
        result = {'1-0': 1.0, '0-1': 0.0, '1/2-1/2': 0.5}.get(outcome, 0.5)

    for r in records:
        r['result'] = result

    return records
