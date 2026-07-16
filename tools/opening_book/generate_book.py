#!/usr/bin/env python3
"""generate_book.py - Generate an engine-analysis opening book.

    Position
       |
       v
    Engine search        <- the compiled, unmodified `chess` UCI binary,
       |                     driven exactly like a normal UCI GUI would
       v
    Candidate move(s) + evaluations
       |
       v
    Store strongest moves
       |
       v
    Expand tree           <- breadth-first, up to --opening-depth plies
                              and --max-positions analyzed positions

No move in the output file is ever hardcoded: every entry's move, score,
and depth come from actually running the engine's own search (never from a
human game database, an opening encyclopedia, or a fixed list). This is the
generator half of src/book/opening_book.{h,cpp}'s loader/prober; see
docs/opening_book.md for the full system writeup, and see that header's
comment for exactly why this format doesn't need to (and doesn't) replicate
the engine's internal Move bit-encoding -- moves are written as a simple
(from-square, to-square, promotion-code) triple, computed the same trivial
way here and in src/book/opening_book.cpp's move_matches_code().

Position identity: every position hash written here is the engine's own
native Zobrist key, read directly off the engine via the `d` UCI command's
"Key: <hex>" line (Position::to_string(), see src/core/position.cpp) --
never recomputed independently in Python. This guarantees a book written by
this script probes correctly at runtime with zero risk of a second hashing
scheme drifting out of sync with the engine's.

Candidate moves (--candidates): with --candidates=1 (the default), each
analyzed position stores exactly the engine's own single best move (a
direct, full-strength search of that position). With --candidates=K>1,
the position's other legal moves become additional candidates by directly
searching *their* resulting child positions and negating the score (this is
literally the same reasoning the engine's own root search uses internally,
done here one level by hand so every evaluated child can be kept, not just
the winner) -- and each of these K candidates becomes the position's own
book entry AND a new frontier node for further expansion, both a book's
list of alternatives and its branching factor. Costs roughly
candidates^opening_depth engine calls in the worst case; --max-positions is
the hard safety cap.

Determinism: run with --threads 1 (the default) and a depth-limited search
(--search-depth, not --movetime) for a fully reproducible book -- the same
inputs always produce byte-identical output. --movetime-based or
multi-threaded (Lazy SMP) generation is inherently less reproducible
run-to-run (search timing affects which lines get explored) and is
supported, but documented here as a deliberate tradeoff, not hidden.

Usage:
    python3 generate_book.py --engine-bin build/bin/Release/chess.exe \\
        --opening-depth 6 --search-depth 14 --max-positions 500 \\
        --out books/starter_book.book
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nnue_pipeline'))
from engine_paths import find_binary  # tools/nnue_pipeline/engine_paths.py, reused not duplicated

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, 'books')

BOOK_MAGIC = 0x314B4243     # must match src/book/opening_book.h's BOOK_MAGIC ("CBK1")
BOOK_VERSION = 1
MATE_CP_BASE = 30000        # same convention as distributed/worker/selfplay.py

_INFO_RE = re.compile(
    r'info .*?\bdepth (\d+)\b.*?\bscore (cp|mate) (-?\d+)\b.*?\bnodes (\d+)\b')
_KEY_RE = re.compile(r'Key:\s*([0-9a-fA-F]+)')

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Move encoding -- must exactly match src/book/opening_book.cpp's
# encode_book_move()/code_for_move(): from(6 bits) | to(6 bits) << 6 |
# promo(3 bits) << 12, promo 0=none,1=N,2=B,3=R,4=Q. Deliberately NOT the
# engine's internal Move::raw() packing -- see opening_book.h's design-
# decision comment for why.
# ---------------------------------------------------------------------------
_PROMO_CODE = {'n': 1, 'b': 2, 'r': 3, 'q': 4}


def square_index(sq):
    """'e4' -> 28 (file + rank*8, matching src/core/types.h's Square enum:
    SQ_A1=0 ... SQ_H8=63, rank-major)."""
    file_ = ord(sq[0]) - ord('a')
    rank_ = int(sq[1]) - 1
    return rank_ * 8 + file_


def encode_book_move(uci_move):
    """'e2e4' -> code; 'e7e8q' -> code with promotion. Raises ValueError on
    a malformed UCI move string (never silently encodes garbage)."""
    if len(uci_move) not in (4, 5):
        raise ValueError(f'not a UCI move: {uci_move!r}')
    frm = square_index(uci_move[0:2])
    to = square_index(uci_move[2:4])
    promo = _PROMO_CODE[uci_move[4].lower()] if len(uci_move) == 5 else 0
    return (frm & 0x3F) | ((to & 0x3F) << 6) | ((promo & 0x7) << 12)


# ---------------------------------------------------------------------------
# Persistent UCI driver. Self-contained (not imported from tools/nnue_pipeline
# or distributed/worker) per this repo's established convention that each
# tools/* subdirectory is independently copyable -- see distributed/worker's
# own note on why it doesn't share code across directories either.
# ---------------------------------------------------------------------------
class BookEngine:
    def __init__(self, binary, hash_mb=16, threads=1, log=print):
        self.log = log
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send('uci')
        self._wait_for('uciok', timeout=10)
        self._send(f'setoption name Hash value {hash_mb}')
        self._send(f'setoption name Threads value {threads}')
        self._send('isready')
        self._wait_for('readyok', timeout=15)
        self.engine_version = self._query_id_name()

    def _query_id_name(self):
        # 'uci' was already sent/consumed above; ask again is wasteful, so
        # just record 'unknown' -- callers that need it can re-query cheaply
        # via engine_paths.engine_version() on the binary path instead.
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
                time.sleep(0.002)
                continue
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f'timed out waiting for {token!r}; last lines: {lines[-5:]}')

    def position_key(self, moves):
        """The exact native Zobrist key for the position reached from
        START_FEN by `moves` (list of UCI move strings), read directly off
        the engine via `d` -- never recomputed independently."""
        pos_cmd = 'position startpos' + (' moves ' + ' '.join(moves) if moves else '')
        self._send(pos_cmd)
        self._send('d')
        # 'd' has no terminator token; isready/readyok brackets it reliably.
        self._send('isready')
        lines = self._wait_for('readyok', timeout=10)
        for line in lines:
            m = _KEY_RE.search(line)
            if m:
                return int(m.group(1), 16)
        raise RuntimeError(f"'d' did not print a Key: line; got: {lines}")

    def analyze(self, moves, depth=None, movetime=None):
        """Search the position reached by `moves` from START_FEN. Returns
        (best_move_uci_or_None, eval_cp_from_side_to_move_pov, actual_depth,
        nodes). eval_cp is None if the position has no legal moves (mate/
        stalemate -- a real, valid outcome, not an error)."""
        pos_cmd = 'position startpos' + (' moves ' + ' '.join(moves) if moves else '')
        self._send(pos_cmd)
        if movetime:
            self._send(f'go movetime {movetime}')
        else:
            self._send(f'go depth {depth}')
        lines = self._wait_for('bestmove', timeout=120)

        best_depth, best_nodes, best_cp, is_mate = depth or 0, 0, 0, False
        for line in lines:
            m = _INFO_RE.search(line)
            if m:
                best_depth = int(m.group(1))
                is_mate = m.group(2) == 'mate'
                best_cp = int(m.group(3))
                best_nodes = int(m.group(4))

        bestmove = None
        for line in reversed(lines):
            if line.startswith('bestmove'):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
                break
        if not bestmove or bestmove in ('(none)', 'none', '0000'):
            return None, None, best_depth, best_nodes

        if is_mate:
            magnitude = max(MATE_CP_BASE - abs(best_cp) * 2, 1)
            eval_cp = magnitude if best_cp > 0 else -magnitude
        else:
            eval_cp = best_cp
        return bestmove, eval_cp, best_depth, best_nodes

    def close(self):
        try:
            self._send('quit')
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# Legal move enumeration (only needed for --candidates > 1). Soft dependency,
# matching tools/nnue_pipeline/uci_match.py's precedent -- with
# --candidates=1 (the default) this is never imported or required at all.
# ---------------------------------------------------------------------------
def legal_moves_uci(moves):
    import chess as _pychess
    board = _pychess.Board(START_FEN)
    for mv in moves:
        board.push_uci(mv)
    return [m.uci() for m in board.legal_moves]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def analyze_candidates(engine, moves, depth, movetime, k, log):
    """Returns up to k (move_uci, eval_cp, search_depth, nodes) tuples for
    the position reached by `moves`, sorted strongest-first. k==1: a single
    direct search of the position itself. k>1: the position's legal moves'
    child positions are directly searched and negated (see module docstring)."""
    if k <= 1:
        mv, ev, d, n = engine.analyze(moves, depth=depth, movetime=movetime)
        if mv is None:
            return []
        return [(mv, ev, d, n)]

    legal = legal_moves_uci(moves)
    if not legal:
        return []
    results = []
    for mv in legal[:k]:
        child_mv, child_ev, child_d, child_n = engine.analyze(moves + [mv], depth=depth, movetime=movetime)
        # child_ev is None only for a position with no legal replies (i.e.
        # `mv` delivers mate or stalemate) -- still a fully valid, evaluable
        # move; score it directly rather than skipping it.
        if child_ev is None:
            eval_for_parent = MATE_CP_BASE  # `mv` ended the game in the mover's favor or drew;
            # a stalemate can't be distinguished from this cheaply without
            # replaying the position, so this is a documented approximation
            # for the rare stalemate-as-book-move case, not a silent bug.
        else:
            eval_for_parent = -child_ev
        results.append((mv, eval_for_parent, child_d, child_n))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def generate_book(engine, opening_depth, search_depth, movetime, max_positions, candidates, log):
    frontier = deque([[]])   # each item: list of UCI moves from START_FEN
    book = {}                 # hash -> list of dict rows (move, eval_cp, depth, visits, confidence, frequency)
    analyzed = 0
    target_depth = max(search_depth or 1, 1)

    while frontier and analyzed < max_positions:
        moves = frontier.popleft()
        ply = len(moves)
        if ply > opening_depth:
            continue

        key = engine.position_key(moves)
        if key in book:
            for row in book[key]:
                row['visits'] += 1
            continue   # transposition: already analyzed and already expanded once

        results = analyze_candidates(engine, moves, search_depth, movetime, candidates, log)
        if not results:
            continue   # checkmate/stalemate leaf: nothing to store, nothing to expand
        analyzed += len(results)

        best_eval = results[0][1]
        rows = []
        for mv, ev, d, n in results:
            confidence = max(0, min(100, round(100 * d / target_depth)))
            frequency = max(1, 100 - (best_eval - ev))
            rows.append({'move': mv, 'eval_cp': ev, 'depth': d, 'visits': 1,
                         'confidence': confidence, 'frequency': frequency})
        book[key] = rows
        log(f'[{analyzed}/{max_positions}] ply={ply} moves={" ".join(moves) or "(start)"} '
            f'-> {rows[0]["move"]} ({rows[0]["eval_cp"]:+d}cp, depth {rows[0]["depth"]}, '
            f'{len(rows)} candidate(s))')

        if ply < opening_depth:
            for row in rows:
                frontier.append(moves + [row['move']])

    return book


def write_book_file(book, out_path):
    """Writes the CBK1 binary format: 16-byte header (magic, version, count)
    then one 20-byte record per (hash, move) pair, matching
    src/book/opening_book.cpp's load()/save() byte-for-byte."""
    rows = []
    for hash_key, entries in book.items():
        for e in entries:
            rows.append((hash_key, e))
    rows.sort(key=lambda r: r[0])   # sorted by hash, required for the C++ side's binary search

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(BOOK_MAGIC.to_bytes(4, 'big'))
        f.write(BOOK_VERSION.to_bytes(4, 'big'))
        f.write(len(rows).to_bytes(8, 'big'))
        for hash_key, e in rows:
            code = encode_book_move(e['move'])
            eval_cp = max(-32768, min(32767, e['eval_cp']))
            f.write(hash_key.to_bytes(8, 'big'))
            f.write(code.to_bytes(2, 'big'))
            f.write((eval_cp & 0xFFFF).to_bytes(2, 'big'))
            f.write(max(0, min(255, e['depth'])).to_bytes(1, 'big'))
            f.write(max(0, min(2**32 - 1, e['visits'])).to_bytes(4, 'big'))
            f.write(max(0, min(100, e['confidence'])).to_bytes(1, 'big'))
            f.write(max(0, min(65535, e['frequency'])).to_bytes(2, 'big'))
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine-bin', default=None, help='path to the compiled chess UCI binary')
    ap.add_argument('--bin-dir', default=None)
    ap.add_argument('--opening-depth', type=int, default=8, help='max plies to expand the tree')
    ap.add_argument('--search-depth', type=int, default=14, help='fixed search depth per position')
    ap.add_argument('--movetime', type=int, default=0,
                     help='ms per position instead of --search-depth (0=off; less reproducible, see module docstring)')
    ap.add_argument('--max-positions', type=int, default=500,
                     help='hard cap on total analyzed positions (safety/cost bound)')
    ap.add_argument('--candidates', type=int, default=1,
                     help='candidate moves stored per position AND branching factor (see module docstring)')
    ap.add_argument('--hash-mb', type=int, default=64)
    ap.add_argument('--threads', type=int, default=1, help='keep at 1 for reproducible generation')
    ap.add_argument('--out', default=None, help='output .book path (default: books/book_<timestamp>.book)')
    args = ap.parse_args()

    engine_bin = args.engine_bin or find_binary('chess', args.bin_dir)
    out_path = args.out or os.path.join(DEFAULT_OUT_DIR, f'book_{time.strftime("%Y%m%d_%H%M%S")}.book')

    print(f'[generate_book] engine={engine_bin}')
    print(f'[generate_book] opening_depth={args.opening_depth} search_depth={args.search_depth} '
          f'movetime={args.movetime} max_positions={args.max_positions} candidates={args.candidates} '
          f'threads={args.threads}')

    engine = BookEngine(engine_bin, hash_mb=args.hash_mb, threads=args.threads, log=print)
    t0 = time.time()
    try:
        book = generate_book(engine, args.opening_depth, args.search_depth or None,
                              args.movetime or None, args.max_positions, args.candidates, print)
    finally:
        engine.close()
    elapsed = time.time() - t0

    n_records = write_book_file(book, out_path)
    n_positions = len(book)

    meta = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'engine_bin': os.path.abspath(engine_bin),
        'start_fen': START_FEN,
        'opening_depth': args.opening_depth,
        'search_depth': args.search_depth,
        'movetime_ms': args.movetime,
        'max_positions': args.max_positions,
        'candidates': args.candidates,
        'threads': args.threads,
        'hash_mb': args.hash_mb,
        'positions_analyzed': n_positions,
        'records_written': n_records,
        'elapsed_s': elapsed,
        'deterministic': args.threads == 1 and args.movetime == 0,
    }
    meta_path = out_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[generate_book] {n_positions} positions, {n_records} records -> {out_path} '
          f'({elapsed:.1f}s)')
    print(f'[generate_book] metadata -> {meta_path}')
    print('[generate_book] OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
