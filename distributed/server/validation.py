#!/usr/bin/env python3
"""validation.py - Server-side validation for submitted training positions.

Runs on every record in every upload, before it ever reaches the database.
Deliberately strict: a bad record (malformed FEN, out-of-range score, garbage
engine_version) is rejected with a specific reason rather than silently
coerced, since this data ultimately trains a network -- garbage in here means
a systematically wrong NNUE later, which is far more expensive to debug than
a rejected upload now.
"""
import hashlib

try:
    import chess as _pychess
    HAVE_PYCHESS = True
except ImportError:
    HAVE_PYCHESS = False

MIN_EVAL_CP = -32000
MAX_EVAL_CP = 32000
VALID_RESULTS = (0.0, 0.5, 1.0)
MAX_DEPTH = 64
MAX_NODES = 10_000_000_000


def _structural_fen_check(fen):
    """Fallback FEN sanity check when python-chess isn't installed: 8 ranks,
    only legal piece/digit characters, exactly one king per side."""
    parts = fen.split()
    if len(parts) < 2:
        return 'FEN missing side-to-move field'
    board_part, stm = parts[0], parts[1]
    ranks = board_part.split('/')
    if len(ranks) != 8:
        return f'FEN board part has {len(ranks)} ranks, expected 8'
    if stm not in ('w', 'b'):
        return f"FEN side-to-move {stm!r} is not 'w' or 'b'"
    white_kings = board_part.count('K')
    black_kings = board_part.count('k')
    if white_kings != 1 or black_kings != 1:
        return f'FEN has {white_kings} white king(s), {black_kings} black king(s) (expected 1 each)'
    valid_chars = set('12345678/pnbrqkPNBRQK')
    if not set(board_part) <= valid_chars:
        return 'FEN board part contains invalid characters'
    return None


def validate_position(record):
    """Returns None if `record` (a dict) is valid, else a short reason string."""
    for field in ('fen', 'side_to_move', 'eval_cp', 'result', 'depth', 'nodes', 'engine_version'):
        if field not in record:
            return f'missing field: {field}'

    fen = record['fen']
    if not isinstance(fen, str) or not fen.strip():
        return 'fen is empty'

    if HAVE_PYCHESS:
        try:
            board = _pychess.Board(fen)
        except Exception as e:
            return f'FEN did not parse: {e}'
        if not board.is_valid():
            return f'FEN is not a legal position ({board.status()!r})'
        actual_stm = 'w' if board.turn == _pychess.WHITE else 'b'
    else:
        err = _structural_fen_check(fen)
        if err:
            return err
        actual_stm = fen.split()[1]

    if record['side_to_move'] not in ('w', 'b'):
        return f"side_to_move {record['side_to_move']!r} is not 'w'/'b'"
    if record['side_to_move'] != actual_stm:
        return (f"side_to_move {record['side_to_move']!r} does not match FEN's "
                f"actual side to move {actual_stm!r}")

    try:
        eval_cp = int(record['eval_cp'])
    except (TypeError, ValueError):
        return 'eval_cp is not an integer'
    if not (MIN_EVAL_CP <= eval_cp <= MAX_EVAL_CP):
        return f'eval_cp {eval_cp} out of range [{MIN_EVAL_CP}, {MAX_EVAL_CP}]'

    try:
        result = float(record['result'])
    except (TypeError, ValueError):
        return 'result is not a number'
    if result not in VALID_RESULTS:
        return f'result {result} not one of {VALID_RESULTS}'

    try:
        depth = int(record['depth'])
    except (TypeError, ValueError):
        return 'depth is not an integer'
    if not (0 <= depth <= MAX_DEPTH):
        return f'depth {depth} out of range [0, {MAX_DEPTH}]'

    try:
        nodes = int(record['nodes'])
    except (TypeError, ValueError):
        return 'nodes is not an integer'
    if not (0 <= nodes <= MAX_NODES):
        return f'nodes {nodes} out of range [0, {MAX_NODES}]'

    engine_version = record['engine_version']
    if not isinstance(engine_version, str) or not engine_version.strip():
        return 'engine_version is empty'

    return None


def content_hash(record):
    """Dedup key: identical (fen, eval, result, depth, engine_version) is the
    same data point regardless of which task/worker produced it."""
    key = '|'.join([
        record['fen'], str(int(record['eval_cp'])), str(float(record['result'])),
        str(int(record['depth'])), record['engine_version'],
    ])
    return hashlib.sha256(key.encode('utf-8')).hexdigest()
