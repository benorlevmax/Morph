#!/usr/bin/env python3
"""nnue_format.py - Shared NNUE math/IO for the tools/nnue_pipeline/ scripts.

Pure-Python reference reimplementation of the production NNUE architecture in
src/nnue/nnue.h / src/nnue/nnue.cpp: HalfKP features (kings excluded), 16 king
buckets, 512-wide dual-perspective accumulator, clipped-ReLU activation, 8
output buckets by piece count, and the exact ".nnue" (NNU2) binary layout that
NNUE::load()/write_net() read and write.

This module is training/tooling code, not engine code -- it does not touch
src/search, src/eval, or NNUE inference. It exists so generate.py / train.py /
export.py / test.py can all agree on one feature-indexing and file-format
implementation instead of duplicating it four times.

Ported from tools/nnue_training/reference_nnue.py (Phase A), which was
verified byte-for-byte against the real compiled C++ engine on 10 positions
(see docs/phaseA_nnue_bullet_audit.md) -- every function here is a direct,
line-by-line port of the corresponding C++ function.
"""
import struct

NNUE_HL = 512
NNUE_KING_BUCKETS = 16
NNUE_PIECE_REL = 10
NNUE_FEATURES = NNUE_KING_BUCKETS * 64 * NNUE_PIECE_REL  # 10240
NNUE_OUT_BUCKETS = 8

MAGIC = 0x4B504E32     # "2NPK" (HalfKP v2 marker) -- must match src/nnue/nnue.cpp
VERSION = 2

CR_MIN = 0
CR_MAX = 32767
LINEAR_BIAS = 16384

WHITE, BLACK = 0, 1
# Piece type numbering matches core/types.h: NO_PIECE=0, PAWN=1..KING=6.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

PIECE_CHARS = {
    'P': (WHITE, PAWN), 'N': (WHITE, KNIGHT), 'B': (WHITE, BISHOP),
    'R': (WHITE, ROOK), 'Q': (WHITE, QUEEN), 'K': (WHITE, KING),
    'p': (BLACK, PAWN), 'n': (BLACK, KNIGHT), 'b': (BLACK, BISHOP),
    'r': (BLACK, ROOK), 'q': (BLACK, QUEEN), 'k': (BLACK, KING),
}


def sq(file_, rank_):
    return rank_ * 8 + file_


def file_of(s):
    return s & 7


def rank_of(s):
    return s >> 3


def orient(persp, s):
    return s if persp == WHITE else (s ^ 56)


def king_bucket(king_sq_oriented):
    return (rank_of(king_sq_oriented) // 2) * 4 + (file_of(king_sq_oriented) // 2)


def feature_index(persp, king_sq, piece_color, piece_type, s):
    kb = king_bucket(orient(persp, king_sq))
    rel = orient(persp, s)
    piece_rel = (piece_type - 1) * 2 + (0 if piece_color == persp else 1)
    return kb * (64 * NNUE_PIECE_REL) + rel * NNUE_PIECE_REL + piece_rel


def output_bucket(num_pieces_on_board):
    b = (num_pieces_on_board - 1) // 4
    if b < 0:
        return 0
    if b >= NNUE_OUT_BUCKETS:
        return NNUE_OUT_BUCKETS - 1
    return b


def parse_fen_board(fen):
    """Returns dict: square(0..63, a1=0) -> (color, piece_type), plus side-to-move."""
    parts = fen.split()
    board_part = parts[0]
    stm = WHITE if (len(parts) > 1 and parts[1] == 'w') else BLACK
    ranks = board_part.split('/')
    if len(ranks) != 8:
        raise ValueError(f'malformed FEN board part: {board_part!r}')
    board = {}
    for rank_idx, rank_str in enumerate(ranks):
        rank = 7 - rank_idx
        file_ = 0
        for ch in rank_str:
            if ch.isdigit():
                file_ += int(ch)
            else:
                color, pt = PIECE_CHARS[ch]
                board[sq(file_, rank)] = (color, pt)
                file_ += 1
    return board, stm


def truncating_div(numerator, denom):
    """Replicate C++ integer division truncation-toward-zero for (int64/int32)."""
    q = abs(numerator) // abs(denom)
    if (numerator < 0) != (denom < 0):
        q = -q
    return q


class RefNet:
    """Holds ftBias/ftWeights/outWeights/outBias/scale exactly as the engine's
    binary format lays them out, and reproduces NNUE::evaluate() bit-for-bit."""

    def __init__(self):
        self.ft_bias = [0] * NNUE_HL
        self.ft_weights = [[0] * NNUE_HL for _ in range(NNUE_FEATURES)]
        self.out_weights = [[0] * (2 * NNUE_HL) for _ in range(NNUE_OUT_BUCKETS)]
        self.out_bias = [0] * NNUE_OUT_BUCKETS
        self.scale = 1

    def active_features(self, board, persp, king_sq):
        idx = []
        for s, (color, pt) in board.items():
            if pt == KING:
                continue
            idx.append(feature_index(persp, king_sq, color, pt, s))
        return idx

    def accumulator(self, board, persp, king_sq):
        acc = list(self.ft_bias)
        for f in self.active_features(board, persp, king_sq):
            row = self.ft_weights[f]
            for i in range(NNUE_HL):
                acc[i] += row[i]
        return acc

    def output_scalar(self, acc_own, acc_opp, bucket):
        w = self.out_weights[bucket]
        s = self.out_bias[bucket]
        for i in range(NNUE_HL):
            x = acc_own[i]
            if x < CR_MIN: x = CR_MIN
            if x > CR_MAX: x = CR_MAX
            s += x * w[i]
        for i in range(NNUE_HL):
            x = acc_opp[i]
            if x < CR_MIN: x = CR_MIN
            if x > CR_MAX: x = CR_MAX
            s += x * w[NNUE_HL + i]
        return truncating_div(s, self.scale)

    def evaluate_fen(self, fen):
        board, stm = parse_fen_board(fen)
        wk = next(s for s, (c, pt) in board.items() if c == WHITE and pt == KING)
        bk = next(s for s, (c, pt) in board.items() if c == BLACK and pt == KING)
        king_of = {WHITE: wk, BLACK: bk}

        acc = {}
        for persp in (WHITE, BLACK):
            acc[persp] = self.accumulator(board, persp, king_of[persp])

        n_pieces = len(board)
        bucket = output_bucket(n_pieces)
        other = BLACK if stm == WHITE else WHITE
        return self.output_scalar(acc[stm], acc[other], bucket)

    # --- binary I/O, matching nnue.cpp write_net()/load() exactly ---------
    def save(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<IIIII', MAGIC, VERSION, NNUE_FEATURES, NNUE_HL, NNUE_OUT_BUCKETS))
            f.write(struct.pack('<i', self.scale))
            f.write(struct.pack(f'<{NNUE_HL}h', *self.ft_bias))
            flat_ft = [v for row in self.ft_weights for v in row]
            f.write(struct.pack(f'<{len(flat_ft)}h', *flat_ft))
            flat_out = [v for row in self.out_weights for v in row]
            f.write(struct.pack(f'<{len(flat_out)}h', *flat_out))
            f.write(struct.pack(f'<{NNUE_OUT_BUCKETS}i', *self.out_bias))

    @classmethod
    def load(cls, path):
        net = cls()
        with open(path, 'rb') as f:
            magic, version, feats, hl, buckets = struct.unpack('<IIIII', f.read(20))
            if not (magic == MAGIC and feats == NNUE_FEATURES and hl == NNUE_HL
                    and buckets == NNUE_OUT_BUCKETS):
                raise ValueError(
                    f'{path}: not a compatible .nnue file '
                    f'(magic={magic:#x} feats={feats} hl={hl} buckets={buckets}; '
                    f'expected magic={MAGIC:#x} feats={NNUE_FEATURES} hl={NNUE_HL} '
                    f'buckets={NNUE_OUT_BUCKETS})')
            (net.scale,) = struct.unpack('<i', f.read(4))
            net.ft_bias = list(struct.unpack(f'<{NNUE_HL}h', f.read(2 * NNUE_HL)))
            flat_ft = struct.unpack(f'<{NNUE_FEATURES * NNUE_HL}h', f.read(2 * NNUE_FEATURES * NNUE_HL))
            net.ft_weights = [list(flat_ft[i * NNUE_HL:(i + 1) * NNUE_HL]) for i in range(NNUE_FEATURES)]
            flat_out = struct.unpack(f'<{NNUE_OUT_BUCKETS * 2 * NNUE_HL}h', f.read(2 * NNUE_OUT_BUCKETS * 2 * NNUE_HL))
            net.out_weights = [list(flat_out[i * 2 * NNUE_HL:(i + 1) * 2 * NNUE_HL]) for i in range(NNUE_OUT_BUCKETS)]
            net.out_bias = list(struct.unpack(f'<{NNUE_OUT_BUCKETS}i', f.read(4 * NNUE_OUT_BUCKETS)))
        return net


def check_i16(vals, name):
    """Raise loudly on int16 overflow -- never silently clamp a trained weight."""
    bad = [v for v in vals if not (-32768 <= v <= 32767)]
    if bad:
        raise SystemExit(
            f'QUANTIZATION OVERFLOW in {name}: {len(bad)} value(s) out of int16 range '
            f'(e.g. {bad[0]}). Reduce --qa/--qb or add weight clipping before export.')


def check_i32(vals, name):
    bad = [v for v in vals if not (-2**31 <= v <= 2**31 - 1)]
    if bad:
        raise SystemExit(f'QUANTIZATION OVERFLOW in {name}: {bad[:5]}')
