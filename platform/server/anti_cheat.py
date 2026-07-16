#!/usr/bin/env python3
"""anti_cheat.py - Extra submission-plausibility checks and auto-suspension
for a PUBLIC deployment, layered on top of (not replacing)
distributed/server/validation.py's structural checks (FEN legality, score/
depth/nodes range, side-to-move consistency), which every record already
goes through unchanged.

Honesty note: none of this can *prove* a submission came from a real engine
search -- that would require re-running the search server-side, which
defeats the point of distributing the work. What it does do: catch cheap,
obviously-fabricated data (e.g. "depth 20, 0 nodes") and stop a
malfunctioning or malicious worker after a sustained pattern of bad
submissions, rather than after every single one individually. The primary
defenses against fake submissions remain, in order: per-worker
authenticated bearer tokens (so submissions are attributable), strict
structural validation (validation.py), content-hash deduplication, and
this module's rate-based suspension.
"""

# A real alpha-beta search cannot complete a depth-D iteration having
# visited fewer than D nodes (even a single forced line touches one node
# per ply). This is a deliberately weak, false-positive-resistant floor,
# not an attempt to model real engine node counts -- see module docstring.
#
# Exception: nodes == 0 is a recognized sentinel for "not tracked", not a
# fabricated low count. chess_train gen's bulk self-play exporter (used by
# the DATA_GENERATION task type) has no per-position node telemetry --
# src/search/search.h's SearchResult struct only carries {best, ponder,
# score, depth}, no node count -- so its records legitimately report
# nodes=0. This mirrors the existing convention in
# training_server/dataset/import_data.py's load_from_jsonl(), which already
# treats nodes=0 the same way for the same reason. Skipping the check for
# nodes=0 avoids auto-rejecting every honest DATA_GENERATION submission
# while still catching a worker that reports a nonzero-but-too-low node
# count for a claimed depth.
def plausibility_check(record):
    try:
        depth = int(record['depth'])
        nodes = int(record['nodes'])
    except (KeyError, TypeError, ValueError):
        return None   # malformed types are validation.py's job, not this module's
    if nodes == 0:
        return None   # sentinel: source doesn't track per-position node counts
    if nodes < depth:
        return f'implausible: {nodes} nodes reported for depth {depth} search'
    return None


# Auto-suspension: if a worker racks up too many rejected records in a
# short rolling window, disable it rather than let a broken/malicious
# client keep hammering the endpoint with garbage that a human admin would
# have to notice and act on manually.
AUTO_DISABLE_THRESHOLD = 25
AUTO_DISABLE_WINDOW_MINUTES = 30


def maybe_auto_disable(db, worker_id, log=print):
    import time
    since = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - AUTO_DISABLE_WINDOW_MINUTES * 60))
    count = db.get_worker_rejection_count(worker_id, since)
    if count >= AUTO_DISABLE_THRESHOLD:
        db.disable_worker(worker_id)
        log(f'[anti-cheat] auto-disabled worker {worker_id}: {count} rejected records in the '
            f'last {AUTO_DISABLE_WINDOW_MINUTES} minutes (threshold {AUTO_DISABLE_THRESHOLD})')
        return True
    return False


# ---------------------------------------------------------------------------
# 'network' artifact structural validation (checkpoint verification, server
# side). Before this, upload_artifact() in app.py accepted ANY bytes as a
# 'network' artifact with zero format checking -- a malicious or simply
# broken worker could upload an arbitrary blob, which would then sit in the
# artifact store, get queued for a real ELO_MATCH against the strongest
# network (wasting another worker's compute when it inevitably fails to
# load), or in the worst case reach /admin/artifacts/{id}/accept and get
# shipped as the engine's "improved" network. This mirrors, server-side,
# the same NNU2 header format platform/trainer/train_network.py's
# _local_verify() already checks worker-side after a HONEST worker finishes
# training (see src/nnue/nnue.cpp's NNUE::load/NNUE::save) -- that check
# only protects an honest worker's own upload from a training-side bug; it
# does nothing against a worker that skips training and uploads garbage
# directly. This function is what actually stops garbage/malicious bytes
# from ever becoming a candidate network in the first place.
#
# Constants mirror src/nnue/nnue.cpp / src/nnue/nnue.h exactly -- keep in
# sync if the on-disk format ever changes (NNUE::load would also need
# updating, so this isn't a hidden coupling).
_NNUE_MAGIC = 0x4B504E32          # "2NPK" (HalfKP v2 marker), nnue.cpp's MAGIC
_NNUE_VERSION = 2                 # nnue.cpp's VERSION
_NNUE_FEATURES = 10240            # NNUE_KING_BUCKETS(16) * 64 * NNUE_PIECE_REL(10)
_NNUE_HL = 512                    # NNUE_HL
_NNUE_OUT_BUCKETS = 8             # NNUE_OUT_BUCKETS
_NNUE_HEADER_BYTES = 6 * 4        # magic,version,features,hl,out_buckets,scale (u32/i32 each)
_NNUE_EXPECTED_SIZE = (
    _NNUE_HEADER_BYTES
    + _NNUE_HL * 2                                  # ftBias:      i16[HL]
    + _NNUE_FEATURES * _NNUE_HL * 2                 # ftWeights:   i16[FEATURES][HL]
    + _NNUE_OUT_BUCKETS * 2 * _NNUE_HL * 2          # outWeights:  i16[OUT_BUCKETS][2*HL]
    + _NNUE_OUT_BUCKETS * 4                         # outBias:     i32[OUT_BUCKETS]
)   # == 10,503,224 bytes for this engine's fixed architecture


def validate_network_artifact(file_path, size_bytes):
    """Structurally validates a candidate .nnue file before it's accepted
    into the artifact store as kind='network'. Returns None if valid, or a
    human-readable rejection reason string otherwise. Deliberately does NOT
    try to fully re-verify numerical sanity (that's what the ELO_MATCH
    result -- real games against the current baseline -- is for); this only
    rejects bytes that couldn't possibly be a real Morph NNUE net at all."""
    import struct
    if size_bytes != _NNUE_EXPECTED_SIZE:
        return (f'not a valid Morph .nnue file: size {size_bytes} bytes, '
                f'expected exactly {_NNUE_EXPECTED_SIZE} for this engine\'s '
                f'architecture (features={_NNUE_FEATURES}, hl={_NNUE_HL}, '
                f'out_buckets={_NNUE_OUT_BUCKETS})')
    try:
        with open(file_path, 'rb') as f:
            header = f.read(_NNUE_HEADER_BYTES)
    except OSError as e:
        return f'could not read uploaded file for header validation: {e}'
    if len(header) != _NNUE_HEADER_BYTES:
        return 'not a valid Morph .nnue file: truncated header'
    magic, version, features, hl, out_buckets, _scale = struct.unpack('<IIIIIi', header)
    if magic != _NNUE_MAGIC:
        return f'not a valid Morph .nnue file: bad magic 0x{magic:08X} (expected 0x{_NNUE_MAGIC:08X})'
    if version != _NNUE_VERSION:
        return f'unsupported .nnue version {version} (expected {_NNUE_VERSION})'
    if (features, hl, out_buckets) != (_NNUE_FEATURES, _NNUE_HL, _NNUE_OUT_BUCKETS):
        return (f'.nnue architecture mismatch: file declares features={features} hl={hl} '
                f'out_buckets={out_buckets}, engine expects '
                f'{_NNUE_FEATURES}/{_NNUE_HL}/{_NNUE_OUT_BUCKETS}')
    return None
