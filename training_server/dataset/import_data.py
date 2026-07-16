#!/usr/bin/env python3
"""import_data.py - Dataset handling, stage 1: import + clean + validate.

Pulls positions from one or both of:
  --distributed-db  a distributed/database/*.sqlite3 file (the `positions`
                     table produced by distributed/server/, already
                     schema-validated once at submission time)
  --jsonl           one or more tools/nnue_pipeline/generate.py-style JSONL
                     files (fields: fen, result, eval, depth, engine_version)

Every record, regardless of source, is normalized to one canonical shape:
    {fen, side_to_move, eval_cp, result, depth, nodes, engine_version, source}
re-validated with the exact same rules the distributed server enforces
(distributed/server/validation.py, imported directly -- not reimplemented),
and deduplicated by the same content hash
(sha256(fen|eval|result|depth|engine_version)) used there, so importing the
same distributed DB twice, or a distributed DB that overlaps with a local
JSONL run, never double-counts a position.

Output: a versioned, immutable dataset snapshot under
training_server/datasets/<version>/all.jsonl plus manifest.json recording
where every record came from and how many were rejected/deduped -- this is
the "dataset version" referenced by every experiment's saved config.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
config.add_distributed_server_to_path()
from validation import validate_position, content_hash  # distributed/server/validation.py

try:
    import chess as _pychess
except ImportError:
    _pychess = None


def _side_to_move_from_fen(fen):
    parts = fen.split()
    return parts[1] if len(parts) > 1 and parts[1] in ('w', 'b') else None


def load_from_distributed_db(db_path):
    """Read every row of the `positions` table, already in canonical shape
    modulo column naming (eval_cp/side_to_move/nodes are already present)."""
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f'distributed DB not found: {db_path}')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT fen, side_to_move, eval_cp, result, depth, nodes, engine_version '
            'FROM positions').fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        rec = dict(r)
        rec['source'] = f'distributed:{os.path.basename(db_path)}'
        out.append(rec)
    return out


def load_from_jsonl(path):
    """Read a tools/nnue_pipeline/generate.py JSONL file. Missing fields
    (side_to_move, nodes) are backfilled: side_to_move from the FEN itself,
    nodes as 0 (that pipeline's bulk exporter doesn't expose per-position
    node counts -- see docs/NNUE_TRAINING.md)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stm = rec.get('side_to_move') or _side_to_move_from_fen(rec.get('fen', ''))
            out.append({
                'fen': rec.get('fen'),
                'side_to_move': stm,
                'eval_cp': rec.get('eval'),
                'result': rec.get('result'),
                'depth': rec.get('depth'),
                'nodes': rec.get('nodes', 0),
                'engine_version': rec.get('engine_version', 'unknown'),
                'source': f'jsonl:{os.path.basename(path)}',
            })
    return out


def import_and_clean(distributed_dbs, jsonl_files, log=print):
    raw = []
    counts_by_source = {}
    for db_path in distributed_dbs:
        recs = load_from_distributed_db(db_path)
        raw.extend(recs)
        counts_by_source[f'distributed:{os.path.basename(db_path)}'] = len(recs)
        log(f'[import] {db_path}: {len(recs)} positions')
    for jf in jsonl_files:
        recs = load_from_jsonl(jf)
        raw.extend(recs)
        counts_by_source[f'jsonl:{os.path.basename(jf)}'] = len(recs)
        log(f'[import] {jf}: {len(recs)} positions')

    log(f'[import] {len(raw)} raw positions from {len(distributed_dbs) + len(jsonl_files)} source(s)')

    seen_hashes = set()
    cleaned = []
    rejected = 0
    duplicates = 0
    rejected_reasons = {}
    for rec in raw:
        reason = validate_position(rec)
        if reason is not None:
            rejected += 1
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            continue
        chash = content_hash(rec)
        if chash in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(chash)
        rec['content_hash'] = chash
        cleaned.append(rec)

    log(f'[import] cleaned: {len(cleaned)} accepted, {duplicates} duplicates, {rejected} rejected')
    return cleaned, {
        'raw_count': len(raw), 'accepted_count': len(cleaned), 'duplicates': duplicates,
        'rejected': rejected, 'rejected_reasons': rejected_reasons,
        'counts_by_source': counts_by_source,
    }


def dataset_version_id(cleaned):
    ts = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    content_digest = hashlib.sha256(
        ''.join(sorted(r['content_hash'] for r in cleaned)).encode()).hexdigest()[:10]
    return f'v_{ts}_{content_digest}'


def write_dataset(cleaned, stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, 'all.jsonl')
    with open(data_path, 'w') as f:
        for rec in cleaned:
            f.write(json.dumps(rec) + '\n')

    manifest = dict(stats)
    manifest['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest['file'] = os.path.abspath(data_path)
    manifest['total_positions'] = len(cleaned)
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return data_path, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--distributed-db', nargs='*', default=[],
                     help='distributed/database/*.sqlite3 file(s); default: the standard location')
    ap.add_argument('--jsonl', nargs='*', default=[],
                     help='tools/nnue_pipeline JSONL dataset file(s)')
    ap.add_argument('--use-default-distributed-db', action='store_true',
                     help=f'include {config.DEFAULT_DISTRIBUTED_DB} if it exists')
    ap.add_argument('--out-dir', default=None, help='override the auto-versioned output directory')
    args = ap.parse_args()

    distributed_dbs = list(args.distributed_db)
    if args.use_default_distributed_db and os.path.isfile(config.DEFAULT_DISTRIBUTED_DB):
        distributed_dbs.append(config.DEFAULT_DISTRIBUTED_DB)

    if not distributed_dbs and not args.jsonl:
        sys.exit('no data sources given: pass --distributed-db, --jsonl, or '
                 '--use-default-distributed-db')

    cleaned, stats = import_and_clean(distributed_dbs, args.jsonl)
    if not cleaned:
        sys.exit('no valid positions after cleaning -- nothing to write')

    version = dataset_version_id(cleaned)
    out_dir = args.out_dir or os.path.join(config.DATASETS_DIR, version)
    data_path, manifest = write_dataset(cleaned, stats, out_dir)

    print(f'[import] dataset version: {version}')
    print(f'[import] wrote {manifest["total_positions"]} positions -> {data_path}')
    print(f'[import] manifest -> {os.path.join(out_dir, "manifest.json")}')
    print(version)
    return 0


if __name__ == '__main__':
    sys.exit(main())
