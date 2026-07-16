#!/usr/bin/env python3
"""data_generation.py - Worker-side executor for the DATA_GENERATION task
type: runs the engine's own native bulk self-play exporter (chess_train
gen -- see src/apps/train_main.cpp) and uploads the resulting positions
through the same /tasks/{id}/results endpoint SELF_PLAY tasks already use
(distributed/server/validation.py's schema doesn't care which task type
produced a record).

Why chess_train gen instead of platform/worker/selfplay.py's own
game-driving loop (used for SELF_PLAY tasks): chess_train gen is the
engine's real, already-proven, natively-fast bulk exporter (used
throughout this project's training pipeline, e.g. training_server/) -- for
a DATA_GENERATION task specifically, the point is to exercise that exact
pipeline path (so the resulting dataset matches what the training side
expects), not to re-drive individual games one UCI command at a time from
Python. SELF_PLAY tasks intentionally keep using selfplay.py's
per-game/streaming-upload approach instead, since that task type exists for
steady incremental position generation with progress reporting, not bulk
export.

nodes=0 sentinel: chess_train gen's SearchResult (src/search/search.h) has
no per-position node count (only best/ponder/score/depth) -- every record
from this executor reports nodes=0, which platform/server/anti_cheat.py's
plausibility_check recognizes as "not tracked" rather than "suspiciously
low", matching the same convention training_server/dataset/import_data.py
already uses for the same reason.
"""
import hashlib
import os
import shutil
import subprocess
import tempfile
import time


class DataGenerationError(Exception):
    pass


def _find_train_binary(engine_bin, train_bin_override=None):
    if train_bin_override and os.path.isfile(train_bin_override):
        return train_bin_override
    engine_dir = os.path.dirname(os.path.abspath(engine_bin))
    for name in ('chess_train.exe', 'chess_train'):
        candidate = os.path.join(engine_dir, name)
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which('chess_train')
    if found:
        return found
    raise DataGenerationError(
        f'could not find chess_train next to {engine_bin!r} (tried {engine_dir}) or on PATH -- '
        f'pass --train-bin explicitly. The worker release archive must bundle chess_train '
        f'alongside the chess binary for DATA_GENERATION tasks to work.')


def _parse_line(line, depth, engine_version):
    """Parses one 'chess_train gen --format bullet-ext' output line:
    '<fen> | <eval_cp> | <result> | <scoreSwing> | <bestMoveChanges>' into
    the position-record dict /tasks/{id}/results expects. The trailing two
    fields are chess_train gen's search-instability signal (see search.h's
    SearchResult::scoreSwing/bestMoveChanges) -- see dataset.h's
    save_bullet_ext(), a separate export method from the original 3-field
    save_bullet() bullet's own `convert` utility consumes, so this format
    change is additive and doesn't touch that path. Also accepts the
    original 3-field format (no trailing columns) for compatibility with a
    worker/engine build that predates this change, in which case
    score_swing/best_move_changes are simply omitted (server stores NULL,
    meaning "not recorded", never fabricated as 0). Returns None for
    blank/malformed lines (skipped, not fabricated) rather than raising --
    a handful of unparseable lines in a large bulk file shouldn't lose the
    whole batch."""
    parts = [p.strip() for p in line.split('|')]
    if len(parts) not in (3, 5):
        return None
    fen, eval_cp_s, result_s = parts[0], parts[1], parts[2]
    if not fen:
        return None
    try:
        eval_cp = int(eval_cp_s)
        result = float(result_s)
    except ValueError:
        return None
    fen_fields = fen.split()
    if len(fen_fields) < 2 or fen_fields[1] not in ('w', 'b'):
        return None
    record = {
        'fen': fen,
        'side_to_move': fen_fields[1],
        'eval_cp': eval_cp,
        'result': result,
        'depth': depth,
        'nodes': 0,   # sentinel: chess_train gen has no per-position node telemetry
        'engine_version': engine_version,
    }
    if len(parts) == 5:
        try:
            record['score_swing'] = int(parts[3])
            record['best_move_changes'] = int(parts[4])
        except ValueError:
            pass   # malformed trailing columns -- keep the record, just without the signal
    return record


def _task_seed(task_id):
    """Derives a seed for `chess_train gen --seed` that's unique per task
    invocation, even across concurrent workers on the same or different
    machines. This is defense-in-depth on top of train_main.cpp's own
    random default seed (previously a FIXED constant, 0xC0FFEE -- every
    worker running `gen` with the same --games/--depth/--randomplies
    produced byte-identical games, which the server's content-hash dedup
    then silently discarded in full for every worker after the first; see
    src/apps/train_main.cpp's --seed flag and fresh_random_seed()). Mixes
    the task id (unique per assignment) with hostname/pid/wall-clock time
    so re-running the same task id after a lease expiry+reassignment still
    gets a different seed than the original attempt."""
    mix = f'{task_id}:{os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "")}' \
          f':{os.getpid()}:{time.time_ns()}'
    return int(hashlib.sha256(mix.encode()).hexdigest()[:16], 16)


def run_data_generation(task, client, engine_bin, engine_version, args, log=print):
    """Executes one DATA_GENERATION task: run chess_train gen, parse its
    output, upload every valid record in one batch, mark done. Raises
    DataGenerationError on a hard failure (missing binary, generation
    crash); the task's lease will simply expire and get reassigned, same
    recovery path as any other task type."""
    payload = task['payload']
    games = int(payload.get('games', 10))
    depth = int(payload.get('depth', 6))
    randomplies = int(payload.get('randomplies', 4))

    train_bin = _find_train_binary(engine_bin, getattr(args, 'train_bin', None))
    seed = _task_seed(task['task_id'])
    log(f"[data_generation] task {task['task_id']}: {games} games, depth={depth}, "
        f"randomplies={randomplies}, seed={seed}, using {train_bin}")

    fd, out_path = tempfile.mkstemp(prefix='chess_train_gen_', suffix='.txt')
    os.close(fd)
    try:
        proc = subprocess.run(
            [train_bin, 'gen', '--games', str(games), '--depth', str(depth),
             '--randomplies', str(randomplies), '--seed', str(seed),
             '--format', 'bullet-ext', '--out', out_path],
            capture_output=True, text=True, timeout=max(300, games * 30))
        if proc.returncode != 0:
            raise DataGenerationError(
                f'chess_train gen exited {proc.returncode}: {proc.stderr[-2000:]}')
        log(f'[data_generation] {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "generation complete"}')

        records = []
        skipped = 0
        with open(out_path) as f:
            for line in f:
                rec = _parse_line(line, depth, engine_version)
                if rec is None:
                    if line.strip():
                        skipped += 1
                    continue
                records.append(rec)
        if skipped:
            log(f'[data_generation] skipped {skipped} unparseable line(s) out of the output file')
        if not records:
            raise DataGenerationError('chess_train gen produced no parseable records')

        log(f'[data_generation] uploading {len(records)} positions')
        resp = client.submit_results(task['task_id'], records, done=True)
        if resp:
            log(f"[data_generation] task {task['task_id']}: accepted={resp.get('accepted')} "
                f"duplicates={resp.get('duplicates')} rejected={resp.get('rejected')}")
        return resp
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
