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

from platform_client import ServerUnavailable


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
    output, upload every valid record in --upload-batch-size-sized chunks
    (same knob and same reasoning as platform_worker.py's SELF_PLAY
    TaskRunner, which already streams uploads for exactly this reason).
    Raises DataGenerationError on a hard failure (missing binary,
    generation crash); the task's lease will simply expire and get
    reassigned, same recovery path as any other task type.

    Chunked, not one giant POST: this used to upload every record from a
    batch (chess_train gen --games 200 alone regularly produces 20,000+
    records) in a single /tasks/{id}/results call. PlatformClient's HTTP
    timeout is a fixed 30s, and a 20k+-record submission can take the
    server longer than that just to run every content-hash INSERT --
    the *server* would go on to finish and commit the insert in its
    background thread regardless of the client giving up, but the worker
    would see a ReadTimeout, burn through 8 retries (several minutes) and,
    on the retry that finally got a response in time, see every record
    reported back as a 'duplicate' (because the earlier, timed-out attempt
    had in fact already been saved) -- confusing to read and, if that
    retry sequence itself was unlucky enough to exhaust its 8 attempts,
    genuinely lost the whole batch's positions instead of just one chunk's
    worth. Splitting into small chunks keeps each individual request well
    under the timeout and, same as TaskRunner._flush, confines a
    ServerUnavailable failure (all retries exhausted) to the chunk it
    happened on rather than the entire batch."""
    payload = task['payload']
    games = int(payload.get('games', 10))
    depth = int(payload.get('depth', 6))
    randomplies = int(payload.get('randomplies', 4))
    # .get() default 0.0, not required: older already-queued tasks (from
    # before this field existed) won't have it in their payload, and 0.0
    # reproduces chess_train gen's exact old behavior for them. See
    # src/train/selfplay.h's SelfPlayConfig::randomMoveProb for why this
    # exists -- randomplies alone (a fixed opening-only prefix) stops being
    # enough to avoid duplicate positions once the server's dataset is
    # large, since search is deterministic after the opening and games
    # that transpose into an already-explored position then produce
    # identical, already-collected continuations forever after.
    random_move_prob = float(payload.get('random_move_prob', 0.0))

    train_bin = _find_train_binary(engine_bin, getattr(args, 'train_bin', None))
    seed = _task_seed(task['task_id'])
    log(f"[data_generation] task {task['task_id']}: {games} games, depth={depth}, "
        f"randomplies={randomplies}, random_move_prob={random_move_prob}, seed={seed}, "
        f"using {train_bin}")

    fd, out_path = tempfile.mkstemp(prefix='chess_train_gen_', suffix='.txt')
    os.close(fd)
    try:
        proc = subprocess.run(
            [train_bin, 'gen', '--games', str(games), '--depth', str(depth),
             '--randomplies', str(randomplies),
             '--random-move-prob', str(random_move_prob),
             '--seed', str(seed),
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

        batch_size = max(1, int(getattr(args, 'upload_batch_size', 100) or 100))
        n_chunks = (len(records) + batch_size - 1) // batch_size
        log(f'[data_generation] uploading {len(records)} positions in '
            f'{n_chunks} chunk(s) of up to {batch_size}')

        totals = {'accepted': 0, 'duplicates': 0, 'rejected': 0}
        last_resp = None
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            is_last = (i + batch_size) >= len(records)
            try:
                resp = client.submit_results(task['task_id'], chunk, done=is_last)
            except ServerUnavailable as e:
                log(f'[data_generation] WARNING: failed to upload {len(chunk)} '
                    f'positions after retries: {e} -- {len(chunk)} positions '
                    f'lost for this chunk (continuing with the rest)')
                continue
            if resp is None:
                continue
            totals['accepted'] += resp.get('accepted', 0)
            totals['duplicates'] += resp.get('duplicates', 0)
            totals['rejected'] += resp.get('rejected', 0)
            last_resp = resp

        log(f"[data_generation] task {task['task_id']}: accepted={totals['accepted']} "
            f"duplicates={totals['duplicates']} rejected={totals['rejected']}")
        if last_resp is not None:
            last_resp = dict(last_resp)
            last_resp.update(totals)
        return last_resp
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
