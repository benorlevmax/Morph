#!/usr/bin/env python3
"""elo_match.py - Worker-side executor for the ELO_MATCH task type: play a
real automated match between a candidate NNUE network and a baseline
network, then upload the aggregated W/L/D result.

Reuses tools/nnue_pipeline/uci_match.py's UCIEngine/play_match/elo_estimate
verbatim (imported via sys.path insert, not duplicated) -- that module is
the same one tools/nnue_pipeline/test.py already uses for exactly this
purpose (candidate vs. baseline .nnue A/B testing) and is already proven
there; see its own docstring for why it drives two independent `chess` UCI
processes instead of the single-process chess_match binary (which cannot
load two different EvalFile paths into one process at once).

Note on packaging: this import crosses out of platform/worker/ into
tools/nnue_pipeline/, mirroring the same cross-directory reuse pattern
already used elsewhere in this codebase (e.g. platform/server/database.py
importing distributed/server/db.py). A standalone worker release archive
(see .github/workflows/release.yml) must bundle tools/nnue_pipeline/
alongside platform/worker/ for this task type to work out of the box --
documented in docs/WORKER.md.

baseline_artifact_id is always a real, previously-registered network
artifact (platform/database/schema_extra.sql's match_results table has a
NOT NULL foreign key to it) -- an operator who wants "candidate vs.
classical evaluator" comparisons seeds a placeholder artifact for that (see
docs/TRAINING.md); this executor never silently falls back to a
built-in evaluator on its own.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'tools', 'nnue_pipeline'))
from uci_match import UCIEngine, play_match, elo_estimate  # noqa: E402

from artifacts import fetch_artifact, ArtifactVerificationError  # noqa: E402


class EloMatchError(Exception):
    pass


def run_elo_match(task, client, engine_bin, args, log=print):
    """Executes one ELO_MATCH task end-to-end: download+verify both
    artifacts, play the match, upload the result. Raises EloMatchError on
    any failure that should NOT be silently swallowed (bad artifact,
    engine crash) -- the caller (platform_worker.py's main loop) treats
    that as "this task attempt failed"; the task's lease will expire on
    the server and it gets reassigned to another worker, same recovery
    path as any other crashed/disconnected worker.
    """
    payload = task['payload']
    candidate_id = payload['candidate_artifact_id']
    baseline_id = payload['baseline_artifact_id']
    games = int(payload.get('games', 24))
    match_depth = int(payload.get('match_depth', 5))
    movetime_ms = payload.get('movetime_ms') or None

    log(f"[elo_match] task {task['task_id']}: candidate={candidate_id} "
        f"baseline={baseline_id} games={games} depth={match_depth}")

    try:
        candidate_path = fetch_artifact(client, candidate_id, args.artifacts_cache_dir, log=log)
        baseline_path = fetch_artifact(client, baseline_id, args.artifacts_cache_dir, log=log)
    except ArtifactVerificationError as e:
        raise EloMatchError(f'artifact verification failed, aborting match: {e}') from e

    engine_a = UCIEngine(engine_bin, net_path=candidate_path, use_nnue=True,
                          depth=match_depth, movetime_ms=movetime_ms)
    engine_b = UCIEngine(engine_bin, net_path=baseline_path, use_nnue=True,
                          depth=match_depth, movetime_ms=movetime_ms)

    try:
        wins, losses, draws = play_match(engine_a, engine_b, games)
    except Exception as e:
        raise EloMatchError(f'match play failed: {e}') from e
    finally:
        engine_a.close()
        engine_b.close()

    elo, margin = elo_estimate(wins, losses, draws)
    log(f"[elo_match] task {task['task_id']}: result +{wins} -{losses} ={draws} "
        f"({wins + losses + draws} games), Elo(candidate-baseline) = {elo:+.1f} +/- {margin:.1f}")

    return client.submit_match_result(task['task_id'], candidate_id, baseline_id,
                                       wins, losses, draws)
