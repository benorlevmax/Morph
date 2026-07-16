#!/usr/bin/env python3
"""models_registry.py - models/{current,candidates,rejected}/ lifecycle.

A network's life in this repo, after training_server/pipeline.py exports
and evaluates it:

    experiments/net_XXX/network/net_XXX.nnue   <- permanent, full record
                    |
                    v  (staged here while the controller acts on the verdict)
    models/candidates/net_XXX.nnue
                    |
             accept?  (training_server's own verdict -- already decided
              / reject?  by evaluate.py's benchmark+SPRT-Elo policy)
             /        \\
            v          v
    models/current/    models/rejected/
      net_XXX.nnue        net_XXX.nnue
      net_XXX.json         net_XXX.json (why it was rejected)
      current.nnue  <- always a copy of the latest accepted network
      current.json  <- pointer metadata for current.nnue

experiments/ remains the one full, immutable record of every run (config,
logs, checkpoints, network, results.json) -- models/ is a convenience layer
on top of it for "what's live right now" / "what did we try and reject",
so nothing here duplicates experiments/'s role as the source of truth.
"""
import json
import os
import shutil
import time

import config


def stage_candidate(experiment_id, network_path):
    """Copy the exported .nnue into models/candidates/ while the controller
    decides what to do with it. Returns the staged path."""
    os.makedirs(config.MODELS_CANDIDATES_DIR, exist_ok=True)
    dest = os.path.join(config.MODELS_CANDIDATES_DIR, f'{experiment_id}.nnue')
    shutil.copy2(network_path, dest)
    return dest


def promote(experiment_id, candidate_path, results, config_json):
    """Move a staged candidate into models/current/, and update the
    current.nnue pointer + current.json metadata to point at it."""
    os.makedirs(config.MODELS_CURRENT_DIR, exist_ok=True)
    versioned_dest = os.path.join(config.MODELS_CURRENT_DIR, f'{experiment_id}.nnue')
    shutil.move(candidate_path, versioned_dest)

    pointer_path = os.path.join(config.MODELS_CURRENT_DIR, 'current.nnue')
    shutil.copy2(versioned_dest, pointer_path)

    meta = {
        'experiment_id': experiment_id,
        'promoted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'network_file': versioned_dest,
        'dataset_version': config_json.get('dataset_version'),
        'validation_score': results.get('validation_score'),
        'benchmark_results': results.get('benchmark_results'),
        'elo_match': (results.get('test_report') or {}).get('elo_match'),
        'verdict_reason': results.get('verdict_reason'),
    }
    with open(os.path.join(config.MODELS_CURRENT_DIR, 'current.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    # Per-version copy of the same metadata, so models/current/'s history of
    # versioned .nnue files each keep their own matching .json record.
    with open(os.path.join(config.MODELS_CURRENT_DIR, f'{experiment_id}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return versioned_dest


def reject(experiment_id, candidate_path, results):
    """Move a staged candidate into models/rejected/ with a record of why."""
    os.makedirs(config.MODELS_REJECTED_DIR, exist_ok=True)
    dest = os.path.join(config.MODELS_REJECTED_DIR, f'{experiment_id}.nnue')
    shutil.move(candidate_path, dest)
    meta = {
        'experiment_id': experiment_id,
        'rejected_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'verdict_reason': results.get('verdict_reason'),
        'benchmark_results': results.get('benchmark_results'),
        'elo_match': (results.get('test_report') or {}).get('elo_match'),
    }
    with open(os.path.join(config.MODELS_REJECTED_DIR, f'{experiment_id}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return dest


def get_current_network():
    """Path to the live current network, or None if nothing has been
    promoted yet (fresh repo -- training_server's own evaluate.py falls
    back to the classical evaluator as a baseline in that case, which is
    the correct behavior, not a bug)."""
    pointer = os.path.join(config.MODELS_CURRENT_DIR, 'current.nnue')
    return pointer if os.path.isfile(pointer) else None


def record_results_index(experiment_id, results):
    """Flat, queryable copies of the benchmark/Elo portions of a run's
    results.json under results/benchmarks/ and results/elo_tests/, so you
    can browse trends across every run without opening each
    experiments/net_XXX/results.json individually. experiments/net_XXX/
    still holds the full record; this is an index, not a second copy of
    the source of truth."""
    os.makedirs(config.RESULTS_BENCHMARKS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_ELO_TESTS_DIR, exist_ok=True)

    bench_path = os.path.join(config.RESULTS_BENCHMARKS_DIR, f'{experiment_id}.json')
    with open(bench_path, 'w') as f:
        json.dump({
            'experiment_id': experiment_id,
            'evaluated_at': results.get('evaluated_at'),
            'benchmark_results': results.get('benchmark_results'),
            'verdict': results.get('verdict'),
        }, f, indent=2)

    test_report = results.get('test_report') or {}
    elo_path = os.path.join(config.RESULTS_ELO_TESTS_DIR, f'{experiment_id}.json')
    with open(elo_path, 'w') as f:
        json.dump({
            'experiment_id': experiment_id,
            'evaluated_at': results.get('evaluated_at'),
            'baseline_network': results.get('baseline_network'),
            'baseline_experiment': results.get('baseline_experiment'),
            'elo_match': test_report.get('elo_match'),
            'verify': test_report.get('verify'),
            'verdict': results.get('verdict'),
            'verdict_reason': results.get('verdict_reason'),
        }, f, indent=2)

    return bench_path, elo_path
