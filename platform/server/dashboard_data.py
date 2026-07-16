#!/usr/bin/env python3
"""dashboard_data.py - Live dashboard summaries, sourced entirely from the
real distributed platform database (PlatformDatabase / distributed/server/db.py's
schema + platform/database/schema_extra.sql), not from any file the legacy
automation/pipeline_controller.py + training_server/ system happens to write.

This replaces training_progress.py, which read experiments/net_XXX/*.json,
models/current/current.json, and automation/state.json -- paths belonging to
a separate, non-distributed pipeline that has no connection to the workers,
tasks, and artifacts a real community deployment of platform/server/ actually
produces (see platform/docs/TRAINING.md). A dashboard built on those files
shows nothing when the real distributed system is the one doing work, which
is exactly the bug this module fixes: every figure below is read straight out
of the same `db` object app.py uses to serve workers, so the dashboard and
the production pipeline can never disagree about what's actually happened.
"""
import os
import sys

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'tools', 'nnue_pipeline')
sys.path.insert(0, os.path.abspath(PIPELINE_DIR))


def _candidate_summary(db, artifact):
    """Public-safe view of one un-promoted 'network' artifact: id, when it
    was produced, its multi-candidate experiment grouping (experiment_id /
    seed, see auto_pipeline.py's maybe_queue_training), and its Elo
    progress-so-far against the strongest promoted network, aggregated from
    real match_results rows (0 games if no ELO_MATCH has completed for it
    yet -- never estimated)."""
    meta = artifact.get('metadata') or {}
    results = db.get_match_results_for_artifact(artifact['id'])
    wins = sum(r['wins'] for r in results)
    losses = sum(r['losses'] for r in results)
    draws = sum(r['draws'] for r in results)
    entry = {
        'artifact_id': artifact['id'],
        'created_at': artifact['created_at'],
        'experiment_id': meta.get('experiment_id'),
        'seed': meta.get('seed'),
        'elo_games_so_far': wins + losses + draws,
        'elo_record_so_far': {'wins': wins, 'losses': losses, 'draws': draws},
    }
    if wins + losses + draws:
        from uci_match import elo_estimate
        elo, margin = elo_estimate(wins, losses, draws)
        entry['elo_estimate'] = elo
        entry['elo_margin'] = margin
    return entry


def build_dashboard_summary(db):
    """Everything a community dashboard needs about live distributed
    activity: connected/CPU/GPU workers, active contributors, cumulative
    CPU/GPU contribution, task counts by type and status, positions
    generated, completed training jobs, and the candidate vs. promoted
    network detail -- all read live from `db`."""
    stats = db.get_stats()
    worker_caps = db.get_worker_capability_counts()
    contributors = db.get_active_contributor_count()
    contribution = db.get_contribution_by_hardware()
    task_counts = db.get_task_counts_by_type()
    networks = db.list_artifacts(kind='network')
    candidate_networks = [n for n in networks if not n['accepted']]
    promoted_networks = [n for n in networks if n['accepted']]
    strongest = db.get_strongest_network()

    train_counts = task_counts.get('TRAIN_NETWORK', {})

    return {
        'workers': {
            'registered': stats['total_workers'],
            'active': stats['active_workers'],
            'connected_recently': worker_caps['connected_workers'],
            'cpu_workers': worker_caps['cpu_workers'],
            'gpu_workers': worker_caps['gpu_workers'],
            'capability_unknown': worker_caps['capability_unknown'],
        },
        'active_contributors': contributors,
        'contribution_by_hardware': contribution,
        'tasks': {
            'by_type_and_status': task_counts,
            'by_status_total': stats['tasks_by_status'],
        },
        'positions_generated': stats['total_positions'],
        'completed_training_jobs': train_counts.get('completed', 0),
        'training_jobs_in_progress': train_counts.get('assigned', 0) + train_counts.get('pending', 0),
        'candidate_networks': [_candidate_summary(db, n) for n in candidate_networks],
        'promoted_networks': len(promoted_networks),
        'strongest_network': strongest,
        'result_distribution': stats['result_distribution'],
        'source': 'live distributed platform database',
    }


def build_elo_series(db):
    """Chronological Elo-vs-baseline series, one point per promoted
    ('network', accepted=1) artifact, computed from that artifact's real
    aggregated match_results using the project's existing elo_estimate()
    (tools/nnue_pipeline/uci_match.py) -- the same function
    auto_pipeline.py's promotion logic uses, reused rather than
    reimplemented here."""
    from uci_match import elo_estimate  # tools/nnue_pipeline/uci_match.py

    promoted = sorted(db.list_artifacts(kind='network', accepted_only=True),
                       key=lambda a: a['created_at'])
    series = []
    for net in promoted:
        results = db.get_match_results_for_artifact(net['id'])
        if not results:
            # A network can be promoted with zero match results only as the
            # very first seed baseline (see auto_pipeline.py's
            # maybe_queue_elo_matches) -- report it with games=0 rather than
            # silently dropping it, so the dashboard's network history is
            # complete even for that special case.
            series.append({'artifact_id': net['id'], 'promoted_at': net['created_at'],
                            'elo': 0.0, 'elo_margin': 0.0, 'games': 0})
            continue
        wins = sum(r['wins'] for r in results)
        losses = sum(r['losses'] for r in results)
        draws = sum(r['draws'] for r in results)
        elo, margin = elo_estimate(wins, losses, draws)
        series.append({
            'artifact_id': net['id'],
            'promoted_at': net['created_at'],
            'elo': elo,
            'elo_margin': margin,
            'games': wins + losses + draws,
        })
    return series
