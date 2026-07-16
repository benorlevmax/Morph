#!/usr/bin/env python3
"""evaluate.py - Automatic evaluation stage.

    new network -> engine benchmark -> Elo test -> accept/reject

Reuses tools/nnue_pipeline/export.py (checkpoint -> quantized .nnue) and
tools/nnue_pipeline/test.py (load/verify -> benchmark -> Elo match) via
subprocess, exactly as training_server/training/train.py reuses train.py --
same reasoning: this logic already exists, is already verified end-to-end,
and re-implementing it here would just be a second copy to keep in sync.

The accept/reject policy itself (the one genuinely new piece of logic) is a
simple, explicit, configurable threshold: reject if the verify step fails
(export.py/test.py already exit non-zero for that) or if the candidate's
Elo estimate against the baseline is below --reject-elo-threshold; accept
otherwise. This is intentionally simple and stated in one place so it's easy
to tighten later (e.g. require an SPRT H1 verdict) without touching the rest
of the pipeline.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
from experiment import list_experiments


def find_baseline_network():
    """Most recent ACCEPTED experiment's network, or None (-> compare against
    the classical evaluator, exactly like tools/nnue_pipeline/test.py's
    default when no --baseline-net is given)."""
    accepted = [e for e in list_experiments()
                if e.get('results', {}).get('verdict') == 'accept']
    if not accepted:
        return None, None
    accepted.sort(key=lambda e: e['id'])
    latest = accepted[-1]
    net_path = latest.get('results', {}).get('network_file')
    return net_path, latest['id']


def export_network(checkpoint_path, out_path, qa, qb, log=print):
    cmd = [sys.executable, os.path.join(config.NNUE_PIPELINE_DIR, 'export.py'),
           '--checkpoint', checkpoint_path, '--out', out_path, '--qa', str(qa), '--qb', str(qb)]
    log(f'[evaluate] $ {" ".join(cmd)}')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f'export failed:\n{proc.stdout}\n{proc.stderr}')
    return out_path


def run_test_stage(net_path, baseline_net, bin_dir, bench_depth, match_games, match_depth,
                   elo0, elo1, report_path, log=print):
    cmd = [sys.executable, os.path.join(config.NNUE_PIPELINE_DIR, 'test.py'),
           '--net', net_path, '--bench-depth', str(bench_depth),
           '--games', str(match_games), '--match-depth', str(match_depth),
           '--elo0', str(elo0), '--elo1', str(elo1), '--report', report_path]
    if baseline_net:
        cmd += ['--baseline-net', baseline_net]
    if bin_dir:
        cmd += ['--bin-dir', bin_dir]
    log(f'[evaluate] $ {" ".join(cmd)}')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log(proc.stdout)
    if proc.returncode != 0:
        return None, proc.stdout + proc.stderr
    with open(report_path) as f:
        return json.load(f), proc.stdout


def decide(test_report, reject_elo_threshold):
    """Accept/reject policy. Returns (verdict, reason)."""
    if test_report is None:
        return 'reject', 'verify/benchmark/test stage failed to run (see log)'
    if test_report.get('verify') != 'PASS':
        return 'reject', 'network failed verification against the compiled engine'
    match = test_report.get('elo_match')
    if match is None:
        return 'accept', 'verification passed; Elo match skipped (--skip-match), accepting by default'
    elo = match['elo']
    if elo < reject_elo_threshold:
        return 'reject', f'Elo {elo:+.1f} below reject threshold {reject_elo_threshold:+.1f}'
    return 'accept', f'Elo {elo:+.1f} >= reject threshold {reject_elo_threshold:+.1f}'


def evaluate_experiment(checkpoint_path, exp_dir, experiment_id, qa, qb, bin_dir, bench_depth,
                        match_games, match_depth, elo0, elo1, reject_elo_threshold,
                        baseline_net_override=None, training_metrics=None, log=print):
    net_path = os.path.join(exp_dir, 'network', f'{experiment_id}.nnue')
    export_network(checkpoint_path, net_path, qa, qb, log=log)

    baseline_net, baseline_id = (baseline_net_override, None) if baseline_net_override \
        else find_baseline_network()
    log(f'[evaluate] baseline: {baseline_net or "classical evaluator"} '
        f'({"experiment " + baseline_id if baseline_id else "no accepted experiment yet"})')

    report_path = os.path.join(exp_dir, 'results.json')  # test.py writes here first, we augment it
    test_report, test_log = run_test_stage(
        net_path, baseline_net, bin_dir, bench_depth, match_games, match_depth, elo0, elo1,
        report_path, log=log)

    verdict, reason = decide(test_report, reject_elo_threshold)
    log(f'[evaluate] verdict: {verdict.upper()} -- {reason}')

    results = {
        'network_file': os.path.abspath(net_path),
        'baseline_network': baseline_net, 'baseline_experiment': baseline_id,
        'evaluated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'validation_score': training_metrics,   # final epoch's train/val MSE from training.py
        'test_report': test_report,
        'benchmark_results': (test_report or {}).get('benchmark'),
        'verdict': verdict,
        'verdict_reason': reason,
    }
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results
