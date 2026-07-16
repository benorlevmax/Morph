#!/usr/bin/env python3
"""test_promotion.py - Regression tests for SPRT-gated network promotion
(Blocker 3): proves auto_pipeline.py's maybe_promote_candidates() never
promotes a candidate network on point-estimate Elo alone, only on a
statistically decisive SPRT H1 verdict from the project's existing sprt()
(tools/nnue_pipeline/uci_match.py's sprt(), the same function
tools/nnue_pipeline/test.py already uses for its own accept/reject
verdicts) -- and that maybe_queue_elo_matches() keeps gathering more games
for an inconclusive candidate (up to --max-elo-games) instead of stopping
after one fixed-size batch, which is what previously made the old
point-estimate check the *only* signal a promotion decision could ever be
based on.

Drives auto_pipeline.py's real Stage 2/Stage 3 functions against an
in-memory FakeAdminClient (no real HTTP server, database, or engine binary
needed) so these tests exercise the actual promotion decision code path,
not just the underlying sprt()/elo_estimate() math in isolation, and run in
milliseconds.

Run directly:  python3 test_promotion.py
Run via pytest: pytest test_promotion.py
"""
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'tools', 'nnue_pipeline'))

import auto_pipeline as ap
from uci_match import elo_estimate  # only used here to document/assert margins, not to decide


class ApiErrorNotFound(ap.ApiError):
    pass


class FakeAdminClient:
    """Stands in for auto_pipeline.AdminClient: same .get()/.post() surface
    real Stage 2/Stage 3 code calls, backed by plain Python state instead of
    an HTTP round-trip to a real platform/server/app.py + sqlite database.
    Records every '.../accept' and '.../tasks/typed' call so tests can
    assert exactly what the promotion logic decided to do."""

    def __init__(self, artifacts, match_results_by_artifact=None, baseline_id=None):
        self.artifacts = {a['id']: dict(a) for a in artifacts}
        self.match_results = {k: list(v) for k, v in (match_results_by_artifact or {}).items()}
        self.baseline_id = baseline_id
        self.accepted_ids = []
        self.queued_elo_tasks = []   # list of payload dicts, in call order
        self._pending_tasks = []      # simulates the admin task queue for in-flight checks

    def get(self, path, **params):
        if path == '/artifacts':
            kind = params.get('kind')
            return [a for a in self.artifacts.values() if kind is None or a['kind'] == kind]
        if path == '/artifacts/strongest-network':
            if self.baseline_id is None:
                raise ApiErrorNotFound('GET /artifacts/strongest-network: HTTP 404: none accepted')
            return self.artifacts[self.baseline_id]
        if path == '/admin/tasks':
            status = params.get('status')
            return [t for t in self._pending_tasks if status is None or t['status'] == status]
        if path.startswith('/admin/artifacts/') and path.endswith('/match-results'):
            artifact_id = path.split('/')[3]
            return self.match_results.get(artifact_id, [])
        raise AssertionError(f'FakeAdminClient: unexpected GET {path}')

    def post(self, path, json_body=None):
        if path.startswith('/admin/artifacts/') and path.endswith('/accept'):
            artifact_id = path.split('/')[3]
            self.accepted_ids.append(artifact_id)
            self.artifacts[artifact_id]['accepted'] = True
            self.baseline_id = artifact_id
            return {}
        if path == '/admin/tasks/typed':
            self.queued_elo_tasks.append(json_body)
            task_id = f"t_fake_{len(self.queued_elo_tasks)}"
            self._pending_tasks.append({'id': task_id, 'status': 'pending',
                                         'task_type': json_body['task_type'],
                                         'payload': json_body['payload']})
            return {'task_id': task_id, 'task_type': json_body['task_type']}
        raise AssertionError(f'FakeAdminClient: unexpected POST {path}')


def _args(**overrides):
    """Same defaults parse_args() gives auto_pipeline.py in real use
    (elo0=0, elo1=5 -- a deliberately narrow non-regression band, the same
    kind real engine-testing frameworks like fishtest use, which honestly
    requires thousands of games to resolve decisively). Tests that only need
    to show 'not enough evidence yet' use these production defaults
    directly with a modest game count -- that IS the correct, realistic
    behavior at that scale."""
    ns = argparse.Namespace(sprt_elo0=0.0, sprt_elo1=5.0, sprt_alpha=0.05, sprt_beta=0.05,
                             max_elo_games=400, elo_games=24, elo_match_depth=5)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _decisive_args(**overrides):
    """A wider SPRT hypothesis band (elo0=-10, elo1=40) than the production
    default -- a legitimate, operator-configurable choice (--sprt-elo0/
    --sprt-elo1), used here only so a *decisive* H1/H0 verdict is reachable
    with a small, fast-to-run number of games in these tests, without
    changing the underlying sprt() math at all. The 'inconclusive sample'
    tests deliberately use the tighter production default instead (see
    _args()), since that is the realistic regime for a small batch."""
    return _args(sprt_elo0=-10.0, sprt_elo1=40.0, **overrides)


def _network_artifact(artifact_id, accepted=False):
    return {'id': artifact_id, 'kind': 'network', 'accepted': accepted,
            'created_at': '2026-01-01T00:00:00Z'}


def _match_result(wins, losses, draws):
    return {'wins': wins, 'losses': losses, 'draws': draws, 'games': wins + losses + draws}


class PromotionDecisionTests(unittest.TestCase):
    """Stage 3 (maybe_promote_candidates): the actual promote/reject decision."""

    def test_weaker_candidate_is_rejected_not_promoted(self):
        """A candidate that has been decisively beaten (not just 'didn't
        clearly win') must reach SPRT H0 and never be promoted, regardless
        of how many games were played. Uses _decisive_args()'s wider SPRT
        band so 40 games is actually enough to resolve -- see its docstring."""
        client = FakeAdminClient(
            artifacts=[_network_artifact('baseline', accepted=True),
                       _network_artifact('weak_candidate')],
            match_results_by_artifact={
                'weak_candidate': [_match_result(wins=4, losses=30, draws=6)]},
            baseline_id='baseline')

        ap.maybe_promote_candidates(client, _decisive_args())

        self.assertEqual(client.accepted_ids, [],
                          'a weaker candidate must never be promoted')
        verdict, *_ = ap._sprt_verdict(client, _decisive_args(), 'weak_candidate')
        self.assertEqual(verdict['verdict'], 'H0',
                          'a decisively-losing candidate should reach SPRT H0')

    def test_statistically_insignificant_sample_is_not_promoted(self):
        """Reproduces the audit's real historical failure case: a candidate
        with a small, split sample of games giving a near-zero point-estimate
        Elo and a very wide confidence interval (no real signal either way).
        The old logic ('promote if elo >= threshold', threshold defaulting
        to 0.0) would have promoted this immediately, since 0.0 >= 0.0. The
        SPRT-gated logic must not: with almost no evidence, the verdict has
        to be 'continue' (not enough data), never 'H1'."""
        # 3 wins / 3 losses / 2 draws: score = 0.5, Elo point estimate ~ 0.0,
        # but only 8 games -- exactly the "wide margin, no real signal" shape
        # the audit flagged (that real case was Elo=0.0 +/- 296 over a
        # single small batch).
        results = [_match_result(wins=3, losses=3, draws=2)]
        client = FakeAdminClient(
            artifacts=[_network_artifact('baseline', accepted=True),
                       _network_artifact('flat_candidate')],
            match_results_by_artifact={'flat_candidate': results},
            baseline_id='baseline')

        elo, margin = elo_estimate(3, 3, 2)
        self.assertAlmostEqual(elo, 0.0, delta=1.0,
                                msg='sanity check: this sample\'s point-estimate Elo really is ~0')
        self.assertGreater(margin, 100,
                            msg='sanity check: this sample really is statistically insignificant '
                                '(wide confidence interval, same shape as the audited real case)')

        ap.maybe_promote_candidates(client, _args())

        self.assertEqual(client.accepted_ids, [],
                          'a statistically insignificant (near-zero-signal) sample must never '
                          'be promoted, even though its point-estimate Elo is >= 0')
        verdict, *_ = ap._sprt_verdict(client, _args(), 'flat_candidate')
        self.assertEqual(verdict['verdict'], 'continue',
                          'too little evidence either way should be SPRT \'continue\', not H1')

    def test_decisively_stronger_candidate_is_promoted(self):
        """The positive case: enough games with a real, decisive advantage
        must reach SPRT H1 and get promoted -- the gate has to actually be
        satisfiable, not just permanently closed. Uses _decisive_args()'s
        wider SPRT band so 40 games is actually enough to resolve; see its
        docstring for why that's still a faithful use of the real sprt()."""
        client = FakeAdminClient(
            artifacts=[_network_artifact('baseline', accepted=True),
                       _network_artifact('strong_candidate')],
            match_results_by_artifact={
                'strong_candidate': [_match_result(wins=28, losses=8, draws=4)]},
            baseline_id='baseline')

        ap.maybe_promote_candidates(client, _decisive_args())

        self.assertEqual(client.accepted_ids, ['strong_candidate'],
                          'a decisively stronger, well-evidenced candidate should be promoted')
        self.assertTrue(client.artifacts['strong_candidate']['accepted'])

    def test_no_match_results_yet_is_not_promoted(self):
        """A brand-new candidate with zero recorded games must never be
        promoted (there's nothing to run SPRT against yet)."""
        client = FakeAdminClient(
            artifacts=[_network_artifact('baseline', accepted=True),
                       _network_artifact('untested_candidate')],
            match_results_by_artifact={},
            baseline_id='baseline')

        ap.maybe_promote_candidates(client, _args())

        self.assertEqual(client.accepted_ids, [])


class QueueingTests(unittest.TestCase):
    """Stage 2 (maybe_queue_elo_matches): keeps gathering evidence for an
    inconclusive candidate instead of stopping after one fixed batch, which
    is what made the old logic's promotion decision structurally unable to
    ever be based on more than a single (often insignificant) batch."""

    def test_more_games_are_queued_while_verdict_is_inconclusive(self):
        client = FakeAdminClient(
            artifacts=[_network_artifact('baseline', accepted=True),
                       _network_artifact('flat_candidate')],
            match_results_by_artifact={
                'flat_candidate': [_match_result(wins=3, losses=3, draws=2)]},
            baseline_id='baseline')

        ap.maybe_queue_elo_matches(client, _args())

        self.assertEqual(len(client.queued_elo_tasks), 1,
                          'still-inconclusive candidate should get another ELO_MATCH batch queued')
        self.assertEqual(client.queued_elo_tasks[0]['payload']['candidate_artifact_id'],
                          'flat_candidate')

    def test_no_more_games_queued_once_verdict_is_decisive(self):
        client = FakeAdminClient(
            artifacts=[_network_artifact('baseline', accepted=True),
                       _network_artifact('strong_candidate')],
            match_results_by_artifact={
                'strong_candidate': [_match_result(wins=28, losses=8, draws=4)]},
            baseline_id='baseline')

        ap.maybe_queue_elo_matches(client, _decisive_args())

        self.assertEqual(client.queued_elo_tasks, [],
                          'a candidate that already has a decisive SPRT verdict (H1 here) should '
                          'not have more games queued -- Stage 3 already has enough to decide')

    def test_no_more_games_queued_once_max_elo_games_reached(self):
        # 100 wins / 100 losses / 100 draws: score exactly 0.5 (no signal),
        # but 300 games -- still 'continue' under default elo0/elo1, but at
        # or beyond a small --max-elo-games cap.
        client = FakeAdminClient(
            artifacts=[_network_artifact('baseline', accepted=True),
                       _network_artifact('endless_candidate')],
            match_results_by_artifact={
                'endless_candidate': [_match_result(wins=100, losses=100, draws=100)]},
            baseline_id='baseline')

        args = _args(max_elo_games=200)   # already exceeded by the 300 games above
        verdict, *_ = ap._sprt_verdict(client, args, 'endless_candidate')
        self.assertEqual(verdict['verdict'], 'continue',
                          'sanity check: an exactly-even sample should still be inconclusive')

        ap.maybe_queue_elo_matches(client, args)

        self.assertEqual(client.queued_elo_tasks, [],
                          'must stop queueing more games once --max-elo-games is reached, even '
                          'though the verdict is still \'continue\' (an honest \'too close to '
                          'call\' outcome, not an excuse to spin forever)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
