#!/usr/bin/env python3
"""auto_pipeline.py - The automated community-compute improvement loop
(Phase 2): removes every manual admin step between "workers contribute
data/training/matches" and "the engine's strongest network actually gets
better."

    DATA_GENERATION positions accumulate (workers, unattended)
        -> this controller exports a new dataset artifact once enough
           new positions exist (POST /admin/pipeline/export-dataset)
        -> queues a TRAIN_NETWORK task against it
        -> a trainer-capable worker trains a real network (Phase 1/3) and
           uploads a candidate 'network' artifact
        -> this controller queues an ELO_MATCH task: candidate vs. the
           current strongest network
        -> a worker plays the match and uploads the result
        -> this controller queues further ELO_MATCH batches and reads the
           aggregated match_results for the candidate on every cycle, running
           the project's existing SPRT implementation
           (tools/nnue_pipeline/uci_match.py's sprt()) against them; once
           that reaches a decisive H1 verdict it promotes the candidate
           (POST /admin/artifacts/{id}/accept), an H0 verdict leaves it
           un-accepted (a real, honest "this one didn't improve things"
           outcome, not a failure), and 'continue' queues more games (up to
           --max-elo-games) -- never a promotion decision from a raw Elo
           point estimate alone, see maybe_promote_candidates() below
        -> repeat

This is a pure HTTP client of platform/server/app.py's admin API -- it does
not touch the database directly (unlike automation/pipeline_controller.py,
which is a separate, single-machine controller for the OLDER distributed/ +
training_server/ system and is left completely unmodified; see that file's
own module doc). Running this alongside the server is what makes the
platform/ community-compute system actually closed-loop end to end; without
it, every step above still works, but an admin has to manually call each
endpoint in order (which is exactly what Phases 1's predecessor audit
found: "no automated task-creation loop").

Also queues fresh SELF_PLAY/DATA_GENERATION work whenever the pending queue
runs low, so idle CPU-only contributors always have something real to do
even before enough data exists for a training cycle.

Usage (single cycle, for testing):
    python3 auto_pipeline.py --server http://localhost:8000 \
        --admin-token $CHESS_PLATFORM_ADMIN_TOKEN --once

Usage (continuous daemon, what an operator actually runs):
    python3 auto_pipeline.py --server http://localhost:8000 \
        --admin-token $CHESS_PLATFORM_ADMIN_TOKEN --loop --interval-seconds 300
"""
import argparse
import os
import sys
import time

import requests

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'tools', 'nnue_pipeline')
sys.path.insert(0, os.path.abspath(PIPELINE_DIR))


def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] [auto_pipeline] {msg}', flush=True)


class ApiError(Exception):
    pass


class AdminClient:
    """Thin wrapper over the admin HTTP API -- every call this controller
    makes goes through here so retries/error-handling/auth stay in one
    place."""

    def __init__(self, base_url, admin_token, timeout=60):
        self.base_url = base_url.rstrip('/')
        self.headers = {'X-Admin-Token': admin_token}
        self.timeout = timeout

    def get(self, path, **params):
        resp = requests.get(f'{self.base_url}{path}', headers=self.headers,
                             params=params, timeout=self.timeout)
        if resp.status_code >= 400:
            raise ApiError(f'GET {path}: HTTP {resp.status_code}: {resp.text}')
        return resp.json() if resp.content else None

    def post(self, path, json_body=None):
        resp = requests.post(f'{self.base_url}{path}', headers=self.headers,
                              json=json_body or {}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise ApiError(f'POST {path}: HTTP {resp.status_code}: {resp.text}')
        return resp.json() if resp.content else None


# ---------------------------------------------------------------------------
# Stage 0: keep the CPU task queue non-empty so idle workers always have
# real work, independent of whether a training cycle is due yet.
# ---------------------------------------------------------------------------
def ensure_cpu_queue(client, args):
    pending = client.get('/admin/tasks', status='pending')
    by_type = {}
    for t in pending:
        by_type[t['task_type']] = by_type.get(t['task_type'], 0) + 1

    dg_pending = by_type.get('DATA_GENERATION', 0)
    if dg_pending < args.queue_data_generation_if_below:
        n = args.data_generation_batch_count
        log(f'DATA_GENERATION queue low ({dg_pending} pending, threshold '
            f'{args.queue_data_generation_if_below}) -- queueing {n} more task(s)')
        for _ in range(n):
            client.post('/admin/tasks/typed', {
                'task_type': 'DATA_GENERATION',
                'payload': {'games': args.data_generation_games,
                            'depth': args.data_generation_depth,
                            'randomplies': 6},
            })

    sp_pending = by_type.get('SELF_PLAY', 0)
    if sp_pending < args.queue_selfplay_if_below:
        log(f'SELF_PLAY queue low ({sp_pending} pending, threshold '
            f'{args.queue_selfplay_if_below}) -- queueing {args.selfplay_batch_positions} '
            f'more target positions')
        client.post('/admin/tasks', {
            'total_positions': args.selfplay_batch_positions,
            'chunk_size': args.selfplay_chunk_size,
            'depth': args.selfplay_depth,
            'randomplies': 6,
        })


# ---------------------------------------------------------------------------
# Stage 1: dataset export + TRAIN_NETWORK scheduling
# ---------------------------------------------------------------------------
def maybe_queue_training(client, args):
    resp = client.post('/admin/pipeline/export-dataset', {
        'min_new_positions': args.min_new_positions,
        'max_positions': args.max_dataset_positions,
    })
    if not resp['created']:
        log(f"dataset export: not enough new data yet ({resp.get('reason')})")
        return None

    log(f"dataset export: new dataset artifact {resp['artifact_id']} "
        f"({resp['count']} positions, watermark now position id {resp['max_position_id']})")

    # Don't pile up redundant TRAIN_NETWORK tasks -- only queue a fresh batch
    # if none is currently pending/assigned. Once none are in flight, queue
    # --experiment-candidates of them (default 1, i.e. the original
    # single-candidate behavior is unchanged) against the SAME dataset,
    # tagged with a shared experiment_id and each given a distinct --seed
    # (see train_network.py's _train_cpu) so they train genuinely different
    # candidate networks, not N copies of the same run. maybe_queue_elo_matches
    # and maybe_promote_candidates already iterate over every non-accepted
    # candidate artifact generically, so multiple concurrent candidates are
    # automatically Elo-tested and promotion-evaluated independently with no
    # further changes needed there -- the only thing that changes is how many
    # TRAIN_NETWORK tasks get queued here.
    pending = client.get('/admin/tasks', status='pending') or []
    assigned = client.get('/admin/tasks', status='assigned') or []
    in_flight = [t for t in pending + assigned if t['task_type'] == 'TRAIN_NETWORK']
    if in_flight:
        log(f'{len(in_flight)} TRAIN_NETWORK task(s) already in flight -- not queueing another')
        return resp['artifact_id']

    n_candidates = max(1, args.experiment_candidates)
    experiment_id = None
    if n_candidates > 1:
        experiment_id = f"exp_{resp['artifact_id']}_{int(time.time())}"
        log(f"queueing a {n_candidates}-candidate training experiment {experiment_id} "
            f"against dataset {resp['artifact_id']}")

    queued_ids = []
    for i in range(n_candidates):
        payload = {'dataset_artifact_id': resp['artifact_id'], 'epochs': args.train_epochs,
                   'max_samples': args.max_dataset_positions,
                   'seed': args.train_seed_base + i}
        if experiment_id:
            payload['experiment_id'] = experiment_id
        task = client.post('/admin/tasks/typed', {'task_type': 'TRAIN_NETWORK', 'payload': payload})
        queued_ids.append(task['task_id'])
        log(f"queued TRAIN_NETWORK task {task['task_id']} against dataset {resp['artifact_id']} "
            f"(seed={payload['seed']}" + (f", experiment={experiment_id})" if experiment_id else ")"))

    return resp['artifact_id']


def _aggregate(results):
    wins = sum(r['wins'] for r in results)
    losses = sum(r['losses'] for r in results)
    draws = sum(r['draws'] for r in results)
    return wins, losses, draws


def _sprt_verdict(client, args, candidate_id):
    """Runs the project's existing SPRT implementation
    (tools/nnue_pipeline/uci_match.py's sprt(), the same function
    tools/nnue_pipeline/test.py already uses for its own accept/reject
    verdicts) against every match_results row recorded so far for this
    candidate. Returns (verdict_dict, wins, losses, draws, total_games) --
    verdict_dict['verdict'] is 'H1' (candidate is statistically stronger),
    'H0' (statistically not stronger -- reject), or 'continue' (not enough
    evidence yet either way)."""
    from uci_match import sprt  # tools/nnue_pipeline/uci_match.py, already-proven math

    results = client.get(f"/admin/artifacts/{candidate_id}/match-results") or []
    wins, losses, draws = _aggregate(results)
    total_games = wins + losses + draws
    verdict = sprt(wins, losses, draws, args.sprt_elo0, args.sprt_elo1,
                   alpha=args.sprt_alpha, beta=args.sprt_beta)
    return verdict, wins, losses, draws, total_games


# ---------------------------------------------------------------------------
# Stage 2: queue ELO_MATCH batches for any candidate network whose SPRT
# verdict isn't decisive yet (no result at all, or 'continue' with more
# games still allowed under --max-elo-games).
# ---------------------------------------------------------------------------
def maybe_queue_elo_matches(client, args):
    candidates = [a for a in (client.get('/artifacts', kind='network') or [])
                  if not a['accepted']]
    if not candidates:
        return

    baseline = None
    try:
        baseline = client.get('/artifacts/strongest-network')
    except ApiError:
        pass  # no accepted network yet -- see below

    pending = client.get('/admin/tasks', status='pending') or []
    assigned = client.get('/admin/tasks', status='assigned') or []
    in_flight_candidates = set()
    for t in pending + assigned:
        if t['task_type'] == 'ELO_MATCH':
            payload = t.get('payload') or {}
            if payload.get('candidate_artifact_id'):
                in_flight_candidates.add(payload['candidate_artifact_id'])

    for cand in candidates:
        if cand['id'] in in_flight_candidates:
            continue

        if baseline is None:
            # No accepted network exists yet anywhere -- this candidate
            # becomes the seed baseline directly (an operator/controller
            # can't run an Elo match with nothing to compare against). This
            # mirrors what a human operator would have to do manually via
            # /admin/artifacts/{id}/accept on a fresh deployment.
            log(f"no strongest-network exists yet -- promoting first candidate "
                f"{cand['id']} directly as the seed baseline (no Elo match possible "
                f"against nothing)")
            client.post(f"/admin/artifacts/{cand['id']}/accept")
            baseline = cand
            continue

        if cand['id'] == baseline['id']:
            continue

        # A decisive SPRT verdict (H1 or H0) already exists for this
        # candidate -- Stage 3 handles promotion/rejection; queueing more
        # games would just waste compute on a question that's already
        # statistically answered. Only 'continue' (or no games yet) means
        # more evidence is actually useful.
        verdict, _wins, _losses, _draws, total_games = _sprt_verdict(client, args, cand['id'])
        if verdict['verdict'] != 'continue':
            continue
        if total_games >= args.max_elo_games:
            # Capped without reaching significance -- an honest "still too
            # close to call after a lot of games" outcome, not a failure.
            # Stage 3 will log this and leave the candidate un-promoted.
            continue

        task = client.post('/admin/tasks/typed', {
            'task_type': 'ELO_MATCH',
            'payload': {'candidate_artifact_id': cand['id'],
                        'baseline_artifact_id': baseline['id'],
                        'games': args.elo_games, 'match_depth': args.elo_match_depth},
        })
        log(f"queued ELO_MATCH task {task['task_id']}: candidate={cand['id']} "
            f"vs baseline={baseline['id']} ({total_games} games so far, SPRT still "
            f"'continue')")


# ---------------------------------------------------------------------------
# Stage 3: promote candidates whose aggregated match results reach a
# statistically decisive SPRT H1 verdict -- never on point-estimate Elo
# alone. Reuses the project's existing sprt() (tools/nnue_pipeline/
# uci_match.py), the same function tools/nnue_pipeline/test.py already uses,
# instead of duplicating the statistics here. See
# platform/server/test_promotion.py for regression tests proving a weaker
# candidate and a statistically-insignificant sample (e.g. the historically
# observed Elo=0.0 +/- 296 case) are both correctly rejected.
# ---------------------------------------------------------------------------
def maybe_promote_candidates(client, args):
    from uci_match import elo_estimate  # for the human-readable log line only

    candidates = [a for a in (client.get('/artifacts', kind='network') or [])
                  if not a['accepted']]
    for cand in candidates:
        verdict, wins, losses, draws, total_games = _sprt_verdict(client, args, cand['id'])
        if total_games == 0:
            continue  # no match results recorded yet -- nothing to decide

        elo, margin = elo_estimate(wins, losses, draws)
        log(f"candidate {cand['id']}: {total_games} games (+{wins} -{losses} ={draws}), "
            f"Elo {elo:+.1f} +/- {margin:.1f}, SPRT llr={verdict['llr']:.2f} "
            f"(bounds [{verdict['lower']:.2f}, {verdict['upper']:.2f}]) -> {verdict['verdict']}")

        if verdict['verdict'] == 'H1':
            client.post(f"/admin/artifacts/{cand['id']}/accept")
            log(f"PROMOTED {cand['id']} -- SPRT accepted H1 (candidate is stronger than "
                f"elo0={args.sprt_elo0:+.1f} at alpha={args.sprt_alpha}, beta={args.sprt_beta})")
        elif verdict['verdict'] == 'H0':
            log(f"NOT promoting {cand['id']} -- SPRT accepted H0 (statistically not "
                f"stronger) -- an honest 'did not improve the engine' outcome, not a "
                f"pipeline failure")
        else:
            log(f"candidate {cand['id']}: SPRT verdict still 'continue' after "
                f"{total_games} games -- not enough evidence yet either way, not "
                f"promoting (Stage 2 will queue more games up to --max-elo-games="
                f"{args.max_elo_games} unless already capped)")


# ---------------------------------------------------------------------------
# One full cycle
# ---------------------------------------------------------------------------
def run_cycle(client, args):
    log('=== cycle starting ===')
    ensure_cpu_queue(client, args)
    maybe_queue_training(client, args)
    maybe_queue_elo_matches(client, args)
    maybe_promote_candidates(client, args)
    log('=== cycle done ===')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--server', required=True, help='platform server base URL')
    ap.add_argument('--admin-token', required=True)

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='run a single cycle and exit (default)')
    mode.add_argument('--loop', action='store_true', help='run continuously')
    ap.add_argument('--interval-seconds', type=int, default=300)
    ap.add_argument('--max-cycles', type=int, default=0, help='0 = unlimited in --loop mode')

    ap.add_argument('--queue-data-generation-if-below', type=int, default=2,
                     help='queue more DATA_GENERATION tasks when fewer than this many are pending')
    ap.add_argument('--data-generation-batch-count', type=int, default=3)
    ap.add_argument('--data-generation-games', type=int, default=200)
    ap.add_argument('--data-generation-depth', type=int, default=6)
    ap.add_argument('--queue-selfplay-if-below', type=int, default=1)
    ap.add_argument('--selfplay-batch-positions', type=int, default=5000)
    ap.add_argument('--selfplay-chunk-size', type=int, default=500)
    ap.add_argument('--selfplay-depth', type=int, default=6)

    ap.add_argument('--min-new-positions', type=int, default=2000,
                     help='minimum newly-accepted positions before exporting a training dataset')
    ap.add_argument('--max-dataset-positions', type=int, default=200_000)
    ap.add_argument('--train-epochs', type=int, default=6)
    ap.add_argument('--experiment-candidates', type=int, default=1,
                     help='how many TRAIN_NETWORK candidates to queue per training cycle '
                          'against the same exported dataset, each with a distinct --seed '
                          '(default 1 -- the original single-candidate behavior). Values > 1 '
                          'queue a real multi-candidate experiment: maybe_queue_elo_matches/'
                          'maybe_promote_candidates already Elo-test and SPRT-evaluate every '
                          'non-accepted candidate independently, so this is the only flag '
                          'needed to turn that on.')
    ap.add_argument('--train-seed-base', type=int, default=1,
                     help='first --seed value used when queueing a training cycle; candidate '
                          'i in the batch gets seed = train-seed-base + i')

    ap.add_argument('--elo-games', type=int, default=24,
                     help='games queued per ELO_MATCH batch; more batches are queued '
                          'automatically (Stage 2) while a candidate\'s SPRT verdict is '
                          'still \'continue\'')
    ap.add_argument('--elo-match-depth', type=int, default=5)

    # SPRT-gated promotion (Blocker 3): a candidate is promoted only once its
    # aggregated match_results reach a statistically decisive verdict (see
    # _sprt_verdict/maybe_promote_candidates above) -- never on a raw Elo
    # point estimate alone. elo0/elo1 define the non-regression band being
    # tested (H0: candidate is no better than elo0 Elo stronger than
    # baseline; H1: candidate is at least elo1 Elo stronger); alpha/beta are
    # the SPRT's false-accept/false-reject rate bounds. The defaults here
    # (elo0=0, elo1=5) mirror what real engine-testing frameworks like
    # fishtest use for a deliberately narrow non-regression test.
    ap.add_argument('--sprt-elo0', type=float, default=0.0,
                     help='SPRT H0 Elo bound (candidate assumed no stronger than this)')
    ap.add_argument('--sprt-elo1', type=float, default=5.0,
                     help='SPRT H1 Elo bound (candidate must reach at least this to be promoted)')
    ap.add_argument('--sprt-alpha', type=float, default=0.05,
                     help='SPRT false-accept rate (probability of promoting a candidate that '
                          'is not actually stronger)')
    ap.add_argument('--sprt-beta', type=float, default=0.05,
                     help='SPRT false-reject rate (probability of rejecting a candidate that '
                          'actually is stronger)')
    ap.add_argument('--max-elo-games', type=int, default=400,
                     help='stop queueing more ELO_MATCH games for a candidate once this many '
                          'have been played, even if the SPRT verdict is still \'continue\' -- '
                          'an honest \'still too close to call\' outcome rather than an '
                          'unbounded compute sink')

    # Disk-space management (positions accumulate forever otherwise on a
    # long-running deployment) -- opt-in and conservative by default: never
    # deletes anything the server hasn't already permanently exported into a
    # dataset artifact file (see maybe_prune_positions below).
    ap.add_argument('--prune-after-export', action='store_true',
                     help='after a successful dataset export (Stage 1), also prune raw '
                          '`positions` rows already covered by older exports (see '
                          'maybe_prune_positions / POST /admin/pipeline/prune-positions). '
                          'Off by default -- a deployment with plenty of disk headroom can '
                          'simply never pass this flag and keep every raw position forever.')
    ap.add_argument('--keep-datasets', type=int, default=3,
                     help='when --prune-after-export is set, how many of the most recent '
                          'auto-exported datasets\' worth of raw positions to keep around as '
                          'a buffer before pruning anything older (passed straight through to '
                          'POST /admin/pipeline/prune-positions\' keep_datasets)')

    # Push-notification alerting (opt-in): rather than have an outside
    # process poll this server (which turned out to be unreliable -- some
    # monitoring environments can't reach an arbitrary server IP/port at
    # all, a network restriction on the monitor's side), this loop checks
    # its own server's /admin/system-load each cycle and, if enabled,
    # pushes a notification out via ntfy.sh (see maybe_alert_on_capacity /
    # evaluate_capacity_alert below) when something needs attention.
    ap.add_argument('--ntfy-topic', default='',
                     help='ntfy.sh topic to push capacity/health alerts to (see '
                          'https://ntfy.sh -- subscribe at ntfy.sh/<topic> in a browser '
                          'or the app). Off by default -- leave unset to disable alerting '
                          'entirely.')
    ap.add_argument('--ntfy-reminder-cycles', type=int, default=12,
                     help='while a problem persists, send a reminder notification every '
                          'this many cycles instead of only once (default 12 -- e.g. '
                          'roughly hourly at a 5-minute --interval-seconds)')

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Stage 4 (opt-in): prune raw positions already captured in older exported
# dataset artifacts, so a long-running deployment doesn't fill its disk with
# rows that only ever mattered as an intermediate step toward a dataset
# file that already exists permanently as an artifact. See database.py's
# delete_positions_up_to() docstring and app.py's /admin/pipeline/
# prune-positions for why this is safe and how the keep_datasets buffer
# works. Only runs when --prune-after-export was passed -- most deployments
# with adequate disk simply never enable this.
# ---------------------------------------------------------------------------
def maybe_prune_positions(client, args):
    if not args.prune_after_export:
        return
    resp = client.post('/admin/pipeline/prune-positions', {'keep_datasets': args.keep_datasets})
    if not resp['pruned']:
        log(f"prune: nothing pruned ({resp.get('reason')})")
        return
    log(f"prune: deleted {resp['deleted_count']} position row(s) with id <= "
        f"{resp['deleted_up_to_id']} (kept the {args.keep_datasets} most recent exports' "
        f"worth of raw rows as a buffer)")



# ---------------------------------------------------------------------------
# Stage 5 (opt-in): push notification when the server itself is under
# strain, via ntfy.sh (https://ntfy.sh -- free, no account, a POST to
# ntfy.sh/<topic> delivers to anyone subscribed to that topic in the app
# or a browser tab). This exists because polling the server FROM outside
# (a scheduled external monitor) turned out not to work reliably in
# practice -- some monitoring environments can't reach an arbitrary
# server IP/port at all (network policy on the *monitor's* side, nothing
# wrong with this server). Pushing OUT from here instead sidesteps that
# entirely: this process already runs on the same box as the server with
# full outbound internet access, so there's no inbound reachability
# question to begin with.
#
# evaluate_capacity_alert() is a pure function (snapshot dict in, message
# string or None out) so the threshold logic is unit-testable without a
# real server or network call -- see test_capacity_alert.py.
# ---------------------------------------------------------------------------
def evaluate_capacity_alert(snapshot):
    # snapshot is GET /admin/system-load's response shape. Returns a
    # human-readable alert message describing every crossed threshold
    # (joined, not one call per condition), or None if nothing is wrong.
    # Mirrors the thresholds a from-outside monitor would use, but this
    # one actually has memory/disk/load_average available since it's
    # calling the real admin endpoint locally, not the header-less public
    # /capacity subset an external caller is limited to.
    problems = []

    if snapshot.get('at_worker_capacity'):
        problems.append(
            f"at worker capacity ({snapshot['connected_workers']}/"
            f"{snapshot['max_connected_workers']} connected) -- new volunteers are being "
            f"turned away")

    pending = snapshot.get('pending_tasks', 0)
    if pending >= 100:
        problems.append(f"task queue backlog: {pending} pending")

    mem = snapshot.get('memory') or {}
    if mem.get('used_percent') is not None and mem['used_percent'] >= 85:
        problems.append(f"memory at {mem['used_percent']}%")

    disk = snapshot.get('disk') or {}
    if disk.get('used_percent') is not None and disk['used_percent'] >= 85:
        problems.append(f"disk at {disk['used_percent']}%")

    load = snapshot.get('load_average') or {}
    cpu_count = snapshot.get('cpu_count')
    if load.get('1min') is not None and cpu_count:
        if load['1min'] >= cpu_count * 1.2:
            problems.append(f"load average {load['1min']} ({cpu_count} CPUs)")

    if not problems:
        return None
    return "Morph server: " + "; ".join(problems)


def send_ntfy_notification(topic, message, priority='default', tags=None):
    # POSTs to ntfy.sh -- see the Stage 5 comment above. Network errors
    # here are logged and swallowed, never raised: a failed notification
    # must never crash the improvement loop itself.
    headers = {'Title': 'Morph server', 'Priority': priority}
    if tags:
        headers['Tags'] = tags
    try:
        resp = requests.post(f'https://ntfy.sh/{topic}', data=message.encode('utf-8'),
                             headers=headers, timeout=15)
        if resp.status_code >= 400:
            log(f'ntfy: HTTP {resp.status_code} sending notification: {resp.text}')
    except requests.RequestException as e:
        log(f'ntfy: failed to send notification: {e}')


class CapacityAlertState:
    # Tracks alert state across loop iterations (this is a long-running
    # process, so plain instance state is enough -- no need to persist
    # anything to disk). Behavior: notify immediately when a problem
    # first appears, send a reminder every reminder_cycles while it
    # persists (so a real, ongoing problem doesn't go silent after the
    # first ping), and send one 'resolved' notification when it clears.

    def __init__(self, reminder_cycles):
        self.reminder_cycles = reminder_cycles
        self.was_alerting = False
        self.cycles_since_notify = 0

    def observe(self, message):
        # message is evaluate_capacity_alert()'s return value (str or
        # None). Returns the notification text to actually send this
        # cycle, or None to send nothing.
        if message is not None:
            if not self.was_alerting:
                # A problem just appeared -- always notify right away.
                self.was_alerting = True
                self.cycles_since_notify = 0
                return message
            self.cycles_since_notify += 1
            if self.cycles_since_notify >= self.reminder_cycles:
                # Still going after reminder_cycles more cycles -- ping again
                # so an ongoing problem doesn't go silent after the first alert.
                self.cycles_since_notify = 0
                return message
            return None

        if self.was_alerting:
            self.was_alerting = False
            self.cycles_since_notify = 0
            return "Morph server: back to normal -- all thresholds clear now."

        return None


def maybe_alert_on_capacity(client, args, alert_state):
    if not args.ntfy_topic:
        return
    try:
        snapshot = client.get('/admin/system-load')
    except ApiError as e:
        log(f'capacity check: could not reach /admin/system-load: {e}')
        return
    message = evaluate_capacity_alert(snapshot)
    to_send = alert_state.observe(message)
    if to_send:
        tags = 'warning' if message is not None else 'white_check_mark'
        log(f'capacity alert: {to_send}')
        send_ntfy_notification(args.ntfy_topic, to_send, tags=tags)

def main():
    args = parse_args()
    client = AdminClient(args.server, args.admin_token)
    alert_state = CapacityAlertState(args.ntfy_reminder_cycles)

    if args.loop:
        cycles = 0
        while True:
            run_cycle(client, args)
            maybe_prune_positions(client, args)
            maybe_alert_on_capacity(client, args, alert_state)
            cycles += 1
            if args.max_cycles and cycles >= args.max_cycles:
                log(f'reached --max-cycles={args.max_cycles} -- exiting')
                break
            time.sleep(args.interval_seconds)
    else:
        run_cycle(client, args)
        maybe_prune_positions(client, args)
        maybe_alert_on_capacity(client, args, alert_state)


if __name__ == '__main__':
    main()
