#!/usr/bin/env python3
"""app.py - FastAPI application for the public community-compute platform.

This is the production evolution of distributed/server/app.py (which stays
completely unmodified for local/LAN trusted testing -- see
docs/DISTRIBUTED_DATA_GENERATION.md). Same underlying task-queue/validation/
dedup engine (reused via PlatformDatabase, a subclass of distributed's
Database), plus what a public deployment needs on top:

    Accounts:   POST /accounts/register, /accounts/login, /accounts/logout,
                POST /accounts/api-key/regenerate
    Workers:    POST /register (account-linked, via api_key -- or the legacy
                shared registration_secret if the operator enabled it),
                POST /workers/capabilities (report/update CPU/RAM/GPU/trainer
                capability -- used by capability-aware task assignment)
    Legacy task polling (untyped, distributed/-compatible wire format):
                GET /tasks/next, POST /tasks/{id}/results
    Typed task polling (SELF_PLAY / DATA_GENERATION / ELO_MATCH /
    TRAIN_NETWORK -- what platform/worker/ actually uses):
                GET /tasks/next-typed, POST /tasks/{id}/match-result
    Artifacts (datasets/checkpoints/candidate & accepted NNUE networks):
                GET /artifacts, GET /artifacts/{id}, GET /artifacts/{id}/download,
                GET /artifacts/strongest-network, POST /artifacts/upload
    Community:  GET /leaderboard, GET /dashboard/summary, GET /dashboard/elo-series,
                GET /stats, GET /workers, GET /version, GET /capacity
    Admin:      POST /admin/tasks, GET /admin/tasks, POST /admin/tasks/typed,
                POST /admin/artifacts, POST /admin/pipeline/export-dataset,
                POST /admin/pipeline/prune-positions, GET /admin/system-load,
                POST /admin/artifacts/{id}/accept, GET /admin/artifacts/{id}/match-results,
                POST /admin/workers/{id}/disable, POST /admin/workers/{id}/enable

Security additions over distributed/server/app.py: rate limiting on
registration/login/submission/api-key-regeneration endpoints (ratelimit.py),
an extra submission-plausibility check + auto-suspension of repeatedly-bad
workers (anti_cheat.py), per-account API keys as the primary
worker-registration credential (accounts.py/database.py) instead of one
secret shared by everyone, and timing-safe comparisons (hmac.compare_digest)
for every secret-equality check (admin token, legacy registration secret).
"""
import hashlib
import json
import os
import sys

from fastapi import FastAPI, Depends, HTTPException, Header, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'distributed', 'server'))
import auth as dist_auth  # distributed/server/auth.py, reused for require_admin's pattern
from anti_cheat import plausibility_check, maybe_auto_disable, validate_network_artifact
from platform_config import settings
from database import PlatformDatabase
from ratelimit import limiter
from dashboard_data import build_dashboard_summary, build_elo_series
from system_load import build_system_load_snapshot
from schemas import (
    RegisterUserRequest, UserResponse, LoginRequest, LoginResponse, ApiKeyResponse,
    WorkerRegisterRequest, WorkerRegisterResponse, LeaderboardResponse,
    TaskResponse, SubmitRequest, SubmitResponse, CreateTasksRequest, CreateTasksResponse,
    WorkerCapabilities, TypedTaskResponse, CreateTypedTaskRequest, CreateTypedTaskResponse,
    ArtifactResponse, RegisterArtifactRequest, MatchResultRequest, MatchResultResponse,
    ArtifactUploadResponse, ExportDatasetRequest, ExportDatasetResponse,
    PrunePositionsRequest, PrunePositionsResponse, SystemLoadResponse,
)
from validation import validate_position, content_hash  # distributed/server/validation.py

db = PlatformDatabase(settings.db_path)

app = FastAPI(title='Morph Community Compute Platform', version='1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],   # every endpoint here is either public-read or bearer/API-key
    allow_methods=['GET', 'POST'],   # authenticated, so a permissive CORS policy doesn't
    allow_headers=['*'],              # weaken anything -- see docs/SERVER.md
)


@app.on_event('startup')
def _startup():
    settings.print_startup_banner()


def client_ip(request: Request):
    return request.client.host if request.client else 'unknown'


# ---------------------------------------------------------------------------
# Health / version
# ---------------------------------------------------------------------------
@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/version')
def version():
    """Worker auto-update manifest. A worker checks this on startup and
    periodically (see platform/worker/updater.py); documented in
    docs/WORKER.md."""
    return {'worker_client_version': settings.latest_worker_version}


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
def require_session(authorization: str = Header(default=None)):
    token = dist_auth._bearer_token(authorization)
    user = db.get_user_by_session(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail='missing or invalid session token')
    return user


@app.post('/accounts/register', response_model=UserResponse)
def register_account(req: RegisterUserRequest, request: Request):
    if not limiter.check(f'register:{client_ip(request)}',
                          settings.rate_limit_registrations_per_hour, 3600):
        raise HTTPException(status_code=429, detail='too many registration attempts, try later')
    conn_check = db._conn()
    try:
        existing = conn_check.execute('SELECT 1 FROM users WHERE username = ?',
                                       (req.username,)).fetchone()
    finally:
        conn_check.close()
    if existing:
        raise HTTPException(status_code=409, detail='username already taken')
    user_id = db.create_user(req.username, req.email, req.password)
    return UserResponse(user_id=user_id, username=req.username)


@app.post('/accounts/login', response_model=LoginResponse)
def login(req: LoginRequest, request: Request):
    if not limiter.check(f'login:{client_ip(request)}',
                          settings.rate_limit_login_attempts_per_15min, 900):
        raise HTTPException(status_code=429, detail='too many login attempts, try again later')
    user = db.authenticate_user(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail='invalid username or password')
    token = db.create_session(user['id'])
    return LoginResponse(session_token=token, user_id=user['id'], username=user['username'])


@app.post('/accounts/logout')
def logout(authorization: str = Header(default=None)):
    token = dist_auth._bearer_token(authorization)
    if token:
        db.revoke_session(token)
    return {'logged_out': True}


@app.post('/accounts/api-key/regenerate', response_model=ApiKeyResponse)
def regenerate_api_key(user=Depends(require_session)):
    # Rate-limited per account (not per IP -- a session is already required,
    # so this bounds how fast a compromised/careless session can churn keys,
    # independent of how many IPs it's called from).
    if not limiter.check(f'apikey-regen:{user["id"]}',
                          settings.rate_limit_registrations_per_hour, 3600):
        raise HTTPException(status_code=429, detail='too many key regenerations, try later')
    key = db.regenerate_api_key(user['id'])
    return ApiKeyResponse(api_key=key)


# ---------------------------------------------------------------------------
# Worker registration + auth (legacy task fetch / result upload keep
# distributed/server's exact wire format -- a worker client only needs to
# change how it registers, not how it talks to /tasks/next or
# /tasks/{id}/results)
# ---------------------------------------------------------------------------
@app.post('/register', response_model=WorkerRegisterResponse)
def register_worker(req: WorkerRegisterRequest, request: Request):
    if not limiter.check(f'wregister:{client_ip(request)}',
                          settings.rate_limit_registrations_per_hour, 3600):
        raise HTTPException(status_code=429, detail='too many registration attempts, try later')

    # Load safety valve (see platform_config.py's max_connected_workers
    # docstring): once the server already has as many *connected* workers
    # as it's configured to handle, turn away new registrations rather
    # than accept them and let everyone's experience quietly degrade.
    # Deliberately does NOT affect already-registered workers re-polling
    # or re-authenticating -- only brand-new POST /register calls. An
    # operator can watch this coming via GET /admin/system-load before it
    # ever triggers.
    connected = db.count_connected_workers()
    if connected >= settings.max_connected_workers:
        raise HTTPException(
            status_code=503,
            detail=f'server is at capacity ({connected}/{settings.max_connected_workers} '
                   f'connected workers) -- please try again later')

    if req.api_key:
        user = db.get_user_by_api_key(req.api_key)
        if user is None:
            raise HTTPException(status_code=401, detail='invalid API key')
        worker_id, token = db.register_worker_for_user(
            user['id'], req.hostname, req.engine_version, req.threads)
        return WorkerRegisterResponse(worker_id=worker_id, worker_token=token,
                                      task_lease_seconds=settings.task_lease_seconds,
                                      linked_account=user['username'])

    if req.registration_secret:
        import hmac
        if not settings.registration_secret or not hmac.compare_digest(
                req.registration_secret, settings.registration_secret):
            raise HTTPException(status_code=401, detail='invalid registration secret')
        worker_id, token = db.register_worker(req.hostname, req.engine_version, req.threads)
        return WorkerRegisterResponse(worker_id=worker_id, worker_token=token,
                                      task_lease_seconds=settings.task_lease_seconds,
                                      linked_account=None)

    raise HTTPException(status_code=400,
                        detail='provide either api_key (recommended -- see /accounts/register) '
                               'or registration_secret')


def require_worker(authorization: str = Header(default=None)):
    token = dist_auth._bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail='missing bearer token')
    worker = db.authenticate_worker(token)
    if worker is None:
        raise HTTPException(status_code=401, detail='invalid or disabled worker token')
    return worker


@app.post('/workers/capabilities')
def report_capabilities(caps: WorkerCapabilities, worker=Depends(require_worker)):
    """A worker calls this right after registering, and again whenever its
    declared resource limits change (e.g. the operator toggles
    --allow-gpu-training). assign_next_typed_task() reads the most recent
    row persisted here to decide whether TRAIN_NETWORK tasks are eligible
    for this worker -- see database.py's set_worker_capabilities/
    get_worker_capabilities and docs/WORKER.md."""
    db.set_worker_capabilities(worker['id'], caps.model_dump())
    return {'worker_id': worker['id'], 'capabilities': caps.model_dump()}


@app.get('/tasks/next')
def next_task(worker=Depends(require_worker)):
    """Legacy/distributed-compatible untyped polling -- only ever hands out
    SELF_PLAY tasks (via the original FIFO assign_next_task, no capability
    awareness). Kept unchanged so distributed/'s own LAN worker and any
    existing SELF_PLAY-only client keep working verbatim. New workers should
    use GET /tasks/next-typed instead."""
    task = db.assign_next_task(worker['id'], settings.task_lease_seconds)
    if task is None:
        # A bare Response (no body) is required here, not JSONResponse(content=None):
        # JSONResponse would still serialize a `null` body, and h11 frames 204
        # responses as zero-length per RFC -- sending any body bytes then raises
        # "Too much data for declared Content-Length" on every empty-queue poll
        # (found live under stress-test load, since polling with no available
        # task is the single most common request in real deployment).
        return Response(status_code=204)
    return TaskResponse(task_id=task['id'], target_positions=task['target_positions'],
                         depth=task['depth'], randomplies=task['randomplies'])


@app.get('/tasks/next-typed', response_model=TypedTaskResponse)
def next_typed_task(worker=Depends(require_worker)):
    """Capability-aware typed polling -- returns SELF_PLAY, DATA_GENERATION,
    ELO_MATCH, or TRAIN_NETWORK tasks (see assign_next_typed_task).
    TRAIN_NETWORK tasks are only ever handed to a worker whose last-reported
    capabilities (POST /workers/capabilities) have trainer_capable=true.
    Returns 204 with no body when the queue has nothing this worker is
    eligible for right now."""
    task = db.assign_next_typed_task(worker['id'], settings.task_lease_seconds)
    if task is None:
        # See next_task() above for why this must be a bare Response, not
        # JSONResponse(content=None).
        return Response(status_code=204)
    return TypedTaskResponse(task_id=task['id'], task_type=task['task_type'],
                             payload=task.get('payload') or {})


@app.post('/tasks/{task_id}/results', response_model=SubmitResponse)
def submit_results(task_id: str, req: SubmitRequest, worker=Depends(require_worker)):
    if not limiter.check(f'submit:{worker["id"]}', settings.rate_limit_submissions_per_minute, 60):
        raise HTTPException(status_code=429,
                            detail=f'submission rate limit exceeded '
                                   f'({settings.rate_limit_submissions_per_minute}/min) -- '
                                   f'batch uploads less frequently')

    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f'unknown task_id {task_id!r}')
    if task['status'] == 'assigned' and task['assigned_worker_id'] not in (None, worker['id']):
        raise HTTPException(status_code=409,
                            detail=f"task {task_id!r} is currently assigned to a different worker")

    # Extra plausibility pass (anti_cheat.py) BEFORE handing records to
    # db.submit_positions (which still runs the full structural
    # validate_position/content_hash pipeline unchanged) -- records failing
    # either are rejected and logged for auto-suspension tracking. Bulk
    # DATA_GENERATION submissions legitimately report nodes=0 (no
    # per-position telemetry from chess_train gen) -- plausibility_check
    # treats that as a recognized sentinel, not a fabrication signal.
    records = []
    extra_rejections = 0
    for p in req.positions:
        rec = p.model_dump()
        reason = plausibility_check(rec)
        if reason:
            extra_rejections += 1
            db.log_rejection(worker['id'], 'plausibility', reason)
            continue
        records.append(rec)

    result = db.submit_positions(task_id, worker['id'], records)
    for reason in result['rejected_reasons']:
        db.log_rejection(worker['id'], 'validation', reason)

    total_rejected = result['rejected'] + extra_rejections
    if total_rejected > 0:
        maybe_auto_disable(db, worker['id'])

    updated = db.get_task(task_id)
    return SubmitResponse(
        accepted=result['accepted'], duplicates=result['duplicates'],
        rejected=total_rejected, rejected_reasons=result['rejected_reasons'],
        task_status=updated['status'], task_accepted_total=updated['accepted_positions'],
        task_target=updated['target_positions'])


@app.post('/tasks/{task_id}/match-result', response_model=MatchResultResponse)
def submit_match_result(task_id: str, req: MatchResultRequest, worker=Depends(require_worker)):
    """ELO_MATCH result upload: candidate vs. baseline artifact, aggregated
    W/L/D from a paired-opening, color-reversed match (worker-side, wraps
    tools/nnue_pipeline/test.py's uci_match -- see docs/WORKER.md). One
    call completes the task; a duplicate call against an already-completed
    task_id is rejected the same way distributed/'s content-hash dedup
    rejects duplicate positions, just at task granularity instead of
    per-record."""
    if not limiter.check(f'submit:{worker["id"]}', settings.rate_limit_submissions_per_minute, 60):
        raise HTTPException(status_code=429, detail='submission rate limit exceeded')

    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f'unknown task_id {task_id!r}')
    if task.get('task_type') != 'ELO_MATCH':
        raise HTTPException(status_code=400,
                            detail=f"task {task_id!r} is not an ELO_MATCH task")
    if task['status'] == 'completed':
        raise HTTPException(status_code=409,
                            detail=f'task {task_id!r} already has a submitted match result '
                                   f'(duplicate submission rejected)')
    if task['assigned_worker_id'] not in (None, worker['id']):
        raise HTTPException(status_code=409,
                            detail=f"task {task_id!r} is currently assigned to a different worker")

    # Anti-cheat: a worker reporting more (or fewer) decisive/drawn games
    # than it was actually assigned would let a single forged submission
    # swing the aggregated Elo estimate arbitrarily -- e.g. claiming
    # wins=10000 on a task that only assigned 6 games would blow past
    # --min-elo-games and force a promotion in one shot, bypassing the
    # whole point of aggregating multiple real match results. The task's
    # own payload (set when auto_pipeline.py queued it) is the only
    # server-trusted source of how many games this task authorized -- any
    # mismatch means the worker either mis-reported or is attempting to
    # inflate its influence on promotion.
    # task['payload'] is the raw TEXT column here (db.get_task doesn't
    # deserialize it -- only assign_next_typed_task does, for its own
    # response shape), so it must be json.loads'd, not treated as a dict.
    raw_payload = task.get('payload')
    task_payload = json.loads(raw_payload) if raw_payload else {}
    expected_games = int(task_payload.get('games', 0))
    reported_games = req.wins + req.losses + req.draws
    if expected_games > 0 and reported_games != expected_games:
        raise HTTPException(
            status_code=400,
            detail=f'reported game count ({reported_games} = {req.wins}W/{req.losses}L/{req.draws}D) '
                   f'does not match the {expected_games} games this task assigned -- rejected')

    if db.get_artifact(req.candidate_artifact_id) is None:
        raise HTTPException(status_code=404, detail='unknown candidate_artifact_id')
    if db.get_artifact(req.baseline_artifact_id) is None:
        raise HTTPException(status_code=404, detail='unknown baseline_artifact_id')

    pgn_path = None
    if req.pgn_base64:
        import base64
        pgn_dir = os.path.join(settings.artifacts_dir, 'match_pgns')
        os.makedirs(pgn_dir, exist_ok=True)
        pgn_path = os.path.join(pgn_dir, f'{task_id}.pgn')
        with open(pgn_path, 'wb') as f:
            f.write(base64.b64decode(req.pgn_base64))

    result = db.submit_match_result(task_id, worker['id'], req.candidate_artifact_id,
                                     req.baseline_artifact_id, req.wins, req.losses, req.draws,
                                     pgn_path=pgn_path)
    return MatchResultResponse(**result)


# ---------------------------------------------------------------------------
# Artifacts (datasets / checkpoints / candidate & accepted NNUE networks)
# ---------------------------------------------------------------------------
@app.get('/artifacts', response_model=list[ArtifactResponse])
def list_artifacts(kind: str = None, accepted_only: bool = False):
    return db.list_artifacts(kind=kind, accepted_only=accepted_only)


@app.get('/artifacts/strongest-network', response_model=ArtifactResponse)
def get_strongest_network():
    art = db.get_strongest_network()
    if art is None:
        raise HTTPException(status_code=404,
                            detail='no network artifact has been accepted yet -- a fresh '
                                   'deployment must be seeded with a baseline network first, '
                                   'see docs/TRAINING.md')
    return art


@app.get('/artifacts/{artifact_id}', response_model=ArtifactResponse)
def get_artifact(artifact_id: str):
    art = db.get_artifact(artifact_id)
    if art is None:
        raise HTTPException(status_code=404, detail=f'unknown artifact_id {artifact_id!r}')
    return art


@app.get('/artifacts/{artifact_id}/download')
def download_artifact(artifact_id: str, worker=Depends(require_worker)):
    """A worker MUST verify the sha256 in GET /artifacts/{id} against the
    downloaded bytes before trusting/executing anything derived from this
    file -- see docs/WORKER.md. Requiring worker auth (rather than public,
    unauthenticated download) keeps artifact fetches attributable, same
    reasoning as every other worker-facing endpoint here."""
    art = db.get_artifact(artifact_id)
    if art is None:
        raise HTTPException(status_code=404, detail=f'unknown artifact_id {artifact_id!r}')
    if not os.path.isfile(art['file_path']):
        raise HTTPException(status_code=410,
                            detail=f'artifact {artifact_id!r} is registered but its file is '
                                   f'missing on the server (operator error -- report this)')
    return FileResponse(art['file_path'], filename=f'{artifact_id}.bin',
                        media_type='application/octet-stream',
                        headers={'X-Artifact-SHA256': art['sha256']})


@app.post('/artifacts/upload', response_model=ArtifactUploadResponse)
def upload_artifact(kind: str = Form(...), task_id: str = Form(None),
                    metadata_json: str = Form(None), file: UploadFile = File(...),
                    worker=Depends(require_worker)):
    """A worker uploads a produced artifact -- a candidate .nnue net (from a
    TRAIN_NETWORK task), a dataset (from DATA_GENERATION), or a training
    checkpoint. The server computes sha256/size itself from the received
    bytes (never trusts a client-supplied hash) and stores the file
    content-addressed under settings.artifacts_dir, so re-uploading
    identical bytes is a no-op cost-wise even if it creates a second row.
    Newly-uploaded 'network' artifacts start with accepted=0 -- promotion to
    the strongest network happens only after a passing ELO_MATCH, never on
    upload alone (see mark_artifact_accepted, called by the automated
    promotion step in platform/server/auto_pipeline.py once enough
    match_results support it)."""
    if kind not in ('dataset', 'checkpoint', 'network'):
        raise HTTPException(status_code=400, detail="kind must be one of 'dataset'/'checkpoint'/'network'")
    if not limiter.check(f'submit:{worker["id"]}', settings.rate_limit_submissions_per_minute, 60):
        raise HTTPException(status_code=429, detail='submission rate limit exceeded')

    hasher = hashlib.sha256()
    size = 0
    tmp_path = os.path.join(settings.artifacts_dir, f'.upload-{worker["id"]}-{os.getpid()}.tmp')
    with open(tmp_path, 'wb') as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_artifact_upload_bytes:
                out.close()
                os.remove(tmp_path)
                raise HTTPException(status_code=413,
                                    detail=f'artifact exceeds max upload size '
                                           f'({settings.max_artifact_upload_bytes} bytes)')
            hasher.update(chunk)
            out.write(chunk)
    sha256 = hasher.hexdigest()

    # Structural checkpoint verification for 'network' artifacts: reject
    # anything that isn't a real Morph .nnue file BEFORE it enters the
    # artifact store, rather than letting it become a candidate that wastes
    # an ELO_MATCH worker's compute (or worse, gets manually accepted).
    # See anti_cheat.validate_network_artifact's docstring for why this is
    # necessary on top of the worker-side _local_verify check in
    # platform/trainer/train_network.py (that one only protects an honest
    # worker's own training bugs, not a worker that uploads garbage on
    # purpose).
    if kind == 'network':
        reason = validate_network_artifact(tmp_path, size)
        if reason is not None:
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail=f'network artifact rejected: {reason}')

    final_dir = os.path.join(settings.artifacts_dir, kind)
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, sha256)
    if os.path.isfile(final_path):
        os.remove(tmp_path)   # identical content already stored -- dedup by content hash
    else:
        os.replace(tmp_path, final_path)

    metadata = json.loads(metadata_json) if metadata_json else None
    artifact_id = db.create_artifact(kind, final_path, sha256, size,
                                     created_by_task_id=task_id,
                                     created_by_worker_id=worker['id'],
                                     metadata=metadata, accepted=False)

    # TRAIN_NETWORK tasks have no separate "results" submission endpoint --
    # uploading the trained checkpoint/candidate network IS the task's
    # completion signal (see database.py's complete_task_for_worker, same
    # ownership guard as submit_match_result). Every other task type keeps
    # its own explicit completion path (submit_results' target-reached
    # check for SELF_PLAY/DATA_GENERATION, submit_match_result for
    # ELO_MATCH) -- this only fires for TRAIN_NETWORK so an unrelated
    # artifact upload never accidentally completes a different task.
    if task_id:
        task = db.get_task(task_id)
        if task and task.get('task_type') == 'TRAIN_NETWORK':
            db.complete_task_for_worker(task_id, worker['id'])

    return ArtifactUploadResponse(artifact_id=artifact_id, sha256=sha256, size_bytes=size)


# ---------------------------------------------------------------------------
# Community / dashboard (all public-read, no auth -- this is what a
# contributor's own browser or the leaderboard API reads; the community
# no longer has a bundled frontend -- see README.md)
# ---------------------------------------------------------------------------
@app.get('/stats')
def stats():
    return db.get_stats()


@app.get('/capacity')
def capacity():
    """Public, unauthenticated subset of GET /admin/system-load -- exists
    specifically for external monitoring tools that can't send a custom
    X-Admin-Token header (e.g. a sandboxed scheduled-check environment
    with no header support in its fetch tool) or reach the server except
    over a plain HTTP GET. Deliberately excludes load average/memory/disk
    (still admin-only, see /admin/system-load) -- those are more
    revealing of exact infra headroom than is worth exposing publicly;
    worker-capacity status and queue depth are the two signals worth
    that tradeoff, since they're also the two a would-be volunteer might
    reasonably want to see before trying to join (no point registering
    if the server is already full)."""
    connected = db.count_connected_workers()
    task_counts = db.get_task_counts_by_type()
    pending_tasks = sum(counts.get('pending', 0) for counts in task_counts.values())
    return {
        'connected_workers': connected,
        'max_connected_workers': settings.max_connected_workers,
        'at_worker_capacity': connected >= settings.max_connected_workers,
        'pending_tasks': pending_tasks,
    }


@app.get('/workers')
def workers():
    return db.list_workers()


@app.get('/leaderboard', response_model=LeaderboardResponse)
def leaderboard(limit: int = 50):
    entries = db.get_leaderboard(min(max(limit, 1), 200))
    return LeaderboardResponse(entries=entries, anonymous_positions=db.get_anonymous_positions())


@app.get('/dashboard/summary')
def dashboard_summary():
    return build_dashboard_summary(db)


@app.get('/dashboard/elo-series')
def dashboard_elo_series():
    return {'series': build_elo_series(db)}


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
def require_admin(x_admin_token: str = Header(default=None)):
    import hmac
    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(status_code=401, detail='missing or invalid admin token')
    return True


@app.post('/admin/tasks', response_model=CreateTasksResponse)
def create_tasks(req: CreateTasksRequest, _=Depends(require_admin)):
    chunk_size = req.chunk_size or settings.default_chunk_size
    task_ids, batch_label = db.create_tasks_bulk(
        req.total_positions, chunk_size, req.depth, req.randomplies, req.batch_label)
    return CreateTasksResponse(batch_label=batch_label, task_ids=task_ids,
                               total_positions=req.total_positions, chunk_size=chunk_size)


@app.post('/admin/tasks/typed', response_model=CreateTypedTaskResponse)
def create_typed_task(req: CreateTypedTaskRequest, _=Depends(require_admin)):
    """Creates one typed task (DATA_GENERATION / ELO_MATCH / TRAIN_NETWORK /
    SELF_PLAY) with an arbitrary JSON payload -- see database.py's
    create_typed_task for the expected payload shape per type. This is the
    endpoint the automated improvement-loop controller
    (platform/server/auto_pipeline.py) calls to advance the pipeline (queue
    a DATA_GENERATION batch, then a TRAIN_NETWORK once enough positions
    exist, then an ELO_MATCH once a candidate exists, etc.)."""
    task_id = db.create_typed_task(req.task_type, req.payload, batch_label=req.batch_label)
    return CreateTypedTaskResponse(task_id=task_id, task_type=req.task_type)


@app.post('/admin/artifacts', response_model=ArtifactResponse)
def register_artifact(req: RegisterArtifactRequest, _=Depends(require_admin)):
    """Seeds an artifact from a file that already exists on the server's
    own filesystem -- e.g. an operator placing an initial baseline .nnue
    net at CHESS_PLATFORM_ARTIFACTS_DIR/seed/baseline.nnue and registering
    it here before any ELO_MATCH task can have something to compare
    candidates against. sha256/size are computed server-side from the
    actual file, never trusted from the request."""
    if req.kind not in ('dataset', 'checkpoint', 'network'):
        raise HTTPException(status_code=400, detail="kind must be one of 'dataset'/'checkpoint'/'network'")
    if not os.path.isfile(req.file_path):
        raise HTTPException(status_code=404, detail=f'no such file: {req.file_path!r}')
    hasher = hashlib.sha256()
    size = 0
    with open(req.file_path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
    artifact_id = db.create_artifact(req.kind, req.file_path, hasher.hexdigest(), size,
                                     metadata=req.metadata, accepted=req.accepted)
    return db.get_artifact(artifact_id)


@app.post('/admin/pipeline/export-dataset', response_model=ExportDatasetResponse)
def export_dataset(req: ExportDatasetRequest, _=Depends(require_admin)):
    """Called by the automated improvement-loop controller
    (platform/server/auto_pipeline.py), not directly by workers. Exports
    every accepted position newer than the last auto-exported dataset (the
    watermark is the max position id recorded in the most recent
    kind='dataset', metadata.source='auto_pipeline' artifact -- no separate
    watermark table needed) into a new JSONL file in tools/nnue_pipeline's
    format, registers it as a 'dataset' artifact via the same
    create_artifact() path /admin/artifacts uses, and returns its id. If
    fewer than req.min_new_positions are available, does nothing and
    reports created=False so the controller can skip queueing a
    TRAIN_NETWORK task this cycle."""
    watermark = 0
    for art in db.list_artifacts(kind='dataset'):
        meta = art.get('metadata') or {}
        if meta.get('source') == 'auto_pipeline':
            watermark = max(watermark, int(meta.get('max_position_id', 0)))

    rows, max_id = db.export_positions_range(watermark, req.max_positions)
    total_in_corpus = db.count_all_positions()
    if len(rows) < req.min_new_positions:
        return ExportDatasetResponse(created=False,
                                      reason=f'only {len(rows)} new position(s), '
                                             f'need {req.min_new_positions}',
                                      count=len(rows), max_position_id=watermark,
                                      total_positions_in_corpus=total_in_corpus)

    out_dir = os.path.join(settings.artifacts_dir, 'auto_datasets')
    os.makedirs(out_dir, exist_ok=True)
    dataset_id_hint = f'auto_{max_id}'
    out_path = os.path.join(out_dir, f'{dataset_id_hint}.jsonl')
    with open(out_path, 'w') as f:
        for r in rows:
            # score_swing/best_move_changes are the optional search-instability
            # signal (see database.py's export_positions_range) -- included
            # whenever the source position recorded them (NULL otherwise, which
            # json.dumps writes as `null`); train.py's load_jsonl_datasets()
            # uses them to prioritize quality/difficulty on later truncation,
            # falling back to its original random-truncate behavior when absent.
            f.write(json.dumps({'fen': r['fen'], 'eval_cp': r['eval_cp'],
                                 'result': r['result'],
                                 'score_swing': r.get('score_swing'),
                                 'best_move_changes': r.get('best_move_changes')}) + '\n')

    hasher = hashlib.sha256()
    size = 0
    with open(out_path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)

    artifact_id = db.create_artifact(
        'dataset', out_path, hasher.hexdigest(), size,
        metadata={'source': 'auto_pipeline', 'max_position_id': max_id,
                  'count': len(rows), 'min_position_id_exclusive': watermark})

    return ExportDatasetResponse(created=True, artifact_id=artifact_id,
                                  count=len(rows), max_position_id=max_id,
                                  total_positions_in_corpus=total_in_corpus)


@app.post('/admin/pipeline/prune-positions', response_model=PrunePositionsResponse)
def prune_positions(req: PrunePositionsRequest, _=Depends(require_admin)):
    """Called by the automated improvement-loop controller
    (platform/server/auto_pipeline.py), opt-in via --prune-after-export --
    disk-space management for long-running deployments (raw positions
    accumulate forever otherwise; see database.py's delete_positions_up_to
    docstring for why this is safe once a position has been exported).

    Walks the auto_pipeline-sourced 'dataset' artifacts and keeps the
    keep_datasets most recent exports' worth of raw rows untouched, deleting
    everything covered by any OLDER export. Never deletes anything newer
    than an actual exported watermark -- if there isn't at least one export
    older than the kept set yet (i.e. fewer than keep_datasets + 1
    auto-exported datasets exist), does nothing and reports pruned=False,
    since there's no old-enough watermark to safely prune up to.

    Ranks exports by max_position_id (numeric, descending) rather than by
    list_artifacts()'s created_at ordering: created_at (database.py's
    now_iso()) only has one-second resolution, so two exports triggered in
    quick succession (a realistic scenario -- e.g. a catch-up cycle, or a
    short --interval-seconds) can tie and make created_at-based "newest"
    ambiguous. max_position_id is strictly increasing by construction
    (export_dataset always exports positions newer than the prior
    watermark), so it's an unambiguous, race-free ordering for this
    specific purpose."""
    watermarks = []
    for art in db.list_artifacts(kind='dataset'):
        meta = art.get('metadata') or {}
        if meta.get('source') == 'auto_pipeline':
            watermarks.append(int(meta.get('max_position_id', 0)))
    watermarks.sort(reverse=True)

    if len(watermarks) <= req.keep_datasets:
        return PrunePositionsResponse(
            pruned=False,
            reason=f'only {len(watermarks)} auto-exported dataset(s) so far, '
                   f'need at least {req.keep_datasets + 1} (keep_datasets={req.keep_datasets} '
                   f'plus one older one to prune up to) before pruning anything')

    # watermarks is newest-first (list_artifacts orders by created_at DESC);
    # index [keep_datasets] is the watermark of the newest export NOT in the
    # kept set -- everything at or before it has already been captured in
    # that export or an even older one, and is safe to drop. The
    # keep_datasets most recent exports (indices 0..keep_datasets-1) keep
    # their raw rows untouched.
    prune_up_to = watermarks[req.keep_datasets]
    if prune_up_to <= 0:
        return PrunePositionsResponse(pruned=False, reason='nothing to prune yet')

    deleted = db.delete_positions_up_to(prune_up_to)
    return PrunePositionsResponse(pruned=True, deleted_count=deleted,
                                   deleted_up_to_id=prune_up_to)


@app.get('/admin/system-load', response_model=SystemLoadResponse)
def system_load(_=Depends(require_admin)):
    """Point-in-time capacity snapshot for external monitoring -- built
    for small, single-instance deployments (see platform_config.py's
    max_connected_workers) where an operator has no existing
    infrastructure-monitoring stack and just wants a cheap thing to poll
    (e.g. from a cron job or scheduled task) and get alerted from. Not a
    time series -- callers that want trends should poll this repeatedly
    and keep their own history.

    connected_workers/at_worker_capacity reflect the exact same check
    POST /register enforces (see there), so 'at_worker_capacity: true'
    here means new registrations are currently being turned away.
    pending_tasks is a queue-depth signal: a large and growing number
    without more connected workers to drain it is itself worth flagging,
    independent of raw CPU/memory. See system_load.py for how
    load_average/memory/disk are computed (pure /proc parsing, no
    extra dependency)."""
    connected = db.count_connected_workers()
    task_counts = db.get_task_counts_by_type()
    pending_tasks = sum(counts.get('pending', 0) for counts in task_counts.values())
    snapshot = build_system_load_snapshot(
        connected, settings.max_connected_workers, pending_tasks, settings.artifacts_dir)
    return SystemLoadResponse(**snapshot)


@app.post('/admin/artifacts/{artifact_id}/accept')
def accept_artifact(artifact_id: str, _=Depends(require_admin)):
    """Manually promotes a candidate 'network' artifact to accepted=1 (the
    current strongest network). In the fully automated loop this normally
    happens once enough ELO_MATCH results support promotion (see
    get_match_results_for_artifact); this endpoint exists for an operator
    to intervene -- accept a candidate early, or re-accept a previous
    network -- without needing direct DB access."""
    if db.get_artifact(artifact_id) is None:
        raise HTTPException(status_code=404, detail=f'unknown artifact_id {artifact_id!r}')
    db.mark_artifact_accepted(artifact_id)
    return db.get_artifact(artifact_id)


@app.get('/admin/tasks')
def list_tasks(status: str = None, _=Depends(require_admin)):
    """Returns raw task rows from db.list_tasks(), except 'payload' is
    normalized from its raw TEXT-column JSON string into a real JSON
    object first -- db.get_task()/db.list_tasks() intentionally return the
    unparsed column (only assign_next_typed_task deserializes it, for its
    own response shape) since most direct DB callers don't need it parsed.
    But this is a public HTTP API: a client (e.g. auto_pipeline.py's
    maybe_queue_elo_matches, which reads payload.candidate_artifact_id off
    exactly this endpoint to dedup in-flight ELO_MATCH tasks) has no way to
    know 'payload' is doubly-JSON-encoded, and would get an AttributeError
    trying to call .get() on a string. Normalizing here, once, at the API
    boundary, is simpler and safer than requiring every HTTP client to know
    to json.loads() a field that looks like it's already a JSON object."""
    tasks = db.list_tasks(status)
    for t in tasks:
        raw = t.get('payload')
        if isinstance(raw, str):
            try:
                t['payload'] = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                t['payload'] = {}
        elif raw is None:
            t['payload'] = {}
    return tasks


@app.get('/admin/artifacts/{artifact_id}/match-results')
def artifact_match_results(artifact_id: str, _=Depends(require_admin)):
    """All ELO_MATCH results recorded against this artifact as a candidate
    -- used by the automated promotion step
    (platform/server/auto_pipeline.py) to decide whether enough evidence
    exists to accept it as the new strongest network."""
    if db.get_artifact(artifact_id) is None:
        raise HTTPException(status_code=404, detail=f'unknown artifact_id {artifact_id!r}')
    return db.get_match_results_for_artifact(artifact_id)


@app.post('/admin/workers/{worker_id}/disable')
def disable_worker(worker_id: str, _=Depends(require_admin)):
    db.disable_worker(worker_id)
    return {'worker_id': worker_id, 'disabled': True}


@app.post('/admin/workers/{worker_id}/enable')
def enable_worker(worker_id: str, _=Depends(require_admin)):
    conn = db._conn()
    try:
        conn.execute('UPDATE workers SET disabled = 0 WHERE id = ?', (worker_id,))
        conn.commit()
    finally:
        conn.close()
    return {'worker_id': worker_id, 'disabled': False}
