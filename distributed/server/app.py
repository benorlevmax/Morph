#!/usr/bin/env python3
"""app.py - FastAPI application for the distributed NNUE data-generation
server.

Endpoints:
    GET  /health                          liveness check, no auth
    POST /register                        worker registration (registration secret -> worker token)
    GET  /tasks/next                      worker: fetch/lease the next pending task
    POST /tasks/{task_id}/results         worker: upload generated positions
    GET  /stats                           public aggregate dataset statistics
    GET  /workers                         public worker list (no tokens)
    POST /admin/tasks                     admin: create a bulk generation job
    GET  /admin/tasks                     admin: list tasks (optionally by status)
    POST /admin/workers/{worker_id}/disable   admin: revoke a worker's token

This module owns no engine logic whatsoever -- it only stores/serves task and
position records that a worker (which does run the real engine) submits.
"""
import os
import sys

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auth
from config import settings
from db import Database
from models import (
    RegisterRequest, RegisterResponse, TaskResponse, SubmitRequest, SubmitResponse,
    CreateTasksRequest, CreateTasksResponse,
)

db = Database(settings.db_path)
auth.configure(db, settings)

app = FastAPI(title='Morph Distributed Data Generation', version='1.0')


@app.on_event('startup')
def _startup():
    settings.print_startup_banner()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/register', response_model=RegisterResponse)
def register(req: RegisterRequest):
    if req.registration_secret != settings.registration_secret:
        raise HTTPException(status_code=401, detail='invalid registration secret')
    worker_id, token = db.register_worker(req.hostname, req.engine_version, req.threads)
    return RegisterResponse(worker_id=worker_id, worker_token=token,
                             task_lease_seconds=settings.task_lease_seconds)


@app.get('/tasks/next')
def next_task(worker=Depends(auth.require_worker)):
    task = db.assign_next_task(worker['id'], settings.task_lease_seconds)
    if task is None:
        return JSONResponse(status_code=204, content=None)
    return TaskResponse(task_id=task['id'], target_positions=task['target_positions'],
                         depth=task['depth'], randomplies=task['randomplies'])


@app.post('/tasks/{task_id}/results', response_model=SubmitResponse)
def submit_results(task_id: str, req: SubmitRequest, worker=Depends(auth.require_worker)):
    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f'unknown task_id {task_id!r}')
    # Lenient on stale assignment (a lease that expired mid-generation and was
    # reassigned): still accept the work rather than discard it, but do not
    # accept results for a task assigned to a *different currently-active*
    # worker to avoid two workers' results silently overwriting one another's
    # progress accounting in confusing ways.
    if (task['status'] == 'assigned' and task['assigned_worker_id'] not in (None, worker['id'])):
        raise HTTPException(
            status_code=409,
            detail=f"task {task_id!r} is currently assigned to a different worker")

    records = [p.model_dump() for p in req.positions]
    result = db.submit_positions(task_id, worker['id'], records)
    updated = db.get_task(task_id)
    return SubmitResponse(
        accepted=result['accepted'], duplicates=result['duplicates'],
        rejected=result['rejected'], rejected_reasons=result['rejected_reasons'],
        task_status=updated['status'], task_accepted_total=updated['accepted_positions'],
        task_target=updated['target_positions'])


@app.get('/stats')
def stats():
    return db.get_stats()


@app.get('/workers')
def workers():
    return db.list_workers()


@app.post('/admin/tasks', response_model=CreateTasksResponse)
def create_tasks(req: CreateTasksRequest, _=Depends(auth.require_admin)):
    chunk_size = req.chunk_size or settings.default_chunk_size
    task_ids, batch_label = db.create_tasks_bulk(
        req.total_positions, chunk_size, req.depth, req.randomplies, req.batch_label)
    return CreateTasksResponse(batch_label=batch_label, task_ids=task_ids,
                               total_positions=req.total_positions, chunk_size=chunk_size)


@app.get('/admin/tasks')
def list_tasks(status: str = None, _=Depends(auth.require_admin)):
    return db.list_tasks(status)


@app.post('/admin/workers/{worker_id}/disable')
def disable_worker(worker_id: str, _=Depends(auth.require_admin)):
    db.disable_worker(worker_id)
    return {'worker_id': worker_id, 'disabled': True}
