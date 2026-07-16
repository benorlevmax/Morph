#!/usr/bin/env python3
"""models.py - Pydantic request/response schemas for the distributed server."""
from typing import List, Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    registration_secret: str
    hostname: str
    engine_version: str
    threads: int = Field(ge=1, le=256, default=1)


class RegisterResponse(BaseModel):
    worker_id: str
    worker_token: str
    task_lease_seconds: int


class TaskResponse(BaseModel):
    task_id: str
    target_positions: int
    depth: int
    randomplies: int


class PositionRecord(BaseModel):
    fen: str
    side_to_move: str
    eval_cp: int
    result: float
    depth: int
    nodes: int
    engine_version: str
    # Optional search-instability signal (see search.h's SearchResult::
    # scoreSwing/bestMoveChanges) -- absent for a worker/executor that
    # doesn't report it, in which case db.py stores NULL ("not recorded"),
    # never a fabricated 0.
    score_swing: Optional[int] = None
    best_move_changes: Optional[int] = None


class SubmitRequest(BaseModel):
    positions: List[PositionRecord]
    done: bool = False   # worker signals it does not intend to generate more for this task


class SubmitResponse(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    rejected_reasons: List[str]
    task_status: str
    task_accepted_total: int
    task_target: int


class CreateTasksRequest(BaseModel):
    total_positions: int = Field(gt=0)
    depth: int = Field(ge=1, le=32, default=6)
    randomplies: int = Field(ge=0, le=40, default=6)
    chunk_size: Optional[int] = Field(gt=0, default=None)
    batch_label: Optional[str] = None


class CreateTasksResponse(BaseModel):
    batch_label: str
    task_ids: List[str]
    total_positions: int
    chunk_size: int
