#!/usr/bin/env python3
"""schemas.py - Pydantic schemas for the platform server's own endpoints
(accounts, worker registration-by-account, leaderboard). Task/submission/
worker-list schemas are reused directly from distributed/server/models.py
(imported, not redefined) since that data shape hasn't changed at all.
"""
import os
import sys
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', '..', 'distributed', 'server'))
from models import (  # noqa: F401 -- re-exported for app.py's convenience
    TaskResponse, PositionRecord, SubmitRequest, SubmitResponse,
    CreateTasksRequest, CreateTasksResponse,
)


class RegisterUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r'^[A-Za-z0-9_\-]+$')
    email: Optional[str] = None
    password: str = Field(min_length=8, max_length=200)


class UserResponse(BaseModel):
    user_id: str
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    session_token: str
    user_id: str
    username: str


class ApiKeyResponse(BaseModel):
    api_key: str
    warning: str = 'This key is shown once. Store it now; it cannot be retrieved again, only regenerated.'


class WorkerRegisterRequest(BaseModel):
    hostname: str
    engine_version: str
    threads: int = Field(ge=1, le=256, default=1)
    api_key: Optional[str] = None
    registration_secret: Optional[str] = None

    @field_validator('threads')
    @classmethod
    def _sane_threads(cls, v):
        return v


class WorkerRegisterResponse(BaseModel):
    worker_id: str
    worker_token: str
    task_lease_seconds: int
    linked_account: Optional[str] = None


class LeaderboardEntry(BaseModel):
    username: str
    positions_generated: int
    workers_count: int
    last_active_at: Optional[str]


class LeaderboardResponse(BaseModel):
    entries: List[LeaderboardEntry]
    anonymous_positions: int


# ---------------------------------------------------------------------------
# Typed tasks (SELF_PLAY / DATA_GENERATION / ELO_MATCH / TRAIN_NETWORK) --
# see database.py's create_typed_task/assign_next_typed_task. Distinct from
# distributed/server/models.py's TaskResponse (re-exported above), which
# stays the wire format for plain untyped SELF_PLAY polling so existing
# worker code needs zero changes there.
# ---------------------------------------------------------------------------
class GpuBackendInfo(BaseModel):
    """One detected GPU backend (platform/worker/capabilities.py's
    detect_gpu_backends()) -- a worker may report several of these, e.g. a
    laptop with both an integrated Intel GPU and a discrete NVIDIA GPU.
    `trainable` reflects whether platform/trainer/train_network.py can
    actually dispatch real GPU training to it via bullet_lib (only
    cuda/rocm are real training backends there -- see capabilities.py's
    module doc), not just whether hardware was found."""
    vendor: str            # 'nvidia' | 'amd' | 'intel' | 'unknown'
    backend: str           # 'cuda' | 'rocm' | 'xpu' | 'level_zero' | 'opencl' | 'openvino' | 'vulkan'
    name: str
    trainable: bool
    detected_via: str      # e.g. 'torch.cuda', 'nvidia-smi', 'rocm-smi', 'sycl-ls', 'openvino.Core'


class WorkerCapabilities(BaseModel):
    cpu_cores: int = Field(ge=1, le=1024)
    ram_mb: int = Field(ge=0)
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_backends: List[GpuBackendInfo] = Field(default_factory=list)
    best_gpu_backend: Optional[str] = None   # 'cuda' | 'rocm' | None -- what train_network.py will try
    trainer_capable: bool = False


class TypedTaskResponse(BaseModel):
    task_id: str
    task_type: str
    payload: dict


class CreateTypedTaskRequest(BaseModel):
    task_type: str
    payload: dict
    batch_label: Optional[str] = None

    @field_validator('task_type')
    @classmethod
    def _known_type(cls, v):
        allowed = ('SELF_PLAY', 'DATA_GENERATION', 'ELO_MATCH', 'TRAIN_NETWORK')
        if v not in allowed:
            raise ValueError(f'task_type must be one of {allowed}')
        return v


class CreateTypedTaskResponse(BaseModel):
    task_id: str
    task_type: str


class ArtifactResponse(BaseModel):
    id: str
    kind: str
    sha256: str
    size_bytes: int
    accepted: bool
    created_by_task_id: Optional[str] = None
    created_by_worker_id: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: str


class RegisterArtifactRequest(BaseModel):
    """Admin-only: seed an artifact whose file already exists on the
    server's filesystem (e.g. the initial baseline NNUE net an operator
    ships with the deployment) without going through the worker upload
    endpoint. sha256 is recomputed server-side and must match."""
    kind: str
    file_path: str
    accepted: bool = False
    metadata: Optional[dict] = None


class MatchResultRequest(BaseModel):
    candidate_artifact_id: str
    baseline_artifact_id: str
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    draws: int = Field(ge=0)
    pgn_base64: Optional[str] = None   # optional: full game record for audit/replay


class MatchResultResponse(BaseModel):
    games: int
    wins: int
    losses: int
    draws: int


class ArtifactUploadResponse(BaseModel):
    artifact_id: str
    sha256: str
    size_bytes: int


class ExportDatasetRequest(BaseModel):
    """Admin-only (called by platform/server/auto_pipeline.py, the automated
    improvement-loop controller): export accepted positions newer than the
    last auto-exported dataset into a new JSONL 'dataset' artifact, in
    tools/nnue_pipeline's format, ready for a TRAIN_NETWORK task."""
    min_new_positions: int = Field(ge=1, default=2000)
    max_positions: int = Field(ge=1, default=200_000)


class ExportDatasetResponse(BaseModel):
    created: bool
    reason: Optional[str] = None
    artifact_id: Optional[str] = None
    count: int = 0
    max_position_id: int = 0


class PrunePositionsRequest(BaseModel):
    """Admin-only (called by platform/server/auto_pipeline.py, opt-in via
    --prune-after-export): delete raw `positions` rows that have already
    been captured in an exported 'dataset' artifact, to keep the
    positions table from growing without bound. Never deletes a position
    that hasn't been exported into a dataset artifact yet -- see
    database.py's delete_positions_up_to() docstring.

    keep_datasets controls the safety margin: the server keeps the
    keep_datasets most recent auto_pipeline dataset exports' worth of raw
    positions rows completely untouched, and only deletes positions
    covered by an OLDER export than that (never anything from the kept
    set, and never anything not yet exported at all). keep_datasets=1
    (the minimum) keeps just the single most recent export's rows;
    keep_datasets=3 (the default) keeps the last 3 exports' worth of raw
    rows around as a buffer, e.g. in case a dataset file needs to be
    regenerated or cross-checked against the live table. Requires at
    least keep_datasets + 1 exports to exist before anything is ever
    pruned (there must be an older export to prune up to)."""
    keep_datasets: int = Field(ge=1, default=3)


class PrunePositionsResponse(BaseModel):
    pruned: bool
    reason: Optional[str] = None
    deleted_count: int = 0
    deleted_up_to_id: int = 0
