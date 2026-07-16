#!/usr/bin/env python3
"""config.py - Shared paths for the training_server/ backend.

Deliberately reuses (imports, never copies) the already-verified NNUE math/
IO from tools/nnue_pipeline/ (nnue_format.py, engine_paths.py, uci_match.py)
rather than re-implementing it a third time -- training_server is a backend
that runs in this repo, not something meant to be copied to a remote
machine the way distributed/worker/ is, so importing across directories is
fine here.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAINING_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

NNUE_PIPELINE_DIR = os.path.join(REPO_ROOT, 'tools', 'nnue_pipeline')
DISTRIBUTED_DIR = os.path.join(REPO_ROOT, 'distributed')

DEFAULT_DISTRIBUTED_DB = os.path.join(DISTRIBUTED_DIR, 'database', 'distributed.sqlite3')
DATASETS_DIR = os.path.join(TRAINING_SERVER_DIR, 'datasets')
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, 'experiments')
BULLET_TRAINER_DIR = os.path.join(REPO_ROOT, 'tools', 'nnue_training', 'bullet_trainer')


def add_nnue_pipeline_to_path():
    if NNUE_PIPELINE_DIR not in sys.path:
        sys.path.insert(0, NNUE_PIPELINE_DIR)


def add_distributed_server_to_path():
    server_dir = os.path.join(DISTRIBUTED_DIR, 'server')
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
