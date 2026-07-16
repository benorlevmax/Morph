#!/usr/bin/env python3
"""config.py - Shared paths for the automation/ self-improvement loop.

Sits one layer above training_server/: training_server/pipeline.py already
does import -> batch -> train -> export -> evaluate for a single run. This
package repeatedly drives that single-run pipeline, decides whether to keep
each result, and maintains the models/ and results/ directories the rest of
the project (and a human) can look at without digging into experiments/.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AUTOMATION_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_SERVER_DIR = os.path.join(REPO_ROOT, 'training_server')
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, 'experiments')

MODELS_DIR = os.path.join(REPO_ROOT, 'models')
MODELS_CURRENT_DIR = os.path.join(MODELS_DIR, 'current')
MODELS_CANDIDATES_DIR = os.path.join(MODELS_DIR, 'candidates')
MODELS_REJECTED_DIR = os.path.join(MODELS_DIR, 'rejected')

RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
RESULTS_BENCHMARKS_DIR = os.path.join(RESULTS_DIR, 'benchmarks')
RESULTS_ELO_TESTS_DIR = os.path.join(RESULTS_DIR, 'elo_tests')

STATE_FILE = os.path.join(AUTOMATION_DIR, 'state.json')
LOGS_DIR = os.path.join(AUTOMATION_DIR, 'logs')
CONTROLLER_LOG_FILE = os.path.join(LOGS_DIR, 'controller.log')

DEFAULT_DISTRIBUTED_DB = os.path.join(REPO_ROOT, 'distributed', 'database', 'distributed.sqlite3')
NNUE_PIPELINE_DIR = os.path.join(REPO_ROOT, 'tools', 'nnue_pipeline')

# Where --auto-generate appends locally self-played positions between
# cycles. A single accumulating file (not one-per-run) so "how many new
# positions exist" is just a line count.
GENERATED_DIR = os.path.join(AUTOMATION_DIR, 'generated')
AUTO_GENERATED_JSONL = os.path.join(GENERATED_DIR, 'auto_positions.jsonl')


def ensure_dirs():
    for d in (MODELS_CURRENT_DIR, MODELS_CANDIDATES_DIR, MODELS_REJECTED_DIR,
              RESULTS_BENCHMARKS_DIR, RESULTS_ELO_TESTS_DIR, LOGS_DIR, GENERATED_DIR):
        os.makedirs(d, exist_ok=True)
