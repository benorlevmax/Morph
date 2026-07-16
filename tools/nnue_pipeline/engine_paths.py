#!/usr/bin/env python3
"""engine_paths.py - Locate built engine binaries for the tools/nnue_pipeline/
scripts (chess_train, chess, chess_match), and small subprocess helpers shared
by generate.py / test.py.

Does not build anything and does not touch engine source -- it only looks for
binaries that must already exist (see CLAUDE.md's "Build" section: `cmake -S .
-B build ...` then `cmake --build build --config Release`).
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Candidate build output directories, in priority order. Windows/MSVC layout
# (per CLAUDE.md) puts binaries in build/bin/Release/*.exe; a Ninja/Unix
# config puts them directly in build/bin/*.
_CANDIDATE_DIRS = [
    os.path.join(REPO_ROOT, 'build', 'bin', 'Release'),
    os.path.join(REPO_ROOT, 'build', 'bin'),
]

_EXE_NAMES = {
    'chess_train': ['chess_train.exe', 'chess_train'],
    'chess': ['chess.exe', 'chess'],
    'chess_match': ['chess_match.exe', 'chess_match'],
}


def find_binary(name, bin_dir=None):
    """Find one of chess_train/chess/chess_match. `bin_dir`, if given,
    is searched first (and exclusively, if the binary is found there)."""
    assert name in _EXE_NAMES, name
    search_dirs = ([bin_dir] if bin_dir else []) + _CANDIDATE_DIRS
    tried = []
    for d in search_dirs:
        if not d:
            continue
        for fname in _EXE_NAMES[name]:
            p = os.path.join(d, fname)
            tried.append(p)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        f"could not find '{name}' binary. Tried:\n  " + "\n  ".join(tried) +
        f"\n\nBuild the engine first (see CLAUDE.md):\n"
        f"  cmake -S . -B build -G \"Visual Studio 17 2022\" -A x64\n"
        f"  cmake --build build --config Release\n"
        f"Or pass --bin-dir to point at your build's bin/ directory.")


def engine_version(chess_bin, timeout=10):
    """Query the running engine's UCI 'id name' string, e.g. 'Morph 0.5'."""
    try:
        proc = subprocess.run([chess_bin], input="uci\nquit\n",
                               capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return f'unknown (query failed: {e})'
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('id name '):
            return line[len('id name '):].strip()
    return 'unknown'


def run_uci(chess_bin, commands, timeout=30):
    """Run a `chess` UCI binary with a list of command strings, return stdout."""
    proc = subprocess.run([chess_bin], input="\n".join(commands) + "\n",
                           capture_output=True, text=True, timeout=timeout)
    return proc.stdout
