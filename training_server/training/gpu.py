#!/usr/bin/env python3
"""gpu.py - GPU/toolchain detection and engine recommendation.

Two independent things need to be true to use the real Bullet trainer
(GPU-accelerated): a Rust toolchain (`cargo`) to build/run it, and an NVIDIA
GPU (`nvidia-smi`) for it to actually accelerate on. Bullet can technically
run on CPU too, but is not a sensible CPU trainer (it's built for CUDA
throughput) -- so "GPU acceleration" here specifically means the Bullet path,
and "CPU fallback" means the bundled NumPy reference trainer
(tools/nnue_pipeline/train.py's --engine reference), which needs neither.
"""
import shutil
import subprocess


def has_cargo():
    return shutil.which('cargo') is not None


def gpu_info():
    """Returns a dict: {'available': bool, 'names': [...], 'raw': str|None}.
    Never raises -- absence of nvidia-smi just means no GPU."""
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        return {'available': False, 'names': [], 'raw': None}
    try:
        out = subprocess.run(
            [nvidia_smi, '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return {'available': False, 'names': [], 'raw': out.stderr}
        names = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        return {'available': bool(names), 'names': names, 'raw': out.stdout}
    except Exception as e:
        return {'available': False, 'names': [], 'raw': str(e)}


def recommend_engine(requested='auto'):
    """Returns (engine, reason) where engine is 'bullet' or 'reference'."""
    if requested in ('bullet', 'reference'):
        return requested, 'explicitly requested'

    gpu = gpu_info()
    if gpu['available'] and has_cargo():
        return 'bullet', f"GPU detected ({', '.join(gpu['names'])}) and cargo is available"
    if gpu['available'] and not has_cargo():
        return 'reference', (f"GPU detected ({', '.join(gpu['names'])}) but no Rust toolchain "
                             f"('cargo' not found) -- install rustup to use it")
    return 'reference', 'no GPU detected (nvidia-smi unavailable or reports no devices)'


def print_report():
    gpu = gpu_info()
    cargo = has_cargo()
    engine, reason = recommend_engine()
    print('=== GPU / training-engine report ===')
    print(f'  GPU available: {gpu["available"]}  {gpu["names"]}')
    print(f'  cargo (Rust) available: {cargo}')
    print(f'  recommended engine: {engine}  ({reason})')
    return {'gpu': gpu, 'cargo': cargo, 'recommended_engine': engine, 'reason': reason}


if __name__ == '__main__':
    print_report()
