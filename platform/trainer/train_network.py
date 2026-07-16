#!/usr/bin/env python3
"""train_network.py - Worker-side executor for the TRAIN_NETWORK task type.
Only ever dispatched to a worker that reported trainer_capable=true (see
platform/server/database.py's assign_next_typed_task).

HISTORY / WHY THIS CHANGED: this executor used to shell out to the engine's
native CPU reference trainer (`chess_train train`, src/train/trainer.cpp)
and upload its output as a 'checkpoint' artifact, never 'network' -- because
the only bridge from that trainer's output to a loadable HalfKP .nnue file,
`chess_train distill`, is documented in the engine's own source
(src/apps/train_main.cpp) to emit a fixed material-baseline net regardless
of training input. That path is a structural dead end, not something this
executor can route around.

tools/nnue_pipeline/ (train.py + export.py + nnue_format.py) is a *separate,
already-correct* implementation of the engine's real production
architecture (HalfKP, 10,240 features, 16 king buckets, 512-wide
dual-perspective accumulator, clipped-ReLU, 8 output buckets -- see
src/nnue/nnue.h) that trains, quantizes, and writes the exact same binary
.nnue format src/nnue/nnue.cpp's write_net()/load() read. This was verified
for real as part of repairing this pipeline: a network trained end-to-end
through this path (1) loads successfully in the compiled engine, (2)
produces evaluations that exactly match a pure-Python reference
implementation on 8 fixed positions (proving correct feature indexing /
king-bucket mapping / output-bucket selection / quantization), and (3)
produces genuinely different evaluations than the old in-code
material-baseline stub net. This executor drives that real pipeline, and
uploads a real 'network' artifact.

GPU DISPATCH (Phase 3, multi-vendor): before choosing a trainer, this
executor calls the worker's own real, multi-vendor capability detection
(platform/worker/capabilities.py -- NVIDIA via torch.cuda/nvidia-smi, AMD
via torch.cuda(hip)/rocm-smi/rocminfo, Intel via
torch.xpu/xpu-smi/sycl-ls/OpenVINO; never self-reported) fresh, right
before training, and independently confirms a Rust toolchain and the
bullet_trainer crate are present. bullet_lib (the real trainer this wraps)
only implements CUDA and ROCm compute backends -- confirmed from its own
crates/gpu/Cargo.toml -- so a detected Intel-only (or other non-CUDA/ROCm)
GPU is reported honestly but never dispatched to the trainer; only a real
CUDA or ROCm device, with cargo and the crate present, triggers
tools/nnue_pipeline's --engine bullet --gpu-backend {cuda,rocm} (the real
GPU path, tools/nnue_training/bullet_trainer, built with the matching
cargo feature). Any failure in that attempt (missing toolchain, compile
error, runtime error) is caught and logged, and this executor falls back
to the CPU --engine reference path automatically -- a GPU-capable worker
still completes the task on a bad bullet run instead of failing it
outright. No fake/simulated GPU work, no self-reported-only capability
flags gating which engine actually runs: the choice is made from the same
honest detection function used for capability reporting, called again at
the moment of use.

Dataset format: the dataset_artifact_id payload field must point at a JSONL
dataset in tools/nnue_pipeline's format (one `{"fen": ..., "eval"/"eval_cp":
..., "result": ...}` object per line) -- the same format generate.py writes
and the same field names DATA_GENERATION positions already use server-side.

Before uploading, this executor performs its own local load+verify pass
(RefNet.load() plus a real UCI round-trip against the actual engine binary
being used) and refuses to upload if either fails -- a broken/corrupt
export should never reach the server, let alone another worker's ELO_MATCH.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'worker'))
from artifacts import fetch_artifact, ArtifactVerificationError  # noqa: E402
from capabilities import detect_capabilities  # noqa: E402

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'tools', 'nnue_pipeline')
sys.path.insert(0, os.path.abspath(PIPELINE_DIR))

BULLET_TRAINER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'tools', 'nnue_training', 'bullet_trainer')

# Stored-scale for a bullet-quantised export -- see
# tools/nnue_training/bullet_trainer/src/main.rs's QA/QB derivation:
# scale == QB when QA == EVAL_SCALE, which main.rs enforces.
BULLET_ENGINE_STORED_SCALE = 128


class TrainNetworkError(Exception):
    pass


def _run_pipeline_script(script_name, cmd_args, log, timeout, cwd=None):
    """Runs one tools/nnue_pipeline/*.py script as a subprocess of the
    same Python interpreter running this worker. Subprocess (not an
    in-process import) so a crash/hang in the trainer can't take the
    worker process down with it, matching how every other executor in
    this platform (data_generation.py, elo_match.py) shells out to
    independently-testable scripts."""
    script_path = os.path.join(PIPELINE_DIR, script_name)
    cmd = [sys.executable, script_path] + cmd_args
    log(f'[train_network] running: {" ".join(cmd)}')
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        raise TrainNetworkError(f'{script_name} timed out after {timeout}s') from e
    for line in proc.stdout.strip().splitlines():
        log(f'[train_network]   {line}')
    if proc.returncode != 0:
        raise TrainNetworkError(
            f'{script_name} exited {proc.returncode}:\n{proc.stderr[-2000:]}')
    return proc.stdout


def _gpu_training_available(log):
    """Honest, re-checked-at-use-time GPU training availability. Real,
    multi-vendor GPU detection (platform/worker/capabilities.py -- NVIDIA
    via torch.cuda/nvidia-smi, AMD via torch.cuda(hip)/rocm-smi/rocminfo,
    Intel via torch.xpu/xpu-smi/sycl-ls/OpenVINO; same function used for
    capability reporting, never a self-reported flag, called fresh here),
    a Rust toolchain on PATH, and the bullet_trainer crate present on disk.

    Detecting a GPU is not the same as being able to train on it: this
    platform's real GPU trainer (tools/nnue_training/bullet_trainer,
    wrapping jw1912/bullet) only has CUDA and ROCm compute backends --
    confirmed directly from bullet_lib's own crates/gpu/Cargo.toml
    ("Compile and execute tensor DAGs on CUDA/ROCm devices"). There is no
    SYCL/Level-Zero/oneAPI/DirectML backend, so a detected Intel-only GPU
    (or any other vendor/backend capabilities.py might detect in the
    future) cannot be dispatched to bullet -- this function returns None
    for that case, same as "no GPU at all", and logs exactly why, rather
    than either pretending to use it or silently miscategorizing it as
    "no GPU detected".

    Returns the backend string ('cuda' or 'rocm') to pass to `train.py
    --gpu-backend`, or None if CPU training should be used instead."""
    caps = detect_capabilities()
    backends = caps.get('gpu_backends') or []
    best_backend = caps.get('best_gpu_backend')

    if not backends:
        log('[train_network] GPU check: no GPU backend detected on this machine '
            '(checked NVIDIA/CUDA, AMD/ROCm, Intel -- see platform/worker/capabilities.py) '
            '-- using CPU reference trainer')
        return None

    if best_backend is None:
        # Something was detected (e.g. an Intel GPU) but nothing this
        # platform's trainer can actually use -- report it honestly
        # instead of silently falling through to the "no GPU" message.
        summary = ', '.join(f"{b['name']} ({b['vendor']}/{b['backend']})" for b in backends)
        log(f'[train_network] GPU check: detected {summary}, but bullet_lib (this platform\'s '
            f'GPU trainer) has no compute backend for {"/".join(sorted({b["backend"] for b in backends}))} '
            f'-- only cuda (NVIDIA) and rocm (AMD) are real training backends -- '
            f'falling back to CPU reference trainer')
        return None

    gpu_name = caps.get('gpu_name')
    if shutil.which('cargo') is None:
        log(f"[train_network] GPU check: trainable GPU detected ({gpu_name}) but no Rust "
            f"toolchain ('cargo' not on PATH) -- falling back to CPU reference trainer")
        return None
    if not os.path.isdir(BULLET_TRAINER_DIR):
        log(f"[train_network] GPU check: trainable GPU detected ({gpu_name}) and cargo present, "
            f"but {BULLET_TRAINER_DIR} is missing -- falling back to CPU reference trainer")
        return None

    log(f"[train_network] GPU check: real trainable GPU detected ({gpu_name}), cargo present, "
        f"bullet_trainer crate present -- attempting --engine bullet --gpu-backend {best_backend}")
    return best_backend


def _local_verify(engine_bin, net_path, log):
    """Load-and-verify sanity check performed by the worker itself, before
    ever uploading: (1) the pure-Python reference implementation can parse
    the file and compute an evaluation, (2) the real compiled engine binary
    (the same one this worker uses for everything else) can load it via UCI
    and produce a matching evaluation on the same position. A mismatch or
    load failure here means the export is broken -- refuse to upload rather
    than push a bad candidate onto the server and waste a future ELO_MATCH
    worker's time on it."""
    from nnue_format import RefNet  # noqa: E402
    from engine_paths import run_uci  # noqa: E402

    net = RefNet.load(net_path)  # raises ValueError on a malformed/incompatible file
    check_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    py_eval = net.evaluate_fen(check_fen)

    out = run_uci(engine_bin, [
        'setoption name Use NNUE value true',
        f'setoption name EvalFile value {net_path}',
        f'position fen {check_fen}',
        'eval',
        'quit',
    ], timeout=30)
    engine_eval = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('eval ') and line.endswith(' cp'):
            engine_eval = int(line.split()[1])
            break
    if engine_eval is None:
        raise TrainNetworkError(
            f'local verify failed: engine produced no eval line when loading {net_path}. '
            f'engine stdout:\n{out}')
    if engine_eval != py_eval:
        raise TrainNetworkError(
            f'local verify failed: python reference eval ({py_eval} cp) does not match '
            f'the compiled engine\'s eval ({engine_eval} cp) for the same network and '
            f'position -- refusing to upload a network with a feature-indexing, '
            f'king-bucket, or quantization-scale mismatch.')
    log(f'[train_network] local verify OK: python={py_eval} cp, engine={engine_eval} cp '
        f'(features={len(net.ft_weights)}, hl={len(net.ft_bias)}, buckets={len(net.out_bias)})')


def _train_gpu(dataset_path, workdir, epochs, backend, log):
    """Attempts the real GPU path (tools/nnue_pipeline/train.py --engine
    bullet --gpu-backend <backend> -> tools/nnue_training/bullet_trainer,
    built with the matching cargo feature). `backend` is 'cuda' or 'rocm'
    (whatever _gpu_training_available's capabilities.py-driven detection
    picked as the best real GPU on this machine -- never hardcoded).
    Returns the exported .nnue path on success. Raises TrainNetworkError on
    any failure -- the caller is expected to catch this and fall back to
    CPU, not propagate it as a task failure."""
    ckpt_dir = os.path.join(workdir, 'bullet_checkpoints')
    net_path = os.path.join(workdir, 'candidate_gpu.nnue')

    _run_pipeline_script('train.py', [
        '--engine', 'bullet', '--gpu-backend', backend,
        '--data', dataset_path, '--out', ckpt_dir,
        '--epochs', str(epochs),
    ], log, timeout=max(1800, epochs * 300))

    net_id = os.path.basename(ckpt_dir.rstrip('/\\')) or 'candidate'
    quantised_path = os.path.join(ckpt_dir, net_id, 'quantised.bin')
    if not os.path.isfile(quantised_path):
        raise TrainNetworkError(
            f'bullet training reported success but {quantised_path} was not produced')

    _run_pipeline_script('export.py', [
        '--bullet-quantised', quantised_path, '--out', net_path,
        '--scale', str(BULLET_ENGINE_STORED_SCALE),
    ], log, timeout=120)

    if not os.path.isfile(net_path) or os.path.getsize(net_path) == 0:
        raise TrainNetworkError('export.py (bullet path) reported success but produced no .nnue file')

    return net_path, {
        'engine': f'bullet (real GPU training via tools/nnue_training/bullet_trainer, '
                  f'backend={backend})',
        'gpu_used': True, 'gpu_backend': backend,
    }


def _train_cpu(dataset_path, workdir, epochs, max_samples, qa, qb, log, seed=1):
    """The proven CPU reference path (Phase 1). Returns the exported .nnue
    path and metadata dict. `seed` drives both train.py's weight
    initialization (new_params(seed)) and its dataset shuffle/truncation
    order (load_jsonl_datasets(..., seed=seed)) -- varying it across
    concurrently-queued TRAIN_NETWORK tasks against the same dataset (see
    auto_pipeline.py's maybe_queue_training) produces genuinely different
    candidate networks to compare, not just repeated runs of the same
    training with a different name."""
    ckpt_dir = os.path.join(workdir, 'checkpoints')
    net_path = os.path.join(workdir, 'candidate.nnue')

    _run_pipeline_script('train.py', [
        '--data', dataset_path, '--out', ckpt_dir,
        '--epochs', str(epochs), '--max-samples', str(max_samples),
        '--seed', str(seed),
    ], log, timeout=max(900, epochs * 180))

    ckpt_path = os.path.join(ckpt_dir, 'latest.npz')
    if not os.path.isfile(ckpt_path):
        raise TrainNetworkError('train.py reported success but produced no latest.npz checkpoint')

    train_mse = val_mse = trained_epoch = None
    metrics_path = os.path.join(ckpt_dir, 'metrics.jsonl')
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            lines = [ln for ln in f if ln.strip()]
        if lines:
            last = json.loads(lines[-1])
            train_mse = last.get('train_mse')
            val_mse = last.get('val_mse')
            trained_epoch = last.get('epoch')
            log(f'[train_network] real training loss: train_mse={train_mse} '
                f'val_mse={val_mse} (epoch {trained_epoch})')

    _run_pipeline_script('export.py', [
        '--checkpoint', ckpt_path, '--out', net_path,
        '--qa', str(qa), '--qb', str(qb),
    ], log, timeout=120)

    if not os.path.isfile(net_path) or os.path.getsize(net_path) == 0:
        raise TrainNetworkError('export.py reported success but produced no .nnue file')

    return net_path, {'engine': 'reference (CPU NumPy HalfKP trainer)', 'gpu_used': False,
                       'train_mse': train_mse, 'val_mse': val_mse, 'trained_epoch': trained_epoch}


def run_train_network(task, client, engine_bin, args, log=print):
    """Executes one TRAIN_NETWORK task end-to-end: download+verify the
    dataset artifact, train a real HalfKP network (GPU via bullet if a real
    GPU + toolchain are available, else CPU reference -- see module doc),
    quantize/export it to the engine's production .nnue format, verify it
    loads correctly (both in the pure-Python reference and in the real
    compiled engine), and upload it as a genuine 'network' artifact. Raises
    TrainNetworkError on a hard failure -- the task's lease expires and it
    gets reassigned, same recovery path as every other task type."""
    payload = task['payload']
    dataset_id = payload['dataset_artifact_id']
    epochs = int(payload.get('epochs', 6))
    qa = int(payload.get('qa', 256))
    qb = int(payload.get('qb', 256))
    max_samples = int(payload.get('max_samples', 200_000))
    # seed/experiment_id: optional multi-candidate-experiment fields (see
    # auto_pipeline.py's maybe_queue_training). Absent for a plain
    # single-candidate task (default seed=1, matching train.py's own
    # default, and experiment_id=None), so ordinary single-candidate
    # behavior is unchanged.
    seed = int(payload.get('seed', 1))
    experiment_id = payload.get('experiment_id')

    log(f"[train_network] task {task['task_id']}: dataset={dataset_id} "
        f"epochs={epochs} qa={qa} qb={qb} seed={seed}"
        + (f" experiment={experiment_id}" if experiment_id else ""))

    try:
        dataset_path = fetch_artifact(client, dataset_id, args.artifacts_cache_dir, log=log)
    except ArtifactVerificationError as e:
        raise TrainNetworkError(f'dataset artifact verification failed, aborting: {e}') from e

    workdir = tempfile.mkdtemp(prefix='train_network_')
    try:
        net_path = None
        engine_meta = None

        gpu_backend = _gpu_training_available(log)
        if gpu_backend is not None:
            try:
                net_path, engine_meta = _train_gpu(dataset_path, workdir, epochs, gpu_backend, log)
            except TrainNetworkError as e:
                log(f'[train_network] GPU training path failed ({e}) -- falling back to CPU '
                    f'reference trainer for this task instead of failing it outright')
                net_path = None

        if net_path is None:
            net_path, engine_meta = _train_cpu(dataset_path, workdir, epochs, max_samples,
                                                qa, qb, log, seed=seed)

        _local_verify(engine_bin, net_path, log)

        metadata = {
            'epochs': epochs, 'qa': qa, 'qb': qb, 'seed': seed,
            'source_dataset_artifact_id': dataset_id,
        }
        if experiment_id:
            metadata['experiment_id'] = experiment_id
        metadata.update(engine_meta)
        log(f'[train_network] uploading network ({os.path.getsize(net_path)} bytes, '
            f"engine={engine_meta['engine']})")
        resp = client.upload_artifact('network', net_path, task_id=task['task_id'],
                                       metadata=metadata)
        log(f"[train_network] task {task['task_id']}: network artifact "
            f"{resp['artifact_id']} uploaded, task marked complete")
        return resp
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
