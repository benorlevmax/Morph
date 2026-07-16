#!/usr/bin/env python3
"""capabilities.py - Detects this machine's CPU/RAM/GPU resources and
reports them to the server via POST /workers/capabilities, so the server's
capability-aware task assignment (assign_next_typed_task in
platform/server/database.py) can route TRAIN_NETWORK tasks only to workers
that opted in as trainer-capable.

Design note on trainer_capable: this is a worker-owner OPT-IN flag
(--trainer-capable on the command line), not an automatic "has a GPU"
detection. Two reasons: (1) the reference CPU trainer
(src/train/trainer.cpp, driven by chess_train train) is a fully valid,
already-proven fallback when no supported GPU backend is available -- a
CPU-only machine can still usefully run TRAIN_NETWORK tasks, just slower;
(2) training is far more resource-intensive and long-running than
self-play/data-generation, so a contributor should explicitly consent to
their machine being tied up that way rather than have it triggered purely
by "a GPU happens to be present." GPU presence is still detected and
reported in full for the operator's own visibility and for
platform/trainer/train_network.py's backend selection -- it just isn't
what flips trainer_capable.

Design note on multi-vendor detection: this module does NOT assume NVIDIA/
CUDA is the only GPU vendor or that `nvidia-smi` is a sufficient detection
method. It separately probes for NVIDIA (CUDA), AMD (ROCm), and Intel
GPUs, through several independent signals per vendor (a Python ML
framework if one happens to be installed, PLUS a vendor CLI tool that
works even on a bare install with no Python ML stack). All of this is
purely for detection/reporting: the trainer this platform actually ships
(tools/nnue_training/bullet_trainer, wrapping jw1912/bullet) only has
CUDA and ROCm compute backends -- confirmed directly from bullet_lib's
own crates/gpu/Cargo.toml ("Compile and execute tensor DAGs on CUDA/ROCm
devices"), which has no SYCL/Level-Zero/oneAPI/DirectML backend. So an
Intel GPU (or any other vendor) is detected and reported honestly, but
platform/trainer/train_network.py can only ever dispatch real GPU training
to a detected NVIDIA or AMD backend; anything else falls back to the
proven CPU reference trainer with a clear log line saying why.
"""
import json
import os
import re
import shutil
import subprocess


def _cpu_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=True)
        if n:
            return n
    except Exception:
        pass
    return os.cpu_count() or 1


def _ram_mb():
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        return 0


def _run(cmd, timeout=5):
    """Runs `cmd`, returns (returncode, stdout) or (None, '') on any
    failure (binary missing, timeout, permission error, etc.) -- detection
    code must never raise just because a vendor's tool isn't installed,
    which is the overwhelmingly common case for any given vendor on any
    given machine."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout
    except Exception:
        return None, ''


# ---------------------------------------------------------------------------
# NVIDIA (CUDA) -- trainable via bullet_lib's `cuda` feature.
# ---------------------------------------------------------------------------
def _detect_nvidia():
    backends = []

    try:
        import torch  # noqa: local import -- optional dependency
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # torch.version.hip is set (non-None) when this is actually a
            # ROCm build of PyTorch reporting through the torch.cuda
            # namespace for API-compat reasons -- that's an AMD GPU, not
            # NVIDIA, and _detect_amd() below is what should claim it.
            if getattr(torch.version, 'hip', None) is None:
                for i in range(torch.cuda.device_count()):
                    backends.append({
                        'vendor': 'nvidia', 'backend': 'cuda',
                        'name': torch.cuda.get_device_name(i),
                        'trainable': True, 'detected_via': 'torch.cuda',
                    })
    except Exception:
        pass

    if not backends:
        nvidia_smi = shutil.which('nvidia-smi')
        if nvidia_smi:
            rc, out = _run([nvidia_smi, '--query-gpu=name', '--format=csv,noheader'])
            if rc == 0:
                for name in (line.strip() for line in out.splitlines()):
                    if name:
                        backends.append({
                            'vendor': 'nvidia', 'backend': 'cuda', 'name': name,
                            'trainable': True, 'detected_via': 'nvidia-smi',
                        })
    return backends


# ---------------------------------------------------------------------------
# AMD (ROCm) -- trainable via bullet_lib's `rocm` feature.
# ---------------------------------------------------------------------------
def _detect_amd():
    backends = []

    try:
        import torch  # noqa: local import -- optional dependency
        if (torch.cuda.is_available() and torch.cuda.device_count() > 0
                and getattr(torch.version, 'hip', None) is not None):
            for i in range(torch.cuda.device_count()):
                backends.append({
                    'vendor': 'amd', 'backend': 'rocm',
                    'name': torch.cuda.get_device_name(i),
                    'trainable': True, 'detected_via': 'torch.cuda(hip)',
                })
    except Exception:
        pass

    if not backends:
        rocm_smi = shutil.which('rocm-smi')
        if rocm_smi:
            # --showproductname is rocm-smi's stable machine-parseable name query.
            rc, out = _run([rocm_smi, '--showproductname'])
            if rc == 0:
                names = [m.strip() for m in re.findall(r'Card series:\s*(.+)', out)]
                if not names:
                    # Older/alternate rocm-smi output formats don't always match
                    # the regex above -- fall back to "present, name unknown"
                    # rather than dropping a real detection because of a
                    # cosmetic parsing miss.
                    names = ['AMD GPU (rocm-smi detected, name unparsed)']
                for name in names:
                    backends.append({
                        'vendor': 'amd', 'backend': 'rocm', 'name': name,
                        'trainable': True, 'detected_via': 'rocm-smi',
                    })

    if not backends:
        rocminfo = shutil.which('rocminfo')
        if rocminfo:
            rc, out = _run([rocminfo], timeout=10)
            if rc == 0 and 'Device Type:' in out and 'GPU' in out:
                names = re.findall(r'Marketing Name:\s*(.+)', out)
                for name in (names or ['AMD GPU (rocminfo detected, name unparsed)']):
                    backends.append({
                        'vendor': 'amd', 'backend': 'rocm', 'name': name.strip(),
                        'trainable': True, 'detected_via': 'rocminfo',
                    })
    return backends


# ---------------------------------------------------------------------------
# Intel (Arc / integrated) -- detected via several real signals (PyTorch
# XPU, Intel's xpu-smi, oneAPI's sycl-ls enumerating Level Zero devices,
# OpenVINO's device list), but NOT trainable by this platform's trainer --
# bullet_lib has no Intel backend (see module docstring). Reported so an
# operator/contributor can see their hardware was recognized, and so a
# future trainer backend has real detection to build on rather than
# needing this rewritten again.
# ---------------------------------------------------------------------------
def _detect_intel():
    backends = []

    try:
        import torch  # noqa: local import -- optional dependency
        xpu = getattr(torch, 'xpu', None)
        if xpu is not None and xpu.is_available() and xpu.device_count() > 0:
            for i in range(xpu.device_count()):
                try:
                    name = xpu.get_device_name(i)
                except Exception:
                    name = 'Intel GPU (torch.xpu)'
                backends.append({
                    'vendor': 'intel', 'backend': 'xpu', 'name': name,
                    'trainable': False, 'detected_via': 'torch.xpu',
                })
    except Exception:
        pass

    if not backends:
        xpu_smi = shutil.which('xpu-smi')
        if xpu_smi:
            rc, out = _run([xpu_smi, 'discovery', '--json'])
            if rc == 0:
                try:
                    data = json.loads(out)
                    devices = data.get('device_list', data if isinstance(data, list) else [])
                    for dev in devices:
                        name = dev.get('device_name') or dev.get('name') or 'Intel GPU'
                        backends.append({
                            'vendor': 'intel', 'backend': 'level_zero', 'name': name,
                            'trainable': False, 'detected_via': 'xpu-smi',
                        })
                except (ValueError, AttributeError):
                    # xpu-smi is present and returned 0 but not in the JSON
                    # shape we expected -- still real evidence a device
                    # exists, just report it without a parsed name rather
                    # than silently dropping the detection.
                    backends.append({
                        'vendor': 'intel', 'backend': 'level_zero',
                        'name': 'Intel GPU (xpu-smi detected, unparsed)',
                        'trainable': False, 'detected_via': 'xpu-smi',
                    })

    if not backends:
        # sycl-ls ships with the Intel oneAPI DPC++/C++ Compiler and lists
        # every SYCL device it can see, including Level Zero (native
        # Intel GPU backend) and OpenCL devices. A line looks like:
        #   [level_zero:gpu][level_zero:0] Intel(R) ... Arc(TM) A770 Graphics ...
        sycl_ls = shutil.which('sycl-ls')
        if sycl_ls:
            rc, out = _run([sycl_ls])
            if rc == 0:
                for line in out.splitlines():
                    if 'gpu' not in line.lower():
                        continue
                    if 'intel' not in line.lower():
                        continue  # sycl-ls can also list non-Intel devices via other backends
                    backend = 'level_zero' if 'level_zero' in line.lower() else 'opencl'
                    m = re.search(r'\]\s*(.+?)\s*(?:\d+\.\d+|\[|$)', line)
                    name = m.group(1).strip() if m else line.strip()
                    backends.append({
                        'vendor': 'intel', 'backend': backend, 'name': name,
                        'trainable': False, 'detected_via': 'sycl-ls',
                    })

    if not backends:
        # OpenVINO is inference-only (not a training backend), but
        # Core().available_devices is a real, reliable way to confirm an
        # Intel GPU is present and driver-visible even with none of the
        # above CLI tools installed, since openvino ships as a pip package.
        try:
            from openvino.runtime import Core  # noqa: optional dependency
            core = Core()
            for dev in core.available_devices:
                if not dev.startswith('GPU'):
                    continue
                try:
                    name = core.get_property(dev, 'FULL_DEVICE_NAME')
                except Exception:
                    name = f'Intel GPU ({dev})'
                backends.append({
                    'vendor': 'intel', 'backend': 'openvino', 'name': str(name),
                    'trainable': False, 'detected_via': 'openvino.Core',
                })
        except Exception:
            pass

    return backends


# ---------------------------------------------------------------------------
# Vulkan -- last-resort, vendor-agnostic presence signal (NOT used to
# classify vendor/trainability; only fills in a generic entry if literally
# nothing vendor-specific above found anything, so a GPU behind an unusual
# driver stack still shows up as "something is here" rather than
# vanishing entirely).
# ---------------------------------------------------------------------------
def _detect_vulkan_fallback():
    vulkaninfo = shutil.which('vulkaninfo')
    if not vulkaninfo:
        return []
    rc, out = _run([vulkaninfo, '--summary'], timeout=8)
    if rc != 0:
        return []
    names = re.findall(r'deviceName\s*=\s*(.+)', out)
    backends = []
    for name in names:
        name = name.strip()
        vendor = 'unknown'
        low = name.lower()
        if 'nvidia' in low:
            vendor = 'nvidia'
        elif 'amd' in low or 'radeon' in low:
            vendor = 'amd'
        elif 'intel' in low:
            vendor = 'intel'
        backends.append({
            'vendor': vendor, 'backend': 'vulkan', 'name': name,
            'trainable': False, 'detected_via': 'vulkaninfo',
        })
    return backends


_BACKEND_TRAIN_PRIORITY = {'cuda': 0, 'rocm': 1}  # lower = preferred when multiple GPUs present


def detect_gpu_backends():
    """Returns every GPU backend detected on this machine, across all
    vendors -- a list of {vendor, backend, name, trainable, detected_via}
    dicts, not a single yes/no. `trainable` reflects whether
    platform/trainer/train_network.py can actually dispatch real GPU
    training to it (only cuda/rocm, per bullet_lib's real feature set --
    see module docstring), not just whether hardware was found."""
    backends = []
    backends += _detect_nvidia()
    backends += _detect_amd()
    backends += _detect_intel()
    if not backends:
        backends += _detect_vulkan_fallback()

    # De-duplicate identical (vendor, backend, name) triples that multiple
    # detection methods might independently report (e.g. both torch.cuda
    # and nvidia-smi finding the same single GPU).
    seen = set()
    deduped = []
    for b in backends:
        key = (b['vendor'], b['backend'], b['name'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
    return deduped


def best_trainable_backend(backends):
    """Picks which single backend platform/trainer/train_network.py should
    actually try to use for real GPU training, or None if nothing
    trainable was detected. NVIDIA/cuda is preferred over AMD/rocm only as
    an arbitrary, documented tie-break (bullet's CUDA path has seen more
    real-world use across the wider engine-dev community than its ROCm
    path as of this writing) -- either is a fully real, supported backend,
    this is not a value judgement about hardware quality."""
    trainable = [b for b in backends if b['trainable']]
    if not trainable:
        return None
    trainable.sort(key=lambda b: _BACKEND_TRAIN_PRIORITY.get(b['backend'], 99))
    return trainable[0]


def detect_capabilities(trainer_capable=False, gpu_name_override=None):
    """Returns the dict POST /workers/capabilities expects (see
    platform/server/schemas.py's WorkerCapabilities): {cpu_cores, ram_mb,
    gpu_available, gpu_name, gpu_backends, best_gpu_backend,
    trainer_capable}.

    gpu_available/gpu_name are kept for backward compatibility with any
    existing dashboard/consumer that only reads those two fields --
    gpu_available is True if ANY vendor's GPU was detected (trainable or
    not), and gpu_name is the best TRAINABLE backend's name if one exists,
    else the first detected GPU's name (so an Intel-only machine still
    reports something meaningful there instead of a misleading blank).
    gpu_backends is the full multi-vendor detail; best_gpu_backend is
    what train_network.py actually acts on."""
    backends = detect_gpu_backends()
    best = best_trainable_backend(backends)

    gpu_available = len(backends) > 0
    if gpu_name_override:
        gpu_name = gpu_name_override
    elif best is not None:
        gpu_name = f"{best['name']} ({best['backend']})"
    elif backends:
        gpu_name = f"{backends[0]['name']} ({backends[0]['backend']}, not trainable by this platform)"
    else:
        gpu_name = None

    return {
        'cpu_cores': _cpu_cores(),
        'ram_mb': _ram_mb(),
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'gpu_backends': backends,
        'best_gpu_backend': best['backend'] if best else None,
        'trainer_capable': bool(trainer_capable),
    }
