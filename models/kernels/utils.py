import functools
import importlib
import operator

from typing import Callable

import torch
import triton
import triton.language as tl

from packaging.version import Version

torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

def is_hip() -> bool:
    return torch.version.hip is not None

def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper

def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))

def is_npu_available() -> bool:
    """Detect Ascend NPU availability."""
    try:
        from transformers.utils import is_torch_npu_available

        return is_torch_npu_available()
    except Exception:
        return False
    
def set_large_grf_mode(kernel_args: dict):
    """Set large GRF mode for XPU devices."""
    # On XPU triton installed along with pytorch-xpu will be called `pytorch-triton-xpu`,
    # triton XPU installed from source will be called `triton`.
    if compare_version("pytorch-triton-xpu", operator.ge, "3.6.0") or compare_version("triton", operator.ge, "3.6.0"):
        kernel_args["grf_mode"] = "256"
    else:
        # API was changed in https://github.com/intel/intel-xpu-backend-for-triton/pull/5430
        kernel_args["grf_mode"] = "large"

def get_npu_core_count(default: int = 20) -> int:
    """Return NPU vector core count.
    Fallback to `default` if Triton runtime or NPU device is unavailable.
    """
    try:
        utils = triton.runtime.driver.active.utils
        props = utils.get_device_properties(0)
        return int(props.get("num_vectorcore", default))
    except Exception:
        return default