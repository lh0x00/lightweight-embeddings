"""Device selection and ONNX session helpers."""

from __future__ import annotations

import logging
import multiprocessing
import os
from typing import Any

logger = logging.getLogger(__name__)


def select_device(prefer: str = "auto") -> str:
    """Resolve the runtime device.

    ``prefer`` may be ``"auto"``, ``"cpu"`` or ``"cuda"``. When ``"auto"`` we
    only return ``"cuda"`` if torch can actually initialize the runtime — this
    avoids hard crashes inside containers without GPU drivers.
    """
    if prefer == "cpu":
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        # Force initialization to surface "no driver" errors early.
        torch.cuda.init()
        torch.cuda.current_device()
        return "cuda"
    except Exception as exc:  # pragma: no cover - hardware dependent
        if prefer == "cuda":
            logger.warning("CUDA requested but unavailable: %s", exc)
        return "cpu"


def configure_threading(
    *,
    onnx_intra_threads: int | None,
    torch_num_threads: int | None,
) -> dict[str, int]:
    """Apply sensible defaults for thread parallelism.

    Returns the resolved values for observability/logging.
    """
    cpus = multiprocessing.cpu_count()
    intra = onnx_intra_threads or cpus
    torch_threads = torch_num_threads or max(1, cpus // 2)

    # Disable HuggingFace tokenizer fork warning + parallelism (we batch in
    # service threads anyway).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Helpful when running with gunicorn/uvicorn workers.
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    try:
        import torch

        torch.set_num_threads(torch_threads)
    except Exception:  # pragma: no cover
        pass

    return {"cpu_count": cpus, "onnx_intra_threads": intra, "torch_threads": torch_threads}


def build_ort_session_options(intra_threads: int, inter_threads: int = 1) -> Any:
    """Construct ``onnxruntime.SessionOptions`` tuned for low-latency serving.

    Returns an opaque object; callers pass it through ``model_kwargs``.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:  # pragma: no cover
        return None

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, intra_threads)
    so.inter_op_num_threads = max(1, inter_threads)
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    so.log_severity_level = 3  # quiet ORT info logs
    return so


def ort_providers(device: str) -> list[Any]:
    """Pick ONNX Runtime providers appropriate for the device."""
    if device == "cuda":
        return [
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]
