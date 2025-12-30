"""
Benchmark detector and liveness ONNX models across execution providers.

The script measures average latency and FPS for the YuNet detector and
anti-spoofing models on CPU, CUDA, and DirectML (when available), then writes
the results to benchmarks.md for quick deployment decisions.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnxruntime as ort

try:
    import cpuinfo  # type: ignore
except ImportError:
    cpuinfo = None

try:
    import GPUtil  # type: ignore
except ImportError:
    GPUtil = None

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "detector": REPO_ROOT / "models" / "detector_quantized.onnx",
    "liveness": REPO_ROOT / "models" / "best_model_quantized.onnx",
}


@dataclass
class BenchmarkResult:
    model: str
    provider_label: str
    provider: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark detector and liveness models across execution providers.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of timed runs per provider.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup runs to stabilize provider initialization.",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=DEFAULT_MODELS["detector"],
        help="Path to detector ONNX model.",
    )
    parser.add_argument(
        "--liveness",
        type=Path,
        default=DEFAULT_MODELS["liveness"],
        help="Path to liveness ONNX model.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        choices=["cpu", "cuda", "directml"],
        help="Subset of providers to benchmark (defaults to all available).",
    )
    parser.add_argument(
        "--detector-size",
        type=int,
        default=320,
        help="Input size (square) for detector benchmark tensor.",
    )
    parser.add_argument(
        "--liveness-size",
        type=int,
        default=128,
        help="Input size (square) for liveness benchmark tensor.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def get_provider_candidates() -> List[Tuple[str, str]]:
    return [
        ("CPU", "CPUExecutionProvider"),
        ("CUDA", "CUDAExecutionProvider"),
        ("DirectML", "DmlExecutionProvider"),
    ]


def resolve_providers(filter_args: Iterable[str] | None) -> List[Tuple[str, str]]:
    available = set(ort.get_available_providers())
    logging.info("Available providers reported by ONNX Runtime: %s", ", ".join(available))

    # Map CLI-friendly names to provider IDs
    filter_set = {p.lower() for p in filter_args} if filter_args else None
    providers: List[Tuple[str, str]] = []

    for label, provider in get_provider_candidates():
        if filter_set is not None:
            if label.lower() not in filter_set and provider.lower() not in filter_set:
                continue
        if provider in available:
            providers.append((label, provider))
        else:
            logging.info("Skipping %s: %s not available", label, provider)

    if not providers:
        raise RuntimeError("No execution providers available for benchmarking.")
    return providers


def create_session(model_path: Path, provider: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.intra_op_num_threads = 0  # Use default thread pool
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=[provider],
    )


def run_benchmark(
    model_label: str,
    model_path: Path,
    provider_label: str,
    provider: str,
    input_shape: Tuple[int, int, int, int],
    warmup_runs: int,
    timed_runs: int,
) -> BenchmarkResult:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = create_session(model_path, provider)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    logging.info(
        "Benchmarking %s on %s (%s) | input shape %s | warmup %d | runs %d",
        model_label,
        provider_label,
        provider,
        dummy_input.shape,
        warmup_runs,
        timed_runs,
    )

    # Warmup
    for _ in range(warmup_runs):
        session.run([], {input_name: dummy_input})

    latencies_ms: List[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        session.run([], {input_name: dummy_input})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(latencies_arr))
    p50_ms = float(np.percentile(latencies_arr, 50))
    p95_ms = float(np.percentile(latencies_arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    logging.info(
        "%s | %s mean %.2f ms | p50 %.2f ms | p95 %.2f ms | fps %.1f",
        model_label,
        provider_label,
        mean_ms,
        p50_ms,
        p95_ms,
        fps,
    )

    return BenchmarkResult(
        model=model_label,
        provider_label=provider_label,
        provider=provider,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        fps=fps,
    )


def collect_system_info() -> Dict[str, str]:
    info = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "onnxruntime": ort.__version__,
    }

    if cpuinfo:
        try:
            cpu = cpuinfo.get_cpu_info()
            info["cpu"] = cpu.get("brand_raw", "Unknown CPU")
        except Exception:
            info["cpu"] = "Unknown CPU"
    else:
        info["cpu"] = "cpuinfo not installed"

    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info["gpu"] = ", ".join(f"{gpu.name} ({gpu.memoryTotal}MB)" for gpu in gpus)
            else:
                info["gpu"] = "No GPU detected"
        except Exception:
            info["gpu"] = "GPU detection failed"
    else:
        info["gpu"] = "GPUtil not installed"

    return info


def render_markdown(
    system_info: Dict[str, str],
    results: List[BenchmarkResult],
    warmup: int,
    runs: int,
) -> str:
    provider_notes = ", ".join(sorted({r.provider for r in results}))
    lines = [
        "# ONNX Face Anti-Spoof Benchmark",
        "",
        f"- Generated: {system_info.get('timestamp', 'N/A')}",
        f"- ONNX Runtime: {system_info.get('onnxruntime', 'unknown')}",
        f"- CPU: {system_info.get('cpu', 'N/A')}",
        f"- GPU: {system_info.get('gpu', 'N/A')}",
        f"- Providers tested: {provider_notes}",
        f"- Warmup runs: {warmup}, Timed runs: {runs}",
        "",
        "| Model | Provider | Mean (ms) | P50 (ms) | P95 (ms) | FPS |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        lines.append(
            f"| {result.model} | {result.provider_label} | "
            f"{result.mean_ms:.2f} | {result.p50_ms:.2f} | {result.p95_ms:.2f} | "
            f"{result.fps:.1f} |"
        )

    lines.extend(
        [
            "",
            "Run `python benchmark.py --help` for options. "
            "Results are synthetic (random inputs) to make device comparisons repeatable.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    setup_logging()

    try:
        providers = resolve_providers(args.providers)
    except RuntimeError as exc:
        logging.error(str(exc))
        sys.exit(1)

    system_info = collect_system_info()
    results: List[BenchmarkResult] = []

    benchmarks: List[Tuple[str, Path, Tuple[int, int, int, int]]] = [
        ("Detector", args.detector, (1, 3, args.detector_size, args.detector_size)),
        ("Liveness", args.liveness, (1, 3, args.liveness_size, args.liveness_size)),
    ]

    for label, model_path, input_shape in benchmarks:
        for provider_label, provider in providers:
            try:
                result = run_benchmark(
                    model_label=label,
                    model_path=model_path,
                    provider_label=provider_label,
                    provider=provider,
                    input_shape=input_shape,
                    warmup_runs=args.warmup,
                    timed_runs=args.runs,
                )
                results.append(result)
            except Exception as exc:  # Keep benchmarking rolling even if one provider fails
                logging.error(
                    "Failed to benchmark %s on %s: %s", label, provider_label, exc
                )

    if not results:
        logging.error("No benchmark results produced. Check models and providers.")
        sys.exit(1)

    report = render_markdown(
        system_info=system_info,
        results=results,
        warmup=args.warmup,
        runs=args.runs,
    )

    output_path = REPO_ROOT / "benchmarks.md"
    output_path.write_text(report, encoding="utf-8")
    logging.info("Saved benchmark report to %s", output_path)


if __name__ == "__main__":
    main()
