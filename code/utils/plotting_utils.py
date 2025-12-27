"""
plotting_utils.py
Utilities for creating clean visualizations of LLM inference experiments.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_prefill_scaling(
    results: dict[int, float],
    save_path: Path | str = None,
    show: bool = False,
):
    """
    Create a clean plot showing prefill time vs prompt length.
    Demonstrates O(n^2) scaling of attention computation.
    """
    lengths = sorted(results.keys())
    times = [results[l] for l in lengths]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        lengths,
        times,
        "o-",
        color="#00d4aa",
        linewidth=2.5,
        markersize=10,
        label="Measured prefill time",
    )

    # O(n^2) reference line
    lengths_arr = np.array(lengths)
    a = np.mean([t / (l**2) for l, t in zip(lengths, times)])
    ax.plot(
        lengths,
        a * lengths_arr**2,
        "--",
        color="#ff6b6b",
        linewidth=2,
        alpha=0.7,
        label="O(n^2) reference",
    )

    ax.set_xlabel("Prompt Length (tokens)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Prefill Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Prefill Time vs Prompt Length\n(Compute-Bound: O(n^2) Attention)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = Path("prefill_scaling.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_decode_scaling(
    results: dict[int, float],
    save_path: Path | str = None,
    show: bool = False,
):
    """
    Create a clean plot showing decode time vs output length.
    Demonstrates linear scaling (memory-bandwidth bound).
    """
    lengths = sorted(results.keys())
    times = [results[l] for l in lengths]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        lengths,
        times,
        "o-",
        color="#00d4aa",
        linewidth=2.5,
        markersize=10,
        label="Measured decode time",
    )

    # Linear reference
    lengths_arr = np.array(lengths)
    slope = np.mean([t / l for l, t in zip(lengths, times)])
    ax.plot(
        lengths,
        slope * lengths_arr,
        "--",
        color="#ffd93d",
        linewidth=2,
        alpha=0.7,
        label="O(n) reference (linear)",
    )

    ax.set_xlabel("Output Length (tokens)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Decode Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Decode Time vs Output Length\n(Memory-Bandwidth Bound: Linear Scaling)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = Path("decode_scaling.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_context_effect(
    results: dict[int, float],
    decode_tokens: int,
    save_path: Path | str = None,
    show: bool = False,
):
    """
    Create a plot showing minimal effect of context length on decode speed.
    Demonstrates KV cache efficiency.
    """
    lengths = sorted(results.keys())
    times = [results[l] for l in lengths]
    itls = [t / decode_tokens for t in times]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        range(len(lengths)), itls, color="#00d4aa", edgecolor="#ffffff", linewidth=1.5
    )
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(l) for l in lengths])

    ax.set_xlabel("Context Length (tokens)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Inter-Token Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Context Length Effect on Decode Speed\n(KV Cache Makes Context Nearly Free!)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, axis="y", linestyle="-")

    avg_itl = np.mean(itls)
    ax.axhline(
        y=avg_itl,
        color="#ff6b6b",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Average: {avg_itl:.1f} ms",
    )
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = Path("context_effect.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_benchmark_results(
    results: list[dict],
    save_path: Path | str = None,
    show: bool = False,
):
    """
    Create visualization from benchmark results (TTFT, E2E, TSP).
    Saves each chart as a separate file.
    """
    configs = [
        f"{r['config']['prompt_tokens']}p/{r['config']['output_tokens']}o"
        for r in results
    ]
    ttft_p50 = [r["ttft"]["p50"] for r in results]
    ttft_p95 = [r["ttft"]["p95"] for r in results]
    e2e_p50 = [r["e2e"]["p50"] for r in results]
    e2e_p95 = [r["e2e"]["p95"] for r in results]
    tsp = [r["tsp"] for r in results]

    if save_path is None:
        save_dir = Path(".")
    else:
        save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(configs))
    width = 0.35

    # TTFT chart
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, ttft_p50, width, label="p50", color="#2ecc71")
    ax.bar(x + width / 2, ttft_p95, width, label="p95", color="#e74c3c")
    ax.set_xlabel("Config (prompt/output)", fontsize=11)
    ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_title("Time to First Token", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        save_dir / "ttft.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e"
    )
    print(f"Plot saved to: {save_dir / 'ttft.png'}")
    if show:
        plt.show()
    plt.close()

    # E2E chart
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, e2e_p50, width, label="p50", color="#3498db")
    ax.bar(x + width / 2, e2e_p95, width, label="p95", color="#9b59b6")
    ax.set_xlabel("Config (prompt/output)", fontsize=11)
    ax.set_ylabel("E2E Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Latency", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        save_dir / "e2e_latency.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e"
    )
    print(f"Plot saved to: {save_dir / 'e2e_latency.png'}")
    if show:
        plt.show()
    plt.close()

    # TSP chart
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, tsp, color="#f39c12")
    ax.set_xlabel("Config (prompt/output)", fontsize=11)
    ax.set_ylabel("Tokens/sec", fontsize=11)
    ax.set_title("Throughput (TSP)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        save_dir / "throughput.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e"
    )
    print(f"Plot saved to: {save_dir / 'throughput.png'}")
    if show:
        plt.show()
    plt.close()


def plot_kv_cache_speedup(results: dict, save_dir: Path | str):
    """Plot decode time with vs without KV cache."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))

    contexts = [r["context"] for r in results["with_cache"]]
    times_with = [r["time_ms"] for r in results["with_cache"]]
    times_without = [r["time_ms"] for r in results["without_cache"]]

    x = np.arange(len(contexts))
    width = 0.35

    ax.bar(x - width / 2, times_with, width, label="With KV Cache", color="#2ecc71")
    ax.bar(
        x + width / 2, times_without, width, label="Without KV Cache", color="#e74c3c"
    )

    ax.set_xlabel("Context Length (tokens)", fontsize=11)
    ax.set_ylabel("Decode Time (ms)", fontsize=11)
    ax.set_title("KV Cache Impact on Decode Speed", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        save_dir / "cache_speedup.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#1a1a2e",
    )
    print(f"Plot saved to: {save_dir / 'cache_speedup.png'}")
    plt.close()


def plot_gpu_memory_comparison(results: dict, save_dir: Path | str):
    """Plot KV cache memory overhead as line chart."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not results.get("gpu_memory"):
        return

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))

    contexts = [r["context"] for r in results["gpu_memory"]]
    overhead = [r["kv_cache_overhead_mb"] for r in results["gpu_memory"]]

    ax.plot(
        contexts,
        overhead,
        "o-",
        color="#2ecc71",
        linewidth=2,
        markersize=8,
        label="KV Cache Overhead",
    )

    ax.set_xlabel("Context Length (tokens)", fontsize=11)
    ax.set_ylabel("Memory Overhead (MB)", fontsize=11)
    ax.set_title(
        "KV Cache Memory Overhead vs Context Length", fontsize=13, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "gpu_memory.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e"
    )
    print(f"Plot saved to: {save_dir / 'gpu_memory.png'}")
    plt.close()
