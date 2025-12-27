"""
Lesson 2: Benchmarking Harness for LLM Inference

Measures TTFT, ITL, E2E latency with proper warmup and percentile reporting.

Run with: python benchmarking_harness.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from paths import PLOTS_DIR
from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample
from utils.model_utils import setup_model_and_tokenizer
from utils.plotting_utils import plot_benchmark_results

RESULTS_DIR = Path(PLOTS_DIR) / "lesson2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def measure_request(model, tokenizer, prompt, max_new_tokens, device):
    """Measure TTFT, ITL, and E2E for a single request."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    # Prefill
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        kv_cache = out.past_key_values

    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    if device == "cuda":
        torch.cuda.synchronize()
    first_token_time = time.perf_counter()
    ttft = (first_token_time - start) * 1000

    # Decode (ignore EOS to ensure consistent token count)
    token_times = [first_token_time]
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv_cache, use_cache=True)
            kv_cache = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        if device == "cuda":
            torch.cuda.synchronize()
        token_times.append(time.perf_counter())

    e2e = (token_times[-1] - start) * 1000
    itl = [
        (token_times[i] - token_times[i - 1]) * 1000 for i in range(1, len(token_times))
    ]

    return {
        "prompt_tokens": prompt_len,
        "output_tokens": len(token_times),
        "ttft_ms": ttft,
        "e2e_ms": e2e,
        "itl_ms": itl,
    }


def create_prompts(dataset, tokenizer, task_instruction, target_len, count):
    """Create prompts of exactly target length by repeating content."""
    prompts = []
    for i in range(count):
        sample = dataset[i % len(dataset)]
        messages = build_messages_for_sample(
            sample, task_instruction, include_assistant=False
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt)

        # Adjust to target length by repeating or truncating
        if len(tokens) < target_len:
            # Repeat tokens (excluding special tokens) to reach target length
            content_tokens = tokens[1:-1] if len(tokens) > 2 else tokens
            while len(tokens) < target_len:
                tokens = tokens[:-1] + content_tokens + tokens[-1:]
            tokens = tokens[:target_len]
        else:
            tokens = tokens[:target_len]

        prompts.append(tokenizer.decode(tokens))
    return prompts


def percentiles(values):
    """Return p50, p95, p99 for a list of values."""
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0}
    return {
        "p50": np.percentile(values, 50),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
    }


def run_benchmark(model, tokenizer, prompts, max_new_tokens, device, warmup=3):
    """Run benchmark with warmup, return aggregated metrics."""
    results = []

    for i, prompt in enumerate(prompts):
        metrics = measure_request(model, tokenizer, prompt, max_new_tokens, device)

        if i < warmup:
            print(f"  [WARMUP] {i+1}/{warmup}")
        else:
            results.append(metrics)
            print(
                f"  [MEASURE] {i+1-warmup}/{len(prompts)-warmup}: "
                f"TTFT={metrics['ttft_ms']:.1f}ms, E2E={metrics['e2e_ms']:.1f}ms"
            )

        if device == "cuda":
            torch.cuda.empty_cache()

    # Aggregate
    ttft_vals = [r["ttft_ms"] for r in results]
    e2e_vals = [r["e2e_ms"] for r in results]
    all_itl = [itl for r in results for itl in r["itl_ms"]]
    total_tokens = sum(r["output_tokens"] for r in results)
    total_time = sum(r["e2e_ms"] for r in results) / 1000

    return {
        "ttft": percentiles(ttft_vals),
        "e2e": percentiles(e2e_vals),
        "itl": percentiles(all_itl),
        "tsp": total_tokens / total_time if total_time > 0 else 0,
        "rps": len(results) / total_time if total_time > 0 else 0,
    }


def main():
    # Configuration
    prompt_lengths = [100, 500, 2000]
    output_lengths = [50, 200]
    num_requests = 50
    warmup = 5

    # Load model
    print("Loading model...")
    cfg = load_config()
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=False, use_lora=False, device_map="auto"
    )
    model.eval()
    device = str(next(model.parameters()).device)

    # Load dataset
    print("Loading dataset...")
    train_data, _, _ = load_and_prepare_dataset(cfg)
    task_instruction = cfg.get(
        "task_instruction", "Summarize the following conversation."
    )

    # Run benchmarks
    all_results = []
    for prompt_len in prompt_lengths:
        prompts = create_prompts(
            train_data, tokenizer, task_instruction, prompt_len, num_requests + warmup
        )

        for output_len in output_lengths:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt_len} tokens, Output: {output_len} tokens")
            print("=" * 60)

            result = run_benchmark(
                model, tokenizer, prompts, output_len, device, warmup
            )
            result["config"] = {
                "prompt_tokens": prompt_len,
                "output_tokens": output_len,
            }
            all_results.append(result)

            print(
                f"\n  TTFT: p50={result['ttft']['p50']:.1f}ms, p95={result['ttft']['p95']:.1f}ms"
            )
            print(
                f"  E2E:  p50={result['e2e']['p50']:.1f}ms, p95={result['e2e']['p95']:.1f}ms"
            )
            print(f"  TSP:  {result['tsp']:.1f} tokens/sec")

    # Save results
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate visualization (saves ttft.png, e2e_latency.png, throughput.png)
    plot_benchmark_results(all_results, save_path=RESULTS_DIR / "plots.png")


if __name__ == "__main__":
    main()
