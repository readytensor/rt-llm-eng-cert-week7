"""
Lesson 3: KV Cache Impact - Decode with vs without cache.
"""

import json
import time
from pathlib import Path

import torch

from paths import PLOTS_DIR
from utils.config_utils import load_config
from utils.data_utils import build_messages_for_sample, load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.plotting_utils import (
    plot_gpu_memory_comparison,
    plot_kv_cache_speedup,
)

RESULTS_DIR = Path(PLOTS_DIR) / "lesson3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sync_device(device):
    """Synchronize GPU for accurate timing."""
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "mps" in device:
        torch.mps.synchronize()


def empty_cache(device):
    """Clear GPU memory cache."""
    if "cuda" in device:
        torch.cuda.empty_cache()
    elif "mps" in device:
        torch.mps.empty_cache()


def decode_with_cache(model, input_ids, num_tokens, device):
    """Decode using KV cache."""
    sync_device(device)
    start = time.perf_counter()

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        kv_cache = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(num_tokens - 1):
            out = model(input_ids=next_token, past_key_values=kv_cache, use_cache=True)
            kv_cache = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    sync_device(device)
    return (time.perf_counter() - start) * 1000


def decode_without_cache(model, input_ids, num_tokens, device):
    """Decode WITHOUT cache - recomputes all tokens each step."""
    sync_device(device)
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_tokens):
            out = model(input_ids=input_ids, use_cache=False)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    sync_device(device)
    return (time.perf_counter() - start) * 1000


def create_input_ids(dataset, tokenizer, task_instruction, target_len, device):
    """Create input_ids of target length."""
    sample = dataset[0]
    messages = build_messages_for_sample(
        sample, task_instruction, include_assistant=False
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(prompt)

    # Repeat content to reach target length
    if len(tokens) < target_len:
        content = tokens[1:-1] if len(tokens) > 2 else tokens
        while len(tokens) < target_len:
            tokens = tokens[:-1] + content + tokens[-1:]
    tokens = tokens[:target_len]

    return torch.tensor([tokens], device=device)


def measure_kv_memory(model, input_ids, device):
    """Measure actual KV cache size by inspecting the tensors."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        kv_cache = out.past_key_values

    # Sum up actual KV cache tensor sizes
    total_bytes = 0
    for layer_kv in kv_cache:
        for tensor in layer_kv:  # K and V tensors
            total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024**2)


def reset_peak_memory(device):
    """Reset peak memory stats."""
    if "cuda" in device:
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory(device):
    """Get peak GPU memory allocated in MB."""
    if "cuda" in device:
        return torch.cuda.max_memory_allocated() / (1024**2)
    elif "mps" in device:
        # MPS doesn't have peak tracking, use current
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0


def main():
    print("Loading model...")
    cfg = load_config()
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=False, use_lora=False, device_map="auto"
    )
    model.eval()
    device = str(next(model.parameters()).device)

    train_data, _, _ = load_and_prepare_dataset(cfg)
    task_instruction = cfg.get(
        "task_instruction", "Summarize the following conversation."
    )

    results = {
        "with_cache": [],
        "without_cache": [],
        "memory": [],
    }

    # Experiment 1: Speed comparison
    print("\n" + "=" * 50)
    print("Decode Speed: With vs Without KV Cache")
    print("=" * 50)

    for ctx_len in [64, 128, 256]:
        input_ids = create_input_ids(
            train_data, tokenizer, task_instruction, ctx_len, device
        )
        num_tokens = 20

        # Warmup
        decode_with_cache(model, input_ids.clone(), 3, device)
        decode_without_cache(model, input_ids.clone(), 3, device)

        time_with = decode_with_cache(model, input_ids.clone(), num_tokens, device)
        time_without = decode_without_cache(
            model, input_ids.clone(), num_tokens, device
        )

        results["with_cache"].append({"context": ctx_len, "time_ms": time_with})
        results["without_cache"].append({"context": ctx_len, "time_ms": time_without})

        print(
            f"Context {ctx_len}: with={time_with:.1f}ms, without={time_without:.1f}ms, speedup={time_without/time_with:.1f}x"
        )

        empty_cache(device)

    # Experiment 2: Peak GPU memory during forward pass with vs without KV cache
    print("\n" + "=" * 50)
    print("Peak GPU Memory: Forward Pass With vs Without KV Cache")
    print("=" * 50)

    results["gpu_memory"] = []
    for ctx_len in [256, 512, 1024, 1536, 2048, 3072, 4096]:
        input_ids = create_input_ids(
            train_data, tokenizer, task_instruction, ctx_len, device
        )

        # Forward WITHOUT KV cache - measure peak memory
        empty_cache(device)
        reset_peak_memory(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=False)
        sync_device(device)
        mem_no_cache = get_peak_memory(device)

        # Forward WITH KV cache - measure peak memory
        empty_cache(device)
        reset_peak_memory(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=True)
        sync_device(device)
        mem_with_cache = get_peak_memory(device)

        extra = mem_with_cache - mem_no_cache
        results["gpu_memory"].append(
            {
                "context": ctx_len,
                "without_cache_mb": mem_no_cache,
                "with_cache_mb": mem_with_cache,
                "kv_cache_overhead_mb": extra,
            }
        )

        print(
            f"Context {ctx_len}: without={mem_no_cache:.1f}MB, with={mem_with_cache:.1f}MB, KV overhead={extra:.1f}MB"
        )
        empty_cache(device)

    # Save and plot
    with open(RESULTS_DIR / "kv_cache_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_kv_cache_speedup(results, RESULTS_DIR)
    plot_gpu_memory_comparison(results, RESULTS_DIR)


if __name__ == "__main__":
    main()
