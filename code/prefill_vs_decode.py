"""
Lesson 1: Prefill vs Decode Phase Comparison

This script demonstrates the fundamental difference between the two phases
of LLM inference:

PREFILL PHASE:
- Processes entire prompt in parallel
- Compute-bound (O(n²) attention)
- Cost scales with prompt length

DECODE PHASE:
- Generates tokens one at a time
- Memory-bandwidth-bound (loading weights)
- Cost scales with output length

Uses SAMSum dialogues from config.yaml for realistic prompt lengths.

Run with: python prefill_vs_decode.py
"""

import time
from pathlib import Path

import torch


from paths import PLOTS_DIR
from utils.config_utils import load_config
from utils.data_utils import build_messages_for_sample, load_and_prepare_dataset
from utils.model_utils import setup_model_and_tokenizer
from utils.plotting_utils import (
    plot_context_effect,
    plot_decode_scaling,
    plot_prefill_scaling,
)

# Output directory for plots (lesson-specific subdirectory)
LESSON_PLOTS_DIR = Path(PLOTS_DIR) / "lesson1"
LESSON_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def measure_prefill_time(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
) -> dict[int, float]:
    """
    Measure prefill time for prompts of different lengths.

    Prefill is compute-bound due to O(n²) attention computation.
    We expect time to grow super-linearly with prompt length.
    """
    results = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        actual_length = inputs["input_ids"].shape[1]

        # Warm-up run
        with torch.no_grad():
            _ = model(**inputs, use_cache=True)

        # Timed run (prefill only - single forward pass)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(**inputs, use_cache=True)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        results[actual_length] = elapsed_ms
        print(f"  Prompt length {actual_length:>4} tokens: {elapsed_ms:>8.2f} ms")

        # Clear GPU memory to prevent accumulation across experiments
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def measure_decode_time(
    model,
    tokenizer,
    num_tokens_list: list[int],
    base_prompt: str,
    device: str,
) -> dict[int, float]:
    """
    Measure decode time for generating different numbers of tokens.

    Decode is memory-bandwidth-bound - each step loads model weights.
    We expect roughly linear scaling with output length.
    """
    results = {}

    inputs = tokenizer(base_prompt, return_tensors="pt").to(device)

    for num_tokens in num_tokens_list:
        # Run prefill first
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values

        # Get first token
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        # Time decode phase only
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_tokens - 1):  # -1 because we already have first token
                outputs = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        results[num_tokens] = elapsed_ms
        per_token = elapsed_ms / num_tokens if num_tokens > 0 else 0
        print(
            f"  Generate {num_tokens:>3} tokens: {elapsed_ms:>8.2f} ms ({per_token:.2f} ms/token)"
        )

    return results


def measure_context_effect_on_decode(
    model,
    tokenizer,
    prompts_by_length: list[str],
    decode_tokens: int,
    device: str,
) -> dict[int, float]:
    """
    Measure how context length affects decode speed.

    Key insight: Context length has minimal effect on decode ITL because
    the KV cache stores precomputed attention keys/values.
    """
    results = {}

    for prompt in prompts_by_length:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        actual_len = inputs["input_ids"].shape[1]

        # Prefill
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        # Time decode
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(decode_tokens):
                outputs = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        results[actual_len] = elapsed_ms
        per_token = elapsed_ms / decode_tokens
        print(
            f"  Context {actual_len:>4} tokens → decode {decode_tokens} tokens: {per_token:.2f} ms/token"
        )

    return results


def create_prompts_of_varying_lengths(
    dataset, tokenizer, task_instruction: str, target_lengths: list[int]
) -> list[str]:
    """
    Create prompts of approximately target lengths by selecting/truncating dialogues.
    """
    prompts = []

    # Build all prompts first
    all_prompts = []
    for sample in dataset:
        messages = build_messages_for_sample(
            sample, task_instruction, include_assistant=False
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(prompt)
        all_prompts.append((len(tokens), prompt))

    # Sort by length
    all_prompts.sort(key=lambda x: x[0])

    # For each target length, find closest prompt or truncate
    for target in target_lengths:
        # Find prompt closest to target length
        best_prompt = None
        best_diff = float("inf")

        for length, prompt in all_prompts:
            diff = abs(length - target)
            if diff < best_diff:
                best_diff = diff
                best_prompt = prompt
            if length >= target:
                break

        if best_prompt:
            # Truncate if needed
            tokens = tokenizer.encode(best_prompt)
            if len(tokens) < target:
                tokens = tokens + [tokenizer.bos_token_id] * (target - len(tokens))
            else:
                tokens = tokens[:target]
            truncated = tokenizer.decode(tokens)
            prompts.append(truncated)

    return prompts


def main():
    # Load configuration
    print("Loading configuration...")
    cfg = load_config()

    print(f"\n{'=' * 70}")
    print("PREFILL vs DECODE PHASE COMPARISON")
    print(f"{'=' * 70}")
    print(f"Model:   {cfg['base_model']}")
    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"{'=' * 70}\n")

    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = setup_model_and_tokenizer(
        cfg, use_4bit=False, use_lora=False, device_map="auto"
    )
    model.eval()

    device = str(next(model.parameters()).device)
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Load dataset
    print("Loading dataset...")
    train_dataset, _, _ = load_and_prepare_dataset(cfg)
    task_instruction = cfg.get(
        "task_instruction", "Summarize the following conversation."
    )

    # =========================================================================
    # EXPERIMENT 1: Prefill scaling with prompt length
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Prefill Time vs Prompt Length")
    print("-" * 70)
    print("Prefill is COMPUTE-BOUND due to O(n²) attention.")
    print("Expectation: Time grows super-linearly with prompt length.")
    print("Using SAMSum dialogues of varying lengths.\n")

    target_lengths = [int(2**i) for i in range(10, 13)]
    prompts_varying = create_prompts_of_varying_lengths(
        train_dataset, tokenizer, task_instruction, target_lengths
    )

    prefill_results = measure_prefill_time(model, tokenizer, prompts_varying, device)

    # Show scaling factor
    if len(prefill_results) >= 2:
        lengths = sorted(prefill_results.keys())
        first_len, last_len = lengths[0], lengths[-1]
        len_ratio = last_len / first_len
        time_ratio = prefill_results[last_len] / prefill_results[first_len]
        print(
            f"\n→ Prompt length increased {len_ratio:.1f}x, time increased {time_ratio:.1f}x"
        )
        print(f"  (Super-linear scaling due to O(n²) attention)")

    # Generate plot
    plot_prefill_scaling(
        prefill_results, save_path=LESSON_PLOTS_DIR / "prefill_scaling.png"
    )

    # =========================================================================
    # EXPERIMENT 2: Decode scaling with output length
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Decode Time vs Output Length")
    print("-" * 70)
    print("Decode is MEMORY-BANDWIDTH-BOUND (loading weights each step).")
    print("Expectation: Time grows linearly with output length.\n")

    # Use a medium-length dialogue as base prompt
    sample = train_dataset[0]
    messages = build_messages_for_sample(
        sample, task_instruction, include_assistant=False
    )
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    decode_lengths = [10, 20, 40, 80, 160]
    decode_results = measure_decode_time(
        model, tokenizer, decode_lengths, base_prompt, device
    )

    # Show scaling factor
    if len(decode_results) >= 2:
        lengths = sorted(decode_results.keys())
        first_len, last_len = lengths[0], lengths[-1]
        len_ratio = last_len / first_len
        time_ratio = decode_results[last_len] / decode_results[first_len]
        print(
            f"\n→ Output length increased {len_ratio:.1f}x, time increased {time_ratio:.1f}x"
        )
        print(f"  (Linear scaling - memory bandwidth is the bottleneck)")

    # Generate plot
    plot_decode_scaling(
        decode_results, save_path=LESSON_PLOTS_DIR / "decode_scaling.png"
    )

    # =========================================================================
    # EXPERIMENT 3: Context length effect on decode speed
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Context Length Effect on Decode Speed")
    print("-" * 70)
    print("The KV cache stores precomputed attention, so context length")
    print("should have MINIMAL effect on decode ITL.\n")

    context_lengths = [64, 128, 256, 512, 1024]
    context_prompts = create_prompts_of_varying_lengths(
        train_dataset, tokenizer, task_instruction, context_lengths
    )

    context_results = measure_context_effect_on_decode(
        model, tokenizer, context_prompts, decode_tokens=20, device=device
    )

    # Show that context doesn't dramatically affect decode
    if len(context_results) >= 2:
        lengths = sorted(context_results.keys())
        first_len, last_len = lengths[0], lengths[-1]
        len_ratio = last_len / first_len
        time_ratio = context_results[last_len] / context_results[first_len]
        print(
            f"\n→ Context increased {len_ratio:.1f}x, decode time increased only {time_ratio:.1f}x"
        )
        print(f"  (KV cache makes context length nearly irrelevant for decode!)")

    # Generate plot
    plot_context_effect(
        context_results,
        decode_tokens=20,
        save_path=LESSON_PLOTS_DIR / "context_effect.png",
    )


if __name__ == "__main__":
    main()
