# LLM Engineering & Deployment - Week 7 Code Examples

**Week 8: LLM Inference Optimization**  
Part of the LLM Engineering & Deployment Certification Program

This repository contains code examples for understanding and optimizing LLM inference performance. The module covers:

- **Inference Fundamentals** - Prefill vs decode phases, autoregressive generation
- **Benchmarking** - Measuring TTFT, ITL, E2E latency, throughput
- **KV Cache** - Understanding the memory-speed tradeoff
- **Attention Optimizations** - Flash Attention and Paged Attention concepts
- **Quantization** - Post-training quantization for faster inference
- **Scheduling** - Continuous batching and speculative decoding

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon Mac
- ~8GB+ GPU memory for running experiments with Llama 3.2 1B

---

## Setup

### 1. Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 2. Dependency Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Hugging Face Authentication

Some models (like Llama) require authentication. Log in to Hugging Face:

```bash
huggingface-cli login
```

You'll need to accept the model license on the Hugging Face model page before downloading.

### 4. Configuration

The `code/config.yaml` file contains all configuration settings:

```yaml
base_model: meta-llama/Llama-3.2-1B-Instruct
dataset:
  name: knkarthick/samsum
  cache_dir: ../data/datasets
```

The default configuration uses Llama 3.2 1B Instruct and the SAMSum dialogue summarization dataset. Adjust settings as needed for your hardware.

---

## Running the Code Examples

All scripts should be run from the `code/` directory:

```bash
cd code
```

### Lesson 1: Prefill vs Decode Phases

Demonstrates the fundamental difference between prefill (compute-bound) and decode (memory-bandwidth-bound) phases:

```bash
python prefill_vs_decode.py
```

This script runs three experiments:
1. **Prefill scaling** - Shows O(n²) time growth with prompt length
2. **Decode scaling** - Shows linear time growth with output length  
3. **Context effect** - Shows minimal decode impact from longer contexts (due to KV cache)

Outputs are saved to `plots/lesson1/`.

### Lesson 2: Benchmarking Harness

Sets up a proper benchmarking framework with warmup runs, percentile reporting, and controlled experiments:

```bash
python benchmarking_harness.py
```

Measures key metrics across different prompt/output length combinations:
- **TTFT** (Time To First Token)
- **ITL** (Inter-Token Latency)
- **E2E** (End-to-End Latency)
- **TSP** (Tokens Per Second)
- **RPS** (Requests Per Second)

Results are saved to `plots/lesson2/benchmark_results.json` with visualizations.

### Lesson 3: KV Cache Impact

Compares inference with and without KV cache to demonstrate its importance:

```bash
python kv_cache.py
```

This script:
1. Measures decode speed with vs without cache (expect 10-100x speedup with cache)
2. Measures GPU memory overhead from KV cache storage
3. Shows how cache memory scales with context length

Results are saved to `plots/lesson3/`.

---

## Lessons Overview

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| 1 | Inference Basics | Autoregressive generation, prefill vs decode, compute vs memory bottlenecks |
| 2 | Benchmarking | TTFT, ITL, E2E, TSP, RPS, warmup, percentile reporting |
| 3 | KV Cache | Cache structure, memory cost formula, fragmentation issues |
| 4 | Attention Optimizations | Flash Attention (tiled computation), Paged Attention (memory management) |
| 5 | Quantization | INT8/INT4 weights, GGUF/GPTQ/AWQ formats, KV cache quantization |
| 6 | Scheduling | Continuous batching, speculative decoding, throughput optimization |

---

## Project Structure

```
rt-llm-eng-cert-week8/
├── code/
│   ├── config.yaml              # Model and dataset configuration
│   ├── prefill_vs_decode.py     # Lesson 1: Phase comparison experiments
│   ├── benchmarking_harness.py  # Lesson 2: Benchmarking framework
│   ├── kv_cache.py              # Lesson 3: KV cache experiments
│   ├── paths.py                 # Path configuration
│   └── utils/
│       ├── config_utils.py      # Configuration loading
│       ├── data_utils.py        # Dataset preparation
│       ├── inference_utils.py   # Inference helpers
│       ├── model_utils.py       # Model loading utilities
│       └── plotting_utils.py    # Visualization functions
├── data/
│   └── datasets/                # Cached datasets (auto-downloaded)
├── lessons/
│   ├── lesson1-inference-basics.md
│   ├── lesson2-benchmarking.md
│   ├── lesson3-kv-cache.md
│   ├── lesson4-attention-optimizations.md
│   ├── lesson5-quantization.md
│   └── lesson6-scheduling-optimizations.md
├── plots/                       # Generated plots and results
├── requirements.txt
└── README.md
```

---

## Key Metrics Reference

| Metric | What It Measures | What Drives It |
|--------|------------------|----------------|
| **TTFT** | Time to first token | Prefill computation + scheduling delays |
| **ITL** | Time between tokens | Decode efficiency, memory bandwidth |
| **E2E** | Total request time | TTFT + (output_tokens × ITL) |
| **TSP** | Tokens per second | Batching, hardware utilization |
| **RPS** | Requests per second | Overall system capacity |

---

## Hardware Considerations

- **GPU Memory**: Llama 3.2 1B requires ~4GB in FP16. KV cache adds ~0.5MB per token.
- **Apple Silicon**: Scripts support MPS backend but CUDA provides better timing accuracy.
- **CPU Fallback**: Possible but significantly slower; not recommended for benchmarking.

---

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**

- Share and adapt this material for non-commercial purposes
- Must give appropriate credit and indicate changes made
- Must distribute adaptations under the same license

See [LICENSE](LICENSE) for full terms.

---

## Contact

For questions or issues related to this repository, please refer to the course materials or contact your instructor.
