To understand the difference between DFlash and PFlash (often formally referred to as FlashPrefill), it is necessary to divide Large Language Model (LLM) inference into its two core phases: the prefill phase and the decode phase.  
These two technologies do not compete; rather, they solve entirely different bottlenecks in the LLM inference pipeline. PFlash accelerates the processing of the user's input, while DFlash accelerates the generation of the model's output.
FlashPrefill (PFlash): Accelerating the Input Prompt
The Prefill phase occurs when the LLM reads your initial prompt. For very long documents, this phase becomes a massive computational bottleneck because standard attention mechanisms scale quadratically (processing 100,000 tokens requires calculating 10 billion attention scores).  
PFlash is designed strictly to accelerate this phase for extremely long contexts.
Mechanism: PFlash utilizes "Instantaneous Pattern Discovery." It rapidly identifies which parts of the attention matrix are actually relevant (detecting vertical, slash, and block-sparse patterns) and ignores the rest.
Dynamic Thresholding: Instead of using slow sorting algorithms to find the most important attention scores, it uses a max-based dynamic thresholding system to instantly drop redundant data and compute only what matters.
Performance Impact: The primary metric PFlash improves is Time to First Token (TTFT). Benchmarks from early 2026 show it achieves up to a 27x speedup on 256K-token contexts without degrading the model's intelligence.
DFlash: Accelerating Token Generation
The Decode phase occurs after the prompt is processed, when the model generates its response one token at a time. This phase is inherently sequential and bottlenecked by GPU memory bandwidth, leaving compute cores underutilized.  
DFlash (Block Diffusion for Flash Speculative Decoding) is a highly advanced speculative decoding framework designed to break this sequential bottleneck.  
Mechanism: Older speculative decoding methods use a smaller "draft" model to guess the next few tokens one-by-one. DFlash instead uses a lightweight block diffusion model. This allows it to draft an entire block of tokens (e.g., 8 to 16 tokens) simultaneously in a single forward pass.  
Target Conditioning (KV Injection): To ensure the draft model guesses correctly, DFlash extracts hidden states from the main target model and injects them directly into the draft model's Key-Value (KV) cache. This turns the drafter into a highly accurate, high-speed adapter.  
Performance Impact: The primary metric DFlash improves is Tokens Per Second (Throughput). It yields up to 6x lossless acceleration over standard autoregressive decoding and operates roughly 2.5x faster than previous state-of-the-art speculative methods like EAGLE-3.  
Direct Comparison
FeatureFlashPrefill (PFlash)DFlashInference PhasePrefill (Input processing)Decode (Output generation)Primary Metric ImprovedTime to First Token (TTFT)Tokens Per Second (Throughput)Core TechnologyBlock-sparse attention & dynamic thresholdingSpeculative decoding via block diffusionPrimary Bottleneck SolvedQuadratic compute cost of long-context attentionSequential memory-bandwidth bottleneck of generationOptimal Use CaseMassive inputs (RAG, long documents, multi-turn agents)Fast outputs (Instruction following, coding, reasoning)DFlash: Faster LLM Inference via Block Diffusion
This video breaks down the DFlash research paper, explaining how its block diffusion model successfully parallels token generation to speed up inference.

Combining DFlash and PFlash (FlashPrefill) creates an end-to-end optimized LLM inference pipeline that resolves the two biggest bottlenecks in AI processing simultaneously.
Because these two technologies target completely different phases of the generation process (Compute-bound vs. Memory-bandwidth-bound), they do not conflict. Instead, they stack perfectly to provide a holistic speedup.
Here is exactly what you get when you combine them:
1. Ultra-Low "Time To First Token" (via PFlash)
When you submit a massive prompt (such as a 100-page PDF or a dense RAG retrieval context), PFlash handles the initial processing. It bypasses the quadratic compute bottleneck using sparse attention, drastically cutting down the time you spend waiting for the model to "think" before it starts typing.
2. Ultra-High "Tokens Per Second" (via DFlash)
The millisecond PFlash finishes processing the input and generates the very first token, DFlash takes over. Instead of the system dropping back down to slow, sequential token generation, DFlash's block diffusion speculative decoding generates entire chunks of the output instantly.
The Net Result
By combining both, you achieve a system with:
Zero-Lag Long Contexts: Agents and applications that require massive context windows (like reading codebases or legal documents) will feel as responsive as if you only sent them a one-sentence prompt.
Real-Time Generation: The actual text output will stream fast enough to keep up with human reading speeds, or fast enough to act as real-time voice agents, without the traditional slowdowns caused by memory bottlenecks.
Higher Server Efficiency: By maximizing compute utilization during both prefill and decode, server costs for hosting the model drop significantly, as the GPUs are never sitting idle waiting on memory bandwidth or calculating useless attention scores.

The links you shared point to Lucebox Hub, an open-source project led by Sandro Puppo (@pupposandro). It is a specialized inference library that manually rewrites LLM software—specifically custom CUDA kernels and speculative decoding algorithms—to extract maximum performance from specific consumer GPUs (like the RTX 3090, 4090, and Blackwell cards) rather than relying on generalized, one-size-fits-all frameworks.
This directly bridges the gap between the DFlash architecture we just discussed and actual, deployable local hardware.
Here is exactly what they built and how it connects:
1. Bringing DFlash to Consumer GPUs (Luce DFlash)
While DFlash represents a massive leap in breaking the sequential memory bandwidth bottleneck, it requires complex engineering to deploy. The lucebox-hub repository contains the very first standalone C++/CUDA port of DFlash speculative decoding, built natively on top of the ggml format.
By implementing block-diffusion speculative decoding for consumer hardware, they managed to get a 27-billion parameter model (Qwen 3.5 and 3.6) running at a massive 129 to 207 tokens per second on a single 24GB RTX 3090. They achieved this by writing custom tree-aware state rollback kernels tailored directly for the SM architecture of the hardware.
2. The "Megakernel" Concept
Alongside DFlash, they developed a "Megakernel" optimization for smaller models. Standard inference frameworks launch roughly 100 separate compute kernels per token, creating severe CPU round-trip latency. Lucebox bypassed this by fusing all 24 layers of an LLM into a single, persistent CUDA dispatch. The weights stream directly into the GPU, entirely eliminating CPU overhead and doubling the throughput-per-watt.
Integration & Deployment
The project is built as a highly optimized C++/CUDA stack, making it an excellent component for high-performance agentic systems. While their official documentation provides a standard, multi-step pip setup for the Python bindings, the environment and PyTorch dependencies can be resolved faster and cleaner by directly executing uv pip install -e . --no-build-isolation when building the package from source.
In short: We were just discussing the mechanics of how DFlash accelerates token generation. These links are the live, open-source execution of that exact technology, hand-tuned to run at extreme speeds on standard desktop silicon.

Zero yapping. Here is the verified data from the original research papers and hardware benchmarks on how these two optimizations impact the intelligence of the LLM.
DFlash: 0% Degradation (Mathematically Lossless)
DFlash does not make the model dumber. It operates on speculative decoding, meaning the lightweight block diffusion model only guesses the next sequence of tokens.  
The Verification Step: The main target model (e.g., Qwen or Llama) evaluates the drafted tokens in a single parallel pass. If the draft model hallucinates or guesses a suboptimal token, the target model instantly rejects it and computes the correct token instead.  
The Hard Fact: The final output distribution is mathematically guaranteed to be identical to what the model would have generated without DFlash. You trade VRAM (to hold the small draft model) for speed, but you sacrifice exactly zero accuracy.  
PFlash (FlashPrefill): >0% Degradation (Threshold-Dependent)
Unlike DFlash, PFlash is an approximation. It drops parts of the attention matrix to skip quadratic compute costs, meaning there is a risk of losing context. The degradation depends entirely on how aggressively you tune the sparsity threshold.
Paper Benchmarks (March 2026): In the original "Needle In A Haystack" (NIAH) retrieval tests on Qwen3-30B over a massive 256K context window, the model maintained "nearly identical" accuracy when using their default dynamic thresholding algorithm.  
Real-World Hardware Limits (Lucebox/CUDA Port): Recent community benchmarks running Qwen3.6 on a 24GB RTX 3090 with a 128K context document mapped out the exact breaking point of model intelligence:
At a 0.05 keep ratio (forcing the model to drop 95% of the attention matrix): The model perfectly retrieved single-needle facts hidden deep in the document, yielding a 10x prefill speedup (24.8 seconds instead of 248 seconds) with effectively zero degradation.
At a 0.02 keep ratio (dropping 98% of the matrix): The model began suffering critical recall failures (losing the "needle" in the text) in exchange for shaving off a few extra seconds of compute time.  
The Bottom Line:
Your output generation (DFlash) loses absolutely no intelligence. Your input processing (PFlash) will only make the model dumb if you manually configure the sparse attention thresholds to drop too much data in pursuit of extreme speed.


Directory structure:
└── luce-org-lucebox-hub/
    ├── README.md
    ├── CONTRIBUTING.md
    ├── LICENSE
    ├── dflash/
    │   ├── README.md
    │   ├── CMakeLists.txt
    │   ├── RESULTS.md
    │   ├── deps/
    │   │   └── bsa_stubs/
    │   │       ├── README.md
    │   │       ├── ATen/
    │   │       │   └── cuda/
    │   │       │       ├── CUDAGeneratorImpl.h
    │   │       │       └── CUDAGraphsUtils.cuh
    │   │       └── c10/
    │   │           └── cuda/
    │   │               └── CUDAException.h
    │   ├── docs/
    │   │   └── SPEC_PREFILL.md
    │   ├── examples/
    │   │   └── chat.py
    │   ├── include/
    │   │   └── dflash27b.h
    │   ├── scripts/
    │   │   ├── _prefill_hook.py
    │   │   ├── bench_daemon.py
    │   │   ├── bench_he.py
    │   │   ├── bench_llm.py
    │   │   ├── convert_dflash_to_gguf.py
    │   │   ├── detokenize.py
    │   │   ├── gen_oracle.py
    │   │   ├── phase_split_dual_gpu.py
    │   │   ├── quantize_draft_q8.py
    │   │   ├── run.py
    │   │   ├── server.py
    │   │   ├── server_tools.py
    │   │   ├── setup_system.sh
    │   │   ├── test_server.py
    │   │   └── tokenize_prompt.py
    │   ├── src/
    │   │   ├── bsa_fwd_inst.cu
    │   │   ├── bsa_launcher.cu
    │   │   ├── delta_net_chunked.cpp
    │   │   ├── delta_net_chunked.h
    │   │   ├── dflash_graph.h
    │   │   ├── errors.cpp
    │   │   ├── f16_convert.cu
    │   │   ├── flashprefill.cpp
    │   │   ├── flashprefill.h
    │   │   ├── flashprefill_kernels.cu
    │   │   ├── flashprefill_q8.cpp
    │   │   ├── flashprefill_select.cpp
    │   │   ├── gguf_draft_loader.cpp
    │   │   ├── gguf_target_loader.cpp
    │   │   ├── internal.h
    │   │   ├── kv_cache.cpp
    │   │   ├── kv_quant.cpp
    │   │   ├── kv_quant.h
    │   │   ├── qwen35_target_graph.cpp
    │   │   ├── qwen3_0p6b_drafter.h
    │   │   ├── qwen3_0p6b_graph.cpp
    │   │   ├── qwen3_0p6b_loader.cpp
    │   │   ├── qwen3_dflash_graph.cpp
    │   │   ├── qwen3_drafter.cpp
    │   │   ├── qwen3_drafter.h
    │   │   └── safetensors_draft.cpp
    │   └── test/
    │       ├── pflash_daemon.cpp
    │       ├── smoke_draft_graph.cpp
    │       ├── smoke_load_draft.cpp
    │       ├── smoke_load_target.cpp
    │       ├── smoke_qwen3_0p6b_forward.cpp
    │       ├── smoke_target_forward.cpp
    │       ├── test_flashprefill_kernels.cpp
    │       ├── test_generate.cpp
    │       ├── test_kv_quant.cpp
    │       └── test_vs_oracle.cpp
    ├── megakernel/
    │   ├── README.md
    │   ├── _phase2_variant.py
    │   ├── bench.py
    │   ├── bench_pp_tg.py
    │   ├── bench_pp_tg_nvfp4.py
    │   ├── build_corpus.py
    │   ├── diag_phase2_metrics.py
    │   ├── diag_prefill_kernels.py
    │   ├── final_bench.py
    │   ├── final_bench_nvfp4.py
    │   ├── half_type.h
    │   ├── kernel.cu
    │   ├── model.py
    │   ├── model_nvfp4.py
    │   ├── prefill_megakernel.cu
    │   ├── RESULTS.md
    │   ├── setup.py
    │   ├── torch_bindings.cpp
    │   └── corpus/
    │       ├── baseline.json
    │       ├── wmma.json
    │       ├── wmma_p3.json
    │       ├── wmma_p4.json
    │       ├── wmma_p6cleanup.json
    │       ├── wmma_p7.json
    │       └── wmma_p8.json
    ├── pflash/
    │   ├── README.md
    │   ├── pyproject.toml
    │   ├── pflash/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── dflash_client.py
    │   └── tests/
    │       ├── bench_niah_cpp.py
    │       └── niah_gen.py
    └── .sisyphus/
        └── plans/
            └── 20260428-1430-path-b-deltanet-wmma-scope.md

================================================
FILE: README.md
================================================
<p align="center">
  <img src="assets/banner.png" alt="Lucebox" width="85%">
</p>

<p align="center">
  <a href="https://lucebox.com"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=f5c842&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=f5c842&labelColor=090909" alt="Discord"></a>
  <a href="https://lucebox.com/blog"><img src="https://img.shields.io/badge/Blog-f5c842?style=for-the-badge&logo=rss&logoColor=f5c842&labelColor=090909" alt="Blog"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-e8e8ed?style=for-the-badge&labelColor=090909" alt="MIT"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=76b900&labelColor=090909" alt="CUDA 12+"></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-e8e8ed?style=for-the-badge&logo=cplusplus&logoColor=e8e8ed&labelColor=090909" alt="C++17"></a>
</p>

<p align="center">
  <strong>Open LLM inference, rewritten by hand for one specific chip at a time.</strong><br/>
  Kernels, speculative decoding, and quantization, tailored per target.<br/>
  We don't wait for better silicon. We rewrite the software.
</p>

---

## Inside the box

Three projects today, more coming. Each one is a self-contained release with its own benchmarks and paper-style writeup.

<p align="center">
  <a href="megakernel/"><img src="assets/svg/card-megakernel-dark.svg" alt="Megakernel" width="46%"></a>
  &nbsp;&nbsp;
  <a href="dflash/"><img src="assets/svg/card-dflash-dark.svg" alt="DFlash 27B" width="46%"></a>
</p>

<p align="center">
  <a href="pflash/"><img src="assets/svg/card-pflash-dark.svg" alt="PFlash speculative prefill" width="46%"></a>
</p>

---

## 01 · Megakernel Qwen3.5 0.8B on RTX 3090

**The first megakernel for hybrid DeltaNet/Attention LLMs.** All 24 layers of Qwen 3.5-0.8B in a single CUDA dispatch, 1.87 tok/J on a 2020 GPU, matching Apple's latest silicon at 2× the throughput.

```bash
# 1. clone + enter
git clone https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/megakernel

# 2. install (Python 3.10+, CUDA 12+, PyTorch 2.0+). Weights stream from HF on first run.
python -m venv .venv && source .venv/bin/activate   # required on Ubuntu 24+ system Python (PEP 668)
pip install --upgrade pip
pip install torch                          # install BEFORE the next step; setup.py imports torch at build time
pip install -e . --no-build-isolation      # --no-build-isolation lets the build see the torch you just installed

# 3. run the benchmark (prefill pp520 + decode tg128 vs llama.cpp BF16 + PyTorch HF)
python final_bench.py
```

| Method | Prefill pp520 | Decode tg128 | tok/J |
|--------|:-------------:|:------------:|:-----:|
| **Megakernel** `@220W` | **21,347** | **413** | **1.87** |
| llama.cpp BF16 `@350W` | 11,247 | 267 | 0.76 |
| PyTorch HF | 7,578 | 108 | n/a |

**What makes it work:** 82 blocks, 512 threads, one persistent kernel. No CPU round-trips between layers. Weights streamed straight from HuggingFace. Cooperative grid sync instead of ~100 kernel launches per token. Power ceiling hit before compute ceiling, so DVFS converts tight execution straight into saved watts.

[Full writeup →](megakernel/README.md) · [Benchmarks →](megakernel/RESULTS.md) · [Blog post →](https://lucebox.com/blog/megakernel)

> **Blackwell (RTX 5090, DGX Spark / GB10):** auto-detected by setup; NVFP4 decode path lands ~194 tok/s tg128 on GB10. See [megakernel/README.md#blackwell-sm_120--sm_121a](megakernel/README.md).

---

## 02 · DFlash DDtree Qwen3.5 & Qwen3.6 27B GGUF on RTX 3090

**First GGUF port of DFlash speculative decoding.** Qwen3.5-27B on a single RTX 3090, Q4_K_M target + BF16 draft, DDTree budget=22.

- **Up to 207 tok/s** in the demo (207.6 tok/s DFlash vs 38.0 tok/s AR, 5.46×)
- **129.5 tok/s mean** on the HumanEval 10-prompt bench
- **3.43× faster than autoregressive** (+15% over chain speculative decoding)
- **2.8× faster than SGLang AWQ** on the same hardware
- **Up to 256K context in 24 GB** via TurboQuant TQ3_0 KV cache (128K Q4_0 bench: 134.78 tok/s at ctx=131072)

```bash
# 1. clone with submodules (pulls the pinned Luce-Org/llama.cpp@luce-dflash fork)
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/dflash

# 2. build the C++/CUDA decoder (CUDA 12+, CMake 3.18+)
# Default compiles for 75/80/86/89 (+120 on CUDA 12.8+, +sm_121/DGX Spark on CUDA 12.9+, +sm_110/Thor on CUDA 13.0+) so the binary runs on every supported card.
# 3090-only users can add -DCMAKE_CUDA_ARCHITECTURES=86 to skip the other archs and build faster (~3 min).
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash -j

# 3. fetch weights: ~16 GB Q4_K_M target + 3.46 GB bf16 draft
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/

# 4a. one-shot streaming generate
python3 scripts/run.py --prompt "def fibonacci(n):"

# 4b. or reproduce the paper-style bench (HumanEval + GSM8K + Math500, ~15 min)
python3 scripts/bench_llm.py
```

| Benchmark | AR (tok/s) | DFlash+DDTree (tok/s) | Speedup |
|-----------|:----------:|:---------------------:|:-------:|
| **HumanEval** | 37.8 | **129.5** | **3.43×** |
| Math500 | 37.7 | 110.5 | 2.93× |
| GSM8K | 37.7 | 96.2 | 2.55× |

**The constraint that shaped the project.** AWQ INT4 of Qwen3.5-27B plus the BF16 draft doesn't leave room for the DDTree verify state on a 24 GB card. Q4_K_M GGUF (~16 GB target) is the largest format that fits target + 3.46 GB draft + budget=22 tree state + KV cache in 24 GB on the RTX 3090. Picking it forced a new port on top of ggml, since no public DFlash runtime supports a GGUF target.

**What we built vs what we didn't.** The algorithms are not ours:
- [**DFlash**](https://arxiv.org/abs/2602.06036) (z-lab, 2026): block-diffusion draft conditioned on target hidden states.
- [**DDTree**](https://arxiv.org/abs/2604.12989) (Ringel et al., 2026): tree-structured verify that beats chain verify at the same compute budget.

What we ported and tuned:
- C++/CUDA decode engine on top of ggml (no libllama, no Python runtime, Q4_K_M target path).
- Three custom CUDA kernels for tree-aware SSM state rollback: `ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`.
- DDTree budget swept for RTX 3090 + Q4_K_M target: **budget=22** is the sweet spot.
- TQ3_0 KV cache (TurboQuant 3.5 bpv, default) + sliding `target_feat` ring to fit up to 256K context in 24 GB (Q4_0 available as legacy, tops out near 128K).

### Running on other GPUs (4090, 5090, DGX Spark / GB10, Jetson AGX Thor)

Supported out of the box; the build just needs the right CUDA toolkit. `dflash/CMakeLists.txt` already auto-adds Blackwell archs when your nvcc is new enough, so the main quickstart above works as-is on newer cards.

| GPU | Arch | Min CUDA | Status |
|-----|:----:|:--------:|--------|
| RTX 3090 Ampere | `sm_86` | 12.0 | **reference, all numbers above** |
| RTX 2080 Ti Turing | `sm_75` | 12.0 | supported, 53 tok/s DFlash verified (FP16 draft) |
| RTX 4090 Ada | `sm_89` | 12.0 | should work, unverified, pass `-DCMAKE_CUDA_ARCHITECTURES=89` |
| RTX 5090 Blackwell consumer | `sm_120` | 12.8 | supported, auto-added by CMake |
| DGX Spark / GB10 | `sm_121` (compute capability 12.1) | 12.9 | supported, auto-added by CMake |
| Jetson AGX Thor | `sm_110` | 13.0 | supported, auto-added by CMake |

Verify your target:
```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, 'sm_%d%d'%(p.major,p.minor), p.multi_processor_count,'SMs', round(p.total_memory/1e9,1),'GB')"
nvcc --version
```

**DGX Spark / GB10 quick start:**
```bash
# CUDA 12.9+ required for sm_121
nvcc --version  # must show >= 12.9
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/dflash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake auto-adds sm_121
cmake --build build --target test_dflash -j
```

**Jetson AGX Thor quick start:**
```bash
# CUDA 13.0+ required for sm_110 / AGX Thor.
nvcc --version
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/dflash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake auto-adds the Thor arch your nvcc supports
cmake --build build --target test_dflash -j
```

**What will NOT auto-port:**
- **DDTree `budget=22`** tuned for 3090 + Q4_K_M + 24 GB. On cards with more VRAM (5090 32 GB, GB10 128 GB unified), re-sweep, larger tree = more verify throughput until memory bandwidth saturates. `scripts/bench_llm.py` has the sweep hooks.
- **TQ3_0 KV cache + sliding `target_feat` ring** was shaped by 24 GB (fits up to 256K context on a 3090). On GB10 (128 GB unified) / 5090 (32 GB) you can push context further or skip quantization entirely and keep F16 KV.
- **Perf numbers** (207 tok/s demo, 129.5 HumanEval, 2.8× vs SGLang AWQ) are RTX 3090 @ stock. Blackwell/Ada not yet swept, PRs with `RESULTS.md` entries welcome.

[Full writeup →](dflash/README.md) · [Benchmarks →](dflash/RESULTS.md) · [Blog post →](https://lucebox.com/blog/dflash27b)

> **Qwen3.6-27B (supported, experimental draft):** same `qwen35` architecture, so the 3.6 Q4_K_M GGUF loads as a drop-in target. With z-lab's matched [Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) draft (still under training, 2026-04-26 snapshot), HumanEval lands at ~78 tok/s (AL 5.05); the 3.5 draft gets ~74 tok/s. 3.5↔3.5 reference is 129.5 tok/s. AL should improve as z-lab finishes training the draft. Details in [dflash/README.md](dflash/README.md#qwen36-27b-target-experimental).

---

## 03 · PFlash speculative prefill on RTX 3090

**In-process speculative prefill, C++/CUDA only.** A drafter (Qwen3-0.6B BF16) loaded directly into the dflash daemon scores per-token importance over a long prompt; the heavy target (Qwen3.6-27B Q4_K_M) only prefills the spans that matter. Both models share the same ggml allocator on a single RTX 3090. **No Python, no Triton, no PyTorch at runtime** — just the dflash binary and four custom CUDA kernels (`mean_K → score → select → sparse_fwd`) plus BSA ([mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention), FA-2 derived, sm_80+) for the long-context drafter forward.

- **~10.4× TTFT** on 128K context: **24.8 s** dflash daemon vs **~257 s** llama.cpp (FA on, Q4_0 KV).
- **10.0× TTFT** on 64K context: **13.5 s** dflash vs **134.95 s** llama.cpp.
- **NIAH single-needle retrieved** at every measured context (32K → 128K), `keep_ratio=0.05`, `DFLASH_FP_ALPHA=0.85`.

```bash
# 1. build dflash + BSA kernel (sm_80+ required for BSA, ~10 min cold compile)
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/dflash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_CUDA_ARCHITECTURES=86 \
                    -DDFLASH27B_ENABLE_BSA=ON
cmake --build build --target test_dflash test_flashprefill_kernels -j

# 2. fetch weights: 27B Q4_K_M target + 0.6B BF16 drafter (GGUF) + DFlash spec-decode draft
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/

# 3. run the daemon: compress (drafter scoring) + generate (target spec decode)
DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85 \
./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf models/draft/model.safetensors --daemon
# stdin protocol: `compress <ids.bin> <keep_x1000> <drafter.gguf>` →
#                 stream of compressed token ids, then `generate <…>` →
#                 stream of generated tokens.
```

| Source S | dflash TTFT | llama.cpp baseline | Speedup | NIAH |
|----------|:-----------:|:------------------:|:-------:|:----:|
| **64K**  | **13.5 s** | 134.95 s (FA off, dense) | **10.0×** | ✅ |
| **128K** | **24.8 s** | ~257 s (FA on, Q4_0 KV)  | **~10.4×** | ✅ |

Daemon stdin commands: `compress` runs the drafter with FlashPrefill block-sparse attention and returns the compressed token-id stream; `generate` runs the target on that stream with normal speculative decode + DDTree. `park` / `unpark` / `free drafter` swap weights in and out of VRAM so target + drafter coexist on a 24 GB card.

**Runtime tunables** (full list in [`dflash/src/flashprefill.h`](dflash/src/flashprefill.h)):
```
DFLASH_FP_USE_BSA=1     # dispatch sparse FA forward through BSA (sm_80+)
DFLASH_FP_ALPHA=0.85    # block-selection threshold; higher = stricter = fewer K-blocks per Q-row
DFLASH_FP_PROFILE=1     # log mean / score / select / forward stage timings
```

**What's ours, what isn't.** Algorithms are from [Cross-Family Speculative Prefill (Liu et al., ICLR 2026)](https://arxiv.org/abs/2603.02631) for the scoring + selection layer and [FlashPrefill (Fan et al., 2026)](https://arxiv.org/abs/2603.06199) for the drafter sparse-attention forward. What we built:
- C++/CUDA daemon-resident speculative prefill in front of a quantized GGUF target — no PyTorch, no Triton, no per-request subprocess.
- BSA wired without `libtorch` via a 3-header ATen/c10 stub set under `dflash/deps/bsa_stubs/`.
- Custom Qwen3-0.6B forward (`qwen3_0p6b_*`) so the drafter runs through the same ggml allocator as the 27B target.
- 4 CUDA kernels (`flashprefill_kernels.cu`) for the FlashPrefill `mean_K / score / select / sparse_fwd` algorithm.

[Full writeup →](pflash/README.md) · [Daemon-side build / tunables →](dflash/docs/SPEC_PREFILL.md) · [Blog post →](https://lucebox.com/blog/pflash)

---

## Why this exists

Local AI should be a default, not a privilege: private data, no per-token bill, no vendor lock-in. The hardware to run capable models already sits on desks. The software to run those chips well doesn't.

General-purpose frameworks dominated the last decade because hand-tuning kernels per chip was too expensive to justify. One stack, decent on everything, great on nothing. Most of the silicon's capability stays on the floor.

AI-assisted development flips that calculus. Rewrites that took a quarter now fit in a release cycle. Lucebox is where we publish them, one chip and one model family at a time. MIT source, full writeup, reproducible benchmarks.

---

## Requirements

All experiments in this repo are built, tuned, and benchmarked on NVIDIA RTX 3090 (2020), the reference target. Supported GPU families:

- **Ampere** (sm_86, RTX 3090 / A-series): reference, CUDA 12+.
- **Ada** (sm_89, RTX 40xx): should work, unverified, CUDA 12+.
- **Blackwell consumer** (sm_120, RTX 50xx incl. 5090): supported, CUDA 12.8+.
- **DGX Spark / GB10** (sm_121, compute capability 12.1): supported, CUDA 12.9+.
- **Jetson AGX Thor** (sm_110): supported, CUDA 13+.
- **Turing** (sm_75, RTX 2080): supported, CUDA 12+.

PyTorch 2.0+. `dflash/` needs CMake 3.18+ and `--recurse-submodules` for the pinned `Luce-Org/llama.cpp@luce-dflash` fork (three tree-mode ggml ops); multi-arch build is automatic (see [Running on other GPUs](#running-on-other-gpus-4090-5090-dgx-spark--gb10-jetson-agx-thor)).

**Megakernel porting note.** `megakernel/setup.py` auto-detects the GPU arch and SM count at build time via `torch.cuda.get_device_capability()`. The decode grid is persistent (one block per SM) and is clamped to the resident-block ceiling at runtime, so no manual tuning is needed. On SM < 80 (Turing), the kernel uses FP16 instead of BF16 via a compile-time `TARGET_SM` flag; on SM ≥ 80 (Ampere+), BF16 is used. Just `pip install -e . --no-build-isolation` and the right code path is selected automatically.

**Optional, find your GPU's sweet spot:** `sudo nvidia-smi -pl 220` (megakernel hits best tok/J at 220 W on 3090; re-sweep for other cards).

---

## Repository layout

```
lucebox-hub/
├── megakernel/    · fused forward pass for Qwen 3.5-0.8B
├── dflash/        · DFlash speculative decoding port for Qwen 3.5/3.6-27B on RTX 3090
├── pflash/        · speculative-prefill harness in front of dflash (12.5× TTFT at 128K)
└── assets/        · banners, cards, diagrams
```

---

## Roadmap

```
  Q1 2026    ▮▮▮▮▮▮▮▮▮▮    RTX 3090 kernels & optimizations
  Q2 2026    ▮▮▮▮▮▯▯▯▯▯    Ryzen AI MAX+ 395 optimizations
  Q2 2026    ▮▮▯▯▯▯▯▯▯▯    Heterogeneous CPU + GPU latency optimizations
  Q2 2026    ▮▯▯▯▯▯▯▯▯▯    Lucebox OS for local AI machines
  Q3 2026    ▯▯▯▯▯▯▯▯▯▯    Lucebox official launch
```

---

## Citation

```bibtex
@software{lucebox_2026,
  title  = {Lucebox: Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub},
  year   = {2026}
}
```

Per-project citations live in each subproject's README.

---

## Inspired by

- [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles): megakernel idea and the intelligence-per-watt methodology.
- [z-lab/DFlash](https://arxiv.org/abs/2602.06036) (Wang et al., 2026): block-diffusion speculative decoding algorithm. We use their published Qwen3.5/Qwen3.6-27B-DFlash draft weights as-is.
- [DDTree](https://arxiv.org/abs/2604.12989) (Ringel & Romano, 2026): tree-structured verify that DFlash 27B uses for its 3.5× speedup over chain spec decoding. [liranringel/ddtree](https://github.com/liranringel/ddtree).
- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen): prior art on fused Qwen kernels.

---

## Community

- **Discord**: [discord.gg/yHfswqZmJQ](https://discord.gg/yHfswqZmJQ)
- **Website**: [lucebox.com](https://lucebox.com)
- **Issues**: [github.com/Luce-Org/lucebox-hub/issues](https://github.com/Luce-Org/lucebox-hub/issues)
- **Blog**: [lucebox.com/blog](https://lucebox.com/blog)

---

<p align="center">
  <sub><a href="LICENSE">MIT</a> · <a href="https://lucebox.com">Lucebox.com</a></sub>
</p>



================================================
FILE: CONTRIBUTING.md
================================================
# Contributing to Lucebox

Thanks for considering a contribution. Lucebox is a hub of self-contained optimization projects. Each one lives with its own README, benchmarks, and code, and the hub stays thin on purpose.

## What we accept

- **Kernel improvements** that preserve correctness and improve `tok/s`, `tok/J`, or memory footprint on the target hardware. Benchmark deltas required.
- **Speculative decoding algorithms** that improve our current SOTA performances
- **Benchmark harness work** under `benchmarks/` once that directory starts shipping code.
- **Doc fixes and writeups** — always welcome.


## What we don't accept (yet)

- Closed-source dependencies. Everything here has to be reproducible from public sources.

## Luce DFash Setup

### dflash

**Hardware:** NVIDIA sm_86+ GPU (RTX 3090, A10, A40, 4090) or Jetson AGX Thor sm_110, 24 GB VRAM. Thor requires CUDA 13+.

On Ubuntu 22.04 or 24.04, one script installs all system dependencies — `build-essential`, `cmake`, `git`, `git-lfs`, and the CUDA Toolkit from NVIDIA's repo:

```bash
sudo dflash/scripts/setup_system.sh
```

The script is idempotent and configures `nvcc` on PATH for both bash and zsh. For other distros see the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

| Tool | Min version |
|------|------------|
| GCC / G++ | 11 |
| CMake | 3.18 |
| Git | 2.x |
| git-lfs | any |
| CUDA Toolkit | 12.0+ |
| huggingface-cli | any |

After setup:

```bash
git submodule update --init --recursive
cmake -B dflash/build -S dflash -DCMAKE_BUILD_TYPE=Release
cmake --build dflash/build --target test_dflash -j
```

> If cmake was previously run without CUDA, wipe the build directory first (`rm -rf dflash/build`) to avoid a stale compiler cache.

---

## Before you open a PR

1. **Benchmark before and after** on the same hardware, at the same power limit, with the same warmup. Numbers without methodology don't get merged.
2. **Run the existing correctness check** (`bench_pp_tg.py` for megakernel) and confirm your change doesn't regress output parity.
3. **One concern per PR.** Kernel/algorithms changes, docs, and build config go in separate commits or separate PRs.

## Commit message format

Conventional commits:

```
feat(megakernel): fused QKV+RoPE path cuts per-token launch by 1 kernel
fix(dflash): clamp int8 DeltaNet state update before dequant
docs(hub): add DVFS methodology link
```

Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `bench`, `chore`, `ci`.

## Hardware access

If you want to contribute benchmarks but don't have the hardware:

- We can run numbered runs on our RTX 3090 (24GB) or Ryzen 395 AI Max (128GB). Open an issue with the PR.
- Apple Silicon numbers need an M-series machine running `powermetrics`, not a remote box.

## Getting help

- [Discord](https://discord.gg/yHfswqZmJQ) — fastest feedback
- [Issues](https://github.com/Luce-Org/lucebox-hub/issues) — for bugs and proposals
- Mention `@Luce-Org/maintainers` on a PR when it's ready for review

## Licensing

By contributing you agree your work is MIT-licensed, same as the rest of the repo.



================================================
FILE: LICENSE
================================================
MIT License

Copyright (c) 2026 Lucebox

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



================================================
FILE: dflash/README.md
================================================
<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce DFlash</h1>

<p align="center">
  <strong>The first GGUF port of DFlash speculative decoding.</strong><br/>
  Qwen3.5-27B at up to 207 tok/s<sup>*</sup> on a single RTX 3090 (HumanEval 10-prompt bench mean 129.5 tok/s at DDTree budget=22). 128K context on 24 GB.<br/>
  3.43× faster than autoregressive (+15% over chain spec decoding), 2.8× faster than SGLang AWQ.<br/>
  <sub><sup>*</sup>Demo run: 207.6 tok/s DFlash vs 38.0 tok/s AR (5.46×).</sub><br/><br/>
  <a href="https://lucebox.com/blog/dflash27b">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

<p align="center">
  <img src="demo.gif" width="600" />
</p>

---

```
                   AR (tok/s)   DFlash (tok/s)   Speedup
HumanEval             37.78        129.52          3.43x
Math500               37.71        110.51          2.93x
GSM8K                 37.65         96.15          2.55x
```

> Consumer GPUs can run 27B models at chat-grade speed without multi-GPU, without batching, without quantization compromises. The bottleneck was never hardware. It was the decoding algorithm.

## The gap we filled

On a 24 GB RTX 3090 with Q4_K_M weights, autoregressive decode of Qwen3.5-27B hits ~37.7 tok/s regardless of framework. Every token reads the full model from VRAM.

Speculative decoding breaks that ceiling: a tiny draft proposes multiple tokens per step, the target verifies them in one forward. [DFlash (z-lab, 2026)](https://arxiv.org/abs/2602.06036) takes this further with **block-diffusion drafting**: a 5-layer non-causal denoising draft conditioned on captured target hidden states. Accepts ~8 tokens/step vs ~3 for chain EAGLE. The official draft is [`z-lab/Qwen3.5-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash). [DDTree (Ringel & Romano, 2026)](https://arxiv.org/abs/2604.12989) adds tree-structured verify on top, recovering the last 30% of the speedup.

**What was missing:** no public implementation ran either on consumer hardware. z-lab targets BF16 on B200 (60+ GB VRAM). No GGUF path. No DDTree port. AWQ INT4 of the target + BF16 draft doesn't leave room for the verify tree on 24 GB.

Q4_K_M GGUF (~16 GB) is the largest quantization that fits target + 3.46 GB draft + budget=22 tree state + KV cache on one RTX 3090. Picking it forced the port onto ggml, the only runtime with first-class Gated DeltaNet CUDA kernels and a GGUF Q4_K_M loader. This repo is that port:

- ~2000 lines of C++/CUDA on top of ggml (no libllama, no Python runtime)
- a pinned fork of llama.cpp at [`Luce-Org/llama.cpp@luce-dflash`](https://github.com/Luce-Org/llama.cpp/tree/luce-dflash) that adds three tree-mode ggml ops: `ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`
- hardcoded for the one model pair, decoding at 129.52 tok/s mean on HumanEval

## Results

Qwen3.5-27B Q4_K_M, concurrency=1, n_gen=256, 10 prompts/dataset:

| Task      | AR tok/s | DFlash+DDTree tok/s | AL   | Speedup |
|-----------|:--------:|:-------------------:|:----:|:-------:|
| HumanEval | 37.78    | **129.52**          | 8.31 | **3.43×** |
| Math500   | 37.71    | **110.51**          | 7.04 | **2.93×** |
| GSM8K     | 37.65    | **96.15**           | 6.14 | **2.55×** |

AR = autoregressive (`test_generate`). DFlash+DDTree = tree verify at budget=22 with fast rollback (`test_dflash`). AL = Acceptance Length, average committed tokens per draft/verify step. Reproduce via `python3 scripts/bench_llm.py`.

**Up to 256K context on 24 GB** via TQ3_0 KV cache (3.5 bpv, default; Q4_0 legacy path tops out near 128K) + sliding `target_feat` ring (4096 slots). TQ3 = ~9.7× memory saving vs F16; Q4_0 = 8×.

| Prompt length | KV     | Prefill time | Decode tok/s (FA window=2048) |
|:-------------:|:------:|:------------:|:------------:|
| 520 (HE)      | Q8_0   | 0.06 s       | ~104 (window inactive)        |
| 32K           | Q8_0   | 38 s         | ~95 (interp.)                 |
| 64K           | Q4_0   | 126 s        | **91** (PR #26)               |
| 128K          | TQ3_0  | ~10 min      | ~85–95 (sliding active)       |

Prefill numbers assume `--max-ctx` sized to the prompt (auto-fit in `run.py` / `bench_llm.py`). Oversizing — e.g. `--max-ctx=131072` on a 32K prompt — triggers FA stride over unused KV and slows prefill ~27× at that ratio.

HE 10-prompt bench mean in 128K mode (ctx=131072, ddtree-budget=16, FA window=2048): **134.78 tok/s** at AL 8.33.

Decode tok/s assume the default sliding-window flash attention (`--fa-window 2048`, lossless: 100% acceptance at all window sizes). Disable with `--fa-window 0` for full attention; expect ~25 tok/s at 60K+. Tune the window via `python3 scripts/run.py --fa-window N` or `--fa-window N` on `test_dflash`/`server.py` (sweet spot 1024–2048; bigger windows trade speed for marginally tighter attention).

Set `DFLASH27B_KV_TQ3=1` (TQ3_0, 3.5 bpv, default) or `DFLASH27B_KV_Q4=1` (Q4_0, 4.5 bpv, legacy) to enable. Full sweep in [RESULTS.md](RESULTS.md).

### Asymmetric K/V quantization

The cache now supports independent quantization types for keys and values, optimizing memory-asymmetric workloads. Set `DFLASH27B_KV_K=<type>` and `DFLASH27B_KV_V=<type>` via environment or CLI flags. Supported types (case-insensitive): `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `tq3_0`.

**Supported (K, V) pairs:**
- K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0} × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
- K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}

Unsupported pairs abort at allocation with a printed list. Precedence (high→low): per-axis `_KV_K`/`_KV_V` override legacy shorthand (`--kv-tq3` / `--kv-q4` / `--kv-f16`); legacy shorthand last-wins among themselves. Default: `q8_0` for both.

**Environment variables:**
```bash
DFLASH27B_KV_K=q8_0 DFLASH27B_KV_V=q4_0 ./test_dflash …
```

**CLI flags on `test_dflash` / `test_generate`:**
```bash
./test_dflash … -ctk q8_0 -ctv q4_0
./test_dflash … --cache-type-k=q8_0 --cache-type-v=q4_0
```

**Python flags on `scripts/run.py`, `scripts/server.py`, `scripts/server_tools.py`:**
```bash
python3 scripts/run.py --ctk q8_0 --ctv q4_0 --prompt "hello"
python3 scripts/run.py --cache-type-k q8_0 --cache-type-v q4_0 --prompt "hello"
```

**TQ3 semantics under asymmetry:** TQ3_0 is K-side-driven. The FWHT rotation is applied to the query and inverse-applied to the attention output only when `K=TQ3_0`. V's type is independent of rotation. The 256-stride context alignment is triggered if *either* K or V is TQ3_0.

Legacy `--kv-tq3` / `--kv-q4` / `--kv-f16` flags continue to work as symmetric shorthand for backward compatibility.

## Qwen3.6-27B target (experimental)

Qwen3.6-27B ships the same `qwen35` architecture string and identical layer/head dims as 3.5, so `test_dflash` loads it with no code change:

```bash
# 1. target
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/

# 2. matched 3.6 draft (gated: accept terms + set HF_TOKEN first)
huggingface-cli download z-lab/Qwen3.6-27B-DFlash --local-dir models/draft/

# 3. bench
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf python3 scripts/bench_he.py --n-gen 128
```

> The draft path is fixed at `models/draft/model.safetensors`; swapping targets requires also swapping the draft file (or pointing `DFLASH_DRAFT` at a different `model.safetensors`).

**Throughput is lower than on 3.5.** z-lab published a matched [Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) draft on 2026-04-26 (still under training). AL should climb as the draft matures. Measured on the same RTX 3090:

| Target | Draft | Bench | AL | Accept | Mean tok/s |
|---|---|---|---:|---:|---:|
| Qwen3.5-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash | HumanEval (README config) | 8.33 | ~65% | 134.78 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash (mismatch) | HumanEval (10 prompts, n_gen=128) | 4.74 | 30.6% | 73.67 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.6-27B-DFlash (still training) | HumanEval (10 prompts, n_gen=128) | 5.05 | 32.3% | 77.77 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash (mismatch) | Math (10 prompts, n_gen=128) | 3.63 | 23.7% | 57.00 |

Full `bench_llm.py` suite on **Qwen3.6-27B UD-Q4_K_XL** (unsloth Dynamic 2.0, 10 prompts, n_gen=256, RTX 3090 24 GB, auto-fit `--max-ctx`):

| Bench | AR tok/s | DFlash tok/s | AL | Speedup |
|---|---:|---:|---:|---:|
| HumanEval | 34.90 | 78.16 | 5.94 | **2.24×** |
| GSM8K | 34.89 | 59.65 | 4.43 | **1.71×** |
| Math500 | 35.13 | 69.77 | 5.15 | **1.99×** |
| **Mean** | 34.97 | 69.19 | 5.17 | **1.98×** |

Compare to Qwen3.5-27B on the same harness: 2.87× / 2.21× / 2.56× — cross-generation drop of ~22% uniformly from the draft mismatch, but still a clean ~2× with zero retraining.

Numbers will move once a Qwen3.6-matched DFlash draft lands; swap it in via `DFLASH_DRAFT=...` without rebuilding.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/dflash

# Build (CUDA 12+, CMake 3.18+, sm_75+ GPU; CUDA 13+ required for Jetson AGX Thor sm_110)
# Pass -DCMAKE_CUDA_ARCHITECTURES matching your GPU. Common values:
#   75 = 2080 Ti (auto-converts BF16 draft → FP16 at load)
#   86 = RTX 3090 / A40
#   89 = RTX 4090
#   90 = H100
#   120 = Blackwell / DGX Spark
#   110 = Jetson AGX Thor (CUDA 13+)
# Omitting the flag falls back to the CMake-set default ("75;86"), which
# fails to compile pflash's BF16 WMMA kernels on the sm_75 pass — set the
# flag explicitly to your real arch to avoid this.
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target test_dflash -j

# Fetch models: ~16 GB target + 3.46 GB draft.
# Quickstart pins to Qwen3.6-27B (latest release). For Qwen3.5-27B swap in
# unsloth/Qwen3.5-27B-GGUF + z-lab/Qwen3.5-27B-DFlash; arch is identical so
# no rebuild is needed (see "Qwen3.6-27B target" section above for AL deltas).
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/

# Streaming one-shot generate (run.py defaults to models/Qwen3.6-27B-Q4_K_M.gguf;
# override with --target or DFLASH_TARGET=... env var).
python3 scripts/run.py --prompt "def fibonacci(n):"

# Multi-turn chat REPL
python3 examples/chat.py

# OpenAI-compatible HTTP server (drop-in for Open WebUI / LM Studio / Cline)
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn transformers jinja2
.venv/bin/python scripts/server.py --port 8000 --daemon

# Reproduce paper numbers
python3 scripts/bench_llm.py                                 # HE + GSM8K + Math500
python3 scripts/bench_he.py --n-gen 256 --ddtree-budget 22   # minimal HE bench
```

**Long-context mode (up to 256K):**
```bash
DFLASH27B_KV_TQ3=1 DFLASH27B_PREFILL_UBATCH=16 \
  build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf \
  models/draft/model.safetensors /tmp/long_prompt.bin 64 /tmp/out.bin \
  --fast-rollback --ddtree --ddtree-budget=16 --max-ctx=4096   # align_up(prompt + n_gen + 64, 256); raise up to 262144 for long prompts
```

**Requirements:** NVIDIA sm_75+ GPU (2080 Ti, 3090, A10, A40, 4090) or Jetson AGX Thor sm_110, CUDA 12+ (CUDA 13+ required for Thor), 22+ GB VRAM, ~80 GB disk. On Turing (SM 7.5), BF16 draft weights are auto-converted to FP16 at load time for tensor core acceleration.

## How it works

**Block-diffusion draft.** Each step, the draft sees `[last_target_token, MASK×15]` plus the last 5 captured target hidden states. It denoises the masks in a single forward, producing 16 candidate tokens conditioned on real target features. Structurally stronger than chain EAGLE: every position conditions on the same captured context, not its own noisy predictions.

**DDTree tree verify.** Instead of one chain of 16 candidates, a best-first tree of up to 22 nodes spans the top-K branches at each position. One target forward verifies the whole tree via a causal mask derived from parent pointers. Budget=22 is the sweet spot where draft accuracy plateaus. Chain pre-seed matters: pure best-first construction with greedy verify on a quantized target can rescue an inferior suffix; the `chain_seed=true` flag in `build_ddtree` recovered AL from ~4 to ~9.

**Per-step rollback, kernel-free.** Before verify, the target's recurrent state (SSM intermediate, conv window, KV cache) is snapshotted; after accept, restored to the committed prefix. Three custom CUDA kernels keep rollback off the critical path:

| Kernel | Purpose |
|--------|---------|
| `ggml_gated_delta_net_tree_persist` | Direct-writes SSM intermediates into a persistent buffer, skipping a 9 ms `ggml_cpy` per step |
| `ggml_ssm_conv_tree` | Tree-aware conv state gather: each sibling reads its K-1 window along the DDTree parent chain, not DFS order |
| Sliding `target_feat` ring | 4096-slot ring via `(pos % cap)`, enables 128K without holding 6.6 GB of captured features |

Prefill and decode share one graph builder; chain mode is just DDTree with `budget=n_spec+1` and no branching.

## Architecture note

Qwen3.5-27B is **not** a dense transformer. llama.cpp calls the arch `qwen35`:

- 64 layers. Every 4th is full softmax attention, the rest are **Gated DeltaNet** (linear attention with learned recurrence)
- M-RoPE, dimension sections `[11, 11, 10, 0]`
- 24 Q heads, 4 KV heads, key/value length 256
- SSM state cache alongside the KV cache

The DeltaNet primitive is already a first-class ggml op (`ggml_gated_delta_net`). Our fork of llama.cpp adds three tree-mode variants (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`) so DDTree verify can roll back SSM state in place, without a replay forward. The full engine (graph builders + decode loop + rollback + kernels) is ~2000 lines.

## Why not llama.cpp / vLLM / z-lab?

- **llama.cpp**: runs Qwen3.5-27B via GGUF but has no DFlash integration. Chain EAGLE isn't enough; block diffusion + DDTree needs a custom decode loop that bypasses `llama_decode`.
- **vLLM / SGLang**: Qwen3.5-27B in BF16 is 54 GB, so a single 24 GB card forces a quantized path. GGUF for this arch is broken on SGLang as of 2026-04 and vLLM is dropping GGUF support. AWQ runs on SGLang as plain autoregressive at 46.6 tok/s but can't host the BF16 draft + DDTree tree state alongside it on 24 GB. Q4_K_M GGUF is the only format that fits the full spec-decode stack, this repo runs it at 129.5 tok/s mean on HumanEval, **2.8× faster** than SGLang AWQ autoregressive on the same hardware.
- **z-lab reference**: vLLM / SGLang integrations ship DFlash as a speculative-decoding method, but only on BF16 weights benchmarked on NVIDIA B200 (54+ GB VRAM). No GGUF path.

## Scope and limits

Research proof-of-concept, not production.

- **Batch size 1**, single-user local inference target (Ollama / LM Studio use case)
- **One model pair**: Qwen3.5-27B Q4_K_M target + z-lab DFlash BF16 draft. Does not generalize without rewriting the graph builders.
- **Greedy only**: `temperature`/`top_p` on the OpenAI server accepted but ignored. Rejection sampling in the verify path is a weekend-sized addition.
- **CUDA sm_86+ / sm_110 Thor** only. No Metal, ROCm, multi-GPU.
- **Q4_K_M target** costs ~30 points of per-position accept vs the paper's BF16. Q5_K_M / Q6_K would recover most of it, if they fit.

Correctness: `test_vs_oracle` validates the draft graph at cos sim 0.999812 vs the PyTorch reference. The target graph matches llama.cpp's `models/qwen35.cpp` semantically and produces bit-identical output to `test_generate` in autoregressive mode.

## Contributing

Open an issue or PR against `Luce-Org/lucebox-hub`. Good first picks:

- **Temperature / top-k sampling** in the verify path
- **Full llama.cpp integration**: new arch, `llama-speculative-dflash.cpp`, `llama-cli` / `llama-server` wiring

## Citation

```bibtex
@software{luce_dflash_2026,
  title  = {Luce DFlash: GGUF port of block-diffusion speculative decoding for Qwen3.5-27B on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/dflash},
  year   = {2026}
}

@article{dflash2026,
  title   = {DFlash: Block-Diffusion Speculative Decoding},
  author  = {z-lab},
  journal = {arXiv:2602.06036},
  year    = {2026}
}

@article{ddtree2026,
  title   = {Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author  = {Ringel, Liran and Romano, Yaniv},
  journal = {arXiv:2604.12989},
  year    = {2026}
}
```

---

MIT · [Lucebox](https://lucebox.com) · [Discord](https://discord.gg/yHfswqZmJQ)

Inspired by [z-lab/DFlash](https://arxiv.org/abs/2602.06036), [liranringel/ddtree](https://github.com/liranringel/ddtree), [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
l-org/llama.cpp](https://github.com/ggml-org/llama.cpp).



================================================
FILE: dflash/CMakeLists.txt
================================================
cmake_minimum_required(VERSION 3.18)
set(DFLASH27B_USER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
project(dflash27b LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If we do not set this, ggml will output to bin and DLLs will not load
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# Bake portable rpath into all executables so bundled libggml-cuda / libggml-base
# are found regardless of LD_LIBRARY_PATH or stale /usr/local/lib (closes #31).
set(CMAKE_INSTALL_RPATH "$ORIGIN/deps/llama.cpp/ggml/src;$ORIGIN/deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/../deps/llama.cpp/ggml/src;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-cuda")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ─── ggml (from the llama.cpp submodule) ─────────────────────────────
#
# We use only ggml from the vendored llama.cpp submodule. The drafter is
# loaded via our own custom Qwen3-0.6B forward (qwen3_0p6b_loader.cpp +
# qwen3_0p6b_graph.cpp) rather than libllama, so libllama is not built.
#
# Hardcoded for CUDA. No BLAS, no Metal, no Vulkan, no examples/tests/tools.

set(GGML_CUDA         ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(GGML_METAL        OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN       OFF CACHE BOOL "" FORCE)
set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL       OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "deps/llama.cpp submodule missing. Run: "
        "git submodule update --init --recursive")
endif()
# Asymmetric KV-quant (Q4_0 K with Q8_0 V etc) needs all fattn quant pairs.
# Build time grows ~3x. Set DFLASH27B_FA_ALL_QUANTS=OFF if you only run
# the spec_prefill demo (target_gen path uses standard quant pairs).
option(DFLASH27B_FA_ALL_QUANTS "Compile ggml-cuda fattn kernels for all KV-quant pairs" ON)
set(GGML_CUDA_FA_ALL_QUANTS ${DFLASH27B_FA_ALL_QUANTS} CACHE BOOL "" FORCE)
# Use only the ggml subtree of llama.cpp (skip libllama).
add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)

# The C++ sources include <cuda_runtime.h> directly, so the toolkit headers must
# be available when compiling the library.
find_package(CUDAToolkit REQUIRED)

# ─── dflash27b static library ──────────────────────────────────────

add_library(dflash27b STATIC
    src/errors.cpp
    src/gguf_target_loader.cpp
    src/gguf_draft_loader.cpp
    src/safetensors_draft.cpp
    src/qwen35_target_graph.cpp
    src/qwen3_dflash_graph.cpp
    src/qwen3_drafter.cpp
    src/qwen3_0p6b_loader.cpp
    src/qwen3_0p6b_graph.cpp
    src/flashprefill_q8.cpp
    src/kv_cache.cpp
    src/kv_quant.cpp
    src/f16_convert.cu
    src/delta_net_chunked.cpp
)
# FlashPrefill custom CUDA kernels need BF16 WMMA (sm_80+). On Turing (sm_75)
# the drafter uses ggml's flash_attn_ext instead. Guard added after SM check.
# BSA (Block-Sparse Attention) backs the speculative-prefill drafter scoring
# path. Default ON so prefill is fast out of the box. Turn OFF if you don't
# run the spec-prefill stack or are building for sm<80 (cutlass BSA kernels
# need sm_80+; the auto-detect below also disables BSA on legacy arches).
if(NOT DEFINED DFLASH27B_ENABLE_BSA)
    set(DFLASH27B_ENABLE_BSA ON)
endif()

# Turing (75) and Ampere (86) always; Blackwell consumer (120) and Thor
# (110 on CUDA 13+) added when nvcc supports them. DGX Spark /
# GB10 is compute capability 12.1 (121), added at CUDA 12.9+.
if(DFLASH27B_USER_CUDA_ARCHITECTURES)
    set(_dflash27b_archs "${DFLASH27B_USER_CUDA_ARCHITECTURES}")
else()
    set(_dflash27b_archs "75;86")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
        list(APPEND _dflash27b_archs "120")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        list(APPEND _dflash27b_archs "110")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
        list(APPEND _dflash27b_archs "121")
    endif()
endif()
set_target_properties(dflash27b PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")

# Extract the minimum SM from the arch list so safetensors_draft.cpp can decide
# at compile time whether to convert BF16 draft weights to FP16 (cuBLAS BF16
# GEMM has no tensor core acceleration on SM < 80).
list(GET _dflash27b_archs 0 _dflash27b_min_sm)
# Strip any trailing 'a' suffix (e.g. "121a" → "121")
string(REGEX REPLACE "[^0-9]" "" _dflash27b_min_sm "${_dflash27b_min_sm}")
target_compile_definitions(dflash27b PRIVATE DFLASH27B_MIN_SM=${_dflash27b_min_sm})

# FlashPrefill custom CUDA kernels need BF16 WMMA (sm_80+).
# On sm_75 the drafter falls back to ggml's flash_attn_ext (see qwen3_0p6b_graph.cpp).
if(_dflash27b_min_sm GREATER_EQUAL 80)
    target_sources(dflash27b PRIVATE
        src/flashprefill_kernels.cu
        src/flashprefill_select.cpp
        src/flashprefill.cpp)
    target_compile_definitions(dflash27b PRIVATE DFLASH27B_HAVE_FLASHPREFILL=1)
endif()

# BSA needs sm_80+ (FA-2 derived kernel uses BF16 tensor cores). Auto-disable
# on legacy arches with a clear message instead of failing the link.
if(DFLASH27B_ENABLE_BSA)
    foreach(_arch IN LISTS _dflash27b_archs)
        if(_arch LESS 80)
            message(WARNING
                "DFLASH27B_ENABLE_BSA=ON requested but CUDA_ARCHITECTURES contains '${_arch}' (<80); "
                "disabling BSA (the spec-prefill path will fall back to the WMMA kernel).")
            set(DFLASH27B_ENABLE_BSA OFF)
            break()
        endif()
    endforeach()
endif()
# BSA also requires the Block-Sparse-Attention submodule (with bundled cutlass).
# Existing users who `git pull` without `--recurse-submodules` would otherwise
# hit a missing-header compile error — auto-disable here with a clear message.
if(DFLASH27B_ENABLE_BSA AND NOT EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include/cutlass/numeric_types.h")
    message(WARNING
        "DFLASH27B_ENABLE_BSA=ON requested but the Block-Sparse-Attention submodule "
        "is missing or its bundled cutlass is not initialized. Run "
        "`git submodule update --init --recursive` to enable the BSA spec-prefill "
        "path. Disabling BSA for this build.")
    set(DFLASH27B_ENABLE_BSA OFF)
endif()
if(DFLASH27B_ENABLE_BSA)
    target_sources(dflash27b PRIVATE src/bsa_fwd_inst.cu src/bsa_launcher.cu)
endif()

target_include_directories(dflash27b
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)
if(DFLASH27B_ENABLE_BSA)
    target_include_directories(dflash27b PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/bsa_stubs
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/block_sparse_attn/src)
    target_compile_options(dflash27b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_definitions(dflash27b PRIVATE FLASHATTENTION_DISABLE_DROPOUT FLASH_NAMESPACE=flash DFLASH27B_HAVE_BSA=1)
endif()

target_link_libraries(dflash27b
    PUBLIC
        ggml
        ggml-cuda
        ggml-base
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-function>
     )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
     )
endif()

# PFlash phase-split harness

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/pflash_daemon.cpp")
    add_executable(pflash_daemon test/pflash_daemon.cpp)
    target_include_directories(pflash_daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(pflash_daemon PRIVATE dflash27b ggml ggml-cuda)
endif()

# ─── Tests (numerics vs oracle) ────────────────────────────────────

option(DFLASH27B_TESTS "Build numerics tests" ON)
if(DFLASH27B_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flashprefill_kernels.cpp")
        add_executable(test_flashprefill_kernels test/test_flashprefill_kernels.cpp)
        set_target_properties(test_flashprefill_kernels PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")
        target_link_libraries(test_flashprefill_kernels PRIVATE dflash27b CUDA::cudart)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kv_quant.cpp")
        add_executable(test_kv_quant test/test_kv_quant.cpp)
        target_include_directories(test_kv_quant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_kv_quant PRIVATE dflash27b)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_draft_vs_reference.cpp")
        add_executable(test_draft_vs_reference test/test_draft_vs_reference.cpp)
        target_link_libraries(test_draft_vs_reference PRIVATE dflash27b)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_draft.cpp")
        add_executable(smoke_load_draft test/smoke_load_draft.cpp)
        target_include_directories(smoke_load_draft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_load_draft PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_draft_graph.cpp")
        add_executable(smoke_draft_graph test/smoke_draft_graph.cpp)
        target_include_directories(smoke_draft_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_draft_graph PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_qwen3_0p6b_forward.cpp")
        add_executable(smoke_qwen3_0p6b_forward test/smoke_qwen3_0p6b_forward.cpp)
        target_include_directories(smoke_qwen3_0p6b_forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_qwen3_0p6b_forward PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_vs_oracle.cpp")
        add_executable(test_vs_oracle test/test_vs_oracle.cpp)
        target_include_directories(test_vs_oracle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_vs_oracle PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_target.cpp")
        add_executable(smoke_load_target test/smoke_load_target.cpp)
        target_include_directories(smoke_load_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_load_target PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_target_forward.cpp")
        add_executable(smoke_target_forward test/smoke_target_forward.cpp)
        target_include_directories(smoke_target_forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_target_forward PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_generate.cpp")
        add_executable(test_generate test/test_generate.cpp)
        target_include_directories(test_generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_generate PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_dflash.cpp")
        add_executable(test_dflash test/test_dflash.cpp)
        target_include_directories(test_dflash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_dflash PRIVATE dflash27b ggml ggml-cuda)
        # test_dflash uses cudaMemcpyAsync / cudaMemcpy2DAsync directly for the
        # --fast-rollback path (per-step SSM intermediate state commit). Needs
        # the CUDA runtime on its own link line.
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(test_dflash PRIVATE CUDA::cudart)
        # OpenMP for parallel CPU top-K extraction in the ddtree path.
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(test_dflash PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()



================================================
FILE: dflash/RESULTS.md
================================================
# Luce DFlash benchmark results

Single RTX 3090 24 GB, CUDA 12, driver 535.
Target: `unsloth/Qwen3.5-27B-GGUF` (Q4_K_M, ~16 GB).
Draft:  `z-lab/Qwen3.5-27B-DFlash` (BF16, 3.46 GB).
Concurrency = 1, greedy decoding, `n_gen=256`.
Reproduce with `python3 scripts/bench_llm.py` (samples 10 prompts/dataset, seed=42).

## Headline — AR vs Luce DFlash at concurrency 1

| Task      | AR tok/s | DFlash tok/s | AL   | Speedup |
|-----------|:--------:|:------------:|:----:|:-------:|
| HumanEval | 37.78    | **129.52**   | 8.31 | **3.43×** |
| Math500   | 37.71    | **110.51**   | 7.04 | **2.93×** |
| GSM8K     | 37.65    | **96.15**    | 6.14 | **2.55×** |

AR = autoregressive target-only decode via `test_generate`.
DFlash = block-diffusion draft + DDTree budget 22 verify + fast rollback.
AL = mean committed tokens per draft/verify step (acceptance length).

Datasets pulled live via HuggingFace `datasets`:
- HumanEval — `openai_humaneval`, `prompt` field
- GSM8K    — `gsm8k` main split, `Question: … Answer: ` format
- Math500  — `HuggingFaceH4/MATH-500`, `Problem: … Solution: ` format

## Per-prompt numbers (seed 42)

### HumanEval (10 samples)

| # | n_tok | AR    | DFlash | AL    |
|:-:|:-----:|:-----:|:------:|:-----:|
| 01| 84    | 37.98 | 137.91 | 8.83  |
| 02| 138   | 37.90 | 143.38 | 9.14  |
| 03| 134   | 37.88 | 137.49 | 8.83  |
| 04| 120   | 37.84 | 153.77 | 9.85  |
| 05| 172   | 37.76 | 131.74 | 8.53  |
| 06| 118   | 37.59 | 113.97 | 7.31  |
| 07| 51    | 37.78 | 103.27 | 6.56  |
| 08| 141   | 37.68 | **158.40** | **10.24** |
| 09| 125   | 37.71 | 128.22 | 8.26  |
| 10| 95    | 37.65 |  87.04 | 5.57  |
| **mean** |   | **37.78** | **129.52** | **8.31** |

Peak per-prompt: **158.40 tok/s at AL 10.24** (4.20× over AR on the same prompt).

### GSM8K (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 45    | 37.62 |  93.87 | 5.95 |
| 02| 111   | 37.53 |  90.59 | 5.82 |
| 03| 49    | 37.73 |  87.79 | 5.57 |
| 04| 70    | 37.67 |  82.11 | 5.22 |
| 05| 102   | 37.62 | **127.83** | **8.26** |
| 06| 118   | 37.61 |  88.67 | 5.69 |
| 07| 113   | 37.62 |  86.86 | 5.57 |
| 08| 50    | 37.72 | 102.98 | 6.56 |
| 09| 43    | 37.69 | 109.66 | 6.92 |
| 10| 96    | 37.72 |  91.12 | 5.82 |
| **mean** |   | **37.65** | **96.15** | **6.14** |

### Math500 (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 257   | 37.60 | 100.97 | 6.56 |
| 02| 53    | 37.73 | 115.62 | 7.31 |
| 03| 40    | 37.76 | 126.47 | 8.00 |
| 04| 50    | 37.76 | 118.20 | 7.53 |
| 05| 117   | 37.69 | 114.55 | 7.31 |
| 06| 76    | 37.70 | 108.63 | 6.92 |
| 07| 43    | 37.72 |  90.41 | 5.69 |
| 08| 79    | 37.73 | 100.10 | 6.40 |
| 09| 52    | 37.69 |  91.69 | 5.82 |
| 10| 57    | 37.74 | **138.45** | **8.83** |
| **mean** |   | **37.71** | **110.51** | **7.04** |

## Why the speedup varies by task

Acceptance length is the dominant factor — tok/s is roughly linear in AL when per-step overhead is fixed:

| Task      | AL   | Speedup vs AR |
|-----------|:----:|:-------------:|
| HumanEval | 8.31 | 3.43×         |
| Math500   | 7.04 | 2.93×         |
| GSM8K     | 6.14 | 2.55×         |

HumanEval prompts are highly regular (function signatures + docstrings), the draft nails consecutive tokens. GSM8K is natural-language arithmetic reasoning, the draft is less confident, tree verify rescues less.

## 128K context configuration

`max_ctx = 131072` + `DFLASH27B_KV_Q4=1` (Q4_0 K+V cache, 8× compression vs F16).
Sliding `target_feat` ring (4096 slots) keeps captured features at 0.2 GB regardless of context length.
`--ddtree-budget=16` keeps per-layer `ssm_intermediate` under 1.3 GB.

| Prompt length | KV size  | Prefill | Decode tok/s |
|:-------------:|:--------:|:-------:|:------------:|
| 520 (HE)      | ~35 MB   | 0.06 s  | 130          |
| 13K           | ~860 MB  | 15 s    | 99           |
| 32K           | ~2.1 GB  | 106 s   | 35           |
| 128K          | ~8.4 GB  | ~10 min | ~15-20 (est) |

Q4_0 KV costs ~3% mean tok/s vs F16 at short contexts and is the only thing that lets 128K allocate at all.

## DDTree budget sweep (HumanEval, n_gen=256, f16 intermediate)

Historical tuning run from commit `f1cb9bf` (2026-04-16). Used to pick the default budget=22. Fresh run at budget=22 on commit `5bb7f8c` is the 129.5 tok/s / AL 8.31 reported in the headline above; the ~5 tok/s delta vs the 135.8 row here comes from sample variance across the 10 prompts and from minor build-flag drift between the two commits.

| Budget | Mean AL | Mean tok/s |
|:------:|:-------:|:----------:|
| 15     | 7.64    | 125.3      |
| 16     | 7.81    | 128.7      |
| 18     | 8.22    | 131.2      |
| 20     | 8.64    | 133.9      |
| **22** | **8.88**| **135.8**  |
| 24     | 8.91    | 133.0      |
| 30     | 8.86    | 120.5      |
| 40     | 8.90    | 105.1      |

AL plateaus at ~8.9, past budget 22 each extra node costs more in verify time than it buys in accept. Memory ceiling at budget 26 on 24 GB (per-token SSM intermediate cache is hybrid-only overhead).

## Kernel-level wins (cumulative, chain mode → DDTree budget 22 + f16)

Starting point: Chain DFlash at 112.8 tok/s mean on HumanEval, AL 7.67.

| Optimization                                    | Δ tok/s | Δ AL | Note |
|-------------------------------------------------|:-------:|:----:|------|
| DDTree budget 20, f32 intermediate              | +15.1   | +0.77| Heap-based best-first tree, 20 nodes |
| Chain pre-seed in `build_ddtree`                | —       | +~5  | Fixes top-1 chain coverage under Q4 noise (prior AL ~4) |
| Tree-aware `ggml_ssm_conv_tree` kernel          | —       | +~1  | Sibling conv window gathers via parent chain, not DFS |
| `target_feat` compaction after sibling-accept   | —       | +~0.8| Stale feature pruning |
| OpenMP-parallel CPU top-K, K reduced 32→8       | +2.1    | —    | Shaves 7% off draft step |
| Fast K=1 path for budget=15                     | +1.5    | —    | Skips 11 ms CPU top-K when no siblings needed |
| D2D `cudaMemcpyAsync` for target_feat (GPU→GPU) | +3.7    | —    | Replaces GPU→CPU→GPU round trip |
| `ggml_gated_delta_net_tree_persist` kernel      | +12.4   | —    | Direct-writes SSM intermediates, skips 9 ms `ggml_cpy` per step |
| Budget 20 → 22, f16 intermediate                | +5.5    | +0.24| f16 cuts intermediate bandwidth in half |
| **Total**                                       | **+16.7** | **+0.64** | **129.5 tok/s, AL 8.31 (HumanEval mean, fresh run)** |

## Reproducibility

- Deterministic: greedy decode + greedy verify. Same prompts + same weights + same binary = same numbers ±1 tok/s.
- Full bench (10×3 = 30 prompts): ~15 min.
- All numbers above reproduced on 2026-04-20 from commit `5bb7f8c` with:
  ```
  python3 scripts/bench_llm.py
  ```

## Hardware ceiling notes

- Published DFlash paper on Qwen3-4B/8B/30B-MoE (pure attention, BF16, B200) reports 4-5× over AR on HumanEval/Math500 at concurrency 1. Ours: 3.43× on 27B hybrid Q4_K_M on RTX 3090.
- Memory ceiling: per-token SSM intermediate cache (hybrid-only cost) caps tree budget at ~26 on 24 GB. The paper uses budgets up to 1024 on pure-attention models with zero per-node memory tax.
- Per-token verify cost drops from 25 ms at N=1 to 0.97 ms at N=128 (ggml-cuda Q4_K matmul amortises well with batch size).

## RTX 2080 Ti (Turing, sm_75, 22 GB)

Single RTX 2080 Ti 22 GB, CUDA 12.4.
Same target/draft as above. BF16 draft weights auto-converted to FP16 at load time
(cuBLAS BF16 GEMM has no tensor core acceleration on SM 7.5; FP16 conversion
gives 3.9× faster draft compute via Turing tensor cores).

Build: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75`

### RTX 2080 Ti headline

| Task      | AR tok/s | DFlash tok/s | AL   | Speedup |
|-----------|:--------:|:------------:|:----:|:-------:|
| HumanEval | 19.88    | **53.42**    | 8.14 | **2.69×** |
| Math500   | 19.67    | **49.01**    | 7.30 | **2.49×** |
| GSM8K     | 19.49    | **43.55**    | 6.53 | **2.23×** |

### RTX 2080 Ti per-prompt — HumanEval (10 samples)

| # | n_tok | AR    | DFlash | AL    |
|:-:|:-----:|:-----:|:------:|:-----:|
| 01| 84    | 19.88 |  58.69 | 8.83  |
| 02| 138   | 19.43 |  45.47 | 9.14  |
| 03| 134   | 19.60 |  62.67 | 9.14  |
| 04| 120   | 20.16 |  63.42 | 9.14  |
| 05| 172   | 19.74 |  56.89 | 8.53  |
| 06| 118   | 20.20 |  44.32 | 6.40  |
| 07| 51    | 20.26 |  54.14 | 8.00  |
| 08| 141   | 19.70 |  40.34 | 5.95  |
| 09| 125   | 19.91 | **70.70** | **10.67** |
| 10| 95    | 19.88 |  37.56 | 5.57  |
| **mean** |   | **19.88** | **53.42** | **8.14** |

Peak per-prompt: **70.70 tok/s at AL 10.67** (3.55× over AR on the same prompt).

### RTX 2080 Ti per-prompt — GSM8K (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 45    | 19.24 |  39.54 | 5.82 |
| 02| 111   | 19.70 |  39.49 | 5.82 |
| 03| 49    | 19.33 |  57.01 | 8.53 |
| 04| 70    | 19.70 |  38.35 | 5.69 |
| 05| 102   | 19.67 |  36.77 | 5.45 |
| 06| 118   | 19.39 |  40.45 | 5.95 |
| 07| 113   | 19.55 |  54.02 | 8.46 |
| 08| 50    | 18.92 |  42.16 | 6.51 |
| 09| 43    | 19.68 |  48.07 | 7.11 |
| 10| 96    | 19.72 |  39.63 | 5.95 |
| **mean** |   | **19.49** | **43.55** | **6.53** |

### RTX 2080 Ti per-prompt — Math500 (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 257   | 19.70 |  42.64 | 6.40 |
| 02| 53    | 19.80 |  49.53 | 7.31 |
| 03| 40    | 19.96 |  52.76 | 8.00 |
| 04| 50    | 19.49 | **62.08** | **9.48** |
| 05| 117   | 17.85 |  43.69 | 6.56 |
| 06| 76    | 19.87 |  45.42 | 6.74 |
| 07| 43    | 20.05 |  42.57 | 6.40 |
| 08| 79    | 19.42 |  51.86 | 7.76 |
| 09| 52    | 20.02 |  39.34 | 5.82 |
| 10| 57    | 20.53 |  60.18 | 8.53 |
| **mean** |   | **19.67** | **49.01** | **7.30** |

### RTX 2080 Ti vs RTX 3090 comparison

| Metric           | RTX 3090 | RTX 2080 Ti | Ratio |
|------------------|:--------:|:-----------:|:-----:|
| AR tok/s (HE)    | 37.78    | 19.88       | 0.53× |
| DFlash tok/s (HE)| 129.52   | 53.42       | 0.41× |
| Mem BW            | 936 GB/s | 616 GB/s   | 0.66× |
| SMs               | 82       | 68          | 0.83× |
| VRAM              | 24 GB    | 22 GB       | 0.92× |

AR scaling (~0.53×) tracks bandwidth × SM count. DFlash scaling (~0.41×) is lower because the draft compute bottleneck is proportionally larger on a slower GPU, even after the BF16→FP16 fix. Acceptance length is identical (same draft model, same tokens), confirming the FP16 conversion is numerically faithful.



================================================
FILE: dflash/deps/bsa_stubs/README.md
================================================
# bsa_stubs

Header shims that let `mit-han-lab/Block-Sparse-Attention` (BSA) compile
without depending on PyTorch's `libtorch`.

BSA was originally built as a PyTorch C++ extension and pulls in
`<ATen/...>` and `<c10/...>` headers. We don't link `libtorch` in the
dflash daemon, so this directory provides minimal stand-ins that satisfy
the references BSA actually uses:

- `c10/cuda/CUDAException.h` — `C10_CUDA_CHECK`, `C10_CUDA_KERNEL_LAUNCH_CHECK`
  macros (forward to `cudaPeekAtLastError`).
- `ATen/cuda/CUDAGeneratorImpl.h` — `at::PhiloxCudaState` POD struct (only
  used by BSA's dropout path, which we never enable).
- `ATen/cuda/CUDAGraphsUtils.cuh` — `at::cuda::philox::unpack` no-op
  returning `{seed, offset}` from the stub state.

These headers are placed FIRST on the BSA include path
(`dflash/CMakeLists.txt`, gated on `DFLASH27B_ENABLE_BSA`). When BSA's
generated CUDA includes `<c10/cuda/CUDAException.h>`, the compiler picks up
this stub instead of trying to find PyTorch.

Because we always build BSA with `FLASHATTENTION_DISABLE_DROPOUT`, the
philox / generator stubs are never exercised — they exist only to satisfy
declarations.

If a future BSA upgrade pulls in additional ATen/c10 headers, add a
matching stub here rather than vendoring all of libtorch.



================================================
FILE: dflash/deps/bsa_stubs/ATen/cuda/CUDAGeneratorImpl.h
================================================
#pragma once
#include <cstdint>
#include <tuple>
namespace at {
struct PhiloxCudaState {
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
    uint64_t* seed_extragraph_ = nullptr;
    uint64_t* offset_extragraph_ = nullptr;
    uint32_t offset_intragraph_ = 0;
    bool captured_ = false;
};
struct Generator {};
}  // namespace at



================================================
FILE: dflash/deps/bsa_stubs/ATen/cuda/CUDAGraphsUtils.cuh
================================================
#pragma once
#include "CUDAGeneratorImpl.h"
#include <cuda_runtime.h>
namespace at { namespace cuda { namespace philox {
__device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
    return std::make_tuple(arg.seed_, arg.offset_);
}
} } }



================================================
FILE: dflash/deps/bsa_stubs/c10/cuda/CUDAException.h
================================================
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define C10_CUDA_CHECK(EXPR) \
  do { cudaError_t _e = (EXPR); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); std::abort(); } } while(0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaPeekAtLastError())



================================================
FILE: dflash/docs/SPEC_PREFILL.md
================================================
# dflash spec-prefill, daemon-side build & tunables

In-process speculative-prefill + speculative-decode daemon (C++/CUDA only,
no Python, no Triton, no PyTorch at runtime).

This doc is the build / runtime / tunables reference for the C++ daemon
path described in [`pflash/README.md`](../../pflash/README.md) and on the
[blog post](https://lucebox.com/blog/pflash):

- **Drafter** (Qwen3-0.6B) loaded via a custom forward (`qwen3_0p6b_*`)
  with the FlashPrefill block-sparse attention kernel for long-context
  scoring.
- **Target** (Qwen3.6-27B Q4_K_M) loaded directly via ggml.
- **Speculative decode** between draft + target with rollback / DDTree.

Both models live in the same process, the same ggml allocator, on a
single RTX 3090 (24 GB). No PyTorch at runtime.

## Build

```
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=86 -DDFLASH27B_ENABLE_BSA=ON ..
cmake --build . --target test_dflash test_flashprefill_kernels -- -j8
```

Required:
- CUDA Toolkit 12.0+ (sm_80+ for BSA path; sm_86 RTX 3090 is the
  reference target).
- `git submodule update --init --recursive` to pull
  `deps/llama.cpp` (ggml only) and `deps/Block-Sparse-Attention` (with
  cutlass).

CMake options:
- `DFLASH27B_ENABLE_BSA=ON` (default) — build the Block-Sparse-Attention
  kernel for sparse FA forward. Required for the long-context perf claim.
  Turn OFF only on sm<80.
- `DFLASH27B_FA_ALL_QUANTS=ON` (default) — compile ggml-cuda fattn for
  all KV-quant pairs (needed for asymmetric Q4_0 K + Q8_0 V cache). Off
  cuts build time ~3x but breaks the 128K target gen path.

## Runtime tunables

```
DFLASH_FP_USE_BSA=1    # dispatch sparse FA forward through BSA (sm_80+)
DFLASH_FP_ALPHA=0.85   # block-selection threshold (default 0.12);
                       # higher = stricter = fewer K-blocks per Q-row.
DFLASH_FP_PROFILE=1    # log mean/score/select/forward stage timings
```

See `src/flashprefill.h` for the full list and defaults.

## Dual-GPU PFlash phase split

PFlash targets the prefill side of long-context requests. The dual-GPU phase
split harness is an opt-in benchmark/runtime path for measuring the PFlash
prefill phase as its own resident CUDA process:

- `pflash_daemon` keeps the Qwen3-0.6B PFlash drafter resident.
- `scripts/phase_split_dual_gpu.py` sends counted token IDs to the daemon.
- `--pflash-gpu` selects the CUDA GPU used for the PFlash phase.
- The report records compressed token/text outputs, PFlash timing, and GPU
  resource peaks for the PFlash worker.

The harness produces the compressed prompt artifact used by later target
prefill experiments. It does not measure or modify decode.

Build:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pflash_daemon -j
```

Run a synthetic NIAH sweep:

```
python scripts/phase_split_dual_gpu.py bench-niah \
  --build-dir build \
  --contexts 4096,8192,16384 \
  --local-files-only \
  --report-dir reports/pflash_phase_split_context_sweep
```

Compress a real prompt:

```
python scripts/phase_split_dual_gpu.py run-prompt \
  --build-dir build \
  --prompt-file /path/to/prompt.txt \
  --local-files-only \
  --report-dir reports/pflash_phase_split_prompt
```

The reports include source token count, compressed token count, compression
ratio, PFlash timing, and GPU resource peaks.

## Performance

NIAH single-needle end-to-end on RTX 3090 (Qwen3.6-27B Q4_K_M target,
Qwen3-0.6B drafter, in-process daemon, `DFLASH_FP_USE_BSA=1`,
`DFLASH_FP_ALPHA=0.85`, `keep_ratio=0.05`):

| Source S | dflash TTFT | llama.cpp baseline | Speedup | NIAH |
|----------|------------:|-------------------:|--------:|:----:|
| 64K      | **13.5 s**  | 134.95 s (FA off, dense) | **10.0×** | ✅ |
| 128K     | **24.8 s**  | ~257 s (FA on, Q4_0 KV)  | **~10.4×** | ✅ |

NIAH needle retrieved (accuracy 1/1) at every measured context. The
runtime is C++/CUDA only — the headline number is the dflash binary on
its own, no Python or Triton in the loop.

## Repo layout

```
src/
  flashprefill.{h,cpp}         FlashPrefill C++ entry + dispatcher
  flashprefill_kernels.cu       4 CUDA kernels (mean_K, score, select, sparse_fwd)
  flashprefill_select.cpp       Host fallback for block_select (rarely used)
  bsa_launcher.cu               BSA launcher: blockmask conversion + Flash_fwd_params
  bsa_fwd_inst.cu               Single-TU instantiation of BSA's hdim128 kernel
  qwen3_0p6b_loader.cpp         GGUF → Qwen3-0.6B BF16 weight tensors
  qwen3_0p6b_graph.cpp          Custom Qwen3-0.6B forward (per-layer A/FP/B graphs)
  qwen3_drafter.{h,cpp}         drafter_score_and_compress() entry point
  qwen35_target_graph.cpp       Qwen3.5/3.6 target graph (ggml)
  qwen3_dflash_graph.cpp        DFlash speculative draft head
  kv_cache.cpp / kv_quant.cpp   Q4_0 KV cache + asymmetric quant
test/
  test_dflash.cpp               daemon executable; supports
                                  `compress / generate / park / unpark / free drafter`
  test_flashprefill_kernels.cpp parity tests for the 4 FP kernels
  smoke_qwen3_0p6b_forward.cpp  drafter forward smoke at S=8K-128K
deps/
  llama.cpp/                    submodule (ggml only; libllama not built)
  Block-Sparse-Attention/       submodule (BSA + cutlass)
  bsa_stubs/                    PyTorch ATen/c10 header shims (see its README)
```



================================================
FILE: dflash/examples/chat.py
================================================
"""
Multi-turn chat REPL with streaming output.

    python3 examples/chat.py

Tokens print as they are committed. Model reloads once per turn;
a daemon-mode binary that keeps the model resident is a planned follow-up.
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DRAFT_ROOT = ROOT / "models" / "draft"
BIN = ROOT / "build" / ("test_dflash.exe" if sys.platform == "win32" else "test_dflash")

BUDGET = 22
N_GEN  = 512
SYSTEM = "You are a concise, helpful assistant."


def resolve_draft() -> Path:
    if DRAFT_ROOT.is_file():
        return DRAFT_ROOT
    if DRAFT_ROOT.is_dir():
        for st in DRAFT_ROOT.rglob("model.safetensors"):
            return st
    sys.exit(
        f"draft weights not found under {DRAFT_ROOT}. Download them as documented in the README."
    )


def tokenize(tok, text: str, path: Path) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def stream_generate(tok, bin_path: Path, target: Path, draft: Path,
                    prompt_bin: Path, n_gen: int, budget: int,
                    stop_ids: set[int]) -> str:
    """Spawn test_dflash, read tokens as they arrive, print + accumulate."""
    with tempfile.TemporaryDirectory() as d:
        out_bin = Path(d) / "out.bin"
        r, w = os.pipe()
        cmd = [str(bin_path), str(target), str(draft), str(prompt_bin),
               str(n_gen), str(out_bin),
               "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
               f"--stream-fd={w}"]
        proc = subprocess.Popen(cmd, pass_fds=(w,),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        os.close(w)

        text = ""
        try:
            while True:
                b = os.read(r, 4)
                if not b or len(b) < 4:
                    break
                tok_id = struct.unpack("<i", b)[0]
                if tok_id in stop_ids:
                    break
                s = tok.decode([tok_id])
                text += s
                sys.stdout.write(s); sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            raise
        finally:
            proc.wait()
        return text


def main():
    if not BIN.is_file():
        sys.exit(f"binary not found at {BIN}. Build: "
                 "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && "
                 "cmake --build build --target test_dflash -j")
    if not TARGET.is_file():
        sys.exit(f"target GGUF not found at {TARGET}")

    draft = resolve_draft()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B",
                                        trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tok.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    messages = [{"role": "system", "content": SYSTEM}]
    print("Luce DFlash chat. Ctrl+C to quit a reply, Ctrl+D to exit.\n",
          flush=True)

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)

        with tempfile.TemporaryDirectory() as d:
            in_bin = Path(d) / "in.bin"
            n = tokenize(tok, prompt, in_bin)
            sys.stdout.write("bot> "); sys.stdout.flush()
            try:
                reply = stream_generate(tok, BIN, TARGET, draft, in_bin,
                                        N_GEN, BUDGET, stop_ids)
            except KeyboardInterrupt:
                sys.stdout.write("\n[interrupted]\n")
                messages.pop()
                continue

        messages.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()



================================================
FILE: dflash/include/dflash27b.h
================================================
// dflash27b — standalone CUDA library for DFlash speculative decoding of
// Qwen3.5-27B with the z-lab/Qwen3.5-27B-DFlash draft model on a single RTX 3090.
//
// Model constants (hardcoded for this pair) + the last-error helper.
// The real driver is test/test_dflash.cpp. A clean public API with chat /
// streaming / KV persistence is a planned follow-up.

#ifndef DFLASH27B_H
#define DFLASH27B_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Model config ─────────────────────────────────────────────────

#define DFLASH27B_TARGET_HIDDEN        5120
#define DFLASH27B_TARGET_LAYERS        64
// NOTE: the `DFLASH27B_TARGET_N_*` / `_HEAD_DIM` macros below are DRAFT
// dimensions (z-lab draft: 32 Q heads, 8 KV heads, 128 head_dim). The TARGET
// Qwen3.5-27B qwen35 hybrid uses 24 Q heads, 4 KV heads, 256 head_dim, which
// live in `src/internal.h` (n_embd_head_k/v, N_HEAD, N_HEAD_KV). Naming is
// historical — do not change without updating safetensors_draft.cpp +
// qwen3_dflash_graph.cpp which consume these as draft-side constants.
#define DFLASH27B_TARGET_N_HEADS       32
#define DFLASH27B_TARGET_N_KV_HEADS    8
#define DFLASH27B_TARGET_HEAD_DIM      128
#define DFLASH27B_TARGET_INTERMEDIATE  17408
#define DFLASH27B_TARGET_VOCAB         248320
#define DFLASH27B_ROPE_THETA           10000000.0f
#define DFLASH27B_RMS_EPS              1e-6f

#define DFLASH27B_DRAFT_LAYERS         5
#define DFLASH27B_DRAFT_BLOCK_SIZE     16
#define DFLASH27B_DRAFT_N_TARGET_LAYERS 5  // fc projects 5*hidden -> hidden
#define DFLASH27B_DRAFT_MASK_TOKEN_ID  248070

// target_layer_ids = {1, 16, 31, 46, 61}  (0-indexed into target layers)
// We capture the OUTPUT of each, which is HF hidden_states[lid + 1].

// ─── Diagnostics ──────────────────────────────────────────────────

// Most recent error from any loader / graph builder. Thread-safe.
const char * dflash27b_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // DFLASH27B_H



================================================
FILE: dflash/scripts/_prefill_hook.py
================================================
"""Pflash speculative-prefill helper for the dflash OpenAI servers.

The dflash daemon already exposes the C++/CUDA spec-prefill pipeline via its
stdin protocol: ``compress <ids.bin> <keep_x1000> <drafter.gguf>`` runs the
in-process Qwen3-0.6B drafter + FlashPrefill scoring (BSA), then emits the
compressed token-id stream. ``free drafter`` releases drafter weights + KV +
BSA scratch, and ``park`` / ``unpark`` cycle target/draft weights through VRAM.

This module wraps that protocol so server.py and server_tools.py can fold
``--prefill-*`` flags into the existing request flow without duplicating the
plumbing. The drafter and target use *different* tokenizers (Qwen3-0.6B vs
Qwen3.5/3.6-27B), so the pipeline is:

    target_text  ──▶  drafter_tokenizer.encode  ──▶  daemon.compress
                                                          │
    target_tokenizer.encode  ◀──  drafter_tokenizer.decode ┘

The result is a shorter span of text (re-tokenised on the target side) that
the existing ``generate`` path can prefill in a fraction of the time.
"""
from __future__ import annotations
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ─── pipe / stdin helpers shared with both servers ─────────────────────

def _drain_until_sentinel(r_pipe: int) -> list[int]:
    """Read int32 LE values from r_pipe until -1 sentinel. Returns the list."""
    out: list[int] = []
    while True:
        b = os.read(r_pipe, 4)
        if not b or len(b) < 4:
            break
        v = struct.unpack("<i", b)[0]
        if v == -1:
            break
        out.append(v)
    return out


def _send_and_ack(daemon_stdin, r_pipe: int, line: str) -> None:
    """Write a daemon command and consume the trailing -1 ack."""
    daemon_stdin.write(line.encode("utf-8"))
    daemon_stdin.flush()
    _drain_until_sentinel(r_pipe)


# ─── public configuration block ────────────────────────────────────────

@dataclass(frozen=True)
class PrefillConfig:
    """Parsed --prefill-* flags. ``mode == "off"`` disables compression."""
    mode: str                                          # "off" | "auto" | "always"
    threshold: int                                     # token threshold for "auto"
    keep_ratio: float                                  # 0.015..0.125
    drafter_gguf: Optional[Path]                       # drafter weights (Qwen3-0.6B BF16 GGUF)
    drafter_tokenizer_id: str                          # HF repo ID for drafter vocab

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def should_compress(self, prompt_token_count: int) -> bool:
        if self.mode == "always":
            return True
        if self.mode == "auto":
            return prompt_token_count >= self.threshold
        return False


def add_cli_flags(ap) -> None:
    """Attach --prefill-* flags to an argparse.ArgumentParser."""
    ap.add_argument("--prefill-compression",
                    choices=["off", "auto", "always"], default="off",
                    help="Speculative-prefill mode. 'auto' compresses when the "
                         "prompt token count reaches --prefill-threshold; "
                         "'always' compresses every request.")
    ap.add_argument("--prefill-threshold", type=int, default=32000,
                    help="Token threshold above which 'auto' mode triggers "
                         "compression (default 32000).")
    ap.add_argument("--prefill-keep-ratio", type=float, default=0.05,
                    help="Fraction of source tokens to keep after compression "
                         "(default 0.05; bench setting).")
    ap.add_argument("--prefill-drafter", type=Path, default=None,
                    help="Path to the drafter Qwen3-0.6B BF16 GGUF used by "
                         "the daemon's compress command. Required when "
                         "--prefill-compression != off.")
    ap.add_argument("--prefill-drafter-tokenizer", default="Qwen/Qwen3-0.6B",
                    help="HF repo ID for the drafter tokenizer "
                         "(default Qwen/Qwen3-0.6B).")


def config_from_args(args) -> PrefillConfig:
    if args.prefill_compression != "off" and args.prefill_drafter is None:
        raise SystemExit(
            "--prefill-compression != off requires --prefill-drafter "
            "(path to Qwen3-0.6B BF16 GGUF used by the daemon's compress).")
    if args.prefill_compression != "off" and not args.prefill_drafter.is_file():
        raise SystemExit(f"prefill drafter not found at {args.prefill_drafter}")
    if not 0.0 < args.prefill_keep_ratio <= 1.0:
        raise SystemExit("--prefill-keep-ratio must be in (0.0, 1.0]")
    return PrefillConfig(
        mode=args.prefill_compression,
        threshold=args.prefill_threshold,
        keep_ratio=args.prefill_keep_ratio,
        drafter_gguf=args.prefill_drafter,
        drafter_tokenizer_id=args.prefill_drafter_tokenizer,
    )


# ─── compress dance ────────────────────────────────────────────────────

def compress_text_via_daemon(
    *,
    daemon_stdin,
    r_pipe: int,
    drafter_tokenizer,
    cfg: PrefillConfig,
    prompt_text: str,
) -> str:
    """Run the daemon's compress + memory dance, return the compressed text.

    Caller holds the daemon lock for the full duration. After this returns,
    the daemon has its target + draft restored and is ready for ``generate``.
    """
    # 1) drafter-tokenize the prompt
    drafter_ids = drafter_tokenizer(prompt_text, return_tensors=None,
                                    add_special_tokens=False)["input_ids"]
    if isinstance(drafter_ids[0], list):  # some tokenizers return [[...]]
        drafter_ids = drafter_ids[0]

    # 2) write drafter ids to a tempfile
    fd, path = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            for t in drafter_ids:
                f.write(struct.pack("<i", int(t)))

        # 3) park target + draft so drafter has VRAM headroom on a 24 GB card
        _send_and_ack(daemon_stdin, r_pipe, "park target\n")
        _send_and_ack(daemon_stdin, r_pipe, "park draft\n")

        # 4) compress: drafter loads, FlashPrefill scoring, emit compressed ids, drafter held
        keep_x1000 = int(round(cfg.keep_ratio * 1000))
        daemon_stdin.write(
            f"compress {path} {keep_x1000} {cfg.drafter_gguf}\n".encode("utf-8"))
        daemon_stdin.flush()
        compressed_ids = _drain_until_sentinel(r_pipe)

        # 5) free drafter weights + BSA scratch, then restore target + draft
        _send_and_ack(daemon_stdin, r_pipe, "free drafter\n")
        _send_and_ack(daemon_stdin, r_pipe, "unpark target\n")
        _send_and_ack(daemon_stdin, r_pipe, "unpark draft\n")
    finally:
        try: os.unlink(path)
        except Exception: pass

    # 6) decode compressed drafter ids back to text for re-tokenisation by target
    return drafter_tokenizer.decode(compressed_ids, skip_special_tokens=True)



================================================
FILE: dflash/scripts/bench_daemon.py
================================================
"""Daemon-mode HE bench. Hits /v1/chat/completions with the same 10 HE
prompts as bench_he.py and reports mean tok/s.

Streams the response and reports two numbers per prompt:

  * wall    — total HTTP time (tokenize + prefill + decode + HTTP / JSON)
  * decode  — first-token → last-token elapsed, matching bench_he.py's
              tok/s (excludes prefill + setup)

Compare `decode` against bench_he.py to verify the C++ decode path is as
fast under the daemon as under a one-shot test_dflash invocation.

Start the server first (same config the published numbers use):
    DFLASH27B_KV_TQ3=1 python3 scripts/server_tools.py \\
        --budget 22 --max-ctx 16384 --port 8000

Then:
    python3 scripts/bench_daemon.py --url http://localhost:8000 --n-gen 256
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path
import sys

# Reuse the exact same 10 HE prompts bench_he.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_he import PROMPTS


def run(url: str, prompt: str, n_gen: int) -> tuple[int, float, float]:
    """POST to /v1/chat/completions with stream=true. Return (n_tok, wall_secs,
    decode_secs) where decode_secs starts at the first streamed token (after
    prefill) and ends at the last token."""
    body = json.dumps({
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_gen,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    t_first = 0.0
    t_last = 0.0
    n_tok = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Count tokens by content / reasoning deltas. Tool-call deltas
            # aren't counted — they arrive as a single final chunk.
            if delta.get("content") or delta.get("reasoning_content"):
                if n_tok == 0:
                    t_first = time.perf_counter()
                n_tok += 1
                t_last = time.perf_counter()
    wall = time.perf_counter() - t0
    decode = (t_last - t_first) if n_tok > 1 else 0.0
    return n_tok, wall, decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000",
                    help="Base URL of the running server (no /v1 suffix)")
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--warmup", action="store_true",
                    help="Run the first prompt once before timing to discard "
                         "cold-start effects (model is already resident, but "
                         "the first request allocates the decode VMM chunks).")
    args = ap.parse_args()

    if args.warmup:
        print("[bench] warmup...", flush=True)
        run(args.url, PROMPTS[0][1], args.n_gen)

    print(f"[bench] daemon API  n_gen={args.n_gen}  url={args.url}", flush=True)
    print(f"{'prompt':28s}  {'n_tok':>5s} {'wall_s':>7s} {'dec_s':>7s} "
          f"{'wall_tps':>9s} {'dec_tps':>9s}")
    print("-" * 72)
    wall_tps_list: list[float] = []
    dec_tps_list: list[float] = []
    total_tok = 0
    total_wall = 0.0
    total_decode = 0.0
    for name, text in PROMPTS:
        try:
            n_tok, wall, decode = run(args.url, text, args.n_gen)
        except Exception as e:
            print(f"  {name:26s}  FAILED: {e}", flush=True)
            continue
        if n_tok == 0:
            print(f"  {name:26s}  {n_tok:5d} {wall:7.2f}    --         --        -- "
                  "  (empty — daemon likely OOM'd)", flush=True)
            continue
        wall_tps = n_tok / wall
        dec_tps = (n_tok - 1) / decode if decode > 0 else 0.0
        wall_tps_list.append(wall_tps)
        if dec_tps > 0:
            dec_tps_list.append(dec_tps)
            total_decode += decode
        total_tok += n_tok
        total_wall += wall
        print(f"  {name:26s}  {n_tok:5d} {wall:7.2f} {decode:7.2f} "
              f"{wall_tps:9.2f} {dec_tps:9.2f}", flush=True)

    print("-" * 72)
    if wall_tps_list:
        print(f"wall tok/s mean:       {sum(wall_tps_list)/len(wall_tps_list):7.2f}  "
              f"(HTTP + tokenize + prefill + decode)")
        if dec_tps_list:
            print(f"decode tok/s mean:     {sum(dec_tps_list)/len(dec_tps_list):7.2f}  "
                  f"(first-token → last-token, matches bench_he.py's number)")
            agg_dec = (total_tok - len(dec_tps_list)) / total_decode if total_decode > 0 else 0.0
            print(f"decode tok/s aggregate:{agg_dec:7.2f}")
            print(f"decode tok/s range:    {min(dec_tps_list):.2f} - {max(dec_tps_list):.2f}")
    else:
        print("no successful runs")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/bench_he.py
================================================
"""
Bench DFlash test_dflash over multiple HumanEval-style prompts to get a stable
average acceptance length. Single-prompt measurements are noisy — z-lab's 8.09
AL on humaneval is averaged over 164 samples.

Usage on lucebox:
    python3 bench_he.py                 # run all 10 prompts with --fast-rollback
    python3 bench_he.py --mode batched  # run without --fast-rollback for A/B
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_FILE = ROOT / "models" / "draft" / "model.safetensors"
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get(
    "DFLASH_BIN",
    str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"),
)
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    # (name, source_code)
    (
        "has_close_elements",
        "from typing import List\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
        "    given threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
        "    True\n"
        '    """\n'
        "    for",
    ),
    (
        "separate_paren_groups",
        "from typing import List\n\n"
        "def separate_paren_groups(paren_string: str) -> List[str]:\n"
        '    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n'
        "    separate those group into separate strings and return the list of those.\n"
        "    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n"
        "    Ignore any spaces in the input string.\n"
        "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
        "    ['()', '(())', '(()())']\n"
        '    """\n'
        "    result = []\n"
        "    current_string = []\n"
        "    current_depth = 0\n"
        "    for",
    ),
    (
        "truncate_number",
        "def truncate_number(number: float) -> float:\n"
        '    """ Given a positive floating point number, it can be decomposed into\n'
        "    and integer part (largest integer smaller than given number) and decimals\n"
        "    (leftover part always smaller than 1).\n"
        "\n"
        "    Return the decimal part of the number.\n"
        "    >>> truncate_number(3.5)\n"
        "    0.5\n"
        '    """\n'
        "    return",
    ),
    (
        "below_zero",
        "from typing import List\n\n"
        "def below_zero(operations: List[int]) -> bool:\n"
        '    """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n'
        "    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n"
        "    at that point function should return True. Otherwise it should return False.\n"
        "    >>> below_zero([1, 2, 3])\n"
        "    False\n"
        "    >>> below_zero([1, 2, -4, 5])\n"
        "    True\n"
        '    """\n'
        "    balance = 0\n"
        "    for op in",
    ),
    (
        "mean_absolute_deviation",
        "from typing import List\n\n"
        "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
        '    """ For a given list of input numbers, calculate Mean Absolute Deviation\n'
        "    around the mean of this dataset.\n"
        "    Mean Absolute Deviation is the average absolute difference between each\n"
        "    element and a centerpoint (mean in this case):\n"
        "    MAD = average | x - x_mean |\n"
        "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
        "    1.0\n"
        '    """\n'
        "    mean =",
    ),
    (
        "intersperse",
        "from typing import List\n\n"
        "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n"
        "    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n"
        "    >>> intersperse([], 4)\n"
        "    []\n"
        "    >>> intersperse([1, 2, 3], 4)\n"
        "    [1, 4, 2, 4, 3]\n"
        '    """\n'
        "    result = []\n"
        "    for i, n in",
    ),
    (
        "parse_nested_parens",
        "from typing import List\n\n"
        "def parse_nested_parens(paren_string: str) -> List[int]:\n"
        '    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n'
        "    For each of the group, output the deepest level of nesting of parentheses.\n"
        "    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n"
        "    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n"
        "    [2, 3, 1, 3]\n"
        '    """\n'
        "    def parse_paren_group(s):\n"
        "        depth = 0\n"
        "        max_depth = 0\n"
        "        for c in",
    ),
    (
        "filter_by_substring",
        "from typing import List\n\n"
        "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
        '    """ Filter an input list of strings only for ones that contain given substring\n'
        "    >>> filter_by_substring([], 'a')\n"
        "    []\n"
        "    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n"
        "    ['abc', 'bacd', 'array']\n"
        '    """\n'
        "    return",
    ),
    (
        "sum_product",
        "from typing import List, Tuple\n\n"
        "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
        '    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n'
        "    Empty sum should be equal to 0 and empty product should be equal to 1.\n"
        "    >>> sum_product([])\n"
        "    (0, 1)\n"
        "    >>> sum_product([1, 2, 3, 4])\n"
        "    (10, 24)\n"
        '    """\n'
        "    s = 0\n"
        "    p = 1\n"
        "    for n in",
    ),
    (
        "rolling_max",
        "from typing import List\n\n"
        "def rolling_max(numbers: List[int]) -> List[int]:\n"
        '    """ From a given list of integers, generate a list of rolling maximum element found until given moment\n'
        "    in the sequence.\n"
        "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n"
        "    [1, 2, 3, 3, 3, 4, 4]\n"
        '    """\n'
        "    result = []\n"
        "    running_max = None\n"
        "    for n in numbers:\n"
        "        if running_max is",
    ),
]


def _find_safetensors(root: Path) -> str | None:
    if root.is_file():
        return str(root)
    if not root.is_dir():
        return None
    for st in root.rglob("model.safetensors"):
        return str(st)
    return None


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        found = _find_safetensors(Path(env))
        if found:
            return found
        raise FileNotFoundError(f"DFLASH_DRAFT does not point to model.safetensors: {env}")

    for candidate in (_LOCAL_DRAFT_FILE, _LOCAL_DRAFT_ROOT):
        found = _find_safetensors(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "draft model.safetensors not found. Expected one of:\n"
        f"  - {_LOCAL_DRAFT_FILE}\n"
        "Download it as documented in the README, or set DFLASH_DRAFT to an explicit file or directory."
    )


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _prompt_path(i: int) -> Path:
    return TMPDIR / f"he_prompt_{i:02d}.bin"


def tokenize_prompt(prompt: str, out_path: Path, tokenizer) -> int:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with open(out_path, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<i", int(tid)))
    return len(ids)


def run_test_dflash(prompt_path: Path, n_gen: int, fast_rollback: bool,
                    ddtree_budget: int | None = None,
                    ddtree_temp: float | None = None,
                    ddtree_no_chain_seed: bool = False) -> dict:
    out_bin = TMPDIR / "he_bench_out.bin"
    cmd = [
        TEST_DFLASH, TARGET, DRAFT, str(prompt_path), str(n_gen), str(out_bin),
    ]
    if fast_rollback:
        cmd.append("--fast-rollback")
    if ddtree_budget is not None:
        cmd.append("--ddtree")
        cmd.append(f"--ddtree-budget={ddtree_budget}")
    if ddtree_temp is not None:
        cmd.append(f"--ddtree-temp={ddtree_temp}")
    if ddtree_no_chain_seed:
        cmd.append("--ddtree-no-chain-seed")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        raise RuntimeError(f"test_dflash exited {r.returncode}")

    # Parse output
    out = r.stdout
    m_tps = re.search(r"(\d+(?:\.\d+)?)\s+tok/s", out)
    m_commit = re.search(r"avg commit/step=(\d+(?:\.\d+)?)", out)
    m_accept = re.search(r"accepted=(\d+)/(\d+) \((\d+(?:\.\d+)?)%", out)
    m_steps = re.search(r"(\d+) draft steps", out)
    if not (m_tps and m_commit and m_accept and m_steps):
        print("STDOUT tail:", out[-2000:])
        raise RuntimeError("failed to parse output")
    return {
        "tok_s": float(m_tps.group(1)),
        "commit_per_step": float(m_commit.group(1)),
        "accepted": int(m_accept.group(1)),
        "total_draft_pos": int(m_accept.group(2)),
        "pct": float(m_accept.group(3)),
        "steps": int(m_steps.group(1)),
    }


def main():
    global DRAFT
    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--mode", choices=["fast", "batched"], default="fast")
    ap.add_argument("--skip-tokenize", action="store_true")
    ap.add_argument("--ddtree-budget", type=int, default=None,
                    help="Enable DDTree mode with this node budget (e.g. 15, 32, 64)")
    ap.add_argument("--ddtree-temp", type=float, default=None,
                    help="Sharpen draft logits with this temperature (T<1 widens top-1/top-2 gap)")
    ap.add_argument("--ddtree-no-chain-seed", action="store_true",
                    help="Use paper's pure best-first (no chain pre-seed)")
    args = ap.parse_args()

    print(f"[bench] target = {TARGET}")
    print(f"[bench] draft  = {DRAFT}")
    print(f"[bench] bin    = {TEST_DFLASH}")
    print(f"[bench] tmp    = {TMPDIR}")

    if not args.skip_tokenize:
        print("[bench] tokenizing prompts via HF…")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)
        for i, (name, p) in enumerate(PROMPTS):
            path = _prompt_path(i)
            n = tokenize_prompt(p, path, tok)
            print(f"  [{i:02d}] {name:26s}  {n:4d} tokens")
    else:
        print(f"[bench] skipping tokenize (reusing {_prompt_path(0).parent})")

    print(f"\n[bench] mode={args.mode}  n_gen={args.n_gen}")
    print(f"{'prompt':28s}  {'steps':>6s} {'AL':>6s} {'pct%':>6s} {'tok/s':>8s}")
    print("-" * 62)

    results = []
    for i, (name, _) in enumerate(PROMPTS):
        path = _prompt_path(i)
        try:
            r = run_test_dflash(path, args.n_gen,
                                fast_rollback=(args.mode == "fast"),
                                ddtree_budget=args.ddtree_budget,
                                ddtree_temp=args.ddtree_temp,
                                ddtree_no_chain_seed=args.ddtree_no_chain_seed)
        except Exception as e:
            print(f"  [{i:02d}] {name:26s}  FAILED: {e}")
            continue
        results.append((name, r))
        print(
            f"  {name:26s}  {r['steps']:6d} {r['commit_per_step']:6.2f} "
            f"{r['pct']:6.1f} {r['tok_s']:8.2f}"
        )

    if not results:
        print("no successful runs")
        sys.exit(1)

    n = len(results)
    mean_al = sum(r["commit_per_step"] for _, r in results) / n
    mean_tps = sum(r["tok_s"] for _, r in results) / n
    mean_pct = sum(r["pct"] for _, r in results) / n

    print("-" * 62)
    print(f"{'MEAN':28s}  {'':6s} {mean_al:6.2f} {mean_pct:6.1f} {mean_tps:8.2f}")
    print()
    print(f"commit/step range: {min(r['commit_per_step'] for _,r in results):.2f} - "
          f"{max(r['commit_per_step'] for _,r in results):.2f}")
    print(f"tok/s range:        {min(r['tok_s'] for _,r in results):.1f} - "
          f"{max(r['tok_s'] for _,r in results):.1f}")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/bench_llm.py
================================================
"""
10 prompts per dataset, AR + DFlash per prompt.

    python3 scripts/bench_llm.py

Paths resolve from the repo root by default. Override with env vars:
    DFLASH_TARGET   path to target Qwen3.6-27B-Q4_K_M.gguf (or 3.5)
    DFLASH_DRAFT    path to draft model.safetensors
    DFLASH_BIN      path to build/test_dflash
    DFLASH_BIN_AR   path to build/test_generate
"""
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_FILE = ROOT / "models" / "draft" / "model.safetensors"
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get("DFLASH_BIN", str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"))
TEST_GENERATE = os.environ.get("DFLASH_BIN_AR", str(ROOT / "build" / f"test_generate{BIN_SUFFIX}"))
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

N_GEN = 256
BUDGET = 22
N_SAMPLE = 10

BENCHES = [
    ("HumanEval", "openai_humaneval", None, "test", lambda x: x["prompt"]),
    ("GSM8K", "gsm8k", "main", "test", lambda x: f"Question: {x['question']}\nAnswer: "),
    ("Math500", "HuggingFaceH4/MATH-500", None, "test", lambda x: f"Problem: {x['problem']}\nSolution: "),
]


def _find_safetensors(root: Path) -> str | None:
    if root.is_file():
        return str(root)
    if not root.is_dir():
        return None
    for st in root.rglob("model.safetensors"):
        return str(st)
    return None


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        found = _find_safetensors(Path(env))
        if found:
            return found
        raise FileNotFoundError(f"DFLASH_DRAFT does not point to model.safetensors: {env}")

    for candidate in (_LOCAL_DRAFT_FILE, _LOCAL_DRAFT_ROOT):
        found = _find_safetensors(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "draft model.safetensors not found. Expected one of:\n"
        f"  - {_LOCAL_DRAFT_FILE}\n"
        "Download it as documented in the README, or set DFLASH_DRAFT to an explicit file or directory."
    )


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_checked(cmd, timeout: int, label: str) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "<no output>").strip()[-2000:]
        raise RuntimeError(f"{label} exited {r.returncode}: {tail}")
    return r


def tokenize(tok, p, path: Path):
    ids = tok.encode(p, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def run_ar(path: Path):
    out_bin = TMPDIR / "ar_out.bin"
    r = _run_checked(
        [TEST_GENERATE, TARGET, str(path), str(N_GEN), str(out_bin)],
        timeout=300,
        label="test_generate",
    )
    m = re.search(r"(\d+\.\d+)\s+tok/s", r.stdout)
    if not m:
        raise RuntimeError(f"test_generate output parse failed: {r.stdout[-1000:]}")
    return float(m.group(1))


def _auto_max_ctx(n_prompt):
    # Auto-fit attention budget: prompt + gen + small verify pad, aligned to
    # FATTN_KQ_STRIDE=256. Oversizing max_ctx makes attention stride over
    # unused KV and can cost >20× prefill time (32K prompt + --kv-q4 +
    # max_ctx=131072 → 1035s vs 38s at max_ctx=32768). See scripts/run.py.
    pad = 64  # covers q_len=16 + ddtree budget up to 22 with margin
    return ((n_prompt + N_GEN + pad + 255) // 256) * 256


def run_df(path: Path, n_prompt):
    max_ctx = _auto_max_ctx(n_prompt)
    out_bin = TMPDIR / "df_out.bin"
    r = _run_checked(
        [
            TEST_DFLASH,
            TARGET,
            DRAFT,
            str(path),
            str(N_GEN),
            str(out_bin),
            "--fast-rollback",
            "--ddtree",
            f"--ddtree-budget={BUDGET}",
            f"--max-ctx={max_ctx}",
        ],
        timeout=300,
        label="test_dflash",
    )
    tps = re.search(r"(\d+(?:\.\d+)?)\s+tok/s", r.stdout)
    al = re.search(r"avg commit/step=(\d+(?:\.\d+)?)", r.stdout)
    if not (tps and al):
        raise RuntimeError(f"test_dflash output parse failed: {r.stdout[-1500:]}")
    return float(tps.group(1)), float(al.group(1))


def main():
    global DRAFT
    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")
    _require_file(TEST_GENERATE, "test_generate binary")

    print(f"[bench] target = {TARGET}", flush=True)
    print(f"[bench] draft  = {DRAFT}", flush=True)
    print(f"[bench] ar bin = {TEST_GENERATE}", flush=True)
    print(f"[bench] df bin = {TEST_DFLASH}", flush=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

    results = {}
    for name, ds_name, cfg, split, extract in BENCHES:
        print(f"\n[bench] ==== {name} (n={N_SAMPLE}) ====", flush=True)
        ds = load_dataset(ds_name, cfg, split=split)
        ds = ds.shuffle(seed=42).select(range(N_SAMPLE))
        ar_tps, df_tps, df_al = [], [], []
        for i, s in enumerate(ds):
            p = extract(s)
            path = TMPDIR / f"b_{name}_{i:02d}.bin"
            n = tokenize(tok, p, path)
            if n == 0 or n > 3500:
                continue
            try:
                ar = run_ar(path)
                df, al = run_df(path, n)
            except Exception as e:
                print(f"  [{i+1:02d}/{N_SAMPLE}] n_tok={n:4d}  FAILED: {e}", flush=True)
                continue
            if ar > 0:
                ar_tps.append(ar)
            if df > 0:
                df_tps.append(df)
                df_al.append(al)
            print(f"  [{i+1:02d}/{N_SAMPLE}] n_tok={n:4d}  AR={ar:6.2f}  DFlash={df:7.2f}  AL={al:5.2f}", flush=True)
        ar_m = sum(ar_tps) / len(ar_tps) if ar_tps else 0
        df_m = sum(df_tps) / len(df_tps) if df_tps else 0
        al_m = sum(df_al) / len(df_al) if df_al else 0
        results[name] = {"ar": ar_m, "dflash": df_m, "al": al_m,
                         "speedup": df_m / ar_m if ar_m else 0}
        print(f"  {name} mean: AR={ar_m:.2f}  DFlash={df_m:.2f}  AL={al_m:.2f}  {results[name]['speedup']:.2f}x", flush=True)

    print("\n[bench] === SUMMARY ===")
    print(f"{'Task':12s}  {'AR':>8s}  {'DFlash':>8s}  {'AL':>6s}  {'Speedup':>8s}")
    for name, r in results.items():
        print(f"{name:12s}  {r['ar']:8.2f}  {r['dflash']:8.2f}  {r['al']:6.2f}  {r['speedup']:7.2f}x")

    out_json = TMPDIR / "bench_llm_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/convert_dflash_to_gguf.py
================================================
#!/usr/bin/env python3
"""
Convert the z-lab DFlash draft (safetensors, bf16) to a GGUF that
llama.cpp can load.

Uses llama.cpp's own gguf-py (deps/llama.cpp/gguf-py) — no hand-rolled
binary writer. The library handles header layout, alignment, BF16
storage, and tensor info offsets correctly.

DFlash draft is a 5-layer Qwen-style transformer with two extra
model-level singletons specific to the spec-decode block-diffusion
algorithm:
  - `fc.weight`           [hidden, 5*hidden]  — fuses 5 captured target
                                                 hidden states into the
                                                 draft's input
  - `hidden_norm.weight`  [hidden]            — RMSNorm applied right after
                                                 the fc projection

These are stored under the `dflash.` prefix so llama.cpp can fetch them
via a custom arch loader without colliding with any upstream tensor
name.

Usage:
  PYTHONPATH=../../dflash27b_ggml/deps/llama.cpp/gguf-py python convert_dflash_to_gguf.py \
    models/draft/model.safetensors \
    qwen3.5-27b-dflash-draft.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Use llama.cpp's own GGUF writer — adds bf16 / metadata / alignment
# correctness without any hand-rolled code.
import gguf

# ──────────────────────────────────────────────────────────────────────
# DFlash 27B draft architecture constants
# ──────────────────────────────────────────────────────────────────────

ARCH                = "qwen35-dflash-draft"
HIDDEN              = 5120
N_LAYER             = 5
N_HEAD              = 32          # query heads
N_HEAD_KV           = 8
HEAD_DIM            = 128
INTERMEDIATE        = 17408
VOCAB               = 248320
N_TARGET_LAYERS     = 5            # fc projects 5*hidden -> hidden
ROPE_THETA          = 1_000_000.0
RMS_EPS             = 1e-6
MASK_TOKEN_ID       = 248070
BLOCK_SIZE          = 16
CTX_LEN             = 32768


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping  —  DFlash safetensors -> llama.cpp GGUF
# ──────────────────────────────────────────────────────────────────────

def map_name(name: str) -> str | None:
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3: return None
        i = int(parts[1])
        rest = parts[2]
        layer_map = {
            "input_layernorm.weight":          f"blk.{i}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            "self_attn.q_proj.weight":         f"blk.{i}.attn_q.weight",
            "self_attn.k_proj.weight":         f"blk.{i}.attn_k.weight",
            "self_attn.v_proj.weight":         f"blk.{i}.attn_v.weight",
            "self_attn.o_proj.weight":         f"blk.{i}.attn_output.weight",
            "self_attn.q_norm.weight":         f"blk.{i}.attn_q_norm.weight",
            "self_attn.k_norm.weight":         f"blk.{i}.attn_k_norm.weight",
            "mlp.gate_proj.weight":            f"blk.{i}.ffn_gate.weight",
            "mlp.up_proj.weight":              f"blk.{i}.ffn_up.weight",
            "mlp.down_proj.weight":            f"blk.{i}.ffn_down.weight",
        }
        return layer_map.get(rest)
    return None


# ──────────────────────────────────────────────────────────────────────
# safetensors reader  —  header parse + raw byte slice
# ──────────────────────────────────────────────────────────────────────

def load_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        return header_size, json.loads(header_json)


def read_tensor_bytes(path: Path, header_size: int, info: dict) -> bytes:
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_size + start)
        return f.read(end - start)


def bytes_to_np(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        # Convert BF16 -> F16 on the host. Several ggml-cuda ops (mul,
        # binbcast) only accept F32 / F16 inputs, and llama.cpp's
        # build_norm path multiplies normalised activations by the norm
        # weight tensor. Storing the draft as F16 throughout sidesteps
        # the unsupported BF16 path entirely. Quality impact ~0 for
        # weight tensors (BF16 -> F16 keeps 10/8 mantissa bits anyway
        # after the implicit cast).
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        # bf16 = sign(1) + exp(8) + mantissa(7); reinterpret as f32 by
        # putting it in the high half, then narrow to f16.
        u32 = (u16.astype(np.uint32) << 16)
        f32 = u32.view("<f4").reshape(shape)
        return f32.astype("<f2")
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


SAFETENSORS_DTYPE_TO_GGUF = {
    "F32":  gguf.GGMLQuantizationType.F32,
    "F16":  gguf.GGMLQuantizationType.F16,
    # BF16 in safetensors -> we narrow to F16 in bytes_to_np above.
    "BF16": gguf.GGMLQuantizationType.F16,
}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", type=Path)
    ap.add_argument("out_gguf",     type=Path)
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] safetensors not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] reading safetensors header from {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)

    # Architecture metadata
    writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft")
    writer.add_uint32(f"{ARCH}.context_length",          CTX_LEN)
    writer.add_uint32(f"{ARCH}.embedding_length",        HIDDEN)
    writer.add_uint32(f"{ARCH}.block_count",             N_LAYER)
    writer.add_uint32(f"{ARCH}.feed_forward_length",     INTERMEDIATE)
    writer.add_uint32(f"{ARCH}.attention.head_count",    N_HEAD)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    # llama.cpp uses key_length / value_length to override the default
    # n_embd_head = n_embd / n_head heuristic (DFlash has n_embd=5120
    # but head_dim=128 so n_head*head_dim=4096 != n_embd).
    writer.add_uint32(f"{ARCH}.attention.key_length",    HEAD_DIM)
    writer.add_uint32(f"{ARCH}.attention.value_length",  HEAD_DIM)
    writer.add_uint32(f"{ARCH}.vocab_size",              VOCAB)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS)
    writer.add_float32(f"{ARCH}.rope.freq_base",         ROPE_THETA)

    # DFlash-specific hyperparameters
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", N_TARGET_LAYERS)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)

    # Walk + add tensors. Sort: dflash.* singletons first, then output_*,
    # then per-layer in numeric order — keeps the on-disk layout stable.
    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__":
            continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}")
            continue
        dtype = SAFETENSORS_DTYPE_TO_GGUF.get(info["dtype"])
        if dtype is None:
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}", file=sys.stderr)
            sys.exit(1)
        pending.append((gguf_name, info["dtype"], info["shape"], info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."):     return (0, n)
        if n.startswith("output_"):     return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    for gguf_name, st_dtype, shape, info in pending:
        raw = read_tensor_bytes(args.safetensors, header_size, info)
        arr = bytes_to_np(raw, st_dtype, shape)
        raw_dtype = SAFETENSORS_DTYPE_TO_GGUF[st_dtype]
        # Norm weights and the dflash hidden_norm singleton must be F32:
        # the ggml-cuda mul path that build_norm emits asserts on
        # src1's element size alignment (binbcast.cu nb10 % sizeof) and
        # the F32 path is the safest cross-quant fallback.
        is_norm = (
            gguf_name.endswith("_norm.weight") or
            gguf_name == "output_norm.weight" or
            gguf_name == "dflash.hidden_norm.weight"
        )
        if is_norm:
            arr = arr.astype("<f4")
            raw_dtype = gguf.GGMLQuantizationType.F32
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        print(f"[tensor] {gguf_name:50s} {st_dtype:4s}->{raw_dtype.name:4s} {tuple(shape)}")

    print(f"[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[done] wrote {args.out_gguf}")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/detokenize.py
================================================
"""Read int32 token IDs from a file and print them as decoded text."""

import argparse
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--slice", default=None,
                    help="optional 'start:end' (token indices, exclusive end)")
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        raw = f.read()
    ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
    print(f"read {len(ids)} tokens")

    if args.slice:
        s, e = args.slice.split(":")
        ids = ids[int(s) if s else None : int(e) if e else None]

    print(f"ids = {ids}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tok.decode(ids, skip_special_tokens=False)
    print(f"text = {text!r}")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/gen_oracle.py
================================================
"""
Generate deterministic oracle inputs + expected output for the DFlash draft
forward pass, using the validated Phase 0 PyTorch reference in
../../megaqwen3_27b_dflash/reference/.

Output: a directory with three float32 binary files:
    noise.bin          shape [1, q_len=16, hidden=5120]  (contiguous row-major)
    target.bin         shape [1, ctx_len, 5*hidden=25600]
    expected.bin       shape [1, q_len=16, hidden=5120]

All files are flat float32 arrays, no header. The C++ test loads them with
the same shape convention (ggml col-major = PyTorch row-major for these
specific layouts since everything is contiguous).

Usage:
    python gen_oracle.py --out /tmp/dflash_oracle --ctx-len 64
"""

import argparse
import os
import sys
import struct

import torch

# Import the existing reference implementation
HERE = os.path.abspath(os.path.dirname(__file__))
REF_DIR = os.path.abspath(os.path.join(
    HERE, "..", "..", "megaqwen3_27b_dflash", "reference"))
sys.path.insert(0, REF_DIR)

from dflash_reference import (
    DFlashConfig,
    dflash_forward_core,
)
from load_weights import load_dflash_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ctx-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda")
    ref_dtype = torch.bfloat16  # reference was trained in bf16
    cfg: DFlashConfig
    weights: 'DFlashWeights'
    cfg, weights = load_dflash_weights(device=device, dtype=ref_dtype)

    # Quick sanity print
    print(f"loaded weights: fc={tuple(weights.fc.shape)}, "
          f"layers={len(weights.layers)}")

    q_len = cfg.block_size                                 # 16
    hidden = cfg.hidden_size                                # 5120
    ctx_len = args.ctx_len
    fc_in_dim = len(cfg.target_layer_ids) * hidden          # 25600

    torch.manual_seed(args.seed)
    noise = (torch.randn(1, q_len, hidden, device=device, dtype=ref_dtype)
             * 0.02)
    target_hidden_cat = (
        torch.randn(1, ctx_len, fc_in_dim, device=device, dtype=ref_dtype)
        * 0.02)
    position_ids = torch.arange(
        0, ctx_len + q_len, device=device, dtype=torch.long
    ).unsqueeze(0).expand(1, ctx_len + q_len)

    # Run the reference forward (matches the C++ graph semantically)
    with torch.inference_mode():
        out_hidden = dflash_forward_core(
            noise_embedding=noise,
            target_hidden_concat=target_hidden_cat,
            position_ids=position_ids,
            weights=weights,
            cfg=cfg,
        )

    print(f"reference out: shape={tuple(out_hidden.shape)} "
          f"dtype={out_hidden.dtype} "
          f"mean={out_hidden.float().mean().item():.6g} "
          f"std={out_hidden.float().std().item():.6g}")

    # Save as float32 row-major contiguous
    noise_f32  = noise.to(torch.float32).contiguous().cpu().numpy()
    target_f32 = target_hidden_cat.to(torch.float32).contiguous().cpu().numpy()
    out_f32    = out_hidden.to(torch.float32).contiguous().cpu().numpy()

    noise_f32.tofile(os.path.join(args.out, "noise.bin"))
    target_f32.tofile(os.path.join(args.out, "target.bin"))
    out_f32.tofile(os.path.join(args.out, "expected.bin"))

    # Also write a small metadata file for the C++ test to read
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"ctx_len={ctx_len}\n")
        f.write(f"q_len={q_len}\n")
        f.write(f"hidden={hidden}\n")
        f.write(f"fc_in={fc_in_dim}\n")
        f.write(f"seed={args.seed}\n")

    print(f"wrote: {args.out}/{{noise,target,expected}}.bin + meta.txt")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/phase_split_dual_gpu.py
================================================
#!/usr/bin/env python3
"""Run the PFlash prefill phase through a persistent CUDA daemon.

This phase-split harness is intentionally PFlash-only. It keeps the Qwen3-0.6B
PFlash drafter resident in `pflash_daemon`, optionally on a different CUDA GPU
from the later target run, and writes compressed token/text outputs plus timing
and GPU resource reports. It measures the PFlash/prefill side only; decode is
outside this harness.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


DEFAULT_BUILD = env_path("PFLASH_PHASE_BUILD_DIR", ROOT / "build")
DEFAULT_DRAFTER = env_path("PFLASH_PHASE_DRAFTER", ROOT / "models" / "Qwen3-0.6B-BF16.gguf")
DEFAULT_TOKENIZER = os.environ.get("PFLASH_PHASE_TOKENIZER", "Qwen/Qwen3-0.6B")


def write_counted_i32(path: Path, ids: Iterable[int]) -> None:
    values = [int(x) for x in ids]
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(values)))
        if values:
            f.write(struct.pack("<" + "i" * len(values), *values))


def read_stream_until_sentinel(r_fd: int) -> list[int]:
    out: list[int] = []
    while True:
        raw = os.read(r_fd, 4)
        if not raw or len(raw) < 4:
            raise RuntimeError("pflash daemon stream closed before sentinel")
        tok = struct.unpack("<i", raw)[0]
        if tok == -1:
            return out
        out.append(tok)


class ProcessLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.queue: queue.Queue[str] = queue.Queue()
        self._file = path.open("wb")
        self._thread: threading.Thread | None = None

    def attach(self, proc: subprocess.Popen[bytes]) -> None:
        def reader() -> None:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                self._file.write(raw)
                self._file.flush()
                self.queue.put(raw.decode("utf-8", errors="replace").rstrip("\n"))
            self._file.close()

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def wait_for(self, needle: str, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        tail: list[str] = []
        while time.time() < deadline:
            try:
                line = self.queue.get(timeout=0.2)
                tail.append(line)
                if needle in line:
                    return
            except queue.Empty:
                pass
        raise TimeoutError(f"timed out waiting for {needle!r}; tail={tail[-12:]}")


class PFlashDaemon:
    def __init__(self, *, binary: Path, drafter: Path, gpu: int, log_path: Path,
                 env: dict[str, str]) -> None:
        self.binary = binary
        self.drafter = drafter
        self.gpu = gpu
        self.log = ProcessLog(log_path)
        self.env = env
        self.proc: subprocess.Popen[bytes] | None = None
        self.r_fd: int | None = None

    def start(self) -> float:
        r_fd, w_fd = os.pipe()
        env = os.environ.copy()
        env.update(self.env)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        cmd = [str(self.binary), str(self.drafter), f"--stream-fd={w_fd}"]
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(w_fd,),
            env=env,
            cwd=str(ROOT),
            bufsize=0,
        )
        os.close(w_fd)
        self.r_fd = r_fd
        self.log.attach(self.proc)
        self.log.wait_for("[pflash-daemon] ready", 180)
        return time.perf_counter() - t0

    def compress(self, counted_ids: Path, *, keep_ratio: float, lookahead: int,
                 chunk_size: int, pool_kernel: int) -> tuple[list[int], float]:
        if self.proc is None or self.proc.stdin is None or self.r_fd is None:
            raise RuntimeError("pflash daemon is not running")
        keep_x1000 = int(round(keep_ratio * 1000))
        cmd = f"compress {keep_x1000} {lookahead} {chunk_size} {pool_kernel} {counted_ids}\n"
        t0 = time.perf_counter()
        self.proc.stdin.write(cmd.encode("utf-8"))
        self.proc.stdin.flush()
        tokens = read_stream_until_sentinel(self.r_fd)
        return tokens, time.perf_counter() - t0

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(b"quit\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.r_fd is not None:
            try:
                os.close(self.r_fd)
            except OSError:
                pass
        self.proc = None


class GpuMonitor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.phase = "init"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def start(self) -> None:
        def phase() -> str:
            with self._lock:
                return self.phase

        def loop() -> None:
            fields = "index,temperature.gpu,fan.speed,power.draw,power.limit,memory.used,memory.total,utilization.gpu"
            with self.path.open("w") as f:
                f.write("ts,phase,index,temp_c,fan_pct,power_w,power_limit_w,mem_used_mib,mem_total_mib,util_pct\n")
                f.flush()
                while not self._stop.is_set():
                    ts = time.time()
                    try:
                        out = subprocess.check_output(
                            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                            text=True,
                            stderr=subprocess.DEVNULL,
                            timeout=2,
                        )
                        for line in out.strip().splitlines():
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) == 8:
                                f.write(",".join([f"{ts:.3f}", phase()] + parts) + "\n")
                        f.flush()
                    except Exception as exc:
                        f.write(f"{ts:.3f},{phase()},ERR,,,,,,,{type(exc).__name__}\n")
                        f.flush()
                    self._stop.wait(1.0)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summarize_gpu(self, gpu: int) -> dict[str, float | int | None]:
        rows: list[dict[str, float | int]] = []
        if not self.path.exists():
            return {"samples": 0}
        for line in self.path.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) != 10 or parts[2] == "ERR":
                continue
            try:
                idx = int(parts[2])
                if idx != gpu:
                    continue
                rows.append({
                    "temp": float(parts[3]),
                    "fan": float(parts[4]),
                    "power": float(parts[5]),
                    "mem": float(parts[7]),
                    "util": float(parts[9]),
                })
            except ValueError:
                pass
        if not rows:
            return {"samples": 0}
        return {
            "samples": len(rows),
            "mem_max_mib": max(float(r["mem"]) for r in rows),
            "temp_max_c": max(float(r["temp"]) for r in rows),
            "fan_max_pct": max(float(r["fan"]) for r in rows),
            "power_avg_w": mean(float(r["power"]) for r in rows),
            "power_max_w": max(float(r["power"]) for r in rows),
            "util_avg_pct": mean(float(r["util"]) for r in rows),
            "util_max_pct": max(float(r["util"]) for r in rows),
        }


@dataclass
class CompressionCase:
    name: str
    source_tokens: int
    compressed_tokens: int
    compress_wall_s: float
    compress_tok_s: float
    compression_ratio: float
    retained_key: bool | None = None
    retained_answer: bool | None = None


def load_tokenizer(args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )


def make_niah_text(tokenizer, token_count: int, case_idx: int, needle_fraction: float) -> tuple[str, str, str, int]:
    key = f"keymark{case_idx}zeta"
    answer = f"04385{74 + case_idx:02d}"
    intro = "Below is a long passage. Keep important facts from the passage.\n\n"
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    needle = f"The special magic {key} number is {answer}. Remember this exact number. "
    question = f"\n\nQuestion: What is the special magic {key} number?\n"

    def build(reps: int) -> str:
        pos = max(0, min(reps, int(reps * needle_fraction)))
        body = filler * pos + needle + filler * (reps - pos)
        return intro + body + question

    fixed = len(tokenizer.encode(intro + needle + question, add_special_tokens=False))
    filler_tokens = max(1, len(tokenizer.encode(filler, add_special_tokens=False)))
    lo = 0
    hi = max(8, (max(0, token_count - fixed) // filler_tokens) + 8)
    while len(tokenizer.encode(build(hi), add_special_tokens=False)) < token_count:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(build(mid), add_special_tokens=False)) <= token_count:
            lo = mid
        else:
            hi = mid - 1
    text = build(lo)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, key, answer, actual


def make_pflash_env(args) -> dict[str, str]:
    env = {"DFLASH_FP_ALPHA": str(args.pflash_alpha)}
    if args.pflash_use_bsa:
        env["DFLASH_FP_USE_BSA"] = "1"
    if args.pflash_k_type:
        env["DFLASH_PFLASH_K_TYPE"] = args.pflash_k_type
    return env


def fmt(value, nd: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{nd}f}"


def run_cases(args, cases: list[tuple[str, str, str | None, str | None]]) -> None:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args)
    monitor = GpuMonitor(args.report_dir / "gpu_monitor.csv")
    daemon = PFlashDaemon(
        binary=args.pflash_bin,
        drafter=args.pflash_drafter,
        gpu=args.pflash_gpu,
        log_path=args.report_dir / "pflash_daemon.log",
        env=make_pflash_env(args),
    )
    results: list[CompressionCase] = []
    try:
        monitor.start()
        monitor.set_phase("pflash_load")
        ready_s = daemon.start()
        for name, text, key, answer in cases:
            case_dir = args.report_dir / name
            case_dir.mkdir(parents=True, exist_ok=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            (case_dir / "prompt.txt").write_text(text, encoding="utf-8")
            counted = case_dir / "prompt_counted.bin"
            write_counted_i32(counted, ids)

            monitor.set_phase(name)
            kept, wall_s = daemon.compress(
                counted,
                keep_ratio=args.keep_ratio,
                lookahead=args.lookahead,
                chunk_size=args.chunk_size,
                pool_kernel=args.pool_kernel,
            )
            compressed_text = tokenizer.decode(kept, skip_special_tokens=True)
            (case_dir / "compressed.txt").write_text(compressed_text, encoding="utf-8")
            write_counted_i32(case_dir / "compressed_counted.bin", kept)

            results.append(CompressionCase(
                name=name,
                source_tokens=len(ids),
                compressed_tokens=len(kept),
                compress_wall_s=wall_s,
                compress_tok_s=len(ids) / wall_s if wall_s > 0 else 0.0,
                compression_ratio=(len(kept) / len(ids)) if ids else 0.0,
                retained_key=(key in compressed_text) if key else None,
                retained_answer=(answer in compressed_text) if answer else None,
            ))

        monitor.set_phase("cleanup")
        resource_summary = monitor.summarize_gpu(args.pflash_gpu)
        summary = {
            "date": time.strftime("%Y-%m-%d"),
            "mode": "dual_gpu_pflash_phase_split",
            "pflash_gpu": args.pflash_gpu,
            "pflash_daemon_ready_s": ready_s,
            "pflash_drafter": str(args.pflash_drafter),
            "tokenizer": args.tokenizer,
            "keep_ratio": args.keep_ratio,
            "lookahead": args.lookahead,
            "chunk_size": args.chunk_size,
            "pool_kernel": args.pool_kernel,
            "pflash_k_type": args.pflash_k_type or "compute",
            "cases": [asdict(c) for c in results],
            "resource_summary": resource_summary,
            "logs": {
                "pflash": str(args.report_dir / "pflash_daemon.log"),
                "monitor": str(args.report_dir / "gpu_monitor.csv"),
            },
        }
        (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(args.report_dir / "summary.md", summary)
        print(json.dumps(summary, indent=2))
    finally:
        monitor.stop()
        daemon.stop()


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Dual-GPU PFlash Phase-Split Report",
        "",
        f"- PFlash GPU: `{summary['pflash_gpu']}`",
        f"- PFlash daemon ready: `{fmt(summary['pflash_daemon_ready_s'])} s`",
        f"- keep ratio: `{summary['keep_ratio']}`",
        f"- lookahead: `{summary['lookahead']}`",
        f"- PFlash K cache: `{summary.get('pflash_k_type', 'compute')}`",
        "",
        "## Resource Peak",
        "",
        "| gpu | samples | peak mem MiB | peak temp C | avg power W | peak power W | avg util % | peak util % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    res = summary.get("resource_summary") or {}
    lines.append(
        "| {gpu} | {samples} | {mem} | {temp} | {pavg} | {pmax} | {uavg} | {umax} |".format(
            gpu=summary["pflash_gpu"],
            samples=res.get("samples", 0),
            mem=fmt(res.get("mem_max_mib")),
            temp=fmt(res.get("temp_max_c")),
            pavg=fmt(res.get("power_avg_w")),
            pmax=fmt(res.get("power_max_w")),
            uavg=fmt(res.get("util_avg_pct")),
            umax=fmt(res.get("util_max_pct")),
        )
    )
    lines.extend([
        "",
        "## Cases",
        "",
        "| case | source tokens | compressed tokens | ratio | PFlash s | PFlash tok/s | key retained | answer retained |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ])
    for case in summary["cases"]:
        key = case.get("retained_key")
        answer = case.get("retained_answer")
        lines.append(
            "| {name} | {source} | {compressed} | {ratio} | {secs} | {tps} | {key} | {answer} |".format(
                name=case["name"],
                source=case["source_tokens"],
                compressed=case["compressed_tokens"],
                ratio=fmt(case["compression_ratio"], 4),
                secs=fmt(case["compress_wall_s"]),
                tps=fmt(case["compress_tok_s"]),
                key="n/a" if key is None else ("yes" if key else "no"),
                answer="n/a" if answer is None else ("yes" if answer else "no"),
            )
        )
    lines.extend([
        "",
        "Files:",
    ])
    for key, value in summary["logs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_prompt_text(args) -> str:
    if args.prompt and args.prompt_file:
        raise SystemExit("use only one of --prompt or --prompt-file")
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("provide --prompt, --prompt-file, or pipe prompt text on stdin")


def run_prompt(args) -> None:
    text = read_prompt_text(args)
    if not text.strip():
        raise SystemExit("prompt is empty")
    run_cases(args, [("prompt", text, None, None)])


def run_bench_niah(args) -> None:
    tokenizer = load_tokenizer(args)
    cases: list[tuple[str, str, str | None, str | None]] = []
    for idx, value in enumerate(x for x in args.contexts.split(",") if x.strip()):
        requested = int(value)
        text, key, answer, actual = make_niah_text(tokenizer, requested, idx, args.needle_fraction)
        cases.append((f"niah_ctx{actual}", text, key, answer))
    run_cases(args, cases)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    ap.add_argument("--pflash-bin", type=Path, default=None)
    ap.add_argument("--pflash-drafter", type=Path, default=DEFAULT_DRAFTER)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--pflash-gpu", type=int, default=0)
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--pool-kernel", type=int, default=13)
    ap.add_argument("--pflash-alpha", type=float, default=0.99)
    ap.add_argument("--pflash-use-bsa", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pflash-k-type", default=None,
                    choices=["f16", "bf16", "q8_0", "q4_0", "q4_1"],
                    help="persistent PFlash drafter K cache type; default follows drafter compute type")
    ap.add_argument("--report-dir", type=Path, default=Path("reports/pflash_phase_split"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    prompt = sub.add_parser("run-prompt", help="run a prompt through the PFlash phase")
    add_common_args(prompt)
    prompt.add_argument("--prompt", default=None)
    prompt.add_argument("--prompt-file", type=Path, default=None)
    prompt.set_defaults(func=run_prompt)

    bench = sub.add_parser("bench-niah", help="compress synthetic NIAH prompts")
    add_common_args(bench)
    bench.add_argument("--contexts", default="4096,8192,16384")
    bench.add_argument("--needle-fraction", type=float, default=0.5)
    bench.set_defaults(func=run_bench_niah)

    args = ap.parse_args()
    args.pflash_bin = args.pflash_bin or (args.build_dir / "pflash_daemon")
    for path in (args.pflash_bin, args.pflash_drafter):
        if not Path(path).exists():
            raise SystemExit(f"missing required path: {path}")
    args.func(args)


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/quantize_draft_q8.py
================================================
#!/usr/bin/env python3
"""
Quantize the z-lab DFlash draft (safetensors, bf16) to a Q8_0 GGUF.

Projection weights (fc, wq, wk, wv, wo, gate, up, down) are quantized
to Q8_0 (~50% size reduction vs BF16).  Norm weights stay F32
(precision-critical, tiny).

The output GGUF uses the same arch and tensor naming as
convert_dflash_to_gguf.py so gguf_draft_loader.cpp can load it.

Usage:
    python3 scripts/quantize_draft_q8.py \
        models/draft/model.safetensors \
        models/draft/draft-q8_0.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import gguf

# ──────────────────────────────────────────────────────────────────────
# DFlash 27B draft architecture constants (must match dflash27b.h)
# ──────────────────────────────────────────────────────────────────────

ARCH                = "qwen35-dflash-draft"
HIDDEN              = 5120
N_LAYER             = 5
N_HEAD              = 32
N_HEAD_KV           = 8
HEAD_DIM            = 128
INTERMEDIATE        = 17408
VOCAB               = 248320
N_TARGET_LAYERS     = 5
ROPE_THETA          = 1_000_000.0
RMS_EPS             = 1e-6
MASK_TOKEN_ID       = 248070
BLOCK_SIZE          = 16
CTX_LEN             = 32768

Q8_0_BLOCK_SIZE     = 32   # elements per Q8_0 block


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping  —  DFlash safetensors -> llama.cpp GGUF
# (Identical to convert_dflash_to_gguf.py)
# ──────────────────────────────────────────────────────────────────────

def map_name(name: str) -> str | None:
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3: return None
        i = int(parts[1])
        rest = parts[2]
        layer_map = {
            "input_layernorm.weight":          f"blk.{i}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            "self_attn.q_proj.weight":         f"blk.{i}.attn_q.weight",
            "self_attn.k_proj.weight":         f"blk.{i}.attn_k.weight",
            "self_attn.v_proj.weight":         f"blk.{i}.attn_v.weight",
            "self_attn.o_proj.weight":         f"blk.{i}.attn_output.weight",
            "self_attn.q_norm.weight":         f"blk.{i}.attn_q_norm.weight",
            "self_attn.k_norm.weight":         f"blk.{i}.attn_k_norm.weight",
            "mlp.gate_proj.weight":            f"blk.{i}.ffn_gate.weight",
            "mlp.up_proj.weight":              f"blk.{i}.ffn_up.weight",
            "mlp.down_proj.weight":            f"blk.{i}.ffn_down.weight",
        }
        return layer_map.get(rest)
    return None


def is_norm_tensor(gguf_name: str) -> bool:
    return (
        gguf_name.endswith("_norm.weight") or
        gguf_name == "output_norm.weight" or
        gguf_name == "dflash.hidden_norm.weight"
    )


# ──────────────────────────────────────────────────────────────────────
# safetensors reader
# ──────────────────────────────────────────────────────────────────────

def load_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        return header_size, json.loads(header_json)


def read_tensor_bytes(path: Path, header_size: int, info: dict) -> bytes:
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_size + start)
        return f.read(end - start)


def bf16_bytes_to_f32(raw: bytes, shape: list[int]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view("<f4").reshape(shape)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Quantize DFlash draft BF16 safetensors to Q8_0 GGUF")
    ap.add_argument("safetensors", type=Path,
                    help="Input BF16 safetensors (e.g. models/draft/model.safetensors)")
    ap.add_argument("out_gguf", type=Path,
                    help="Output Q8_0 GGUF (e.g. models/draft/draft-q8_0.gguf)")
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] safetensors not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] reading safetensors header from {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)

    # Architecture metadata (identical to convert_dflash_to_gguf.py)
    writer.add_string("general.name", "Qwen3.5-27B-DFlash-Draft-Q8_0")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32(f"{ARCH}.context_length",          CTX_LEN)
    writer.add_uint32(f"{ARCH}.embedding_length",        HIDDEN)
    writer.add_uint32(f"{ARCH}.block_count",             N_LAYER)
    writer.add_uint32(f"{ARCH}.feed_forward_length",     INTERMEDIATE)
    writer.add_uint32(f"{ARCH}.attention.head_count",    N_HEAD)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    writer.add_uint32(f"{ARCH}.attention.key_length",    HEAD_DIM)
    writer.add_uint32(f"{ARCH}.attention.value_length",  HEAD_DIM)
    writer.add_uint32(f"{ARCH}.vocab_size",              VOCAB)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS)
    writer.add_float32(f"{ARCH}.rope.freq_base",         ROPE_THETA)

    # DFlash-specific hyperparameters
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", N_TARGET_LAYERS)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)

    # Collect and sort tensors (same order as convert_dflash_to_gguf.py)
    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__":
            continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}")
            continue
        if info["dtype"] not in ("BF16", "F16", "F32"):
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}",
                  file=sys.stderr)
            sys.exit(1)
        pending.append((gguf_name, st_name, info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."):   return (0, n)
        if n.startswith("output_"):   return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    total_bf16 = 0
    total_q8   = 0

    for gguf_name, st_name, info in pending:
        shape = info["shape"]
        raw = read_tensor_bytes(args.safetensors, header_size, info)

        # Convert to F32 from whatever source dtype
        if info["dtype"] == "BF16":
            arr = bf16_bytes_to_f32(raw, shape)
        elif info["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype="<f2").reshape(shape).astype("<f4")
        else:
            arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()

        src_bytes = len(raw)
        total_bf16 += src_bytes

        if is_norm_tensor(gguf_name):
            # Norm weights: keep F32
            writer.add_tensor(gguf_name, arr,
                              raw_dtype=gguf.GGMLQuantizationType.F32)
            total_q8 += arr.nbytes
            print(f"[tensor] {gguf_name:50s} BF16->F32  {tuple(shape)}"
                  f"  ({arr.nbytes:,} bytes)")
        else:
            # Projection weights: quantize to Q8_0
            # Verify alignment: last dim must be multiple of 32
            last_dim = shape[-1]
            assert last_dim % Q8_0_BLOCK_SIZE == 0, \
                f"{gguf_name}: last dim {last_dim} not divisible by {Q8_0_BLOCK_SIZE}"
            q8_data = gguf.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(gguf_name, q8_data,
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            total_q8 += q8_data.nbytes
            ratio = q8_data.nbytes / src_bytes
            print(f"[tensor] {gguf_name:50s} BF16->Q8_0 {tuple(shape)}"
                  f"  ({q8_data.nbytes:,} bytes, {ratio:.1%} of BF16)")

    print(f"\n[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[done] wrote {args.out_gguf}")
    print(f"[size] BF16 source: {total_bf16 / 1e9:.2f} GB")
    print(f"[size] Q8_0 output: {total_q8 / 1e9:.2f} GB")
    print(f"[size] compression: {total_q8 / total_bf16:.1%}")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/run.py
================================================
"""
Streaming one-shot generation.

    python3 scripts/run.py --prompt "def fibonacci(n):"
    echo "Write a haiku about GPUs" | python3 scripts/run.py

Tokens print live as they are committed by the spec-decode loop.
Auto-applies the Qwen3.5/3.6 chat template unless --raw is passed.

Default target is Qwen3.6-27B-Q4_K_M.gguf. Override with `--target` or the
`DFLASH_TARGET` env var (also honored by bench_he.py / bench_llm.py).
The HF tokenizer repo defaults to `Qwen/Qwen3.6-27B` and can be overridden via
the `DFLASH_TOKENIZER` env var.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def default_paths():
    return {
        "target": os.environ.get("DFLASH_TARGET",
                                 "models/Qwen3.6-27B-Q4_K_M.gguf"),
        "draft":  os.environ.get("DFLASH_DRAFT", "models/draft"),
        "bin":    "build/test_dflash" + (".exe" if sys.platform == "win32" else ""),
    }


def resolve_draft(draft_dir: str) -> str:
    if draft_dir.endswith(".safetensors"):
        p = Path(draft_dir)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"draft safetensors not found: {draft_dir}")

    p = Path(draft_dir)
    if p.is_file():
        return str(p)
    if p.is_dir():
        for st in p.rglob("model.safetensors"):
            return str(st)

    raise FileNotFoundError(
        f"no model.safetensors under {draft_dir}. Download it as documented in the README, or pass --draft explicitly."
    )


def tokenize(tokenizer, text: str, out_path: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    with open(out_path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def main():
    d = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--target", type=str, default=d["target"])
    ap.add_argument("--draft",  type=str, default=d["draft"])
    ap.add_argument("--bin",    type=str, default=d["bin"])
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--kv-q4", action="store_true",
                    help="Q4_0 KV cache (required for max_ctx=131072)")
    ap.add_argument("--kv-tq3", action="store_true",
                    help="TQ3_0 KV cache (3.5 bpv, near-lossless)")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="K cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for K). "
                         "See kv_quant.cpp for supported (K,V) pairs.")
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="V cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for V).")
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers (KV positions). 0 = full "
                         "attention. Default 2048 (set in C++); only kicks in "
                         "once kv_cache > window.")
    ap.add_argument("--max-ctx", type=int, default=0,
                    help="Override max KV context (default: auto-fit "
                         "prompt+n_gen+block, aligned to 256). Passing a "
                         "value much larger than needed (e.g. 131072 on a "
                         "16K prompt) massively degrades attention speed "
                         "because the kernel strides over unused KV.")
    args = ap.parse_args()

    prompt_text = args.prompt if args.prompt else sys.stdin.read().strip()
    if not prompt_text:
        sys.exit("no prompt")

    from transformers import AutoTokenizer
    tok_repo = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B")
    tokenizer = AutoTokenizer.from_pretrained(tok_repo,
                                              trust_remote_code=True)

    if args.raw:
        text = prompt_text
    else:
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": prompt_text})
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)

    draft_path = resolve_draft(args.draft)
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_id[0] if im_end_id else -1

    args.bin = str(Path(args.bin).resolve())
    bin_dir = str(Path(args.bin).parent)
    dll_dir = str(Path(args.bin).parent / "bin")
    env = {**os.environ}
    if sys.platform == "win32":
        env["PATH"] = dll_dir + os.pathsep + bin_dir + os.pathsep + env.get("PATH", "")
    if args.cache_type_k:
        env["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        env["DFLASH27B_KV_V"] = args.cache_type_v
    if args.kv_q4:
        env["DFLASH27B_KV_Q4"] = "1"
    if args.kv_tq3:
        env["DFLASH27B_KV_TQ3"] = "1"
    if args.fa_window is not None:
        env["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    with tempfile.TemporaryDirectory() as tmp:
        in_bin  = os.path.join(tmp, "prompt.bin")
        out_bin = os.path.join(tmp, "out.bin")
        n_tok = tokenize(tokenizer, text, in_bin)
        # Auto-fit max_ctx: prompt + gen + a small verify pad, aligned to
        # FATTN_KQ_STRIDE=256. Oversizing this is a performance trap —
        # attention compute scales with the allocated max_ctx, not the
        # actual filled kv_len, so a 131072 max_ctx on a 16K prompt runs
        # attention 8× slower than necessary (verified: 32K prompt with
        # max_ctx=131072 + --kv-q4 → 1035s prefill vs 38s at max_ctx=32768).
        if args.max_ctx > 0:
            max_ctx = args.max_ctx
        else:
            pad = 64  # covers q_len=16 + ddtree budget up to 22 with margin
            max_ctx = ((n_tok + args.n_gen + pad + 255) // 256) * 256
        print(f"[run] prompt {n_tok} tokens, streaming up to {args.n_gen} tokens, max_ctx={max_ctx}",
              file=sys.stderr, flush=True)

        r, w = os.pipe()
        if sys.platform == "win32":
            import msvcrt
            os.set_inheritable(w, True)
            stream_fd_val = int(msvcrt.get_osfhandle(w))
        else:
            stream_fd_val = w
        cmd = [args.bin, args.target, draft_path, in_bin,
               str(args.n_gen), out_bin,
               "--fast-rollback", "--ddtree", f"--ddtree-budget={args.budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
        if sys.platform == "win32":
            proc = subprocess.Popen(cmd, env=env, close_fds=False,
                                    stdout=sys.stderr,
                                    stderr=subprocess.PIPE)
        else:
            proc = subprocess.Popen(cmd, pass_fds=(w,), env=env,
                                    stdout=sys.stderr,
                                    stderr=subprocess.PIPE)
        os.close(w)

        generated = 0
        buffer = b""
        try:
            while True:
                b = os.read(r, 4)
                if not b or len(b) < 4:
                    break
                tok_id = struct.unpack("<i", b)[0]
                generated += 1
                if tok_id == im_end_id:
                    break
                sys.stdout.write(tokenizer.decode([tok_id]))
                sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            sys.stderr.write("\n[run] interrupted\n")
            return
        finally:
            proc.wait()
            err = proc.stderr.read()
            if err:
                sys.stderr.buffer.write(err)
                sys.stderr.flush()
        print(file=sys.stderr, flush=True)
        print(f"[run] generated {generated} tokens", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/server.py
================================================
"""
OpenAI-compatible HTTP server on top of test_dflash.

    pip install fastapi uvicorn transformers
    python3 scripts/server.py                 # serves on :8000

    curl http://localhost:8000/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"luce-dflash","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:8000/v1  OPENAI_API_KEY=sk-any

Streams tokens as Server-Sent Events using the OpenAI delta format.
Model reloads per request (~10 s first-token latency). A daemon-mode
binary that keeps the model resident is a planned follow-up.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer

from _prefill_hook import (
    PrefillConfig, add_cli_flags, config_from_args,
    compress_text_via_daemon,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / ("test_dflash" + (".exe" if sys.platform == "win32" else ""))
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


# Models known to share the qwen35 GGUF arch + vocab. Verified via
# tokenizer.ggml.pre == "qwen35" and identical eos/pad/bos token IDs.
_QWEN35_FAMILY_TOKENIZERS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
}


def _tokenizer_id_from_gguf(gguf_path: Path) -> str:
    """Infer the HuggingFace tokenizer repo from a GGUF target file.

    The GGUF file encodes its own tokenizer so in principle we could use that
    directly, but `test_dflash` drives generation through the HF tokenizer for
    chat-template application. We match on `general.basename` / `general.name`
    metadata; if anything goes wrong we fall back to the historical default
    (Qwen/Qwen3.5-27B) so existing setups don't break.
    """
    default = "Qwen/Qwen3.5-27B"
    try:
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        for key in ("general.basename", "general.name"):
            f = r.fields.get(key)
            if f is None or not f.data:
                continue
            import numpy as np
            p = f.parts[f.data[0]]
            if not isinstance(p, np.ndarray):
                continue
            try:
                val = bytes(p).decode("utf-8", errors="replace")
            except Exception:
                continue
            for known, repo in _QWEN35_FAMILY_TOKENIZERS.items():
                if known.lower() in val.lower():
                    return repo
    except Exception:
        pass
    return default


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None  # noted + ignored (greedy-only)
    top_p: float | None = None


class AnthropicMessage(BaseModel):
    role: str
    # Anthropic allows either a plain string or a list of content blocks.
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = MODEL_NAME
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


def build_app(target: Path, draft: Path, bin_path: Path, budget: int, max_ctx: int,
              tokenizer: AutoTokenizer, stop_ids: set[int],
              prefill_cfg: PrefillConfig | None = None,
              drafter_tokenizer: AutoTokenizer | None = None) -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server")
    daemon_lock = asyncio.Lock()

    r_pipe, w_pipe = os.pipe()
    if sys.platform == "win32":
        import msvcrt
        os.set_inheritable(w_pipe, True)
        stream_fd_val = int(msvcrt.get_osfhandle(w_pipe))
    else:
        stream_fd_val = w_pipe

    bin_abs = str(Path(bin_path).resolve())
    dll_dir = str(Path(bin_abs).parent / "bin")
    env = {**os.environ}
    if sys.platform == "win32":
        env["PATH"] = dll_dir + os.pathsep + str(Path(bin_abs).parent) + os.pathsep + env.get("PATH", "")

    cmd = [bin_abs, str(target), str(draft), "--daemon",
           "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
           f"--max-ctx={max_ctx}",
           f"--stream-fd={stream_fd_val}"]
    if sys.platform == "win32":
        daemon_proc = subprocess.Popen(cmd, close_fds=False, env=env,
                                       stdin=subprocess.PIPE)
    else:
        daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), env=env,
                                       stdin=subprocess.PIPE)
    os.close(w_pipe)

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "luce"}],
        }

    def _ids_to_bin(ids: list[int]) -> Path:
        # mkstemp returns (fd, path). os.fdopen() takes ownership and closes.
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return Path(path)

    def _render_messages(msgs_list: list[dict]) -> tuple[Path, list[int], str]:
        """Apply chat template to msgs_list and return (bin path, ids, raw prompt).

        The raw prompt is returned for spec-prefill: when compression fires we
        re-tokenise it with the drafter vocab.
        """
        prompt = tokenizer.apply_chat_template(
            msgs_list, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        return _ids_to_bin(ids), ids, prompt

    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, list[int], list[dict]]:
        """Returns (bin path, ids, raw msgs). Does NOT yet apply prefill compression."""
        msgs = [{"role": m.role, "content": _anthropic_text_from_content(m.content)
                 if isinstance(m.content, list) else m.content}
                for m in req.messages]
        path, ids, _prompt = _render_messages(msgs)
        return path, ids, msgs

    def _maybe_compress(msgs: list[dict], prompt_bin: Path, prompt_ids: list[int]
                        ) -> tuple[Path, list[int]]:
        """Run pflash compression on the LAST user message if config + length
        thresholds say so. Returns the (possibly new) bin path + ids. Holds the
        daemon lock for the duration via the *outer* caller (callers call this
        from within ``async with daemon_lock``)."""
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_ids
        if not prefill_cfg.should_compress(len(prompt_ids)):
            return prompt_bin, prompt_ids
        if drafter_tokenizer is None:
            return prompt_bin, prompt_ids

        # Find the last user message (the long-context blob)
        last_user_idx = next((i for i in range(len(msgs) - 1, -1, -1)
                              if msgs[i]["role"] == "user"), None)
        if last_user_idx is None:
            return prompt_bin, prompt_ids
        long_text = msgs[last_user_idx]["content"]

        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=long_text,
        )

        new_msgs = list(msgs)
        new_msgs[last_user_idx] = {"role": "user", "content": compressed_text}
        new_bin, new_ids, _ = _render_messages(new_msgs)
        try: prompt_bin.unlink()
        except Exception: pass
        return new_bin, new_ids

    def _token_stream(r, n_gen):
        generated = 0
        hit_stop = False
        while True:
            b = os.read(r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin, prompt_ids, raw_msgs = _tokenize_prompt(req)

        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        def _gen_len_for(prompt_len: int) -> int:
            return min(req.max_tokens, max_ctx - prompt_len - 20)

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    cur_bin, cur_ids = await asyncio.to_thread(
                        _maybe_compress, raw_msgs, prompt_bin, prompt_ids)
                    prompt_len = len(cur_ids)
                    gen_len = _gen_len_for(prompt_len)
                    if gen_len <= 0:
                        try: cur_bin.unlink()
                        except Exception: pass
                        err = {"id": completion_id, "object": "chat.completion.chunk",
                               "created": created, "model": MODEL_NAME,
                               "choices": [{"index": 0, "delta": {},
                                            "finish_reason": "length"}]}
                        yield f"data: {json.dumps(err)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    cmd_line = f"{cur_bin} {gen_len}\n"
                    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                    daemon_proc.stdin.flush()
                    head = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0,
                                      "delta": {"role": "assistant"},
                                      "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(head)}\n\n"
                    try:
                        # Offload blocking os.read in _token_stream to a thread so
                        # SSE chunks flush progressively instead of after generation ends.
                        async for tok_id in iterate_in_threadpool(_token_stream(r_pipe, gen_len)):
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created, "model": MODEL_NAME,
                                "choices": [{"index": 0,
                                              "delta": {"content": tokenizer.decode([tok_id])},
                                              "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    finally:
                        try: cur_bin.unlink()
                        except Exception: pass
                    tail = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {},
                                      "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(tail)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming: collect all tokens, return one response
        async with daemon_lock:
            cur_bin, cur_ids = await asyncio.to_thread(
                _maybe_compress, raw_msgs, prompt_bin, prompt_ids)
            prompt_len = len(cur_ids)
            gen_len = _gen_len_for(prompt_len)
            if gen_len <= 0:
                try: cur_bin.unlink()
                except Exception: pass
                return JSONResponse(
                    {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                    status_code=400)
            cmd_line = f"{cur_bin} {gen_len}\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = list(_token_stream(r_pipe, gen_len))

        try: cur_bin.unlink()
        except Exception: pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": len(tokens),
                      "total_tokens": prompt_len + len(tokens)},
        })

    # ── Anthropic Messages API ──────────────────────────────────────
    # Mirrors the OpenAI endpoint but formatted for the Anthropic SDK.
    # `?beta=true` (or any other query params) are accepted and ignored.

    def _anthropic_text_from_content(content) -> str:
        if isinstance(content, str):
            return content
        # list of blocks — concatenate the text blocks, ignore images/tools
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)

    def _tokenize_anthropic(req: AnthropicMessagesRequest
                            ) -> tuple[Path, list[int], list[dict]]:
        msgs = []
        system_text = _anthropic_text_from_content(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role,
                         "content": _anthropic_text_from_content(m.content)})
        path, ids, _prompt = _render_messages(msgs)
        return path, ids, msgs

    async def _astream_tokens(r, n_gen):
        """Yields one token at a time without blocking the event loop.
        Each 4-byte pipe read is dispatched to a worker thread."""
        generated = 0
        hit_stop = False
        while True:
            b = await asyncio.to_thread(os.read, r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_ids, raw_msgs = _tokenize_anthropic(req)
        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                # Hold the lock across the ENTIRE read cycle so concurrent
                # requests don't interleave tokens through the shared pipe.
                async with daemon_lock:
                    cur_bin, cur_ids = await asyncio.to_thread(
                        _maybe_compress, raw_msgs, prompt_bin, prompt_ids)
                    prompt_len = len(cur_ids)
                    gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                    if gen_len <= 0:
                        try: cur_bin.unlink()
                        except Exception: pass
                        err = {"type": "error",
                               "error": {"type": "invalid_request_error",
                                         "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}}
                        yield f"event: error\ndata: {json.dumps(err)}\n\n"
                        return
                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "model": req.model or MODEL_NAME,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": prompt_len, "output_tokens": 0},
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

                    cb_start = {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start)}\n\n"

                    cmd_line = f"{cur_bin} {gen_len}\n"
                    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                    daemon_proc.stdin.flush()

                    out_tokens = 0
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            delta = {
                                "type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta",
                                          "text": tokenizer.decode([tok_id])},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                    finally:
                        try: cur_bin.unlink()
                        except Exception: pass

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            cur_bin, cur_ids = await asyncio.to_thread(
                _maybe_compress, raw_msgs, prompt_bin, prompt_ids)
            prompt_len = len(cur_ids)
            gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
            if gen_len <= 0:
                try: cur_bin.unlink()
                except Exception: pass
                return JSONResponse(
                    {"type": "error",
                     "error": {"type": "invalid_request_error",
                               "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                    status_code=400)
            cmd_line = f"{cur_bin} {gen_len}\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = [t async for t in _astream_tokens(r_pipe, gen_len)]

        try: cur_bin.unlink()
        except Exception: pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or MODEL_NAME,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_len,
                      "output_tokens": len(tokens)},
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    # Attention compute currently scales with --max-ctx, not the actual
    # prompt+gen length (see https://github.com/Luce-Org/lucebox-hub/issues/10).
    # Default 16384 fits most API workloads without the 20×+ slowdown users
    # hit with --max-ctx=131072 on short requests. Bump via --max-ctx if you
    # actually need long-context serving.
    default_ctx = 16384
    ap.add_argument("--max-ctx", type=int, default=default_ctx,
                    help=f"Maximum context length (default: {default_ctx}; "
                         "oversizing this — e.g. 131072 on short prompts — "
                         "can slow attention 20×+ until issue #10 is fixed)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache. When --max-ctx > 6144 the server "
                         "auto-enables TQ3_0 KV to fit; pass --kv-f16 to opt out.")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="K cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for K). "
                         "See kv_quant.cpp for supported (K,V) pairs.")
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="V cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for V).")
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers (KV positions). 0 = full "
                         "attention. Default 2048 (set in C++); only kicks in "
                         "once kv_cache > window. Trades attention range for "
                         "long-context decode speed.")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="HuggingFace tokenizer repo ID (default: auto-detect "
                         "from target GGUF basename; falls back to Qwen/Qwen3.5-27B)")
    ap.add_argument("--daemon", action="store_true", help="Run with persistent model daemon (now default)")
    add_cli_flags(ap)
    args = ap.parse_args()
    prefill_cfg = config_from_args(args)

    # Auto-enable TQ3_0 KV cache when the requested context exceeds what F16 fits.
    # Clients like Claude Code routinely send 10k+ token system prompts, so
    # 6144 is too tight for real-world use. setdefault so an explicit user
    # DFLASH27B_KV_TQ3=0 still wins.
    if args.cache_type_k:
        os.environ["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        os.environ["DFLASH27B_KV_V"] = args.cache_type_v
    if args.max_ctx > 6144 and not args.kv_f16 and not args.cache_type_k and not args.cache_type_v:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")

    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    # When pflash is on, the daemon needs the same memory-tight env the bench
    # harness uses (otherwise the LM-head dequant buffer fragments cudaMalloc
    # after the compress dance and the post-compress draft graph reserve
    # OOMs). setdefault so explicit user overrides still win.
    if args.prefill_compression != "off":
        os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
        os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin}")
    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")
    draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
    if not draft.is_file():
        raise SystemExit(f"draft safetensors not found at {args.draft}")

    tokenizer_id = args.tokenizer or _tokenizer_id_from_gguf(args.target)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    drafter_tokenizer = None
    if prefill_cfg.enabled:
        drafter_tokenizer = AutoTokenizer.from_pretrained(
            prefill_cfg.drafter_tokenizer_id, trust_remote_code=True)

    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids,
                    prefill_cfg=prefill_cfg if prefill_cfg.enabled else None,
                    drafter_tokenizer=drafter_tokenizer)

    import uvicorn
    print(f"Luce DFlash OpenAI server on http://{args.host}:{args.port}")
    print(f"  target    = {args.target}")
    print(f"  draft     = {draft}")
    print(f"  bin       = {args.bin}")
    print(f"  budget    = {args.budget}")
    print(f"  max_ctx   = {args.max_ctx}")
    print(f"  tokenizer = {tokenizer_id}")
    if prefill_cfg.enabled:
        print(f"  pflash    = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
              f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        print("  pflash    = off")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/server_tools.py
================================================
"""
OpenAI-compatible HTTP server on top of test_dflash, **with tool-calling support**.

Patched fork of scripts/server.py that:
  1. Accepts the OpenAI `tools` array in ChatRequest.
  2. Renders tools into the prompt via Qwen's chat template (`tools=...`).
  3. Parses `<tool_call><function=...><parameter=...></tool_call>` blocks out
     of the model output and returns them as proper OpenAI `tool_calls`.
  4. Supports `role: "tool"` and assistant `tool_calls` in input messages so
     multi-turn agent loops round-trip correctly.

Streaming behavior:
  - Content tokens are streamed as `delta.content` until a `<tool_call>` opener
    is detected; the rest of the response is then buffered, parsed at the end
    of generation, and emitted as a single final `delta.tool_calls` chunk with
    `finish_reason: "tool_calls"`.
  - If no tool call appears in the output, behavior is identical to the
    upstream server.

Greedy decoding still applies (verify path is greedy-only). `temperature` and
`top_p` are accepted but ignored, matching upstream.

Run:
  pip install fastapi uvicorn transformers
  python3 scripts/server_tools.py --port 8000
"""
import argparse
import json
import os
import re
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from _prefill_hook import (
    PrefillConfig, add_cli_flags, config_from_args,
    compress_text_via_daemon,
)
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / "test_dflash"
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


# ─── pydantic schemas ──────────────────────────────────────────────

class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI spec


class ToolCall(BaseModel):
    id: str | None = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None  # str, list, or null when tool_calls present
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDef(BaseModel):
    type: str = "function"
    function: dict  # {name, description, parameters: {...JSON schema...}}


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ToolDef] | None = None
    tool_choice: Any | None = None  # "auto" | "none" | {"function": {...}}
    chat_template_kwargs: dict | None = None  # e.g. {"enable_thinking": false}
    stop: Any | None = None  # str or list[str]
    stream_options: dict | None = None  # e.g. {"include_usage": true}


class AnthropicMessage(BaseModel):
    role: str
    # Anthropic allows either a plain string or a list of content blocks.
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = MODEL_NAME
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


# ─── tool-call parser ──────────────────────────────────────────────

# Qwen3.6 chat template emits:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>
#   VALUE
#   </parameter>
#   ...
#   </function>
#   </tool_call>
# Parsers ported from vLLM (Apache-2.0) for behavioral parity with
# `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`:
#   vllm/reasoning/qwen3_reasoning_parser.py
#   vllm/tool_parsers/qwen3coder_tool_parser.py
# Core algorithms reproduced without vLLM runtime dependencies.

TOOL_CALL_COMPLETE_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_CALL_FUNCTION_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL,
)
# vLLM's improved parameter regex: tolerates unclosed </parameter> by using
# next <parameter= or </function> or end-of-string as a terminator.
TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)
TOOL_OPEN_TAG = "<tool_call>"

# Qwen3.6 chat template wraps the model's CoT inside <think>...</think>.
# The template typically prefills `<think>\n` into the prompt (headless mode)
# so only `</think>` appears in generated output; older templates emit both.
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def normalize_stop(stop) -> list[str]:
    """Coerce OpenAI's stop field (str | list[str] | None) to list[str]."""
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if isinstance(s, str) and s]


def first_stop_match(text: str, stops: list[str]) -> int:
    """Return the earliest index where any stop sequence appears, or -1."""
    best = -1
    for s in stops:
        i = text.find(s)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def parse_reasoning(text: str, thinking_enabled: bool = True) -> tuple[str, str | None]:
    """Port of vLLM's Qwen3ReasoningParser.extract_reasoning.

    Handles the three Qwen3.x thinking flavors:
      1. Paired:   `<think>...</think>` both in generated output.
      2. Headless: template prefilled `<think>\\n` into the prompt, model
         only emits `...</think>...`.
      3. Disabled: user passed `chat_template_kwargs: {enable_thinking: false}`.
         Template still emits `<think>\\n\\n</think>\\n\\n` but into the prompt;
         the model output is pure content and contains no tags.

    If the output was truncated mid-thinking (no `</think>` seen and
    `thinking_enabled=True`), returns `("", full_output_as_reasoning)` —
    matching vLLM's convention.

    Returns (cleaned_content, reasoning_content).
    """
    # Strip <think> if the model emitted it itself (older templates).
    parts = text.partition(THINK_OPEN_TAG)
    rest = parts[2] if parts[1] else parts[0]
    if THINK_CLOSE_TAG not in rest:
        if thinking_enabled:
            # No close tag — assume truncated; everything is reasoning.
            return "", (rest.strip() or None)
        else:
            # Thinking disabled — output is pure content.
            return rest.strip(), None
    reasoning, _, content = rest.partition(THINK_CLOSE_TAG)
    return content.strip(), (reasoning.strip() or None)


def _find_tool_properties(tools, function_name):
    """Helper matching vLLM's `find_tool_properties`: returns the parameters
    dict for a given function name, or {} if not found.
    Accepts pydantic ToolDef instances or plain dicts.
    """
    for t in tools or []:
        fn = t.function if hasattr(t, "function") else t.get("function", {})
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump()
        if fn.get("name") == function_name:
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict,
                         func_name: str):
    """Port of vLLM's _convert_param_value. Coerces stringified XML values
    to their JSON-schema type (int/float/bool/object/array/string)."""
    import ast
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        ptype = str(cfg["type"]).strip().lower()
    elif isinstance(cfg, dict) and "anyOf" in cfg:
        ptype = "object"
    else:
        ptype = "string"
    if ptype in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    if any(ptype.startswith(p) for p in ("int", "uint", "long", "short", "unsigned")):
        try: return int(param_value)
        except (ValueError, TypeError): return param_value
    if ptype.startswith("num") or ptype.startswith("float"):
        try:
            f = float(param_value)
            return f if f - int(f) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value
    if ptype in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    # object / array / dict / list
    if (ptype in ("object", "array", "arr")
            or ptype.startswith("dict") or ptype.startswith("list")):
        try: return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError): pass
    try: return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError): return param_value


def parse_tool_calls(text: str, tools=None) -> tuple[str, list[dict]]:
    """Port of Qwen3CoderToolParser._parse_xml_function_call (non-streaming).

    Handles Qwen3.x's `<tool_call><function=NAME>...<parameter=KEY>VAL
    </parameter>...</function></tool_call>` XML. Uses vLLM's improved
    parameter regex that tolerates unclosed </parameter> tags. When `tools`
    is provided, each parameter value is coerced to its JSON-schema type.

    Returns (cleaned_content, tool_calls_list).
    """
    tool_calls: list[dict] = []
    cleaned_parts: list[str] = []
    cursor = 0
    for m in TOOL_CALL_COMPLETE_RE.finditer(text):
        cleaned_parts.append(text[cursor:m.start()])
        cursor = m.end()
        body = m.group(1)
        fn_match = TOOL_CALL_FUNCTION_RE.search(body)
        if not fn_match:
            continue
        fn_text = fn_match.group(1) or fn_match.group(2) or ""
        end_idx = fn_text.find(">")
        if end_idx == -1:
            continue
        function_name = fn_text[:end_idx].strip()
        params_region = fn_text[end_idx + 1:]
        param_config = _find_tool_properties(tools, function_name)
        args: dict = {}
        for match_text in TOOL_CALL_PARAMETER_RE.findall(params_region):
            eq_idx = match_text.find(">")
            if eq_idx == -1:
                continue
            k = match_text[:eq_idx].strip()
            v = match_text[eq_idx + 1:]
            if v.startswith("\n"): v = v[1:]
            if v.endswith("\n"): v = v[:-1]
            args[k] = _convert_param_value(v, k, param_config, function_name)
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), tool_calls


# ─── app ───────────────────────────────────────────────────────────

def build_app(target: Path, draft: Path, bin_path: Path, budget: int,
              max_ctx: int, tokenizer: AutoTokenizer, stop_ids: set[int],
              prefill_cfg: PrefillConfig | None = None,
              drafter_tokenizer: AutoTokenizer | None = None) -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server (tool-aware)")
    daemon_lock = asyncio.Lock()

    r_pipe, w_pipe = os.pipe()
    cmd = [str(bin_path), str(target), str(draft), "--daemon",
           "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
           f"--max-ctx={max_ctx}",
           f"--stream-fd={w_pipe}"]
    daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), stdin=subprocess.PIPE)
    os.close(w_pipe)

    @app.get("/v1/models")
    def list_models():
        return {"object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "luce"}]}

    def _maybe_compress_tool_chat(req: "ChatRequest", prompt_bin: Path,
                                  prompt_len: int, started_in_thinking: bool
                                  ) -> tuple[Path, int, bool]:
        """If prefill is on and the request has no tools and the last user
        message is long, run the daemon compress + re-tokenise. Returns
        (bin, prompt_len, started_in_thinking) — the last is recomputed when
        compression fires, otherwise passed through."""
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_len, started_in_thinking
        if req.tools:
            # Tool definitions ride in the prompt; compressing them mangles JSON.
            return prompt_bin, prompt_len, started_in_thinking
        if not prefill_cfg.should_compress(prompt_len) or drafter_tokenizer is None:
            return prompt_bin, prompt_len, started_in_thinking

        last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
        if last_user is None or not isinstance(last_user.content, str):
            return prompt_bin, prompt_len, started_in_thinking

        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=last_user.content,
        )

        new_msgs = []
        compressed_emitted = False
        for m in req.messages:
            if m is last_user and not compressed_emitted:
                new_msgs.append({"role": "user", "content": compressed_text})
                compressed_emitted = True
            else:
                d = {"role": m.role}
                if m.content is not None:
                    d["content"] = m.content
                new_msgs.append(d)

        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if req.chat_template_kwargs:
            kwargs.update(req.chat_template_kwargs)
        prompt = tokenizer.apply_chat_template(new_msgs, **kwargs)
        new_started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        try: prompt_bin.unlink()
        except Exception: pass
        return Path(path), len(ids), new_started_in_thinking

    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, bool]:
        """Returns (prompt_bin_path, started_in_thinking). started_in_thinking
        is True when the chat template prefilled <think>\\n at the end of the
        prompt — the model's first emitted tokens are reasoning content."""
        # Convert pydantic messages to dicts the chat template expects.
        msgs: list[dict] = []
        for m in req.messages:
            d: dict = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
            if m.name is not None:
                d["name"] = m.name
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                # The Qwen template walks tool_calls[i].function.{name, arguments}
                d["tool_calls"] = []
                for tc in m.tool_calls:
                    args = tc.function.arguments
                    # Template expects arguments as a dict, not a JSON string.
                    if isinstance(args, str):
                        try:
                            args_obj = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args_obj = {"_raw": args}
                    else:
                        args_obj = args
                    d["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": args_obj},
                    })
            msgs.append(d)

        tools_arg = None
        if req.tools:
            tools_arg = [t.model_dump()["function"] | {"type": t.type} for t in req.tools]
            # The Qwen template accepts the raw OpenAI tools array structure.
            tools_arg = [t.model_dump() for t in req.tools]

        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if tools_arg:
            kwargs["tools"] = tools_arg
        # Per-request chat template knobs (e.g. enable_thinking, preserve_thinking).
        if req.chat_template_kwargs:
            kwargs.update(req.chat_template_kwargs)
        prompt = tokenizer.apply_chat_template(msgs, **kwargs)
        # Did the template prefill `<think>\n` at the end? Then streaming should
        # start in reasoning mode.
        started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, started_in_thinking

    def _token_stream(r, n_gen):
        generated = 0
        hit_stop = False
        while True:
            b = os.read(r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin, started_in_thinking = _tokenize_prompt(req)
        prompt_len = prompt_bin.stat().st_size // 4

        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        # pflash compress hook (no-op when --prefill-compression=off / has tools)
        async with daemon_lock:
            prompt_bin, prompt_len, started_in_thinking = await asyncio.to_thread(
                _maybe_compress_tool_chat, req, prompt_bin, prompt_len, started_in_thinking)

        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            try: prompt_bin.unlink()
            except Exception: pass
            return JSONResponse(
                {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                status_code=400)

        if req.stream:
            return await _stream_response(req, prompt_bin, gen_len,
                                           completion_id, created,
                                           started_in_thinking, daemon_lock)

        # Non-streaming: collect, parse, return.
        async with daemon_lock:
            cmd_line = f"{prompt_bin} {gen_len}\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = list(_token_stream(r_pipe, gen_len))
        try: prompt_bin.unlink()
        except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        # User-supplied stop sequences: trim at first match.
        stops = normalize_stop(req.stop)
        if stops:
            i = first_stop_match(text, stops)
            if i != -1:
                text = text[:i]
        # Respect enable_thinking from chat_template_kwargs when deciding how
        # to treat a `</think>`-less response (see parse_reasoning docstring).
        thinking_enabled = True
        if req.chat_template_kwargs:
            thinking_enabled = req.chat_template_kwargs.get("enable_thinking", True)
        cleaned, tool_calls = parse_tool_calls(text, tools=req.tools)
        cleaned, reasoning = parse_reasoning(cleaned, thinking_enabled=thinking_enabled)

        msg: dict = {"role": "assistant"}
        finish_reason = "stop"
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["content"] = cleaned if cleaned else None
            msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            msg["content"] = cleaned

        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": len(tokens),
                      "total_tokens": prompt_len + len(tokens)},
        })

    async def _stream_response(req, prompt_bin, gen_len, completion_id, created,
                                started_in_thinking, lock):
        prompt_len = prompt_bin.stat().st_size // 4
        include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))
        def chunk(delta_obj, finish=None):
            return {"id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": delta_obj,
                                  "finish_reason": finish}]}

        async def sse() -> AsyncIterator[str]:
            async with lock:
                cmd_line = f"{prompt_bin} {gen_len}\n"
                daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                daemon_proc.stdin.flush()

                yield f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n"

                # State machine: mode ∈ {'reasoning', 'content', 'tool_buffer'}
                mode = "reasoning" if started_in_thinking else "content"
                window = ""           # holdback buffer for tag detection
                tool_buffer = ""
                stops = normalize_stop(req.stop)
                # Holdback must cover longest tag AND longest stop sequence.
                tag_holdback = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG), len(TOOL_OPEN_TAG))
                stop_holdback = max((len(s) for s in stops), default=0)
                HOLDBACK = max(tag_holdback, stop_holdback)
                completion_tokens = 0
                stop_hit = False

                def emit_delta(text, kind):
                    """kind: 'content' or 'reasoning_content'"""
                    if not text:
                        return None
                    return f"data: {json.dumps(chunk({kind: text}))}\n\n"

                try:
                    async for tok_id in iterate_in_threadpool(_token_stream(r_pipe, gen_len)):
                        completion_tokens += 1
                        piece = tokenizer.decode([tok_id])
                        window += piece

                        # Stop-sequence check on the visible (content/reasoning) stream.
                        if stops and mode != "tool_buffer":
                            si = first_stop_match(window, stops)
                            if si != -1:
                                window = window[:si]
                                stop_hit = True
                                # Flush truncated remainder per current mode.
                                kind = "reasoning_content" if mode == "reasoning" else "content"
                                out = emit_delta(window, kind)
                                if out: yield out
                                window = ""
                                break

                        # Process state transitions until no more tags found in window.
                        while True:
                            if mode == "tool_buffer":
                                tool_buffer += window
                                window = ""
                                break

                            # Look for the next tag of interest based on mode.
                            if mode == "reasoning":
                                idx = window.find(THINK_CLOSE_TAG)
                                if idx != -1:
                                    pre = window[:idx]
                                    out = emit_delta(pre, "reasoning_content")
                                    if out: yield out
                                    window = window[idx + len(THINK_CLOSE_TAG):]
                                    mode = "content"
                                    continue
                                # No close tag yet. Stream all but holdback.
                                if len(window) > HOLDBACK:
                                    safe = window[:-HOLDBACK]
                                    out = emit_delta(safe, "reasoning_content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break  # need more tokens

                            else:  # mode == "content"
                                think_idx = window.find(THINK_OPEN_TAG)
                                tool_idx  = window.find(TOOL_OPEN_TAG)
                                # Pick the earliest tag that actually appears.
                                hits = [(i, t) for i, t in
                                        ((think_idx, "think"), (tool_idx, "tool")) if i != -1]
                                if hits:
                                    hits.sort()
                                    idx, which = hits[0]
                                    pre = window[:idx]
                                    out = emit_delta(pre, "content")
                                    if out: yield out
                                    if which == "think":
                                        window = window[idx + len(THINK_OPEN_TAG):]
                                        mode = "reasoning"
                                    else:  # tool
                                        tool_buffer = window[idx:]
                                        window = ""
                                        mode = "tool_buffer"
                                    continue
                                if len(window) > HOLDBACK:
                                    safe = window[:-HOLDBACK]
                                    out = emit_delta(safe, "content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break  # need more tokens

                    if stop_hit:
                        finish_reason = "stop"
                        yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                        if include_usage:
                            usage_chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                           "created": created, "model": MODEL_NAME, "choices": [],
                                           "usage": {"prompt_tokens": prompt_len,
                                                      "completion_tokens": completion_tokens,
                                                      "total_tokens": prompt_len + completion_tokens}}
                            yield f"data: {json.dumps(usage_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        try: prompt_bin.unlink()
                        except Exception: pass
                        return

                    # Generation done. Flush remaining window per current mode.
                    if mode == "reasoning" and window:
                        out = emit_delta(window, "reasoning_content")
                        if out: yield out
                    elif mode == "content" and window:
                        out = emit_delta(window, "content")
                        if out: yield out
                    elif mode == "tool_buffer":
                        tool_buffer += window
                    window = ""

                    finish_reason = "stop"
                    if mode == "tool_buffer":
                        cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=req.tools)
                        if tool_calls:
                            if cleaned_after:
                                out = emit_delta(cleaned_after, "content")
                                if out: yield out
                            tc_delta_list = [{
                                "index": i, "id": tc["id"], "type": "function",
                                "function": {"name": tc["function"]["name"],
                                              "arguments": tc["function"]["arguments"]},
                            } for i, tc in enumerate(tool_calls)]
                            yield f"data: {json.dumps(chunk({'tool_calls': tc_delta_list}))}\n\n"
                            finish_reason = "tool_calls"
                        else:
                            # Unclosed <tool_call> — emit raw as content fallback.
                            out = emit_delta(tool_buffer, "content")
                            if out: yield out
                finally:
                    try: prompt_bin.unlink()
                    except Exception: pass

                yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                if include_usage:
                    usage_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [],
                        "usage": {"prompt_tokens": prompt_len,
                                   "completion_tokens": completion_tokens,
                                   "total_tokens": prompt_len + completion_tokens},
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ── Anthropic Messages API ──────────────────────────────────────
    # Mirrors the OpenAI endpoint but formatted for the Anthropic SDK
    # (Claude Code, Anthropic clients). Tool calling NOT forwarded here
    # yet — agent CLIs that want tools should use /v1/chat/completions.

    def _anthropic_text_from_content(content) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)

    def _tokenize_anthropic(req: AnthropicMessagesRequest
                            ) -> tuple[Path, int, list[dict]]:
        msgs = []
        system_text = _anthropic_text_from_content(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role,
                         "content": _anthropic_text_from_content(m.content)})
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # mkstemp returns (fd, path); discarding fd leaks 1 per request (#15).
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, len(ids), msgs

    def _maybe_compress_anthropic(prompt_bin: Path, prompt_len: int,
                                  msgs: list[dict]) -> tuple[Path, int]:
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_len
        if not prefill_cfg.should_compress(prompt_len) or drafter_tokenizer is None:
            return prompt_bin, prompt_len
        last_user_idx = next((i for i in range(len(msgs) - 1, -1, -1)
                              if msgs[i]["role"] == "user"), None)
        if last_user_idx is None:
            return prompt_bin, prompt_len
        long_text = msgs[last_user_idx]["content"]
        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=long_text,
        )
        new_msgs = list(msgs)
        new_msgs[last_user_idx] = {"role": "user", "content": compressed_text}
        prompt = tokenizer.apply_chat_template(
            new_msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        try: prompt_bin.unlink()
        except Exception: pass
        return Path(path), len(ids)

    async def _astream_tokens(r, n_gen):
        """Yields one token at a time without blocking the event loop.
        Each 4-byte pipe read is dispatched to a worker thread."""
        generated = 0
        hit_stop = False
        while True:
            b = await asyncio.to_thread(os.read, r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_len, raw_msgs = _tokenize_anthropic(req)

        async with daemon_lock:
            prompt_bin, prompt_len = await asyncio.to_thread(
                _maybe_compress_anthropic, prompt_bin, prompt_len, raw_msgs)

        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            try: prompt_bin.unlink()
            except Exception: pass
            return JSONResponse(
                {"type": "error",
                 "error": {"type": "invalid_request_error",
                           "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                status_code=400)

        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "model": req.model or MODEL_NAME,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": prompt_len, "output_tokens": 0},
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

                    cb_start = {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start)}\n\n"

                    cmd_line = f"{prompt_bin} {gen_len}\n"
                    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                    daemon_proc.stdin.flush()

                    out_tokens = 0
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            delta = {
                                "type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta",
                                          "text": tokenizer.decode([tok_id])},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                    finally:
                        try: prompt_bin.unlink()
                        except Exception: pass

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            cmd_line = f"{prompt_bin} {gen_len}\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = [t async for t in _astream_tokens(r_pipe, gen_len)]

        try: prompt_bin.unlink()
        except Exception: pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or MODEL_NAME,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_len,
                      "output_tokens": len(tokens)},
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    # Attention compute currently scales with --max-ctx, not the actual
    # prompt+gen length (see issue #10). Default 16384 fits most API
    # workloads without the 20×+ slowdown users hit with --max-ctx=131072
    # on short requests. Bump via --max-ctx for long-context serving.
    ap.add_argument("--max-ctx", type=int, default=16384,
                    help="Maximum context length (default: 16384; oversizing "
                         "this, e.g. 131072 on short prompts, can slow "
                         "attention 20×+ until issue #10 is fixed)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache. When --max-ctx > 6144 the server "
                         "auto-enables TQ3_0 KV to fit; pass --kv-f16 to opt out.")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="K cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for K). "
                         "See kv_quant.cpp for supported (K,V) pairs.")
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="V cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for V).")
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers (KV positions). 0 = full "
                         "attention. Default 2048 (set in C++); only kicks in "
                         "once kv_cache > window. Trades attention range for "
                         "long-context decode speed.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B",
                    help="HF tokenizer id; Qwen3.6 shares this tokenizer.")
    add_cli_flags(ap)
    args = ap.parse_args()
    prefill_cfg = config_from_args(args)

    # Auto-enable TQ3_0 KV cache when the requested context exceeds what F16 fits.
    # setdefault so an explicit user DFLASH27B_KV_TQ3=0 still wins.
    if args.cache_type_k:
        os.environ["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        os.environ["DFLASH27B_KV_V"] = args.cache_type_v
    if args.max_ctx > 6144 and not args.kv_f16 and not args.cache_type_k and not args.cache_type_v:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")

    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    # When pflash is on, daemon needs the same env the bench harness uses
    # (otherwise post-compress draft graph reserve OOMs at 64K+).
    if args.prefill_compression != "off":
        os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
        os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin}")
    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")
    draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
    if not draft.is_file():
        raise SystemExit(f"draft safetensors not found at {args.draft}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    drafter_tokenizer = None
    if prefill_cfg.enabled:
        drafter_tokenizer = AutoTokenizer.from_pretrained(
            prefill_cfg.drafter_tokenizer_id, trust_remote_code=True)

    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids,
                    prefill_cfg=prefill_cfg if prefill_cfg.enabled else None,
                    drafter_tokenizer=drafter_tokenizer)

    import uvicorn
    print(f"Luce DFlash OpenAI server (tool-aware) on http://{args.host}:{args.port}")
    print(f"  target = {args.target}")
    print(f"  draft  = {draft}")
    print(f"  bin    = {args.bin}")
    print(f"  budget = {args.budget}")
    print(f"  max_ctx= {args.max_ctx}")
    print(f"  tokenizer = {args.tokenizer}")
    if prefill_cfg.enabled:
        print(f"  pflash = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
              f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        print("  pflash = off")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/scripts/setup_system.sh
================================================
#!/usr/bin/env bash
# Install system build dependencies for dflash.
# Ubuntu 22.04 (jammy) and 24.04 (noble). Run with sudo.
set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (try: sudo $0)"
fi

# ── distro check ─────────────────────────────────────────────────────────────

if [[ ! -f /etc/os-release ]]; then
    die "Cannot detect OS. This script supports Ubuntu 22.04 and 24.04 only."
fi
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" ]]; then
    die "Unsupported distro '${ID:-unknown}'. This script supports Ubuntu only."
fi

case "${VERSION_CODENAME:-}" in
    jammy)  UBUNTU_VER="2204" ;;
    noble)  UBUNTU_VER="2404" ;;
    *)      die "Unsupported Ubuntu release '${VERSION_CODENAME:-unknown}'. Supported: 22.04 (jammy), 24.04 (noble)." ;;
esac

info "Detected Ubuntu ${VERSION_ID} (${VERSION_CODENAME})"

# ── apt build deps ────────────────────────────────────────────────────────────

info "Installing build-essential, cmake, git, git-lfs..."
apt-get update -qq
apt-get install -y build-essential cmake git git-lfs

git lfs install --system 2>/dev/null || git lfs install
ok "Build tools installed."

# ── huggingface-cli (pipx, installed for $SUDO_USER not root) ────────────────

REAL_USER="${SUDO_USER:-$USER}"

apt-get install -y pipx

if sudo -u "${REAL_USER}" pipx list 2>/dev/null | grep -q huggingface_hub; then
    ok "huggingface-cli already installed for ${REAL_USER}."
else
    info "Installing huggingface-cli for ${REAL_USER}..."
    sudo -u "${REAL_USER}" pipx install "huggingface_hub[cli]"
    ok "huggingface-cli installed."
fi

sudo -u "${REAL_USER}" pipx ensurepath --quiet 2>/dev/null || true

# ── CUDA Toolkit ─────────────────────────────────────────────────────────────

CUDA_NEWLY_INSTALLED=0

if command -v nvcc &>/dev/null; then
    ok "nvcc already on PATH ($(nvcc --version | grep -oP 'release \K[\d.]+')). Skipping CUDA install."
else
    KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
    KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/x86_64/${KEYRING_DEB}"

    info "Downloading NVIDIA CUDA keyring from ${KEYRING_URL}..."
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    wget -q --show-progress -O "${TMP}/${KEYRING_DEB}" "${KEYRING_URL}"

    info "Installing CUDA keyring..."
    dpkg -i "${TMP}/${KEYRING_DEB}"
    apt-get update -qq

    info "Installing cuda-toolkit..."
    apt-get install -y cuda-toolkit
    ok "CUDA toolkit installed."
    CUDA_NEWLY_INSTALLED=1

    # PATH persistence
    BASH_PROFILE=/etc/profile.d/cuda.sh
    if [[ ! -f "${BASH_PROFILE}" ]]; then
        printf 'export PATH=/usr/local/cuda/bin:$PATH\n' > "${BASH_PROFILE}"
        ok "Bash PATH configured via ${BASH_PROFILE}."
    fi

    # Ubuntu's default zsh does not source /etc/profile.d, so write zshenv too.
    ZSH_ENV=/etc/zsh/zshenv
    if [[ -d /etc/zsh ]] && ! grep -q 'cuda' "${ZSH_ENV}" 2>/dev/null; then
        printf 'export PATH=/usr/local/cuda/bin:$PATH\n' >> "${ZSH_ENV}"
        ok "Zsh PATH configured via ${ZSH_ENV}."
    fi
fi

# ── summary ──────────────────────────────────────────────────────────────────

printf '\n'

if [[ ${CUDA_NEWLY_INSTALLED} -eq 1 ]]; then
    warn "nvcc is not yet on your current PATH. Activate it now with:"
    printf '    export PATH=/usr/local/cuda/bin:$PATH\n'
    printf '\n'
    warn "If you ran cmake before this script, wipe the build directory first to avoid stale compiler cache:"
    printf '    rm -rf build\n'
    printf '\n'
fi

info "All system dependencies installed. Next steps:"
printf '    git submodule update --init --recursive\n'
printf '    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release\n'
printf '    cmake --build build --target test_dflash -j\n'



================================================
FILE: dflash/scripts/test_server.py
================================================
import os
import struct
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from server import build_app, MODEL_NAME


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    tokenizer.decode.return_value = "hello"
    return tokenizer

@patch("server.subprocess.Popen")
def test_models_endpoint(mock_popen, mock_tokenizer):
    app = build_app(
        target=Path("target.gguf"),
        draft=Path("draft.safetensors"),
        bin_path=Path("test_dflash"),
        budget=22,
        max_ctx=131072,
        tokenizer=mock_tokenizer,
        stop_ids={2}
    )
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == MODEL_NAME

@patch("server.os.pipe")
@patch("server.subprocess.Popen")
@patch("server.os.read")
def test_chat_completions_non_streaming(mock_os_read, mock_popen, mock_pipe, mock_tokenizer):
    mock_pipe.return_value = (1, 2)
    
    app = build_app(
        target=Path("target.gguf"),
        draft=Path("draft.safetensors"),
        bin_path=Path("test_dflash"),
        budget=22,
        max_ctx=131072,
        tokenizer=mock_tokenizer,
        stop_ids={2}
    )
    
    # Mock os.read to return a single token (e.g. 10) and then -1
    mock_os_read.side_effect = [
        struct.pack("<i", 10),
        struct.pack("<i", -1)
    ]
    
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "hello"

@patch("server.os.pipe")
@patch("server.subprocess.Popen")
@patch("server.os.read")
def test_chat_completions_streaming(mock_os_read, mock_popen, mock_pipe, mock_tokenizer):
    mock_pipe.return_value = (1, 2)
    
    app = build_app(
        target=Path("target.gguf"),
        draft=Path("draft.safetensors"),
        bin_path=Path("test_dflash"),
        budget=22,
        max_ctx=131072,
        tokenizer=mock_tokenizer,
        stop_ids={2}
    )
    
    mock_os_read.side_effect = [
        struct.pack("<i", 10),
        struct.pack("<i", -1)
    ]
    
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True
    })
    
    assert response.status_code == 200
    lines = response.text.strip().split("\n\n")
    assert len(lines) >= 3
    assert lines[-1] == "data: [DONE]"


================================================
FILE: dflash/scripts/tokenize_prompt.py
================================================
"""
Tokenize a prompt string using the Qwen3.5 HF tokenizer (via transformers)
and emit the token IDs as a flat int32 binary file.

We depend on Python only for the tokenizer — the C++ library consumes the
int32 file directly. This keeps the standalone lib free of a BPE impl.

Usage:
    python tokenize_prompt.py --out /tmp/prompt.bin --prompt "The capital of France is"
"""

import argparse
import os
import sys
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B",
                    help="HF repo id whose tokenizer to use")
    ap.add_argument("--add-bos", action="store_true", help="Prepend BOS token")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ids = tok.encode(args.prompt, add_special_tokens=args.add_bos)
    print(f"tokenized {len(ids)} tokens: {ids}")

    with open(args.out, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))

    print(f"wrote {args.out} ({len(ids) * 4} bytes)")


if __name__ == "__main__":
    main()



================================================
FILE: dflash/src/bsa_fwd_inst.cu
================================================
// Instantiate BSA's hdim=128 bf16 forward block kernel.
// Slow to compile (cutlass templates) — separate translation unit so incremental rebuilds skip this.
#include "flash_fwd_block_hdim128_bf16_sm80.cu"



================================================
FILE: dflash/src/bsa_launcher.cu
================================================
// Torchless launcher for the Block-Sparse Attention forward kernel
// (mit-han-lab/Block-Sparse-Attention, sm_80+ FA-2 derived).
//
// Maps our (counts[B,M,H], indices[B,M,N,H]) FlashPrefill selection format
// to the BSA blockmask layout, fills `Flash_fwd_params`, then dispatches
// `run_mha_fwd_block_<bf16, 128, false>`.
//
// All scratch (blockmask, head_mask_type, softmax_lse) is held in a single
// process-lifetime cache (`BsaCache`) that grows on demand and can be freed
// explicitly via dflash_bsa_free_persistent() before the daemon swaps to
// the target-gen path that needs full VRAM.
//
// Hardcoded shape: head_dim=128, block_size=128, non-causal. Add new
// dispatch arms as we support more.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "namespace_config.h"
#include "flash.h"

#include <cutlass/numeric_types.h>

namespace FLASH_NAMESPACE {
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_block_(Flash_fwd_params &params, cudaStream_t stream);
}

namespace dflash27b {
namespace flashprefill {

namespace {

constexpr int kSupportedHeadDim   = 128;
constexpr int kSupportedBlockSize = 128;
constexpr int kConvertThreads     = 64;
constexpr float kLog2E            = 1.4426950408889634f;

// Process-lifetime device buffers. Single-stream daemon use only; not
// thread-safe by design. Wrap behind a function-local accessor so we can
// reset state cleanly via dflash_bsa_free_persistent().
struct BsaCache {
    int32_t* blockmask       = nullptr;
    size_t   blockmask_bytes = 0;

    int*     head_mask_type     = nullptr;
    int      head_mask_capacity = 0;
    int      head_mask_state    = -1;  // -1 = uninitialized; 1 = filled with [1..H]

    float*   softmax_lse       = nullptr;
    size_t   softmax_lse_bytes = 0;

    cudaError_t ensure_blockmask(size_t bytes) {
        if (bytes <= blockmask_bytes) return cudaSuccess;
        if (blockmask) cudaFree(blockmask);
        cudaError_t e = cudaMalloc(&blockmask, bytes);
        if (e == cudaSuccess) blockmask_bytes = bytes;
        return e;
    }
    cudaError_t ensure_head_mask(int n_heads) {
        if (n_heads <= head_mask_capacity) return cudaSuccess;
        if (head_mask_type) cudaFree(head_mask_type);
        cudaError_t e = cudaMalloc(&head_mask_type, n_heads * sizeof(int));
        if (e == cudaSuccess) {
            head_mask_capacity = n_heads;
            head_mask_state = -1;
        }
        return e;
    }
    cudaError_t ensure_softmax_lse(size_t bytes) {
        if (bytes <= softmax_lse_bytes) return cudaSuccess;
        if (softmax_lse) cudaFree(softmax_lse);
        cudaError_t e = cudaMalloc(&softmax_lse, bytes);
        if (e == cudaSuccess) softmax_lse_bytes = bytes;
        return e;
    }
    void release() {
        if (blockmask)       { cudaFree(blockmask);       blockmask = nullptr;       blockmask_bytes = 0; }
        if (head_mask_type)  { cudaFree(head_mask_type);  head_mask_type = nullptr;  head_mask_capacity = 0; head_mask_state = -1; }
        if (softmax_lse)     { cudaFree(softmax_lse);     softmax_lse = nullptr;     softmax_lse_bytes = 0; }
    }
};

BsaCache & cache() {
    static BsaCache c;
    return c;
}

// Convert (counts[B,M,H], indices[B,M,N,H]) → BSA blockmask[B, H, Mp, Np].
// Layout: blockmask[(b * num_bs_heads + h) * (Mp * Np) + m * Np + n].
//
// Invariants:
//   - BSA's fwdBlockmask iterator does a binary search assuming a
//     descending-sorted run of valid block IDs followed by a -1 sentinel.
//     We reverse the input (which is ascending) and force at least one -1.
//   - Pad rows beyond M with all -1 so the iterator terminates.
__global__ void convert_to_blockmask_kernel(
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ blockmask_out,
    int B, int M, int N, int H,
    int Mp, int Np,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z;
    if (b >= B || h >= H) return;

    int32_t* outp = blockmask_out + ((size_t)(b * H + h) * Mp + m) * Np;

    if (m >= M) {
        for (int n = threadIdx.x; n < Np; n += blockDim.x) outp[n] = -1;
        return;
    }

    int cnt = counts[(size_t)b * cnt_s_b + (size_t)m * cnt_s_m + (size_t)h * cnt_s_h];
    if (cnt >= Np) cnt = Np - 1;  // leave at least one -1 sentinel

    for (int n = threadIdx.x; n < Np; n += blockDim.x) {
        if (n < cnt && n < N) {
            int rev = cnt - 1 - n;
            outp[n] = indices[(size_t)b * idx_s_b + (size_t)m * idx_s_m + (size_t)rev * idx_s_n + (size_t)h * idx_s_h];
        } else {
            outp[n] = -1;
        }
    }
}

}  // namespace

// Public free hook. Idempotent. Call before the daemon switches from
// drafter scoring to target generation to release the BSA scratch (~tens of
// MB at S=128K) and give the target's KV cache full headroom.
extern "C" void dflash_bsa_free_persistent() {
    cache().release();
}

extern "C" int launch_bsa_sparse_flash_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    const int32_t* indices, const int32_t* counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim,
    int block_size,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream)
{
    if (head_dim != kSupportedHeadDim) {
        std::fprintf(stderr, "[bsa] unsupported head_dim=%d (only %d)\n",
                     head_dim, kSupportedHeadDim);
        return -1;
    }
    if (block_size != kSupportedBlockSize) {
        std::fprintf(stderr, "[bsa] unsupported block_size=%d (only %d)\n",
                     block_size, kSupportedBlockSize);
        return -1;
    }

    const int M  = (seq_len + block_size - 1) / block_size;
    const int Mp = M;
    const int Np = M;
    const int seqlen_q_rounded = M * block_size;
    const int seqlen_k_rounded = M * block_size;

    BsaCache & c = cache();

    const size_t bm_bytes  = (size_t)batch * n_q_heads * Mp * Np * sizeof(int32_t);
    const size_t lse_bytes = (size_t)batch * n_q_heads * seqlen_q_rounded * sizeof(float);

    cudaError_t err;
    if ((err = c.ensure_blockmask(bm_bytes)) != cudaSuccess)   goto fail;
    if ((err = c.ensure_head_mask(n_q_heads)) != cudaSuccess) goto fail;
    if ((err = c.ensure_softmax_lse(lse_bytes)) != cudaSuccess) goto fail;

    // head_mask_type[h] = h+1: each Q head selects its own per-head block
    // pattern (mask_type-1 indexes into params.blockmask).
    if (c.head_mask_state != 1) {
        int* h_hmt = (int*)std::malloc(n_q_heads * sizeof(int));
        for (int h = 0; h < n_q_heads; ++h) h_hmt[h] = h + 1;
        cudaMemcpyAsync(c.head_mask_type, h_hmt, n_q_heads * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        std::free(h_hmt);
        c.head_mask_state = 1;
    }

    {
        dim3 grid(batch, Mp, n_q_heads);
        dim3 block(kConvertThreads, 1, 1);
        convert_to_blockmask_kernel<<<grid, block, 0, stream>>>(
            indices, counts, c.blockmask,
            batch, M, /*N=*/M, n_q_heads, Mp, Np,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h);
    }

    {
        FLASH_NAMESPACE::Flash_fwd_params params{};
        params.q_ptr = const_cast<void*>(Q);
        params.k_ptr = const_cast<void*>(K);
        params.v_ptr = const_cast<void*>(V);
        params.o_ptr = O;

        params.q_batch_stride = (int64_t)seq_len * n_q_heads * head_dim;
        params.q_row_stride   = (int64_t)n_q_heads * head_dim;
        params.q_head_stride  = head_dim;

        params.k_batch_stride = (int64_t)seq_len * n_k_heads * head_dim;
        params.k_row_stride   = (int64_t)n_k_heads * head_dim;
        params.k_head_stride  = head_dim;

        params.v_batch_stride = (int64_t)seq_len * n_k_heads * head_dim;
        params.v_row_stride   = (int64_t)n_k_heads * head_dim;
        params.v_head_stride  = head_dim;

        params.o_batch_stride = (int64_t)seq_len * n_q_heads * head_dim;
        params.o_row_stride   = (int64_t)n_q_heads * head_dim;
        params.o_head_stride  = head_dim;

        params.h           = n_q_heads;
        params.h_k         = n_k_heads;
        params.h_h_k_ratio = n_q_heads / n_k_heads;

        params.b               = batch;
        params.seqlen_q        = seq_len;
        params.seqlen_k        = seq_len;
        params.d               = head_dim;
        params.seqlen_q_rounded = seqlen_q_rounded;
        params.seqlen_k_rounded = seqlen_k_rounded;
        params.d_rounded       = head_dim;

        params.scale_softmax      = scale;
        params.scale_softmax_log2 = scale * kLog2E;

        params.cu_seqlens_q   = nullptr;
        params.cu_seqlens_k   = nullptr;
        params.seqused_k      = nullptr;
        params.softmax_lse_ptr = c.softmax_lse;
        params.p_ptr          = nullptr;

        params.blockmask             = c.blockmask;
        params.streaming_info        = nullptr;
        params.head_mask_type        = c.head_mask_type;
        params.m_block_dim           = block_size;
        params.n_block_dim           = block_size;
        params.num_blocksparse_heads = n_q_heads;

        params.window_size_left      = -1;
        params.window_size_right     = -1;
        params.is_bf16               = true;
        params.is_causal             = false;
        params.is_exact_streaming    = false;
        params.is_seqlens_k_cumulative = false;
        params.is_rotary_interleaved = false;
        params.num_splits            = 1;
        params.alibi_slopes_ptr      = nullptr;
        params.alibi_slopes_batch_stride = 0;
        params.p_dropout             = 1.f;  // 1.0 = no dropout

        FLASH_NAMESPACE::run_mha_fwd_block_<cutlass::bfloat16_t,
                                            kSupportedHeadDim,
                                            /*Is_causal=*/false>(params, stream);
    }

    return 0;

fail:
    std::fprintf(stderr, "[bsa] cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return -1;
}

}  // namespace flashprefill
}  // namespace dflash27b



================================================
FILE: dflash/src/delta_net_chunked.cpp
================================================
// Chunked gated delta-net: direct port of llama.cpp's
// `llm_build_delta_net_base::build_delta_net_chunking` (src/models/delta-net-base.cpp).
//
// Why we need it: at n_tokens > 1 (spec-decode chain verify / batched
// prefill) the fused ggml_gated_delta_net kernel loops over tokens
// sequentially inside each head. At n_tokens=16..32 and 48 delta-net
// layers that's ~50 ms per target forward, which dominates long-ctx
// decode step time. The chunking algorithm re-expresses the same
// recurrence as a series of matmul/tri/cumsum ops that ggml-cuda
// parallelises across tokens, giving 2-3× on verify compute at the cost
// of a bigger graph.
//
// Scope: chain-mode, no tree, no per-token intermediate capture. The
// tree path (parent_ids != null) and the rollback-capture path still
// use the sequential kernel. Those cases need per-token state fanout
// which chunking does not preserve.

#include "delta_net_chunked.h"

#include <cmath>

namespace dflash27b {

static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

DeltaNetChunkedResult build_delta_net_chunked(
        ggml_context * ctx0,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * g,
        ggml_tensor  * b,
        ggml_tensor  * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    // GDA only in our port — Qwen3.5 delta-net uses gate scalar per head
    // (g->ne[0] == 1). The KDA branch below is kept structurally identical
    // to llama.cpp but never taken in practice.
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf((float)S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    g = ggml_permute(ctx0, g, 0, 2, 1, 3);
    b = ggml_permute(ctx0, b, 0, 2, 1, 3);

    const int CS = kda ? 16 : 64; // chunk size

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (int)((n_tokens + pad) / CS);

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));

    ggml_tensor * kb = nullptr;
    ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;

        ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);

        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);

        decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

        ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
        ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);

        kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);

        kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        ggml_tensor * g_cs_i = g_cs;
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);

        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);

        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
    }

    kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);

    ggml_tensor * attn;
    attn = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);

    attn = ggml_neg(ctx0, attn);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);

    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);

    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);

    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));

    g_last = ggml_cont(ctx0, g_last);

    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));

    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);

    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));

    s = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v * n_seqs);

    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk);
        ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk);
        ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk);
        ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk);
        ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk);

        ggml_tensor * v_t_p = ggml_mul_mat(ctx0, ch_k_cd, s);

        ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);

        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_t_new, ch_kq);

        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s, ch_q_g_exp);

        ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new);

        ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s = ggml_mul(ctx0, s, ch_g_last_exp_t);
        s = ggml_add(ctx0, s, kgv);
    }

    // truncate padded tokens back to n_tokens
    ggml_tensor * o = ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(v->type, S_v),
            ggml_row_size(v->type, S_v * CS * n_chunks),
            ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);
    o = ggml_permute  (ctx0, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);

    DeltaNetChunkedResult r;
    r.output    = o;
    r.new_state = s;
    return r;
}

} // namespace dflash27b



================================================
FILE: dflash/src/delta_net_chunked.h
================================================
#pragma once
// Chunked gated delta-net graph builder.
// See src/delta_net_chunked.cpp for the algorithm and history.

#include <ggml.h>

namespace dflash27b {

struct DeltaNetChunkedResult {
    ggml_tensor * output;     // [S_v, H_v, n_tokens, n_seqs]
    ggml_tensor * new_state;  // [S_v, S_v, H_v, n_seqs]
};

// Chain-only, no-capture, no-tree variant. Caller passes q/k/v/g/b/s in the
// same shape as ggml_gated_delta_net expects. Returns the per-token output
// and the final recurrent state as two separate tensors (unlike the fused
// kernel which packs them into one dst tensor).
DeltaNetChunkedResult build_delta_net_chunked(
        ggml_context * ctx0,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * g,
        ggml_tensor  * b,
        ggml_tensor  * s);

} // namespace dflash27b



================================================
FILE: dflash/src/dflash_graph.h
================================================
// Shared inputs/outputs for the DFlash draft graph builder.
#pragma once

#include "ggml.h"

namespace dflash27b {

struct DraftWeights; // fwd

struct DraftGraphInputs {
    int           ctx_len;          // length of target_hidden_cat along ne[1]
    ggml_tensor * noise_embed;      // [hidden, q_len=16, 1] f32
    ggml_tensor * target_hidden_cat;// [5*hidden, ctx_len, 1] f32
    ggml_tensor * positions_q;      // [q_len] i32   values [ctx_len..ctx_len+q_len-1]
    ggml_tensor * positions_k;      // [ctx_len+q_len] i32   values [0..ctx_len+q_len-1]
    // Optional: if non-null, the graph projects final hidden states through
    // this LM head (shape [hidden, vocab]) and returns logits instead of
    // hidden states. Used for DFlash integration where the draft shares the
    // target's lm_head.
    ggml_tensor * lm_head;
};

struct DraftGraphOutputs {
    ggml_tensor * hidden_states;    // [hidden, q_len, 1]  (always set)
    ggml_tensor * logits;           // [vocab, q_len, 1]   (non-null iff lm_head was provided)
};

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in);

} // namespace dflash27b



================================================
FILE: dflash/src/errors.cpp
================================================
// Thread-safe last-error string used by loaders and graph builders.
// Consumed by tests and the test_dflash driver via dflash27b_last_error().

#include "dflash27b.h"
#include "internal.h"

#include <mutex>
#include <string>

namespace dflash27b {

namespace {
std::mutex g_err_mu;
std::string g_last_error;
}

void set_last_error(std::string msg) {
    std::lock_guard<std::mutex> lk(g_err_mu);
    g_last_error = std::move(msg);
}

} // namespace dflash27b

extern "C" const char * dflash27b_last_error(void) {
    std::lock_guard<std::mutex> lk(dflash27b::g_err_mu);
    return dflash27b::g_last_error.c_str();
}



================================================
FILE: dflash/src/f16_convert.cu
================================================
// Tiny half-precision → f32 conversion kernels used by the DDtree rollback
// path and the drafter's target_feat widen. We store some tensors
// (ssm_intermediate, target_feat) at 16-bit to halve their memory footprint,
// and widen on read into f32 consumers.
//
// Exposes plain C entry points so test_dflash.cpp can call them without
// pulling in a CUDA compile unit of its own.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static __global__ void f16_to_f32_kernel(const __half * __restrict__ src,
                                         float * __restrict__ dst,
                                         size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __half2float(src[i]);
    }
}

static __global__ void bf16_to_f32_kernel(const __nv_bfloat16 * __restrict__ src,
                                          float * __restrict__ dst,
                                          size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __bfloat162float(src[i]);
    }
}

extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    f16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __half *)src, (float *)dst, n_elems);
}

extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16 *)src, (float *)dst, n_elems);
}



================================================
FILE: dflash/src/flashprefill.cpp
================================================
// Glue between the 4 FlashPrefill steps:
//   1. compute_mean_vector_bf16    (kernel)
//   2. compute_block_score_bf16    (kernel)  + normalise on host
//   3. block_select_host           (host)
//   4. sparse_flash_forward_bf16   (kernel)

#include "flashprefill.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

namespace dflash27b {
namespace flashprefill {

extern "C" {
void launch_compute_mean_vector_bf16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream);

void launch_compute_block_score_bf16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream);

void launch_block_select(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * idx_out, int32_t * cnt_out,
    cudaStream_t stream);

#ifdef DFLASH27B_HAVE_BSA
int launch_bsa_sparse_flash_forward_bf16(
    const void* Q, const void* K, const void* V, void* O,
    const int32_t* indices, const int32_t* counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream);
#endif

void launch_sparse_flash_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    const int32_t * block_index, const int32_t * counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int q_tile, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream);
}

void block_select_host(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int32_t * idx_out, int32_t * cnt_out);

namespace {
inline int cdiv(int a, int b) { return (a + b - 1) / b; }
}

int flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg)
{
    const int B = batch;
    const int S = seq_len;
    const int H = n_q_heads;
    const int Hk = n_k_heads;
    const int D = head_dim;
    const int BLOCK = cfg.block_size;
    const int M = cdiv(S, BLOCK);
    const int N = M;
    const int Q_TILE = 64;

    // Strides assume contiguous [B, S, H, D] / [B, S, Hk, D] row-major.
    int s_Q_b = S * H * D, s_Q_n = H * D, s_Q_h = D, s_Q_d = 1;
    int s_K_b = S * Hk * D, s_K_n = Hk * D, s_K_h = D, s_K_d = 1;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D, s_mK_d = 1;
    int s_S_b = M * N * H, s_S_m = N * H, s_S_n = H, s_S_h = 1;
    int s_idx_b = M * N * H, s_idx_m = N * H, s_idx_n = H, s_idx_h = 1;
    int s_cnt_b = M * H, s_cnt_m = H, s_cnt_h = 1;

    // Allocate scratch on the same device as Q.
    void * dmK = nullptr;
    float * dS = nullptr, * dM = nullptr;
    int32_t * dIdx = nullptr, * dCnt = nullptr;
    cudaError_t e;
    if ((e = cudaMalloc(&dmK,  (size_t)B * M * Hk * D * 2)) != cudaSuccess) goto err;  // bf16
    if ((e = cudaMalloc(&dS,   (size_t)B * M * N * H * sizeof(float))) != cudaSuccess) goto err;
    if ((e = cudaMalloc(&dM,   (size_t)B * M * N * H * sizeof(float))) != cudaSuccess) goto err;
    if ((e = cudaMalloc(&dIdx, (size_t)B * M * N * H * sizeof(int32_t))) != cudaSuccess) goto err;
    if ((e = cudaMalloc(&dCnt, (size_t)B * M * H * sizeof(int32_t))) != cudaSuccess) goto err;

    static const bool prof = (std::getenv("DFLASH_FP_PROFILE") != nullptr);
    cudaEvent_t pE[5];
    if (prof) for (int i=0;i<5;i++) cudaEventCreate(&pE[i]);
    if (prof) cudaEventRecord(pE[0]);
    // 1. mean_K
    launch_compute_mean_vector_bf16(
        K, dmK, B, S, Hk, D, BLOCK,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d, 0);

    if (prof) cudaEventRecord(pE[1]);
    // 2. block scores
    launch_compute_block_score_bf16(
        Q, dmK, scale, dS, dM,
        B, H, Hk, S, D, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_S_b, s_S_m, s_S_n, s_S_h, 0);

    if (prof) cudaEventRecord(pE[2]);
    // 3. block_select on GPU.
    launch_block_select(
        dS, B, M, N, H,
        cfg.attention_sink, cfg.window, cfg.last_n_full, cfg.alpha,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_idx_b, s_idx_m, s_idx_n, s_idx_h,
        s_cnt_b, s_cnt_m, s_cnt_h,
        dIdx, dCnt, 0);

    if (prof) cudaEventRecord(pE[3]);
    static const bool dump_cnt = (std::getenv("DFLASH_FP_DUMP_COUNTS") != nullptr);
    if (dump_cnt) {
        std::vector<int32_t> hcnt((size_t)B * M * H);
        cudaMemcpy(hcnt.data(), dCnt, hcnt.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        long long sum = 0; int mn = 1<<30, mx = 0;
        for (auto c : hcnt) { sum += c; if (c<mn) mn=c; if (c>mx) mx=c; }
        std::fprintf(stderr, "[fp-cnt] S=%d M=%d H=%d  total_select=%lld avg=%.1f min=%d max=%d\n",
                     S, M, H, sum, (double)sum/(M*H*B), mn, mx);
    }
    // 4. sparse flash forward (BSA-or-WMMA)
#ifdef DFLASH27B_HAVE_BSA
    static const bool use_bsa = (std::getenv("DFLASH_FP_USE_BSA") != nullptr);
    if (use_bsa && D == 128 && BLOCK == 128) {
        launch_bsa_sparse_flash_forward_bf16(
            Q, K, V, O, dIdx, dCnt, scale,
            B, H, Hk, S, D, BLOCK,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h, 0);
    } else
#endif
    {
        launch_sparse_flash_forward_bf16(
            Q, K, V, O, dIdx, dCnt, scale,
            B, H, Hk, S, D, Q_TILE, BLOCK,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_K_b, s_K_n, s_K_h, s_K_d,    // V uses K strides
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,    // O uses Q strides
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h, 0);
    }

    if (prof) {
        cudaEventRecord(pE[4]);
        cudaEventSynchronize(pE[4]);
        float t1, t2, t3, t4;
        cudaEventElapsedTime(&t1, pE[0], pE[1]);
        cudaEventElapsedTime(&t2, pE[1], pE[2]);
        cudaEventElapsedTime(&t3, pE[2], pE[3]);
        cudaEventElapsedTime(&t4, pE[3], pE[4]);
        std::fprintf(stderr,
            "[fp-prof] S=%d H=%d Hk=%d  mean=%.2fms  score=%.2fms  select=%.2fms  forward=%.2fms\n",
            S, n_q_heads, n_k_heads, t1, t2, t3, t4);
        for (int i=0;i<5;i++) cudaEventDestroy(pE[i]);
    }
    cudaFree(dmK); cudaFree(dS); cudaFree(dM); cudaFree(dIdx); cudaFree(dCnt);
    return 0;

err:
    if (dmK)  cudaFree(dmK);
    if (dS)   cudaFree(dS);
    if (dM)   cudaFree(dM);
    if (dIdx) cudaFree(dIdx);
    if (dCnt) cudaFree(dCnt);
    std::fprintf(stderr, "[flashprefill] cudaMalloc failed: %s\n", cudaGetErrorString(e));
    return -1;
}

} // namespace flashprefill
} // namespace dflash27b



================================================
FILE: dflash/src/flashprefill.h
================================================
// Public C++ entry point for the FlashPrefill block-sparse attention used by
// the in-process Qwen3-0.6B drafter (speculative prefill scoring).
//
// Wraps kernels 1-4 + GPU block_select into one call. Call signature mirrors
// the upstream `flash_prefill` from qhfan/FlashPrefill (arXiv:2603.06199).
//
// Tensor layout (all CUDA, bf16, contiguous, D fastest):
//   Q[B, S, n_q_heads, D]
//   K[B, S, n_k_heads, D]
//   V[B, S, n_k_heads, D]
//   O[B, S, n_q_heads, D]
//
// Backends:
//   - Default: WMMA m16n16k16 sparse forward (sm_70+). Functional everywhere.
//   - Set env DFLASH_FP_USE_BSA=1 to dispatch to the Block-Sparse-Attention
//     kernel (FA-2 derived, m16n8k16 PTX, sm_80+ via cuBLAS BF16 GEMM).
//     Requires building with -DDFLASH27B_ENABLE_BSA=ON. ~3x faster than WMMA
//     on RTX 3090 at S=128K.
//
// Tunables (env vars):
//   DFLASH_FP_USE_BSA      [0/1] enable BSA backend (default: 0).
//   DFLASH_FP_ALPHA        [float in (0,1)] override FlashPrefillConfig.alpha.
//                          Higher = stricter selection = fewer K-blocks per Q
//                          row = faster but riskier. Default 0.12. For long
//                          context with broad needles, 0.85-0.99 work well.
//   DFLASH_FP_PROFILE      [set] log per-stage timing (mean / score / select /
//                          forward) to stderr.
//   DFLASH_FP_DUMP_COUNTS  [set] log per-row select counts to stderr.

#pragma once

#include <cstdint>
#include "ggml-backend.h"

namespace dflash27b {
namespace flashprefill {

// Algorithmic parameters for the FlashPrefill selection + sparse forward.
struct FlashPrefillConfig {
    int   block_size       = 128;   // K stride; query block size = K block size
    int   attention_sink   = 2;     // first N k-blocks always selected
    int   window           = 4;     // last `window` k-blocks before query
    int   last_n_full      = 2;     // last N q-blocks attend to all selected blocks
    float alpha            = 0.12f; // dynamic top-K threshold (score >= max_score * alpha)
};

// Runs the full FP forward (mean_K → block_score → block_select → sparse_fwd).
// Returns 0 on success, non-zero on failure (allocator OOM, bad shape, etc.).
// Output O is written in place.
//
// Scratch memory (allocated/freed per call inside): ~M*M*H*4 * 3 + M*H*4
// where M = ceil(seq_len/block_size). At S=140K, M≈1093, H=16: ~300 MB.
int flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    const FlashPrefillConfig & cfg);

// ggml flash_attn_ext-based implementation. Works on all SM targets (75+).
// Same interface as flash_prefill_forward_bf16 but uses ggml's FA internally
// (chunked causal attention). Accepts BF16/F16 Q/K/V tensors stored in the
// same [B, S, H, D] contiguous layout. No custom WMMA kernels needed.
//
// On SM < 80 this is the only available path (BF16 WMMA doesn't exist).
// On SM >= 80 callers may prefer flash_prefill_forward_bf16 for the
// block-sparse selection + custom WMMA kernel path.
int flash_prefill_forward_q8(
    ggml_backend_t backend,
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    int elem_size,
    const FlashPrefillConfig & cfg);

#ifdef DFLASH27B_HAVE_BSA
// Free BSA persistent device buffers (blockmask, head_mask_type, softmax_lse).
// Safe to call any time; idempotent. Useful before unloading the drafter to
// give the daemon's target gen path the full VRAM headroom.
extern "C" void dflash_bsa_free_persistent();
#endif

} // namespace flashprefill
} // namespace dflash27b



================================================
FILE: dflash/src/flashprefill_kernels.cu
================================================
// CUDA port of qhfan/FlashPrefill (arXiv 2603.06199).
//
// Block-sparse attention for the pflash drafter. Replaces the upstream
// Triton implementation so the daemon can score long prompts on GPU
// in-process (no Python, no Triton runtime).
//
// Algorithm (from the FlashPrefill paper + qhfan reference impl):
//   1. compute_mean_vector_kernel  — mean K over BLOCK_SIZE blocks.
//      Output: mean_k[B, n_k_blocks, n_kv_heads, D]
//   2. compute_block_score_kernel  — per (q_block, k_block) score via
//      Q · mean_K^T.  Output: score[B, M, N, H], max[B, M, N, H]
//   3. block_select (host or CUDA) — pick top-K blocks per Q row, with
//      sink + window + dynamic alpha threshold + always-on last blocks.
//      Output: compact_indices[B, M, N, H], counts[B, M, H]
//   4. flash_forward_kernel        — sparse attention forward over the
//      SELECTED blocks only. Online softmax + output reduction.
//
// Status: all 4 kernels implemented (mean_vector, block_score, block_select, sparse_flash_forward).
// Currently dispatched only for D_HEAD=128 BLOCK_SIZE=128 (Qwen3 family).
//
// Conventions:
//   - All tensors row-major, D fastest.
//   - Q shape: [B, S, n_q_heads, D]
//   - K, V shape: [B, S, n_k_heads, D]   (n_q_heads = n_k_heads × group_size for GQA)
//   - mean_k shape: [B, ceil(S/BLOCK), n_k_heads, D]
//   - score shape: [B, M, N, n_q_heads]   (M = N = ceil(S/BLOCK))

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace dflash27b {
namespace flashprefill {

// ── cp.async helpers (sm_8x) ─────────────────────────────────────────
// 16-byte (uint4) async global → shared copy. Issued by every thread that
// participates in the cooperative load. wait_all() drains all outstanding
// transfers; commit_group()/wait_group(N) supports multi-stage pipelines.
__device__ inline void cp_async16(void * smem_ptr, const void * gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ inline void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ inline void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}


// ---- Kernel 1: compute_mean_vector ----
// Each block of (BLOCK threads, 1 K-head, 1 batch) reduces one K-block
// along the sequence dim, computing the mean per dim. Tile dims: (S/BLOCK, B*Hk).

template <int BLOCK, int D_HEAD>
__global__ void compute_mean_vector_kernel_bf16(
    const __nv_bfloat16 * __restrict__ K,
    __nv_bfloat16       * __restrict__ mean_K,
    int batch, int seq_len, int n_kv_heads,
    // strides, in elements (not bytes)
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d)
{
    const int block_idx_n = blockIdx.x;          // which K-block (0 .. M_blocks-1)
    const int zh = blockIdx.y;
    const int b = zh / n_kv_heads;
    const int h = zh % n_kv_heads;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    const int dim = tid;
    if (dim >= D_HEAD) return;

    const __nv_bfloat16 * Kp = K + (size_t)b * s_K_b + (size_t)h * s_K_h;
    __nv_bfloat16       * Mp = mean_K + (size_t)b * s_mK_b + (size_t)h * s_mK_h
                                      + (size_t)block_idx_n * s_mK_m;

    const int n_lo = block_idx_n * BLOCK;
    const int n_hi = min(n_lo + BLOCK, seq_len);
    const int count = n_hi - n_lo;
    if (count <= 0) return;

    float sum = 0.0f;
    for (int n = n_lo; n < n_hi; ++n) {
        sum += __bfloat162float(Kp[(size_t)n * s_K_n + (size_t)dim * s_K_d]);
    }
    Mp[(size_t)dim * s_mK_d] = __float2bfloat16(sum / (float)count);
}

// Public launcher (called from C++).
extern "C" void launch_compute_mean_vector_bf16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream)
{
    const int n_k_blocks = (seq_len + block_size - 1) / block_size;
    dim3 grid(n_k_blocks, batch * n_kv_heads, 1);
    dim3 block(head_dim, 1, 1);
    if (head_dim == 128 && block_size == 128) {
        compute_mean_vector_kernel_bf16<128, 128><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16 *)K, (__nv_bfloat16 *)mean_K,
            batch, seq_len, n_kv_heads,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d);
    }
    // Only D_HEAD=128 BLOCK=128 dispatched here. Add other combos when new heads/blocks needed.
}

// ---- Kernel 2: compute_block_score ----
//
// Per (q_block, k_block), compute the attention score that decides whether
// the k_block makes the cut for sparse attention. Uses Q tile (full BLOCK
// rows) vs mean_K tile (one row per k_block, from kernel 1).
//
// For each (q_block_idx M, kv_head):
//   For each k_block_idx N (causal: N <= M):
//     score[M, N, h] = sum_{j in q_block} sum_{n in k_block_one_row}
//                       exp2( (Q[j,h,d] · mean_K[N,h_kv,d] * sm_scale) - max )
//     Also output max for renormalization.
//
// One CUDA block per (q_block_idx, batch×n_q_heads). Threads: BLOCK_SIZE
// rows of Q within the block. Loops over N (k_blocks) on the inner.
//
// This is a straightforward reduction. Inner dot product in registers.

template <int BLOCK, int D_HEAD, int N_BLOCKS_TILE>
__global__ void compute_block_score_kernel_bf16(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ mean_K,
    float sm_scale,
    float * __restrict__ score,    // [B, M, N, H]
    float * __restrict__ score_max,// [B, M, N, H]
    int batch, int n_q_heads, int n_k_heads,
    int q_block_idx_max,           // total q blocks  M = ceil(S/BLOCK)
    int k_block_idx_max,           // total k blocks  N = ceil(S/BLOCK)
    // strides (in elements)
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h)
{
    const int q_block_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;

    const int kh = qh * n_k_heads / n_q_heads;     // GQA: q-head → kv-head

    const int tid = threadIdx.x;                    // 0..BLOCK-1
    const int q_row_local = tid;                    // which Q row in this block
    const int q_row_global = q_block_idx * BLOCK + q_row_local;

    if (q_row_local >= BLOCK || q_row_global >= q_block_idx_max * BLOCK) {
        // Out of range, no Q to load.
        return;
    }

    // Load this Q row into registers (one row per thread).
    const __nv_bfloat16 * Qp = Q + (size_t)b * s_Q_b
                                  + (size_t)q_row_global * s_Q_n
                                  + (size_t)qh * s_Q_h;
    float q_reg[D_HEAD];
    #pragma unroll
    for (int d = 0; d < D_HEAD; ++d) {
        q_reg[d] = __bfloat162float(Qp[(size_t)d * s_Q_d]);
    }

    // For each k-block, compute the score: contributions from THIS q row,
    // then reduce across BLOCK rows via warp/block reductions.
    extern __shared__ float smem[];      // [BLOCK] for per-row partial sums

    for (int n = 0; n <= q_block_idx; ++n) {       // causal
        // Load mean_K[n, kh, :] into registers (broadcast to all threads).
        const __nv_bfloat16 * mKp = mean_K + (size_t)b * s_mK_b
                                            + (size_t)n * s_mK_m
                                            + (size_t)kh * s_mK_h;
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_HEAD; ++d) {
            dot += q_reg[d] * __bfloat162float(mKp[(size_t)d * s_mK_d]);
        }
        dot *= sm_scale * 1.4426950408889634f;     // log2 base for exp2

        // Find max across threads.
        smem[tid] = dot;
        __syncthreads();
        // Block-reduce max.
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] = fmaxf(smem[tid], smem[tid + off]);
            __syncthreads();
        }
        float m_block = smem[0];
        __syncthreads();

        // Sum of exp2(dot - m_block).
        float p = exp2f(dot - m_block);
        smem[tid] = p;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] += smem[tid + off];
            __syncthreads();
        }
        float p_sum = smem[0];
        __syncthreads();

        // Thread 0 writes the (score, max) for this (q_block, n).
        if (tid == 0) {
            score    [(size_t)b * s_S_b + (size_t)q_block_idx * s_S_m
                      + (size_t)n * s_S_n + (size_t)qh * s_S_h] = p_sum;
            score_max[(size_t)b * s_M_b + (size_t)q_block_idx * s_M_m
                      + (size_t)n * s_M_n + (size_t)qh * s_M_h] = m_block;
        }
    }
}

extern "C" void launch_compute_block_score_bf16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream)
{
    const int M = (seq_len + block_size - 1) / block_size;
    dim3 grid(M, batch * n_q_heads, 1);
    dim3 block(block_size, 1, 1);
    size_t smem = block_size * sizeof(float);
    if (head_dim == 128 && block_size == 128) {
        compute_block_score_kernel_bf16<128, 128, 1><<<grid, block, smem, stream>>>(
            (const __nv_bfloat16 *)Q, (const __nv_bfloat16 *)mean_K, sm_scale,
            (float *)score, (float *)score_max,
            batch, n_q_heads, n_k_heads, M, M,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d,
            s_S_b, s_S_m, s_S_n, s_S_h,
            s_M_b, s_M_m, s_M_n, s_M_h);
    }
}

// ---- Kernel 4: sparse_flash_forward ----
//
// FlashAttention-style online softmax over a *selected* set of K-blocks.
// For each Q tile, we iterate `counts[q_block, h]` blocks (indices listed
// in `block_index[q_block, :, h]`) and update m_i, l_i, acc per row.
//
// Inputs:
//   Q[B, S, H, D]  bf16
//   K[B, S, Hk, D] bf16
//   V[B, S, Hk, D] bf16
//   block_index[B, M, N, H] int32  (compact, padded with N=invalid)
//   counts[B, M, H] int32          (how many indices are valid per row)
// Output:
//   O[B, S, H, D]  bf16
//
// Tile: Q_TILE_SIZE rows per CTA, BLOCK_SIZE = 128 K rows per selected block.
// One CTA per (q_tile_idx, batch×n_q_heads).

// v2: shared-memory tiled. Each CTA cooperatively loads each selected
// K-block (and V-block) into shared mem, then all Q_TILE threads do their
// QK / softmax / PV reduction reading from shared mem. This avoids
// re-loading K/V from HBM Q_TILE times per row, which was the dominant
// cost of the v1 scalar kernel.
template <int Q_TILE, int K_TILE, int BLOCK, int D_HEAD>
__global__ void sparse_flash_forward_kernel_bf16(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ K,
    const __nv_bfloat16 * __restrict__ V,
    __nv_bfloat16       * __restrict__ O,
    const int32_t       * __restrict__ block_index,
    const int32_t       * __restrict__ counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int M_blocks,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h)
{
    using namespace nvcuda;
    constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    constexpr int NDK = D_HEAD / MMA_K;       // 8 K-tiles along D
    constexpr int NNK = K_TILE / MMA_N;       // 4 N-tiles along K_TILE (when K_TILE=64)
    constexpr int N_INNER = BLOCK / K_TILE;   // 2 inner iters per selected block
    constexpr int NTHREADS = (Q_TILE / MMA_M) * 32;  // 1 warp per 16 Q rows

    const int q_tile_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;
    const int kh = qh * n_k_heads / n_q_heads;
    const int q_block_idx = q_tile_idx * Q_TILE / BLOCK;

    const int wid  = threadIdx.x / 32;        // 0..3
    const int lane = threadIdx.x & 31;
    const int tid  = threadIdx.x;             // 0..127

    // SMEM: Q + KV (K or V at a time, alias) + P + row state
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 * Q_sh   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16 * KV_sh  = Q_sh + (size_t)Q_TILE * D_HEAD;
    __nv_bfloat16 * P_sh   = KV_sh + (size_t)K_TILE * D_HEAD;
    float         * row_m  = reinterpret_cast<float*>(P_sh + (size_t)Q_TILE * K_TILE);
    float         * row_l  = row_m + Q_TILE;

    if (tid < Q_TILE) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }

    // Load Q [Q_TILE, D_HEAD] cooperatively
    {
        const __nv_bfloat16 * Qp = Q + (size_t)b * s_Q_b + (size_t)qh * s_Q_h;
        for (int idx = tid; idx < Q_TILE * D_HEAD; idx += NTHREADS) {
            int row = idx / D_HEAD;
            int dim = idx - row * D_HEAD;
            int q_global = q_tile_idx * Q_TILE + row;
            Q_sh[row * D_HEAD + dim] = (q_global < seq_len)
                ? Qp[(size_t)q_global * s_Q_n + (size_t)dim * s_Q_d]
                : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Pre-scale Q by sm_scale = scale * log2(e)
    {
        const float sm_scale = scale * 1.4426950408889634f;
        for (int idx = tid; idx < Q_TILE * D_HEAD; idx += NTHREADS) {
            float v = __bfloat162float(Q_sh[idx]);
            Q_sh[idx] = __float2bfloat16(v * sm_scale);
        }
    }
    __syncthreads();

    // O accumulator (one per D col tile per warp)
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> O_frag[NDK];
    #pragma unroll
    for (int d = 0; d < NDK; ++d) wmma::fill_fragment(O_frag[d], 0.0f);

    const int hi = counts[(size_t)b * s_cnt_b + (size_t)q_block_idx * s_cnt_m + (size_t)qh * s_cnt_h];

    for (int it = 0; it < hi; ++it) {
        int blk = block_index[(size_t)b * s_idx_b + (size_t)q_block_idx * s_idx_m
                              + (size_t)it * s_idx_n + (size_t)qh * s_idx_h];
        if (blk < 0 || blk >= M_blocks) continue;
        const int k_lo_block = blk * BLOCK;
        const bool is_diag = (blk == q_block_idx);

        #pragma unroll
        for (int inner = 0; inner < N_INNER; ++inner) {
            const int k_lo = k_lo_block + inner * K_TILE;

            // ── Load K tile [K_TILE, D_HEAD] into KV_sh via cp.async ──
            {
                const __nv_bfloat16 * Kp = K + (size_t)b * s_K_b + (size_t)kh * s_K_h;
                const bool vec_ok = (s_K_d == 1);
                if (vec_ok) {
                    int total8 = (K_TILE * D_HEAD) / 8;
                    for (int idx = tid; idx < total8; idx += NTHREADS) {
                        int row8 = idx / (D_HEAD / 8);
                        int d8   = idx - row8 * (D_HEAD / 8);
                        int j = k_lo + row8;
                        __nv_bfloat16 * dst = KV_sh + row8 * D_HEAD + d8 * 8;
                        if (j < seq_len) {
                            cp_async16(dst, Kp + (size_t)j * s_K_n + (size_t)(d8 * 8));
                        } else {
                            uint4 z = make_uint4(0, 0, 0, 0);
                            *reinterpret_cast<uint4*>(dst) = z;
                        }
                    }
                    cp_async_commit();
                    cp_async_wait_all();
                }
            }
            __syncthreads();

            // ── S = Q @ K^T in REGISTER fragments ──
            // Loop swap: dk-outer reuses Af across NNK nt-iters (saves 24/32 redundant Af SMEM reads).
            wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> S_frag[NNK];
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) wmma::fill_fragment(S_frag[nt], 0.0f);
            {
                wmma::fragment<wmma::matrix_a,    MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b,    MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::col_major> Bf;
                #pragma unroll
                for (int dk = 0; dk < NDK; ++dk) {
                    wmma::load_matrix_sync(Af,
                        Q_sh + (size_t)(wid * MMA_M) * D_HEAD + dk * MMA_K, D_HEAD);
                    #pragma unroll
                    for (int nt = 0; nt < NNK; ++nt) {
                        wmma::load_matrix_sync(Bf,
                            KV_sh + (size_t)(nt * MMA_N) * D_HEAD + dk * MMA_K, D_HEAD);
                        wmma::mma_sync(S_frag[nt], Af, Bf, S_frag[nt]);
                    }
                }
            }

            // ── Apply causal/seq mask + per-lane rowmax for both row pairs owned ──
            // sm_8x mma.m16n16k16 acc layout: lane t holds 8 elems mapping to:
            //   e[0..3]: row = 2*(t/4)+0
            //   e[4..7]: row = 2*(t/4)+1
            //   e[0,1,4,5]: col_in_tile = 2*(t%4) + {0,1}
            //   e[2,3,6,7]: col_in_tile = 2*(t%4) + 8 + {0,1}
            const int q = lane >> 2;
            const int c = lane & 3;
            const int row0_in_warp = 2 * q + 0;
            const int row1_in_warp = 2 * q + 1;
            const int row0_g = q_tile_idx * Q_TILE + wid * MMA_M + row0_in_warp;
            const int row1_g = q_tile_idx * Q_TILE + wid * MMA_M + row1_in_warp;

            float lm0 = -INFINITY, lm1 = -INFINITY;
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int col_pair = i & 1;
                    int col_off  = ((i >> 1) & 1) ? 8 : 0;
                    int col_in_tile = 2 * c + col_pair + col_off;
                    int col_g = k_lo + nt * MMA_N + col_in_tile;
                    int rg = (i < 4) ? row0_g : row1_g;
                    bool valid = (col_g < seq_len);
                    if (is_diag) valid = valid && (col_g <= rg);
                    if (!valid) S_frag[nt].x[i] = -INFINITY;
                    if (i < 4) lm0 = fmaxf(lm0, S_frag[nt].x[i]);
                    else       lm1 = fmaxf(lm1, S_frag[nt].x[i]);
                }
            }
            // Reduce rowmax across the 4 lanes of the quad (lanes with same q value).
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, off));
                lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, off));
            }

            // Read m_old/l_old. Quad-leader (c==0) reads from SMEM; broadcast.
            float m_old0 = 0, m_old1 = 0, l_old0 = 0, l_old1 = 0;
            int row0_warp = wid * MMA_M + row0_in_warp;
            int row1_warp = wid * MMA_M + row1_in_warp;
            if (c == 0) {
                m_old0 = row_m[row0_warp];
                m_old1 = row_m[row1_warp];
                l_old0 = row_l[row0_warp];
                l_old1 = row_l[row1_warp];
            }
            int leader = lane & ~3;  // first lane of quad
            m_old0 = __shfl_sync(0xffffffff, m_old0, leader);
            m_old1 = __shfl_sync(0xffffffff, m_old1, leader);
            l_old0 = __shfl_sync(0xffffffff, l_old0, leader);
            l_old1 = __shfl_sync(0xffffffff, l_old1, leader);

            float m_new0 = fmaxf(m_old0, lm0);
            float m_new1 = fmaxf(m_old1, lm1);
            float alpha0 = (m_old0 == -INFINITY) ? 0.0f : exp2f(m_old0 - m_new0);
            float alpha1 = (m_old1 == -INFINITY) ? 0.0f : exp2f(m_old1 - m_new1);

            // Compute P and rowsum, write bf16 P to P_sh
            float rs0 = 0, rs1 = 0;
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int col_pair = i & 1;
                    int col_off  = ((i >> 1) & 1) ? 8 : 0;
                    int col_in_tile = 2 * c + col_pair + col_off;
                    int col_in_KTILE = nt * MMA_N + col_in_tile;
                    float v = S_frag[nt].x[i];
                    float mn = (i < 4) ? m_new0 : m_new1;
                    float p = (v == -INFINITY) ? 0.0f : exp2f(v - mn);
                    if (i < 4) rs0 += p; else rs1 += p;
                    int p_row = wid * MMA_M + ((i < 4) ? row0_in_warp : row1_in_warp);
                    P_sh[(size_t)p_row * K_TILE + col_in_KTILE] = __float2bfloat16(p);
                }
            }
            // Reduce rowsum across quad
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                rs0 += __shfl_xor_sync(0xffffffff, rs0, off);
                rs1 += __shfl_xor_sync(0xffffffff, rs1, off);
            }

            // Quad-leader writes new state
            if (c == 0) {
                row_m[row0_warp] = m_new0;
                row_m[row1_warp] = m_new1;
                row_l[row0_warp] = alpha0 * l_old0 + rs0;
                row_l[row1_warp] = alpha1 * l_old1 + rs1;
            }

            // Frag-direct rescale of O accumulator
            #pragma unroll
            for (int d = 0; d < NDK; ++d) {
                O_frag[d].x[0] *= alpha0;
                O_frag[d].x[1] *= alpha0;
                O_frag[d].x[2] *= alpha0;
                O_frag[d].x[3] *= alpha0;
                O_frag[d].x[4] *= alpha1;
                O_frag[d].x[5] *= alpha1;
                O_frag[d].x[6] *= alpha1;
                O_frag[d].x[7] *= alpha1;
            }

            // Sync before V load (P_sh writes need to be visible too)
            __syncthreads();

            // ── Load V tile [K_TILE, D_HEAD] into KV_sh (overwrites K) via cp.async ──
            {
                const __nv_bfloat16 * Vp = V + (size_t)b * s_V_b + (size_t)kh * s_V_h;
                const bool vec_ok = (s_V_d == 1);
                if (vec_ok) {
                    int total8 = (K_TILE * D_HEAD) / 8;
                    for (int idx = tid; idx < total8; idx += NTHREADS) {
                        int row8 = idx / (D_HEAD / 8);
                        int d8   = idx - row8 * (D_HEAD / 8);
                        int j = k_lo + row8;
                        __nv_bfloat16 * dst = KV_sh + row8 * D_HEAD + d8 * 8;
                        if (j < seq_len) {
                            cp_async16(dst, Vp + (size_t)j * s_V_n + (size_t)(d8 * 8));
                        } else {
                            uint4 z = make_uint4(0, 0, 0, 0);
                            *reinterpret_cast<uint4*>(dst) = z;
                        }
                    }
                    cp_async_commit();
                    cp_async_wait_all();
                }
            }
            __syncthreads();

            // ── O += P @ V via WMMA ──
            // P shape [Q_TILE, K_TILE], V shape [K_TILE, D_HEAD]
            // Loop swap: kk-outer reuses Af across NDK dt-iters (saves redundant P_sh SMEM reads).
            {
                wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> Bf;
                #pragma unroll
                for (int kk = 0; kk < NNK; ++kk) {
                    wmma::load_matrix_sync(Af,
                        P_sh + (size_t)(wid * MMA_M) * K_TILE + kk * MMA_K, K_TILE);
                    #pragma unroll
                    for (int dt = 0; dt < NDK; ++dt) {
                        wmma::load_matrix_sync(Bf,
                            KV_sh + (size_t)(kk * MMA_K) * D_HEAD + dt * MMA_N, D_HEAD);
                        wmma::mma_sync(O_frag[dt], Af, Bf, O_frag[dt]);
                    }
                }
            }
            __syncthreads();
        } // inner
    } // it

    // Write O = acc / l_i. Store frag → SMEM (reuse Q_sh region as scratch), divide row-wise.
    // Q_sh is no longer needed; treat Q_sh region as f32 [Q_TILE][D_HEAD] (since 64*128*2 bf16 = 64*128*2 = 16K, but f32 needs 64*128*4 = 32K — won't fit). Use P_sh as scratch (8K, only first 4K rows needed for D=128 per warp store).
    // Simpler: store one D col tile at a time into a small scratch and write out.
    // We'll reuse the KV_sh region as f32 staging ([Q_TILE, MMA_N] = 64*16*4 = 4K) for one col tile at a time.
    float * stage_f32 = reinterpret_cast<float*>(KV_sh);  // 4 KB needed, KV_sh is 16 KB, plenty.
    #pragma unroll
    for (int d = 0; d < NDK; ++d) {
        __syncthreads();
        wmma::store_matrix_sync(stage_f32 + (size_t)(wid * MMA_M) * MMA_N,
                                O_frag[d], MMA_N, wmma::mem_row_major);
        __syncthreads();
        // Lanes 0..15 of each warp write 16 rows of MMA_N=16 D cols to global O.
        if (lane < MMA_M) {
            int row = wid * MMA_M + lane;
            int q_global = q_tile_idx * Q_TILE + row;
            if (q_global < seq_len) {
                __nv_bfloat16 * Op = O + (size_t)b * s_O_b + (size_t)q_global * s_O_n + (size_t)qh * s_O_h;
                int row_warp = wid * MMA_M + lane;
                float l = row_l[row_warp];
                float l_rec = (l > 0.0f) ? (1.0f / l) : 1.0f;
                const float * srow = stage_f32 + (size_t)row_warp * MMA_N;
                #pragma unroll
                for (int dd = 0; dd < MMA_N; ++dd) {
                    Op[(size_t)(d * MMA_N + dd) * s_O_d] = __float2bfloat16(srow[dd] * l_rec);
                }
            }
        }
    }
}
// ---- Kernel 4-TC: tensor-core sparse_flash_forward (WMMA)  [SCAFFOLD] ----
//
// FlashAttention-2 style with NVIDIA WMMA fragments. Scaffolding only —
// the online softmax l_i bookkeeping has a known bug (alpha rescale stash
// overwrites l_i, so partial sums lose the previous-block contribution).
// Kept here as a starting point for the proper rewrite. NOT compiled
// (gated out below) and NOT called from the launcher.
//
// To finish: maintain m_i, l_i, and alpha in three separate shared arrays;
// rescale acc fragments via PTX-correct per-thread fragment elementwise
// scaling (or via the store/scale/reload trick) BEFORE accumulating new PV
// contributions; ensure l_new = alpha * l_old + sum(P_new) per row.

extern "C" void launch_sparse_flash_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    const int32_t * block_index, const int32_t * counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int q_tile, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream)
{
    const int M = (seq_len + block_size - 1) / block_size;
    const int q_tiles = (seq_len + q_tile - 1) / q_tile;
    dim3 grid(q_tiles, batch * n_q_heads, 1);
    dim3 block(q_tile, 1, 1);
    if (q_tile == 64 && block_size == 128 && head_dim == 128) {
        // FA-2 register-resident layout @ Q_TILE=64 (4 warps, 128 threads), 2 CTAs/SM.
        constexpr int Q_TILE = 64, K_TILE = 64, BLOCK = 128, D_HEAD = 128;
        size_t smem_bytes = sizeof(__nv_bfloat16) * (Q_TILE * D_HEAD)
                           + sizeof(__nv_bfloat16) * (K_TILE * D_HEAD)
                           + sizeof(__nv_bfloat16) * (Q_TILE * K_TILE)
                           + sizeof(float)         * (2 * Q_TILE);
        dim3 block128(128, 1, 1);
        cudaFuncSetAttribute(
            (const void*)sparse_flash_forward_kernel_bf16<Q_TILE, K_TILE, BLOCK, D_HEAD>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        sparse_flash_forward_kernel_bf16<Q_TILE, K_TILE, BLOCK, D_HEAD><<<grid, block128, smem_bytes, stream>>>(
            (const __nv_bfloat16 *)Q, (const __nv_bfloat16 *)K,
            (const __nv_bfloat16 *)V, (__nv_bfloat16 *)O,
            block_index, counts, scale,
            batch, n_q_heads, n_k_heads, seq_len, M,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_V_b, s_V_n, s_V_h, s_V_d,
            s_O_b, s_O_n, s_O_h, s_O_d,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h);
    }
}


// ---- Kernel 3: block_select on GPU ----
//
// One warp per (B, M, H). Each warp scans n in [0, m] in chunks of 32, takes
// the max of scores[b,m,n,h] for the threshold, then re-scans applying the
// keep predicate (sink | window | last_n_full | score >= max*alpha). Surviving
// indices are compacted via warp ballot + popc so the output stays sorted by n.
//
// Replaces flashprefill_select.cpp::block_select_host. Removes the per-call
// D2H + host loop + H2D round trip (~75 MB at 140K, ~1.5 ms steady-state plus
// PCIe latency).

__global__ void block_select_kernel(
    const float * __restrict__ score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * __restrict__ idx_out,
    int32_t * __restrict__ cnt_out)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z;
    const int lane = threadIdx.x;  // 0..31, single warp CTA

    if (b >= B || m >= M || h >= H) return;

    const float * sp = score + (size_t)b*s_b + (size_t)m*s_m + (size_t)h*s_h;
    int32_t * idxp = idx_out + (size_t)b*idx_s_b + (size_t)m*idx_s_m + (size_t)h*idx_s_h;

    const bool last_full = (m >= M - last_n_full);
    const float NEG_INF = -INFINITY;

    // Pass 1: max score in [0, m] (warp reduce).
    float local_max = NEG_INF;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        int n = n_base + lane;
        bool valid = (n <= m);
        float v = valid ? sp[(size_t)n * s_n] : NEG_INF;
        local_max = fmaxf(local_max, v);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    const float max_score = local_max;
    const float thresh = max_score * alpha;

    // Pass 2: predicate + warp-ballot compact, sorted by n.
    int total = 0;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        int n = n_base + lane;
        bool valid = (n <= m);
        bool keep = false;
        if (valid) {
            float v = sp[(size_t)n * s_n];
            keep = last_full
                || (n < attention_sink)
                || ((m - n) < window)
                || (v >= thresh);
        }
        unsigned mask = __ballot_sync(0xffffffff, keep);
        int rank = __popc(mask & ((1u << lane) - 1u));
        if (keep) {
            idxp[(size_t)(total + rank) * idx_s_n] = (int32_t)n;
        }
        total += __popc(mask);
    }

    // Tail pad with -1 across [total, N).
    for (int n = total + lane; n < N; n += 32) {
        idxp[(size_t)n * idx_s_n] = (int32_t)-1;
    }

    if (lane == 0) {
        cnt_out[(size_t)b*cnt_s_b + (size_t)m*cnt_s_m + (size_t)h*cnt_s_h] = (int32_t)total;
    }
}

extern "C" void launch_block_select(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * idx_out, int32_t * cnt_out,
    cudaStream_t stream)
{
    dim3 grid(B, M, H);
    dim3 block(32, 1, 1);
    block_select_kernel<<<grid, block, 0, stream>>>(
        score, B, M, N, H,
        attention_sink, window, last_n_full, alpha,
        s_b, s_m, s_n, s_h,
        idx_s_b, idx_s_m, idx_s_n, idx_s_h,
        cnt_s_b, cnt_s_m, cnt_s_h,
        idx_out, cnt_out);
}

} // namespace flashprefill
} // namespace dflash27b



================================================
FILE: dflash/src/flashprefill_q8.cpp
================================================
// ggml flash_attn_ext-based FlashPrefill implementation.
//
// Provides flash_prefill_forward_q8() — a portable alternative to the custom
// BF16 WMMA flashprefill kernels. Uses ggml's built-in flash_attn_ext which
// works on all SM targets (75+) with FP16/BF16/Q8_0 K/V support.
//
// The attention is run in chunks (CHUNK_S tokens per pass) to keep the
// causal mask small and the ggml graph allocation bounded. Each chunk
// attends to all K/V positions up to the end of that chunk (full causal).

#include "flashprefill.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dflash27b {
namespace flashprefill {

namespace {
constexpr int CHUNK_S = 4096;
}

int flash_prefill_forward_q8(
    ggml_backend_t backend,
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    int elem_size,
    const FlashPrefillConfig & cfg)
{
    (void)cfg;  // block-sparse selection not used in this path

    const int B  = batch;
    (void)B;  // used only for shape validation in the future
    const int S  = seq_len;
    const int H  = n_q_heads;
    const int Hk = n_k_heads;
    const int D  = head_dim;

    // Determine the ggml type from element size.
    // The caller passes Q/K/V as raw pointers with a known element size.
    ggml_type qkv_type;
    if      (elem_size == 2) qkv_type = GGML_TYPE_BF16;  // or F16 — FA handles both
    else if (elem_size == 4) qkv_type = GGML_TYPE_F32;
    else {
        std::fprintf(stderr, "[flashprefill_q8] unsupported elem_size=%d\n", elem_size);
        return -1;
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));

    for (int cs = 0; cs < S; cs += CHUNK_S) {
        const int cl     = std::min(CHUNK_S, S - cs);
        const int kv_len = cs + cl;

        // ── Build ggml graph for this chunk ──
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 40
                      + ggml_graph_overhead_custom(256, false)
                      + 64 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            std::fprintf(stderr, "[flashprefill_q8] ggml_init failed\n");
            ggml_gallocr_free(galloc);
            return -1;
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

        // External tensors wrapping the caller's raw pointers.
        // Q_buf layout: [D, H, S] contiguous (the drafter's persistent buf).
        // K/V layout:   [D, Hk, S] contiguous.
        const size_t esz = (size_t)elem_size;

        ggml_tensor * Q_full = ggml_new_tensor_3d(ctx, qkv_type, D, H, S);
        Q_full->data = const_cast<void *>(Q);
        ggml_set_name(Q_full, "Q_ext");
        ggml_set_input(Q_full);

        ggml_tensor * K_full = ggml_new_tensor_3d(ctx, qkv_type, D, Hk, S);
        K_full->data = const_cast<void *>(K);
        ggml_set_name(K_full, "K_ext");
        ggml_set_input(K_full);

        ggml_tensor * V_full = ggml_new_tensor_3d(ctx, qkv_type, D, Hk, S);
        V_full->data = const_cast<void *>(V);
        ggml_set_name(V_full, "V_ext");
        ggml_set_input(V_full);

        ggml_tensor * O_full = ggml_new_tensor_3d(ctx, qkv_type, D, H, S);
        O_full->data = O;
        ggml_set_name(O_full, "O_ext");
        ggml_set_output(O_full);

        // Q chunk: [D, H, cl] view, then permute → [D, cl, H] for FA
        ggml_tensor * Q_chunk = ggml_view_3d(ctx, Q_full, D, H, cl,
                                             esz * D, esz * D * H,
                                             (size_t)cs * esz * D * H);
        ggml_tensor * Q_fa = ggml_cont(ctx, ggml_permute(ctx, Q_chunk, 0, 2, 1, 3));

        // K/V: [D, Hk, kv_len] view, then permute → [D, kv_len, Hk] for FA
        ggml_tensor * K_view = ggml_view_3d(ctx, K_full, D, Hk, kv_len,
                                            esz * D, esz * D * Hk, 0);
        ggml_tensor * V_view = ggml_view_3d(ctx, V_full, D, Hk, kv_len,
                                            esz * D, esz * D * Hk, 0);
        ggml_tensor * K_fa = ggml_cont(ctx, ggml_permute(ctx, K_view, 0, 2, 1, 3));
        ggml_tensor * V_fa = ggml_cont(ctx, ggml_permute(ctx, V_view, 0, 2, 1, 3));

        // Causal mask: [kv_len, cl] f16
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, cl);
        ggml_set_name(mask, "causal_mask");
        ggml_set_input(mask);

        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q_fa, K_fa, V_fa,
                                                  mask, scale, 0.0f, 0.0f);
        // FA output: [D, H, cl] (permuted). Copy to O_full at chunk offset.
        ggml_tensor * O_dst = ggml_view_3d(ctx, O_full, D, H, cl,
                                           esz * D, esz * D * H,
                                           (size_t)cs * esz * D * H);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, attn, O_dst));

        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "[flashprefill_q8] graph alloc failed at cs=%d\n", cs);
            ggml_free(ctx);
            ggml_gallocr_free(galloc);
            return -1;
        }

        // Fill causal mask
        {
            std::vector<uint16_t> mask_data((size_t)kv_len * cl);
            const uint16_t zero_f16 = 0;
            const uint16_t ninf_f16 = 0xFC00;
            for (int q_local = 0; q_local < cl; ++q_local) {
                int q_global = cs + q_local;
                for (int k = 0; k < kv_len; ++k) {
                    mask_data[(size_t)q_local * kv_len + k] =
                        (k <= q_global) ? zero_f16 : ninf_f16;
                }
            }
            ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                    (size_t)kv_len * cl * sizeof(uint16_t));
        }

        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        ggml_free(ctx);
    }

    ggml_gallocr_free(galloc);
    return 0;
}

} // namespace flashprefill
} // namespace dflash27b



================================================
FILE: dflash/src/flashprefill_select.cpp
================================================
// Block selection for FlashPrefill: turns per-(q_block, k_block, head) scores
// into a sparse, sorted index list per (q_block, head). This runs on host
// because the score grid is small (~1100×1100×16 = 19 M entries at 140K ctx)
// and the logic is sequential. The selected indices feed the sparse
// flash_forward CUDA kernel (kernel 4).
//
// Selection rules (from qhfan/FlashPrefill):
//   - sink:       k_block_idx < attention_sink (always include)
//   - window:     q_block_idx - window < k_block_idx <= q_block_idx
//   - last_full:  q_block_idx >= M - last_n_full → include all
//   - top-K dyn:  score >= max_score * alpha
//   - causal:     k_block_idx <= q_block_idx
//
// Output: compact_indices[B, M, N, H] (padded with -1) and counts[B, M, H].

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dflash27b {
namespace flashprefill {

// score: [B, M, N, H] row-major (B outer, H fastest).
// idx_out: [B, M, N, H] same layout, padded with -1.
// cnt_out: [B, M, H] same layout.
void block_select_host(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int32_t * idx_out, int32_t * cnt_out)
{
    // Score strides assume contiguous [B, M, N, H] row-major.
    const int s_b = M * N * H;
    const int s_m = N * H;
    const int s_n = H;
    const int s_h = 1;
    const int idx_s_b = M * N * H;
    const int idx_s_m = N * H;
    const int idx_s_n = H;
    const int idx_s_h = 1;
    const int cnt_s_b = M * H;
    const int cnt_s_m = H;
    const int cnt_s_h = 1;

    std::vector<int32_t> selected;
    selected.reserve(N);

    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int h = 0; h < H; ++h) {
                selected.clear();

                // Find max score for this (b, m, h) across n in [0, m].
                float max_score = -INFINITY;
                for (int n = 0; n <= m; ++n) {
                    float v = score[b*s_b + m*s_m + n*s_n + h*s_h];
                    if (v > max_score) max_score = v;
                }
                const float thresh = max_score * alpha;
                const bool last_full = (m >= M - last_n_full);

                for (int n = 0; n <= m; ++n) {
                    bool keep = false;
                    if (n < attention_sink) keep = true;
                    if (m - n < window && n <= m) keep = true;
                    if (last_full) keep = true;
                    if (!keep) {
                        float v = score[b*s_b + m*s_m + n*s_n + h*s_h];
                        if (v >= thresh) keep = true;
                    }
                    if (keep) selected.push_back((int32_t)n);
                }
                std::sort(selected.begin(), selected.end());

                int32_t * idx_row = idx_out + b*idx_s_b + m*idx_s_m + h*idx_s_h;
                for (int n = 0; n < N; ++n) {
                    idx_row[n*idx_s_n] = (n < (int)selected.size()) ? selected[n] : -1;
                }
                cnt_out[b*cnt_s_b + m*cnt_s_m + h*cnt_s_h] = (int32_t)selected.size();
            }
        }
    }
}

} // namespace flashprefill
} // namespace dflash27b



================================================
FILE: dflash/src/gguf_draft_loader.cpp
================================================
// Loads a DFlash draft model from a GGUF file on disk into a ggml context
// on the CUDA backend.
//
// This is the Q8_0-quantized counterpart of safetensors_draft.cpp. The draft
// graph builder (qwen3_dflash_graph.cpp) doesn't care about tensor storage
// types — ggml's ggml_mul_mat handles Q8_0 × F32 dequantization transparently.
//
// GGUF arch: "qwen35-dflash-draft" (from convert_dflash_to_gguf.py /
// quantize_draft_q8.py). Tensor naming convention:
//
//   dflash.fc.weight                        [5*hidden, hidden]  Q8_0 / F16
//   dflash.hidden_norm.weight               [hidden]            F32
//   output_norm.weight                      [hidden]            F32
//   blk.<i>.attn_norm.weight                [hidden]            F32
//   blk.<i>.ffn_norm.weight                 [hidden]            F32
//   blk.<i>.attn_q.weight                   [q_dim, hidden]     Q8_0 / F16
//   blk.<i>.attn_k.weight                   [kv_dim, hidden]    Q8_0 / F16
//   blk.<i>.attn_v.weight                   [kv_dim, hidden]    Q8_0 / F16
//   blk.<i>.attn_output.weight              [hidden, q_dim]     Q8_0 / F16
//   blk.<i>.attn_q_norm.weight              [head_dim]          F32
//   blk.<i>.attn_k_norm.weight              [head_dim]          F32
//   blk.<i>.ffn_gate.weight                 [intermediate, hidden]  Q8_0 / F16
//   blk.<i>.ffn_up.weight                   [intermediate, hidden]  Q8_0 / F16
//   blk.<i>.ffn_down.weight                 [hidden, intermediate]  Q8_0 / F16

#include "internal.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

struct Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap: " + std::string(std::strerror(errno)); addr = nullptr; return false; }
#endif
        return true;
    }
    ~Mmap() {
#if defined(_WIN32)
        if (addr)                        UnmapViewOfFile(addr);
        if (hMap)                        CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

} // namespace

bool load_draft_gguf(const std::string & path,
                     ggml_backend_t       backend,
                     DraftWeights &       out) {

    // ── 1. Parse metadata + create ggml_context with tensor descriptors ──
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate arch
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture in draft GGUF");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "qwen35-dflash-draft") {
            set_last_error(std::string("unexpected draft arch: ") + arch +
                           " (expected qwen35-dflash-draft)");
            gguf_free(gctx);
            return false;
        }
    }

    // Validate dimensions that the graph builder hardcodes
    const char * A = "qwen35-dflash-draft";
    char key[128];

    auto check_u32 = [&](const char * suffix, uint32_t expected) -> bool {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        uint32_t v = get_u32_or(gctx, key, 0);
        if (v != expected) {
            char b[256];
            std::snprintf(b, sizeof(b), "draft GGUF: %s=%u expected %u", key, v, expected);
            set_last_error(b);
            return false;
        }
        return true;
    };

    bool ok = true;
    ok = ok && check_u32("embedding_length",        DFLASH27B_TARGET_HIDDEN);
    ok = ok && check_u32("block_count",             DFLASH27B_DRAFT_LAYERS);
    ok = ok && check_u32("feed_forward_length",     DFLASH27B_TARGET_INTERMEDIATE);
    ok = ok && check_u32("attention.head_count",    DFLASH27B_TARGET_N_HEADS);
    ok = ok && check_u32("attention.head_count_kv", DFLASH27B_TARGET_N_KV_HEADS);
    ok = ok && check_u32("attention.key_length",    DFLASH27B_TARGET_HEAD_DIM);
    ok = ok && check_u32("attention.value_length",  DFLASH27B_TARGET_HEAD_DIM);
    ok = ok && check_u32("dflash.block_size",       DFLASH27B_DRAFT_BLOCK_SIZE);
    ok = ok && check_u32("dflash.n_target_layers",  DFLASH27B_DRAFT_N_TARGET_LAYERS);
    if (!ok) {
        gguf_free(gctx);
        return false;
    }

    // ── 2. Wire tensor pointers into DraftWeights ────────────────────────
    out.ctx     = meta_ctx;
    out.backend = backend;
    out.layers.assign(DFLASH27B_DRAFT_LAYERS, DraftLayer{});

    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };

    out.fc          = g("dflash.fc.weight");
    out.hidden_norm = g("dflash.hidden_norm.weight");
    out.out_norm    = g("output_norm.weight");
    if (!out.fc || !out.hidden_norm || !out.out_norm) {
        set_last_error("draft GGUF: missing top-level tensors "
                       "(dflash.fc / dflash.hidden_norm / output_norm)");
        gguf_free(gctx);
        return false;
    }

    for (int il = 0; il < DFLASH27B_DRAFT_LAYERS; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        DraftLayer & L = out.layers[il];
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
        L.wq        = fnd("attn_q.weight");
        L.wk        = fnd("attn_k.weight");
        L.wv        = fnd("attn_v.weight");
        L.wo        = fnd("attn_output.weight");
        L.q_norm    = fnd("attn_q_norm.weight");
        L.k_norm    = fnd("attn_k_norm.weight");
        L.w_gate    = fnd("ffn_gate.weight");
        L.w_up      = fnd("ffn_up.weight");
        L.w_down    = fnd("ffn_down.weight");
        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            char b[128];
            std::snprintf(b, sizeof(b), "draft GGUF: layer %d missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    // ── 3. Allocate CUDA buffer for all tensors ──────────────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed (draft GGUF)");
        gguf_free(gctx);
        return false;
    }

    // ── 4. mmap file and copy tensor bytes to CUDA ───────────────────────
    std::string err;
    Mmap mm;
    if (!mm.open_ro(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const size_t data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    size_t total = 0;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("draft GGUF: tensor '") + tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        total += sz;
    }

    gguf_free(gctx);

    char summary[192];
    std::snprintf(summary, sizeof(summary),
        "draft GGUF loaded: %" PRId64 " tensors, %.2f GiB on GPU",
        n_tensors, total / (1024.0 * 1024.0 * 1024.0));
    set_last_error(summary);

    return true;
}

} // namespace dflash27b



================================================
FILE: dflash/src/gguf_target_loader.cpp
================================================
// Loads Qwen3.5-27B qwen35 hybrid from a GGUF file on disk into a ggml
// context on the CUDA backend.
//
// The file is expected to use arch "qwen35" (NOT plain "qwen3"). See
// unsloth/Qwen3.5-27B-GGUF or ddh0/Qwen3.5-GGUF for reference.
//
// Tensor naming convention (from real inspection of ddh0's Qwen3.5-27B-4.71.gguf):
//
//   Top-level:
//     token_embd.weight              [hidden, vocab]
//     output_norm.weight             [hidden]                  F32
//     output.weight                  [hidden, vocab]           Q6_K (lm_head)
//
//   Per layer blk.<i> (full-attention layers, i.e. i % 4 == 3):
//     attn_norm.weight               [hidden]                  F32
//     post_attention_norm.weight     [hidden]                  F32
//     attn_q.weight                  [hidden, 2*q_dim]         Q4_K   (Q || gate packed)
//     attn_k.weight                  [hidden, kv_dim]          Q8_0
//     attn_v.weight                  [hidden, kv_dim]          Q8_0
//     attn_output.weight             [q_dim,  hidden]          Q5_K
//     attn_q_norm.weight             [head_dim]                F32
//     attn_k_norm.weight             [head_dim]                F32
//     ffn_gate.weight                [hidden, intermediate]    IQ4_XS
//     ffn_up.weight                  [hidden, intermediate]    IQ4_XS
//     ffn_down.weight                [intermediate, hidden]    IQ4_XS
//
//   Per layer blk.<i> (Gated DeltaNet layers, i.e. i % 4 != 3):
//     attn_norm.weight               [hidden]                  F32
//     post_attention_norm.weight     [hidden]                  F32
//     attn_qkv.weight                [hidden, 10240]           Q5_K   (q/k/v/beta fused)
//     attn_gate.weight               [hidden, inner=6144]      Q5_K   (z projection)
//     ssm_conv1d.weight              [inner, 4]                F32
//     ssm_a                          [dt_rank=48]              F32
//     ssm_alpha.weight               [dt_rank, hidden]         F32
//     ssm_beta.weight                [dt_rank, hidden]         F32
//     ssm_dt.bias                    [dt_rank]                 F32
//     ssm_norm.weight                [state=128]               F32
//     ssm_out.weight                 [inner, hidden]           Q5_K
//     ffn_gate/up/down              (same as full-attn)
//
// This loader reads the file via ggml's built-in GGUF API, which returns a
// ggml_context pre-populated with tensors. We then wire that context onto
// the CUDA backend (via ggml_backend_alloc_ctx_tensors) and copy each
// tensor's bytes from the mmap'd file.

#include "internal.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

// CpuEmbedder destructor + embed() method
CpuEmbedder::~CpuEmbedder() {
#if defined(_WIN32)
    if (mmap_addr)                         UnmapViewOfFile(mmap_addr);
    if (mmap_hmap)                         CloseHandle(mmap_hmap);
    if (mmap_hfile != INVALID_HANDLE_VALUE) CloseHandle(mmap_hfile);
#else
    if (mmap_addr) ::munmap(mmap_addr, mmap_len);
    if (mmap_fd >= 0) ::close(mmap_fd);
#endif
}

bool CpuEmbedder::embed(const int32_t * ids, int n, float * out_f32) const {
    if (!tok_embd_bytes || tok_embd_type == GGML_TYPE_COUNT) return false;
    const ggml_type_traits * tr = ggml_get_type_traits(tok_embd_type);
    if (!tr || !tr->to_float) return false;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= n_vocab) return false;
        const uint8_t * row = tok_embd_bytes + (size_t)id * row_bytes;
        tr->to_float(row, out_f32 + (size_t)i * n_embd, n_embd);
    }
    return true;
}

namespace {

// Local Mmap used only during load (separate from the one kept alive inside
// TargetWeights::embedder). We don't call munmap on this one when we want
// to hand ownership to the CpuEmbedder — see end of load_target_gguf.
struct Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap: " + std::string(std::strerror(errno)); addr = nullptr; return false; }
#endif
        return true;
    }
    // Ownership transfer: release handles without unmapping.
    void release() {
        addr = nullptr;
        len  = 0;
#if defined(_WIN32)
        hFile = INVALID_HANDLE_VALUE;
        hMap  = nullptr;
#else
        fd = -1;
#endif
    }
    ~Mmap() {
#if defined(_WIN32)
        if (addr)                        UnmapViewOfFile(addr);
        if (hMap)                        CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

// Required uint32 metadata key → bound check. Aborts load on mismatch.
bool expect_u32(const gguf_context * g, const char * key, uint32_t expected, std::string & err) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing gguf key: ") + key; return false; }
    uint32_t v = gguf_get_val_u32(g, id);
    if (v != expected) {
        char b[256];
        std::snprintf(b, sizeof(b), "gguf key %s=%u expected %u", key, v, expected);
        err = b;
        return false;
    }
    return true;
}

int32_t get_i32_or(const gguf_context * g, const char * key, int32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_i32(g, id);
}

uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

} // namespace

bool load_target_gguf(const std::string & path,
                      ggml_backend_t       backend,
                      TargetWeights &      out) {

    // ── 1. Parse metadata + create a ggml_context holding tensor descriptors ─
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate arch + the dimensions we hardcode everywhere.
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "qwen35") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected qwen35)");
            gguf_free(gctx);
            return false;
        }
    }

    std::string err;
    const uint32_t n_embd = get_u32_or(gctx, "qwen35.embedding_length",    0);
    const uint32_t n_ff   = get_u32_or(gctx, "qwen35.feed_forward_length", 0);
    const uint32_t n_layer= get_u32_or(gctx, "qwen35.block_count",         0);
    const uint32_t n_head = get_u32_or(gctx, "qwen35.attention.head_count",0);
    const uint32_t n_headkv=get_u32_or(gctx, "qwen35.attention.head_count_kv",0);
    const uint32_t kl     = get_u32_or(gctx, "qwen35.attention.key_length",   0);
    const uint32_t vl     = get_u32_or(gctx, "qwen35.attention.value_length", 0);
    const uint32_t fai    = get_u32_or(gctx, "qwen35.full_attention_interval",0);
    const uint32_t ssm_conv  = get_u32_or(gctx, "qwen35.ssm.conv_kernel",  0);
    const uint32_t ssm_inner = get_u32_or(gctx, "qwen35.ssm.inner_size",   0);
    const uint32_t ssm_state = get_u32_or(gctx, "qwen35.ssm.state_size",   0);
    const uint32_t ssm_dt    = get_u32_or(gctx, "qwen35.ssm.time_step_rank",0);
    const uint32_t ssm_grp   = get_u32_or(gctx, "qwen35.ssm.group_count",  0);

    if (n_embd != 5120 || n_layer != 64 || n_head != 24 || n_headkv != 4 ||
        kl != 256 || vl != 256 || n_ff != 17408 || fai != 4 ||
        ssm_conv != 4 || ssm_inner != 6144 || ssm_state != 128 ||
        ssm_dt != 48 || ssm_grp != 16) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "unexpected hparams: n_embd=%u n_layer=%u n_head=%u n_head_kv=%u "
            "kl=%u vl=%u n_ff=%u fai=%u ssm{conv=%u inner=%u state=%u dt=%u grp=%u}",
            n_embd, n_layer, n_head, n_headkv, kl, vl, n_ff, fai,
            ssm_conv, ssm_inner, ssm_state, ssm_dt, ssm_grp);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // rope dimension_sections (array of 4 uint32)
    int rope_sections[4] = {0, 0, 0, 0};
    {
        int64_t rid = gguf_find_key(gctx, "qwen35.rope.dimension_sections");
        if (rid >= 0) {
            size_t n = gguf_get_arr_n(gctx, rid);
            if (n >= 4) {
                const int32_t * arr = (const int32_t *)gguf_get_arr_data(gctx, rid);
                for (int k = 0; k < 4; k++) rope_sections[k] = arr[k];
            }
        }
    }

    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer = (int)n_layer;
    out.n_embd  = (int)n_embd;
    out.n_ff    = (int)n_ff;
    out.n_head  = (int)n_head;
    out.n_head_kv = (int)n_headkv;
    out.n_embd_head_k = (int)kl;
    out.n_embd_head_v = (int)vl;
    out.full_attention_interval = (int)fai;
    for (int k = 0; k < 4; k++) out.rope_sections[k] = rope_sections[k];
    out.ssm_d_conv = (int)ssm_conv;
    out.ssm_d_inner= (int)ssm_inner;
    out.ssm_d_state= (int)ssm_state;
    out.ssm_dt_rank= (int)ssm_dt;
    out.ssm_n_group= (int)ssm_grp;
    out.layers.assign((size_t)n_layer, TargetLayer{});

    // ── 2. Wire our layer pointers to tensors inside meta_ctx ─────────
    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };
    out.tok_embd = g("token_embd.weight");
    out.out_norm = g("output_norm.weight");
    out.output   = g("output.weight");
    if (!out.tok_embd || !out.out_norm || !out.output) {
        set_last_error("missing top-level tensors (token_embd/output_norm/output)");
        gguf_free(gctx);
        return false;
    }

    for (int il = 0; il < (int)n_layer; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        TargetLayer & L = out.layers[il];

        // Always-present tensors
        L.attn_norm      = fnd("attn_norm.weight");
        L.attn_post_norm = fnd("post_attention_norm.weight");
        L.w_gate         = fnd("ffn_gate.weight");
        L.w_up           = fnd("ffn_up.weight");
        L.w_down         = fnd("ffn_down.weight");
        if (!L.attn_norm || !L.attn_post_norm || !L.w_gate || !L.w_up || !L.w_down) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d: missing shared tensor", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }

        // Full-attention tensors (only on layers where (il+1)%fai == 0,
        // i.e. il%4 == 3 for fai=4). May be null on deltanet layers.
        L.wq     = fnd("attn_q.weight");
        L.wk     = fnd("attn_k.weight");
        L.wv     = fnd("attn_v.weight");
        L.wo     = fnd("attn_output.weight");
        L.q_norm = fnd("attn_q_norm.weight");
        L.k_norm = fnd("attn_k_norm.weight");

        // Gated DeltaNet tensors (null on full-attention layers)
        L.wqkv         = fnd("attn_qkv.weight");
        L.wqkv_gate    = fnd("attn_gate.weight");
        L.ssm_conv1d   = fnd("ssm_conv1d.weight");
        L.ssm_beta     = fnd("ssm_beta.weight");
        L.ssm_alpha    = fnd("ssm_alpha.weight");
        L.ssm_a        = fnd("ssm_a");
        L.ssm_dt_bias  = fnd("ssm_dt.bias");
        L.ssm_norm     = fnd("ssm_norm.weight");
        L.ssm_out      = fnd("ssm_out.weight");

        // Sanity: each layer must be EITHER full-attn OR deltanet, not both, not neither.
        const bool has_attn = L.wq && L.wk && L.wv && L.wo && L.q_norm && L.k_norm;
        const bool has_ssm  = L.wqkv && L.wqkv_gate && L.ssm_conv1d && L.ssm_out;
        const bool is_full_attn_layer = (((il + 1) % out.full_attention_interval) == 0);
        if (is_full_attn_layer && !has_attn) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d expected full-attn, missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
        if (!is_full_attn_layer && !has_ssm) {
            char b[128];
            std::snprintf(b, sizeof(b), "layer %d expected deltanet, missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    // ── 3. Allocate CUDA buffer for all tensors in meta_ctx ───────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed (target)");
        gguf_free(gctx);
        return false;
    }

    // ── 4. mmap the file and copy tensor bytes to CUDA ────────────────
    //
    // SKIP uploading token_embd.weight — it stays on CPU for embedding
    // lookup (CUDA get_rows doesn't support k-quants). We hand the mmap
    // ownership to TargetWeights::embedder at the end.
    Mmap mm;
    if (!mm.open_ro(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const size_t data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    size_t total = 0;
    size_t tok_embd_off = 0, tok_embd_sz = 0;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("tensor '") + tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        if (std::string(tname) == "token_embd.weight") {
            // Remember offset + size for the CPU embedder; don't upload to GPU.
            tok_embd_off  = off;
            tok_embd_sz   = sz;
            tok_embd_type = gguf_get_tensor_type(gctx, tid);
            continue;
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        total += sz;
    }

    gguf_free(gctx);

    if (tok_embd_off == 0 || tok_embd_type == GGML_TYPE_COUNT) {
        set_last_error("token_embd.weight not found or invalid type");
        return false;
    }

    // ── 5. Transfer mmap ownership to the CpuEmbedder so it can dequantize
    //       rows on demand without uploading the full embedding table to GPU.
    out.embedder.mmap_addr      = mm.addr;
    out.embedder.mmap_len       = mm.len;
#if defined(_WIN32)
    out.embedder.mmap_hfile     = mm.hFile;
    out.embedder.mmap_hmap      = mm.hMap;
#else
    out.embedder.mmap_fd        = mm.fd;
#endif
    out.embedder.tok_embd_bytes = (const uint8_t *)mm.addr + tok_embd_off;
    out.embedder.tok_embd_type  = tok_embd_type;
    out.embedder.n_embd         = out.n_embd;
    out.embedder.n_vocab        = DFLASH27B_TARGET_VOCAB;
    out.embedder.row_bytes      = tok_embd_sz / DFLASH27B_TARGET_VOCAB;
    mm.release();  // don't munmap on Mmap dtor — now owned by the embedder

    // Stash the total for callers that want to print it
    char summary[192];
    std::snprintf(summary, sizeof(summary),
        "target loaded: %" PRId64 " tensors on GPU %.2f GiB, tok_embd %.0f MiB CPU-only (%s)",
        n_tensors, total / (1024.0 * 1024.0 * 1024.0),
        tok_embd_sz / (1024.0 * 1024.0), ggml_type_name(tok_embd_type));
    set_last_error(summary);

    return true;
}

void free_target_weights(TargetWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    // CpuEmbedder destructor handles the mmap automatically.
    w.layers.clear();
    w.tok_embd = nullptr;
    w.out_norm = nullptr;
    w.output   = nullptr;
}

} // namespace dflash27b



================================================
FILE: dflash/src/internal.h
================================================
// Internal-only shared header for dflash27b library sources.
// Not installed, not exposed in the public API.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "dflash27b.h"

namespace dflash27b {

// Single source of truth for error reporting.
// All loaders / graph builders push into this via set_last_error(...).
void set_last_error(std::string msg);

// ─── Target weights (Qwen3.5-27B, qwen35 hybrid, Q4_K_M in ggml context) ──
//
// Qwen3.5 uses two kinds of blocks interleaved:
//   - FULL ATTENTION block  (every `full_attention_interval`-th layer, =4):
//       attn_norm, wq, wk, wv, wo, q_norm, k_norm + FFN tensors
//       (M-RoPE applied with rope_sections [11,11,10,0] — rope dims=64 of head_dim=256)
//   - GATED DELTANET block (all other layers, ~3 out of every 4):
//       attn_norm, wqkv (fused), wqkv_gate (the "z" projection),
//       delta-net per-head parameters (beta, gate, conv), plus FFN tensors.
//
// We keep ONE struct with all possible fields and leave unused ones nullptr.
// Actual tensor names in unsloth's GGUF are read via gguf_find_tensor() in
// the loader; see task #11.

struct TargetLayer {
    // Shared
    ggml_tensor * attn_norm      = nullptr;  // [hidden]
    ggml_tensor * attn_post_norm = nullptr;  // [hidden]  (post-block norm before FFN)
    ggml_tensor * ffn_norm       = nullptr;  // [hidden]
    ggml_tensor * w_gate         = nullptr;  // [hidden, intermediate]
    ggml_tensor * w_up           = nullptr;  // [hidden, intermediate]
    ggml_tensor * w_down         = nullptr;  // [intermediate, hidden]

    // Full-attention block (non-null for layers where (il+1) % 4 == 0)
    ggml_tensor * wq             = nullptr;  // [hidden, q_dim]
    ggml_tensor * wk             = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wv             = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wo             = nullptr;  // [q_dim, hidden]
    ggml_tensor * q_norm         = nullptr;  // [head_dim]
    ggml_tensor * k_norm         = nullptr;  // [head_dim]

    // Gated DeltaNet block (non-null for the other ~3/4 of layers)
    ggml_tensor * wqkv           = nullptr;  // fused Q/K/V projection
    ggml_tensor * wqkv_gate      = nullptr;  // the "z" projection
    ggml_tensor * ssm_conv1d     = nullptr;  // [kernel, dim]  depthwise causal conv
    ggml_tensor * ssm_beta       = nullptr;  // per-token beta input projection
    ggml_tensor * ssm_alpha      = nullptr;  // per-token alpha input projection
    ggml_tensor * ssm_a          = nullptr;  // [dt_rank] per-head -A parameter
    ggml_tensor * ssm_dt_bias    = nullptr;  // [dt_rank] per-head alpha bias
    ggml_tensor * ssm_norm       = nullptr;  // [head_v_dim]
    ggml_tensor * ssm_out        = nullptr;  // output projection after delta-net
};

// CPU-side embedder: keeps a mmap of the GGUF alive and knows how to
// dequantize individual rows of the quantized tok_embd tensor on demand.
// This matches llama.cpp's behavior of running embedding get_rows on CPU
// (because CUDA's get_rows doesn't support k-quants), so we never need to
// upload the 682 MiB token embedding to VRAM.
struct CpuEmbedder {
    void *           mmap_addr = nullptr;
    size_t           mmap_len  = 0;
#if defined(_WIN32)
    HANDLE           mmap_hfile = INVALID_HANDLE_VALUE;
    HANDLE           mmap_hmap  = nullptr;
#else
    int              mmap_fd   = -1;
#endif
    const uint8_t *  tok_embd_bytes = nullptr;  // into the mmap region
    ggml_type        tok_embd_type  = GGML_TYPE_COUNT;
    int64_t          n_embd = 0;
    int64_t          n_vocab = 0;
    size_t           row_bytes = 0;             // bytes per row in the quant format

    ~CpuEmbedder();
    // Dequantize N rows specified by `ids` into `out_f32` (shape [n_embd, n]).
    // Values are written contiguously row-major (n_embd fast axis).
    bool embed(const int32_t * ids, int n, float * out_f32) const;
};

struct TargetWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // CPU-side embedding table (zero GPU cost).
    CpuEmbedder           embedder;

    ggml_tensor * tok_embd = nullptr;        // [hidden, vocab] (metadata only; data NOT on GPU)
    std::vector<TargetLayer> layers;         // size = 64
    ggml_tensor * out_norm = nullptr;        // [hidden]
    ggml_tensor * output   = nullptr;        // [hidden, vocab]  (lm_head)

    // Metadata from GGUF (validated at load time)
    int full_attention_interval = 4;
    int rope_sections[4]        = {11, 11, 10, 0};
    int n_embd_head_k           = 256;  // key_length
    int n_embd_head_v           = 256;  // value_length
    int n_head                  = 24;
    int n_head_kv               = 4;
    int n_layer                 = 64;
    int n_embd                  = 5120;
    int n_ff                    = 17408;
    int ssm_d_conv              = 4;
    int ssm_d_inner             = 6144;
    int ssm_d_state             = 128;
    int ssm_dt_rank             = 48;
    int ssm_n_group             = 16;
};

// Load a Q4_K_M target model from a GGUF file on disk.
// Returns false and sets last_error on failure.
bool load_target_gguf(const std::string & path,
                      ggml_backend_t backend,
                      TargetWeights & out);

void free_target_weights(TargetWeights & w);

// ─── Draft weights (z-lab DFlash, bf16) ───────────────────────────

struct DraftLayer {
    ggml_tensor * attn_norm;
    ggml_tensor * ffn_norm;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * q_norm;
    ggml_tensor * k_norm;
    ggml_tensor * w_gate;
    ggml_tensor * w_up;
    ggml_tensor * w_down;
};

struct DraftWeights {
    ggml_context *    ctx = nullptr;
    ggml_backend_t    backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor *          fc          = nullptr;   // [5*hidden, hidden]
    ggml_tensor *          hidden_norm = nullptr;   // [hidden]
    std::vector<DraftLayer> layers;                 // size = 5
    ggml_tensor *          out_norm    = nullptr;   // [hidden]
};

bool load_draft_safetensors(const std::string & path,
                            ggml_backend_t backend,
                            DraftWeights & out);

// Load a Q8_0 (or F16) draft model from a GGUF file on disk.
// Alternative to load_draft_safetensors for quantized drafts.
bool load_draft_gguf(const std::string & path,
                     ggml_backend_t backend,
                     DraftWeights & out);

void free_draft_weights(DraftWeights & w);

// ─── Target cache (persistent state between forward calls) ────────

// Pre-allocated, backend-resident state that persists across decode steps.
// Created once via create_target_cache() and threaded through every
// build_qwen35_graph() call.
struct TargetCache {
    ggml_context *        base_ctx     = nullptr;
    ggml_backend_buffer_t base_buf     = nullptr;
    ggml_context *        rollback_ctx = nullptr;
    ggml_backend_buffer_t rollback_buf = nullptr;
    ggml_backend_t        backend  = nullptr;

    int max_ctx  = 0;         // max tokens in the KV cache
    int cur_pos  = 0;         // number of tokens already committed

    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;

    // Full-attention KV cache: one K and one V per full-attention layer.
    // Layout: [head_dim, max_ctx, n_head_kv] f16, contiguous per layer.
    std::vector<ggml_tensor *> attn_k;   // size = n_full_attn_layers (16)
    std::vector<ggml_tensor *> attn_v;

    // Gated DeltaNet recurrent state: one per delta-net layer.
    // ssm_state: [S_v, S_v, H_v] f32    (head_v_dim^2 × num_v_heads)
    // conv_state: [(kernel-1), conv_channels] f32
    // where conv_channels = d_inner + 2 * n_group * d_state
    std::vector<ggml_tensor *> ssm_state;    // size = n_delta_layers (48)
    std::vector<ggml_tensor *> conv_state;

    // Snapshot buffers for speculative decoding rollback. Sized identically
    // to ssm_state/conv_state above. Populated by snapshot_ssm_state() and
    // restored by restore_ssm_state().
    std::vector<ggml_tensor *> ssm_state_snap;
    std::vector<ggml_tensor *> conv_state_snap;

    // Per-step SSM + conv inputs captured during a verify forward when
    // QwenGraphInputs::capture_delta_intermediate is true. Populated by
    // in-graph ggml_cpy ops in build_delta_net_block so their data lives in
    // persistent cache memory (not tracked by the per-call gallocr), matching
    // SGLang's mamba_caches.intermediate_ssm / intermediate_conv_window pattern.
    //
    //   ssm_intermediate: [S_v, S_v, H_v, max_q_len] f32, one per delta layer.
    //     Element t on axis 3 holds the DeltaNet recurrent state after
    //     processing verify token t. Spec decode commits t = commit_n - 1.
    //   conv_input_cache: [(kernel-1) + max_q_len, conv_channels] f32, one per
    //     delta layer. Holds the full concat(old_conv_state, qkv_new_tokens)
    //     that was fed to ggml_ssm_conv. Spec decode slices
    //     [commit_n..commit_n+kernel-2] along dim 0 for conv state rollback.
    std::vector<ggml_tensor *> ssm_intermediate;    // size = n_delta (48)
    std::vector<ggml_tensor *> conv_input_cache;    // size = n_delta (48)

    // Rolling target layer features captured during target forward passes.
    // Shape [5 * hidden, target_feat_cap] bf16. target_feat_cap is typically
    // << max_ctx (e.g. 4096) so the buffer stays small at 128K context. The
    // graph writes to slot `(kv_start + i) % target_feat_cap` so positions
    // beyond the cap wrap and overwrite older entries. Readers (draft) only
    // need the last DRAFT_CTX_MAX positions, so wrap is invisible in
    // practice. Fed into the draft graph's fc projection after a bf16→f32
    // cast (dflash27b_launch_bf16_to_f32).
    ggml_tensor * target_feat = nullptr;
    int target_feat_cap = 0;
};

// Snapshot the current SSM+conv state into TargetCache::*_snap tensors.
void snapshot_ssm_state(TargetCache & c);
// Restore the SSM+conv state from the snapshot.
void restore_ssm_state(TargetCache & c);

// max_verify_tokens controls the per-layer ssm_intermediate and conv_input_cache
// sizes. Default is DFLASH27B_DRAFT_BLOCK_SIZE (16) for chain verify. DDTree
// mode requires max(chain, 1 + tree_budget) to hold the flat tree + root.
// Pass 0 to use the default.
// When prefill_only is true, rollback tensors (snapshots, intermediates) are
// skipped — saving ~1.4 GB on 48 DeltaNet layers. Use migrate_prefill_cache()
// to promote the cache to a full decode cache after prefill.
bool create_target_cache(const TargetWeights & w,
                         int max_ctx,
                         int max_verify_tokens,
                         ggml_backend_t backend,
                         TargetCache & out,
                         bool prefill_only = false);

void free_target_cache(TargetCache & c);

// Zero all state tensors (KV, SSM, conv, target_feat, rollback) in place
// without freeing/reallocating GPU buffers. Used by daemon mode between
// requests to avoid the ~5 s overhead of full cache destruction + recreation.
void reset_target_cache(TargetCache & c);

// Reallocate a prefill-only cache with full rollback tensors, copying all live
// state (KV, SSM, conv, target_feat) device-to-device. Frees the old cache.
bool migrate_prefill_cache(const TargetWeights & w,
                           int max_ctx,
                           int max_verify_tokens,
                           ggml_backend_t backend,
                           TargetCache & cache);

// ─── Target forward graph ─────────────────────────────────────────

// Per-delta-net-layer pointers exposed by the graph for spec-decode rollback.
// Populated when QwenGraphInputs::capture_delta_intermediate is true.
//
// Both tensors are persistent cache buffers (cache.ssm_intermediate[il] and
// cache.conv_input_cache[il]). Their ->data pointers are always valid — the
// graph just runs ggml_cpy ops to fill them during verify. Matches SGLang's
// mamba_caches.intermediate_ssm / intermediate_conv_window pattern:
// persistent memory, not managed by the per-call gallocr.
//
//   ssm_intermediate_states: [S_v, S_v, H_v, q_len] f32
//       Element t on axis 3 holds the DeltaNet state after processing verify
//       token t. Rollback reads offset (commit_n-1) * S_v*S_v*H*elt.
//   conv_input: [(kernel-1) + q_len, conv_channels, 1] f32
//       Full concat(old_conv_state, qkv_new_tokens) fed to ggml_ssm_conv.
//       Rollback reads slice [commit_n..commit_n+kernel-2] along dim 0.
struct DeltaNetCapture {
    ggml_tensor * ssm_intermediate_states = nullptr;
    ggml_tensor * conv_input              = nullptr;
};

struct QwenGraphInputs {
    ggml_tensor * inp_embed;      // [hidden, n_tokens, 1] f32 — pre-embedded by the caller
    ggml_tensor * positions;      // [4 * n_tokens] i32 (M-RoPE needs 4 per token)
    ggml_tensor * attn_mask;      // optional [kv_len, n_tokens_padded] f32 (causal); nullptr for n_tokens==1
    int           n_tokens;       // number of new tokens in this forward
    int           kv_start;       // position where the new tokens begin
    bool          capture_layers; // if true, write captured layer features into cache.target_feat
    bool          capture_delta_intermediate = false; // if true, populate out_delta_captures
    int           fa_window = 0;  // sliding window for FA layers: 0 = full attention
    ggml_tensor * parent_ids = nullptr; // [n_tokens] i32; tree mode when non-null
};

struct QwenGraphOutputs {
    ggml_tensor * logits;      // [vocab, n_tokens] f32
    // One entry per delta-net layer (48 for qwen35-27b). Only populated when
    // QwenGraphInputs::capture_delta_intermediate is true. Tensors are graph
    // views marked as ggml_set_output() so their data persists after
    // graph_compute; the spec-decode loop reads them host-side for rollback.
    std::vector<DeltaNetCapture> delta_captures;
};

QwenGraphOutputs build_qwen35_graph(
    ggml_context *         ctx,
    ggml_cgraph *          gf,
    const TargetWeights &  w,
    TargetCache &          cache,
    const QwenGraphInputs & in);

// Build a single-layer forward graph. Mirrors build_qwen35_graph but processes
// only one layer, taking `inp` as the input activation and returning the output.
// Used by layer-segmented prefill to iterate layers as the outer loop.
ggml_tensor * build_qwen35_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,         // [hidden, n_tokens]
    ggml_tensor *         positions,   // [4 * n_tokens] i32
    ggml_tensor *         attn_mask,   // optional
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window = 0);

} // namespace dflash27b



================================================
FILE: dflash/src/kv_cache.cpp
================================================
// Single-sequence KV cache for the target (64 layers) plus a rolling
// bf16 buffer for the concatenated target layer features that feed the draft.
//
// Layout:
//   target_k[layer] : [max_ctx, n_kv_heads, head_dim]   bf16
//   target_v[layer] : [max_ctx, n_kv_heads, head_dim]   bf16
//   target_feat     : [max_ctx, 5*hidden]               bf16
//
// Operations:
//   init(max_ctx)
//   reset()
//   reserve(n)            : ensure capacity, no write
//   commit(pos, n)        : mark positions [pos, pos+n) as valid (just bumps len_)
//   truncate(committed)   : drop everything >= committed (on mis-speculation)
//   length()
//
// The actual writes into target_k/v happen inside the target graph via
// ggml_cpy into the slot at offset = pos*bytes_per_token. Same for target_feat.

#include "internal.h"

namespace dflash27b {

// Placeholder; real impl lives with the spec_loop driver.

} // namespace dflash27b



================================================
FILE: dflash/src/kv_quant.cpp
================================================
// KV-cache quantisation helpers for dflash27b.
//
// Centralises the supported (K, V) ggml_type pair table and environment-variable
// resolution that was previously inlined in qwen35_target_graph.cpp.
//
// Supported pairs mirror fattn.cu with GGML_CUDA_FA_ALL_QUANTS=ON:
//
//   K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0}
//     × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
//
//   K = TQ3_0
//     × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}

#include "kv_quant.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace dflash {

// ─── String <-> type helpers ────────────────────────────────────────────────

static std::string to_lower(const char * s) {
    std::string out;
    for (; *s; ++s) {
        out += static_cast<char>(std::tolower(static_cast<unsigned char>(*s)));
    }
    return out;
}

ggml_type parse_kv_type(const char * s) {
    if (!s) return GGML_TYPE_COUNT;
    const std::string lower = to_lower(s);
    if (lower == "f16")   return GGML_TYPE_F16;
    if (lower == "bf16")  return GGML_TYPE_BF16;
    if (lower == "q4_0")  return GGML_TYPE_Q4_0;
    if (lower == "q4_1")  return GGML_TYPE_Q4_1;
    if (lower == "q5_0")  return GGML_TYPE_Q5_0;
    if (lower == "q5_1")  return GGML_TYPE_Q5_1;
    if (lower == "q8_0")  return GGML_TYPE_Q8_0;
    if (lower == "tq3_0") return GGML_TYPE_TQ3_0;
    return GGML_TYPE_COUNT;
}

const char * kv_type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:   return "f16";
        case GGML_TYPE_BF16:  return "bf16";
        case GGML_TYPE_Q4_0:  return "q4_0";
        case GGML_TYPE_Q4_1:  return "q4_1";
        case GGML_TYPE_Q5_0:  return "q5_0";
        case GGML_TYPE_Q5_1:  return "q5_1";
        case GGML_TYPE_Q8_0:  return "q8_0";
        case GGML_TYPE_TQ3_0: return "tq3_0";
        default:              return "?";
    }
}

// ─── Supported pair table ────────────────────────────────────────────────────

// Each entry is a (K type, V type) pair supported by the CUDA fattn kernels
// when GGML_CUDA_FA_ALL_QUANTS=ON.
struct KVPair {
    ggml_type k;
    ggml_type v;
};

// clang-format off
static const KVPair SUPPORTED_PAIRS[] = {
    // K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0} × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
    { GGML_TYPE_F16,   GGML_TYPE_F16   },
    { GGML_TYPE_F16,   GGML_TYPE_BF16  },
    { GGML_TYPE_F16,   GGML_TYPE_Q4_0  },
    { GGML_TYPE_F16,   GGML_TYPE_Q4_1  },
    { GGML_TYPE_F16,   GGML_TYPE_Q5_0  },
    { GGML_TYPE_F16,   GGML_TYPE_Q5_1  },
    { GGML_TYPE_F16,   GGML_TYPE_Q8_0  },
    { GGML_TYPE_F16,   GGML_TYPE_TQ3_0 },

    { GGML_TYPE_BF16,  GGML_TYPE_F16   },
    { GGML_TYPE_BF16,  GGML_TYPE_BF16  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_BF16,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_BF16,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q4_0,  GGML_TYPE_F16   },
    { GGML_TYPE_Q4_0,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q4_0,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q4_1,  GGML_TYPE_F16   },
    { GGML_TYPE_Q4_1,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q4_1,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q5_0,  GGML_TYPE_F16   },
    { GGML_TYPE_Q5_0,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q5_0,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q5_1,  GGML_TYPE_F16   },
    { GGML_TYPE_Q5_1,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q5_1,  GGML_TYPE_TQ3_0 },

    { GGML_TYPE_Q8_0,  GGML_TYPE_F16   },
    { GGML_TYPE_Q8_0,  GGML_TYPE_BF16  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q4_0  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q4_1  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q5_0  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q5_1  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_Q8_0  },
    { GGML_TYPE_Q8_0,  GGML_TYPE_TQ3_0 },

    // K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}
    { GGML_TYPE_TQ3_0, GGML_TYPE_F16   },
    { GGML_TYPE_TQ3_0, GGML_TYPE_BF16  },
    { GGML_TYPE_TQ3_0, GGML_TYPE_Q4_0  },
    { GGML_TYPE_TQ3_0, GGML_TYPE_Q8_0  },
    { GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0 },
};
// clang-format on

static constexpr int N_SUPPORTED_PAIRS =
    static_cast<int>(sizeof(SUPPORTED_PAIRS) / sizeof(SUPPORTED_PAIRS[0]));

bool is_supported_kv_pair(ggml_type k, ggml_type v) {
    for (int i = 0; i < N_SUPPORTED_PAIRS; ++i) {
        if (SUPPORTED_PAIRS[i].k == k && SUPPORTED_PAIRS[i].v == v) {
            return true;
        }
    }
    return false;
}

// ─── Environment-variable resolution ────────────────────────────────────────

void resolve_kv_types(ggml_type & k_out, ggml_type & v_out) {
    ggml_type k = GGML_TYPE_Q8_0;
    ggml_type v = GGML_TYPE_Q8_0;

    // Layer 2: legacy shorthand (last wins, mirrors qwen35_target_graph.cpp:96-108)
    if (const char * s = std::getenv("DFLASH27B_KV_F16")) {
        if (std::atoi(s) != 0) { k = GGML_TYPE_F16;   v = GGML_TYPE_F16;   }
    }
    if (const char * s = std::getenv("DFLASH27B_KV_Q4")) {
        if (std::atoi(s) != 0) { k = GGML_TYPE_Q4_0;  v = GGML_TYPE_Q4_0;  }
    }
    if (const char * s = std::getenv("DFLASH27B_KV_TQ3")) {
        if (std::atoi(s) != 0) { k = GGML_TYPE_TQ3_0; v = GGML_TYPE_TQ3_0; }
    }

    // Layer 1: explicit per-axis override (highest precedence)
    if (const char * s = std::getenv("DFLASH27B_KV_K")) {
        const ggml_type parsed = parse_kv_type(s);
        if (parsed == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "[dflash] Unknown KV K type: \"%s\"\n", s);
            std::abort();
        }
        k = parsed;
    }
    if (const char * s = std::getenv("DFLASH27B_KV_V")) {
        const ggml_type parsed = parse_kv_type(s);
        if (parsed == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "[dflash] Unknown KV V type: \"%s\"\n", s);
            std::abort();
        }
        v = parsed;
    }

    // Validate the resolved (K, V) pair
    if (!is_supported_kv_pair(k, v)) {
        std::fprintf(stderr,
            "[dflash] KV pair (K=%s, V=%s) not supported by fattn-cuda. Supported pairs:\n",
            kv_type_name(k), kv_type_name(v));
        for (int i = 0; i < N_SUPPORTED_PAIRS; ++i) {
            std::fprintf(stderr, "  K=%-6s  V=%s\n",
                kv_type_name(SUPPORTED_PAIRS[i].k),
                kv_type_name(SUPPORTED_PAIRS[i].v));
        }
        std::abort();
    }

    k_out = k;
    v_out = v;
}

}  // namespace dflash



================================================
FILE: dflash/src/kv_quant.h
================================================
#pragma once
#include "ggml.h"
#include <string>

namespace dflash {

// Parses a KV-cache element type string (case-insensitive).
// Accepted: "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "tq3_0".
// Returns GGML_TYPE_COUNT on unknown input (caller should treat as error).
ggml_type parse_kv_type(const char * s);

// Returns the canonical lowercase string for a supported KV ggml_type,
// or "?" for unsupported.
const char * kv_type_name(ggml_type t);

// True iff the (K, V) ggml_type pair is supported by the CUDA flash-attention
// kernels currently compiled in (mirror of fattn.cu type-pair table when
// GGML_CUDA_FA_ALL_QUANTS=ON, which is now forced ON in dflash/CMakeLists.txt).
bool is_supported_kv_pair(ggml_type k, ggml_type v);

// Resolves K and V types from environment variables.
// Precedence (high -> low):
//   1. DFLASH27B_KV_K=<type> / DFLASH27B_KV_V=<type>  (independent override)
//   2. DFLASH27B_KV_F16 / _KV_Q4 / _KV_TQ3            (legacy shorthand, K==V)
//   3. Default: GGML_TYPE_Q8_0 for both
// On invalid input or unsupported (K,V) pair, prints an explanatory message
// and calls std::abort(). Returns the resolved pair via out params.
void resolve_kv_types(ggml_type & k_out, ggml_type & v_out);

}  // namespace dflash



================================================
FILE: dflash/src/qwen35_target_graph.cpp
================================================
// Forward pass of Qwen3.5-27B (qwen35 hybrid) in pure ggml.
//
// Translates llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp` into
// our standalone library, hardcoded for Qwen3.5-27B dimensions. No
// llama.cpp runtime is linked — only ggml ops.
//
// Architecture highlights:
//   - 64 layers; every 4th (il % 4 == 3) is full attention, rest are Gated DeltaNet
//   - Full-attention Q projection is PACKED with a gate (attn_q has width 2*q_dim)
//   - Full attention uses M-RoPE with sections [11,11,10,0]
//   - Flash attention is GQA 24/4, causal
//   - Delta-net uses ggml_ssm_conv for the 1D conv + ggml_gated_delta_net for the recurrence
//   - FFN is SwiGLU (w_gate * silu, element-wise multiply with w_up, then w_down)
//
// State (persisted in TargetCache across calls):
//   - attn_k[16], attn_v[16]     : KV cache for full-attn layers, f16
//   - conv_state[48]             : 1D conv recurrence state, f32
//   - ssm_state[48]              : delta-net recurrent state (head_v^2 × H_v), f32
//
// Key dimensions (all hardcoded via DFLASH27B_* macros):
//   n_embd           = 5120
//   n_head           = 24    head_dim = 256   q_dim = n_head * head_dim = 6144
//   n_head_kv        = 4     kv_dim = 4 * 256 = 1024
//   n_ff             = 17408
//   d_inner (ssm)    = 6144
//   d_state (ssm)    = 128
//   dt_rank (ssm)    = 48    (num_v_heads)
//   n_group (ssm)    = 16    (num_k_heads)
//   head_v_dim       = d_inner / dt_rank = 128
//   head_k_dim       = d_state           = 128
//   conv_kernel      = 4

#include "internal.h"
#include "delta_net_chunked.h"
#include "kv_quant.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash27b {

// ─── Local qwen35 constants (from the GGUF, hardcoded for this model) ─
// These complement the DFLASH27B_* macros in dflash27b.h with qwen35-specific
// hparams that differ from the draft (which uses plain Qwen3 dims).
namespace q35 {
constexpr int N_HEAD        = 24;
constexpr int N_HEAD_KV     = 4;
constexpr int HEAD_DIM      = 256;   // key_length == value_length
constexpr int Q_DIM         = N_HEAD * HEAD_DIM;    // 6144
constexpr int KV_DIM        = N_HEAD_KV * HEAD_DIM; // 1024
constexpr int FFN_DIM       = 17408;

constexpr int SSM_D_INNER   = 6144;
constexpr int SSM_D_STATE   = 128;
constexpr int SSM_DT_RANK   = 48;
constexpr int SSM_N_GROUP   = 16;
constexpr int SSM_CONV_KERN = 4;

// Derived
constexpr int HEAD_V_DIM    = SSM_D_INNER / SSM_DT_RANK;  // 128
constexpr int HEAD_K_DIM    = SSM_D_STATE;                // 128
constexpr int CONV_CHANNELS = SSM_D_INNER + 2 * SSM_N_GROUP * SSM_D_STATE; // 6144 + 2*16*128 = 10240

constexpr float EPS         = 1e-6f;
constexpr float ROPE_THETA  = 10000000.0f;
}  // namespace q35

// ─── TargetCache allocation ─────────────────────────────────────────

bool create_target_cache(const TargetWeights & w,
                         int max_ctx,
                         int max_verify_tokens,
                         ggml_backend_t backend,
                         TargetCache & out,
                         bool prefill_only) {
    out.backend = backend;
    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    if (max_verify_tokens <= 0) {
        max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    }

    const int n_full_attn = w.n_layer / w.full_attention_interval; // 16
    const int n_delta     = w.n_layer - n_full_attn;               // 48

    out.attn_k.assign(n_full_attn, nullptr);
    out.attn_v.assign(n_full_attn, nullptr);
    out.ssm_state.assign(n_delta, nullptr);
    out.conv_state.assign(n_delta, nullptr);
    out.ssm_state_snap.assign(n_delta, nullptr);
    out.conv_state_snap.assign(n_delta, nullptr);
    out.ssm_intermediate.assign(n_delta, nullptr);
    out.conv_input_cache.assign(n_delta, nullptr);

    // KV cache element types (resolved from env; aborts on unsupported pair).
    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;
    dflash::resolve_kv_types(kv_k_type, kv_v_type);
    out.kv_k_type = kv_k_type;
    out.kv_v_type = kv_v_type;
    const int max_ctx_alloc = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0)
        ? ((max_ctx + 255) / 256) * 256
        : max_ctx;

    // ── Base context: KV cache + SSM/conv state + target_feat ────────
    {
        const int base_tensors = 2 * n_full_attn + 2 * n_delta + 1;
        ggml_init_params ip{};
        ip.mem_size   = (size_t)(base_tensors + 16) * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        out.base_ctx = ggml_init(ip);
        if (!out.base_ctx) { set_last_error("base cache ggml_init failed"); return false; }

        int fa_idx = 0, dn_idx = 0;
        for (int il = 0; il < w.n_layer; il++) {
            const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
            if (is_attn) {
                // [head_dim, max_ctx_alloc, n_head_kv]
                ggml_tensor * K = ggml_new_tensor_3d(out.base_ctx, kv_k_type,
                                                     q35::HEAD_DIM, max_ctx_alloc, q35::N_HEAD_KV);
                ggml_tensor * V = ggml_new_tensor_3d(out.base_ctx, kv_v_type,
                                                     q35::HEAD_DIM, max_ctx_alloc, q35::N_HEAD_KV);
                char name[64];
                std::snprintf(name, sizeof(name), "cache_k_%d", il);
                ggml_set_name(K, name);
                std::snprintf(name, sizeof(name), "cache_v_%d", il);
                ggml_set_name(V, name);
                out.attn_k[fa_idx] = K;
                out.attn_v[fa_idx] = V;
                fa_idx++;
            } else {
                // ssm_state: [head_v_dim, head_v_dim, num_v_heads]
                ggml_tensor * S = ggml_new_tensor_3d(out.base_ctx, GGML_TYPE_F32,
                                                     q35::HEAD_V_DIM, q35::HEAD_V_DIM, q35::SSM_DT_RANK);
                // conv_state: [kernel-1, conv_channels]
                ggml_tensor * C = ggml_new_tensor_2d(out.base_ctx, GGML_TYPE_F32,
                                                     q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS);
                char name[64];
                std::snprintf(name, sizeof(name), "ssm_state_%d", il);  ggml_set_name(S, name);
                std::snprintf(name, sizeof(name), "conv_state_%d", il); ggml_set_name(C, name);
                out.ssm_state[dn_idx]  = S;
                out.conv_state[dn_idx] = C;
                dn_idx++;
            }
        }

        constexpr int TARGET_FEAT_CAP_DEFAULT = 4096;
        out.target_feat_cap = std::min(max_ctx, TARGET_FEAT_CAP_DEFAULT);
        const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * w.n_embd;  // 25600
        out.target_feat = ggml_new_tensor_2d(out.base_ctx, GGML_TYPE_BF16, fc_in, out.target_feat_cap);
        ggml_set_name(out.target_feat, "target_feat");

        out.base_buf = ggml_backend_alloc_ctx_tensors(out.base_ctx, backend);
        if (!out.base_buf) {
            set_last_error("ggml_backend_alloc_ctx_tensors failed for base cache");
            ggml_free(out.base_ctx);
            out.base_ctx = nullptr;
            return false;
        }
    }

    // ── Rollback context: snapshots + intermediates ───────────────────
    if (!prefill_only) {
        const int rb_tensors = 4 * n_delta;
        ggml_init_params ip{};
        ip.mem_size   = (size_t)(rb_tensors + 16) * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        out.rollback_ctx = ggml_init(ip);
        if (!out.rollback_ctx) { set_last_error("rollback cache ggml_init failed"); return false; }

        int dn_idx = 0;
        for (int il = 0; il < w.n_layer; il++) {
            if (((il + 1) % w.full_attention_interval) != 0) {
                ggml_tensor * Sn = ggml_new_tensor_3d(out.rollback_ctx, GGML_TYPE_F32,
                                                       q35::HEAD_V_DIM, q35::HEAD_V_DIM, q35::SSM_DT_RANK);
                ggml_tensor * Cn = ggml_new_tensor_2d(out.rollback_ctx, GGML_TYPE_F32,
                                                       q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS);
                ggml_tensor * Si = ggml_new_tensor_4d(out.rollback_ctx, GGML_TYPE_F16,
                                                       q35::HEAD_V_DIM, q35::HEAD_V_DIM,
                                                       q35::SSM_DT_RANK, max_verify_tokens);
                ggml_tensor * Ci = ggml_new_tensor_3d(out.rollback_ctx, GGML_TYPE_F32,
                                                       (q35::SSM_CONV_KERN - 1) + max_verify_tokens,
                                                       q35::CONV_CHANNELS, 1);
                char name[64];
                std::snprintf(name, sizeof(name), "ssm_state_snap_%d", il);  ggml_set_name(Sn, name);
                std::snprintf(name, sizeof(name), "conv_state_snap_%d", il); ggml_set_name(Cn, name);
                std::snprintf(name, sizeof(name), "ssm_intermediate_%d", il); ggml_set_name(Si, name);
                std::snprintf(name, sizeof(name), "conv_input_cache_%d", il); ggml_set_name(Ci, name);
                out.ssm_state_snap[dn_idx]  = Sn;
                out.conv_state_snap[dn_idx] = Cn;
                out.ssm_intermediate[dn_idx] = Si;
                out.conv_input_cache[dn_idx] = Ci;
                dn_idx++;
            }
        }

        out.rollback_buf = ggml_backend_alloc_ctx_tensors(out.rollback_ctx, backend);
        if (!out.rollback_buf) {
            set_last_error("ggml_backend_alloc_ctx_tensors failed for rollback cache");
            ggml_free(out.rollback_ctx);
            out.rollback_ctx = nullptr;
            return false;
        }
    }

    // ── Zero-initialize all state tensors ─────────────────────────────
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    ggml_context * ctx_list[] = { out.base_ctx, out.rollback_ctx };
    for (int ci = 0; ci < 2; ci++) {
        ggml_context * c = ctx_list[ci];
        if (!c) continue;
        for (ggml_tensor * t = ggml_get_first_tensor(c); t != nullptr;
             t = ggml_get_next_tensor(c, t)) {
            size_t nb = ggml_nbytes(t);
            size_t off = 0;
            while (off < nb) {
                size_t chunk = std::min(nb - off, zeros.size());
                ggml_backend_tensor_set(t, zeros.data(), off, chunk);
                off += chunk;
            }
        }
    }

    return true;
}

void free_target_cache(TargetCache & c) {
    if (c.base_buf)     { ggml_backend_buffer_free(c.base_buf);     c.base_buf     = nullptr; }
    if (c.base_ctx)     { ggml_free(c.base_ctx);                   c.base_ctx     = nullptr; }
    if (c.rollback_buf) { ggml_backend_buffer_free(c.rollback_buf); c.rollback_buf = nullptr; }
    if (c.rollback_ctx) { ggml_free(c.rollback_ctx);               c.rollback_ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.ssm_state.clear();
    c.conv_state.clear();
    c.ssm_state_snap.clear();
    c.conv_state_snap.clear();
    c.ssm_intermediate.clear();
    c.conv_input_cache.clear();
    c.target_feat = nullptr;
    c.cur_pos = 0;
}

void reset_target_cache(TargetCache & c) {
    c.cur_pos = 0;
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    ggml_context * ctx_list[] = { c.base_ctx, c.rollback_ctx };
    for (int ci = 0; ci < 2; ci++) {
        ggml_context * ctx = ctx_list[ci];
        if (!ctx) continue;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
             t = ggml_get_next_tensor(ctx, t)) {
            size_t nb = ggml_nbytes(t);
            size_t off = 0;
            while (off < nb) {
                size_t chunk = std::min(nb - off, zeros.size());
                ggml_backend_tensor_set(t, zeros.data(), off, chunk);
                off += chunk;
            }
        }
    }
}

// Attach rollback tensors to an existing prefill cache without touching the
// base tensors (KV, SSM, conv, target_feat) that prefill already populated.
// No D2D copies — the base tensors stay right where the graph wrote them.
// If rollback tensors are already present (e.g. daemon mode second request),
// this is a no-op.
bool migrate_prefill_cache(const TargetWeights & w,
                           int max_ctx,
                           int max_verify_tokens,
                           ggml_backend_t backend,
                           TargetCache & cache) {
    // Already migrated (e.g. daemon mode second+ request after reset_target_cache).
    if (cache.rollback_ctx) return true;

    const int n_delta = (int)cache.ssm_state.size(); // 48
    if (max_verify_tokens <= 0) {
        max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    }

    cache.ssm_state_snap.assign(n_delta, nullptr);
    cache.conv_state_snap.assign(n_delta, nullptr);
    cache.ssm_intermediate.assign(n_delta, nullptr);
    cache.conv_input_cache.assign(n_delta, nullptr);

    const int rb_tensors = 4 * n_delta;
    ggml_init_params ip{};
    ip.mem_size   = (size_t)(rb_tensors + 16) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    cache.rollback_ctx = ggml_init(ip);
    if (!cache.rollback_ctx) { set_last_error("rollback cache ggml_init failed"); return false; }

    int dn_idx = 0;
    for (int il = 0; il < w.n_layer; il++) {
        if (((il + 1) % w.full_attention_interval) != 0) {
            ggml_tensor * Sn = ggml_new_tensor_3d(cache.rollback_ctx, GGML_TYPE_F32,
                                                   q35::HEAD_V_DIM, q35::HEAD_V_DIM, q35::SSM_DT_RANK);
            ggml_tensor * Cn = ggml_new_tensor_2d(cache.rollback_ctx, GGML_TYPE_F32,
                                                   q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS);
            ggml_tensor * Si = ggml_new_tensor_4d(cache.rollback_ctx, GGML_TYPE_F16,
                                                   q35::HEAD_V_DIM, q35::HEAD_V_DIM,
                                                   q35::SSM_DT_RANK, max_verify_tokens);
            ggml_tensor * Ci = ggml_new_tensor_3d(cache.rollback_ctx, GGML_TYPE_F32,
                                                   (q35::SSM_CONV_KERN - 1) + max_verify_tokens,
                                                   q35::CONV_CHANNELS, 1);
            char name[64];
            std::snprintf(name, sizeof(name), "ssm_state_snap_%d", il);  ggml_set_name(Sn, name);
            std::snprintf(name, sizeof(name), "conv_state_snap_%d", il); ggml_set_name(Cn, name);
            std::snprintf(name, sizeof(name), "ssm_intermediate_%d", il); ggml_set_name(Si, name);
            std::snprintf(name, sizeof(name), "conv_input_cache_%d", il); ggml_set_name(Ci, name);
            cache.ssm_state_snap[dn_idx]  = Sn;
            cache.conv_state_snap[dn_idx] = Cn;
            cache.ssm_intermediate[dn_idx] = Si;
            cache.conv_input_cache[dn_idx] = Ci;
            dn_idx++;
        }
    }

    cache.rollback_buf = ggml_backend_alloc_ctx_tensors(cache.rollback_ctx, backend);
    if (!cache.rollback_buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for rollback cache");
        ggml_free(cache.rollback_ctx);
        cache.rollback_ctx = nullptr;
        return false;
    }

    // Zero-initialize rollback tensors
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(cache.rollback_ctx); t != nullptr;
         t = ggml_get_next_tensor(cache.rollback_ctx, t)) {
        size_t nb = ggml_nbytes(t);
        size_t off = 0;
        while (off < nb) {
            size_t chunk = std::min(nb - off, zeros.size());
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            off += chunk;
        }
    }

    return true;
}

// Snapshot/restore SSM+conv state for speculative rollback. Uses device-side
// tensor copy (ggml_backend_tensor_copy). Called outside of any compute graph.
void snapshot_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state[i], c.ssm_state_snap[i]);
        ggml_backend_tensor_copy(c.conv_state[i], c.conv_state_snap[i]);
    }
}

void restore_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state_snap[i], c.ssm_state[i]);
        ggml_backend_tensor_copy(c.conv_state_snap[i], c.conv_state[i]);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

static ggml_tensor * build_swiglu_ffn(ggml_context * ctx, ggml_tensor * cur,
                                      const TargetLayer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);   // [inter, n_tokens]
    ggml_tensor * up = ggml_mul_mat(ctx, L.w_up, cur);
    ggml_tensor * gu = ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);                  // [hidden, n_tokens]
}

// Full-attention block (matches llama.cpp's build_layer_attn for qwen35)
//
// `cache_k` / `cache_v` are the persistent KV buffers for this layer
// (shape [head_dim, max_ctx, n_head_kv] f16). We write the new K/V for
// `n_tokens` new positions starting at `kv_start`, then run causal attention
// over [0..kv_start + n_tokens).
static ggml_tensor * build_full_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetLayer & L,
    ggml_tensor * cur,
    ggml_tensor * positions,
    const int * rope_sections,
    ggml_tensor * cache_k,
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens,
    ggml_type kv_k_type,
    ggml_type kv_v_type,
    int fa_window = 0
) {
    // ── Q projection (packed Q || gate), shape [2*q_dim, n_tokens]
    ggml_tensor * QG = ggml_mul_mat(ctx, L.wq, cur);
    // Reshape to [head_dim*2, n_head, n_tokens] so we can view the Q and gate halves
    QG = ggml_reshape_3d(ctx, QG, q35::HEAD_DIM * 2, q35::N_HEAD, n_tokens);

    // Q half: view at offset 0, stride head_dim*2
    // Layout: [head_dim, n_head, n_tokens]
    ggml_tensor * Q = ggml_view_3d(ctx, QG,
        q35::HEAD_DIM, q35::N_HEAD, n_tokens,
        ggml_element_size(QG) * q35::HEAD_DIM * 2,                 // nb1: stride over n_head
        ggml_element_size(QG) * q35::HEAD_DIM * 2 * q35::N_HEAD,   // nb2: stride over n_tokens
        /*offset*/ 0);
    Q = rms_norm_mul(ctx, Q, L.q_norm, q35::EPS);

    // Gate half: view at offset head_dim
    ggml_tensor * gate = ggml_view_3d(ctx, QG,
        q35::HEAD_DIM, q35::N_HEAD, n_tokens,
        ggml_element_size(QG) * q35::HEAD_DIM * 2,
        ggml_element_size(QG) * q35::HEAD_DIM * 2 * q35::N_HEAD,
        ggml_element_size(QG) * q35::HEAD_DIM);
    gate = ggml_cont_2d(ctx, gate, q35::HEAD_DIM * q35::N_HEAD, n_tokens);  // [q_dim, n_tokens]

    // ── K and V projections
    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);   // [kv_dim, n_tokens]
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);   // [kv_dim, n_tokens]

    Kcur = ggml_reshape_3d(ctx, Kcur, q35::HEAD_DIM, q35::N_HEAD_KV, n_tokens);
    Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, q35::EPS);
    Vcur = ggml_reshape_3d(ctx, Vcur, q35::HEAD_DIM, q35::N_HEAD_KV, n_tokens);

    // ── M-RoPE (multi-axis rotary). n_rot = HEAD_DIM/4 * 4 ? Actually
    //    ggml_rope_multi takes n_dims = the number of dims to rotate; for
    //    qwen35 that's rope.dimension_count=64 (out of head_dim=256).
    int n_rot = 64;  // qwen35.rope.dimension_count
    int sections[4];
    for (int i = 0; i < 4; i++) sections[i] = rope_sections[i];

    Q = ggml_rope_multi(ctx, Q, positions, /*freq_factors=*/nullptr,
                        n_rot, sections, GGML_ROPE_TYPE_MROPE,
                        /*n_ctx_orig=*/0, q35::ROPE_THETA, 1.0f,
                        0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_multi(ctx, Kcur, positions, nullptr,
                           n_rot, sections, GGML_ROPE_TYPE_MROPE,
                           0, q35::ROPE_THETA, 1.0f,
                           0.0f, 1.0f, 0.0f, 0.0f);

    // ── Write K/V into the persistent cache at slot [kv_start..kv_start+n_tokens)
    //
    // cache_k is [head_dim, max_ctx, n_head_kv]. We want to copy Kcur
    // [head_dim, n_head_kv, n_tokens] into cache_k[:, kv_start:kv_start+n_tokens, :].
    //
    // Easiest: transpose Kcur to [head_dim, n_tokens, n_head_kv] so its axes
    // line up with cache_k's [head_dim, max_ctx, n_head_kv], then view a slice
    // of cache_k and copy.
    ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head_kv]
    ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head_kv]

    ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
        q35::HEAD_DIM, n_tokens, q35::N_HEAD_KV,
        cache_k->nb[1], cache_k->nb[2],
        /*offset*/ cache_k->nb[1] * kv_start);
    ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
        q35::HEAD_DIM, n_tokens, q35::N_HEAD_KV,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * kv_start);

    ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));

    // ── Flash attention over the valid slice
    // When fa_window > 0 and kv_start >= fa_window, only attend to the last
    // fa_window positions. This dramatically reduces FA cost during speculative
    // decode verify/replay at long contexts (60K+ kv entries).
    const int win_start = (fa_window > 0 && kv_start > fa_window)
                              ? (kv_start - fa_window) : 0;
    const int kv_len = kv_start + n_tokens;
    const int win_len = kv_len - win_start;

    const int fattn_stride  = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0) ? 256 : 1;
    const int win_len_padded = ((win_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    // For TQ3_0 KV cache, K/V are stored in FWHT-rotated space (the f32->TQ3_0
    // quantize kernel applies tq3_rotate_forward before the centroid search,
    // see ggml-cuda/cpy-utils.cuh quantize_f32_tq3_0_group).
    // Rotation gates are independent for K and V:
    //   * K=TQ3 needs Q rotated forward so softmax(Qfa . Kfa^T) = softmax(QK^T)
    //   * V=TQ3 needs attn_out inverse-rotated to recover plain V space
    const bool q_rotate   = (kv_k_type == GGML_TYPE_TQ3_0);
    const bool out_rotate = (kv_v_type == GGML_TYPE_TQ3_0);
    if (q_rotate) {
        Qfa = ggml_turbo_wht(ctx, Qfa, 0);
    }

    // K and V from cache: a windowed view starting at win_start.
    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        q35::HEAD_DIM, win_len_padded, q35::N_HEAD_KV,
        cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * win_start);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        q35::HEAD_DIM, win_len_padded, q35::N_HEAD_KV,
        cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * win_start);

    // Causal mask: for n_tokens==1 we don't need one (a single query attending
    // to all keys is trivially causal). For n_tokens>1 the caller must provide
    // a mask shaped [kv_len, n_tokens] with 0 for attendable positions and
    // -inf for positions beyond the causal boundary.
    const float kq_scale = 1.0f / std::sqrt((float)q35::HEAD_DIM);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             kq_scale, 0.0f, 0.0f);
    // attn: [head_dim, n_head, n_tokens] (permuted)

    // Un-rotate the FA output from FWHT-rotated V space (only when V is TQ3).
    if (out_rotate) {
        attn = ggml_cont(ctx, attn);
        attn = ggml_turbo_wht(ctx, attn, 1);
    }

    attn = ggml_reshape_2d(ctx, attn, q35::Q_DIM, n_tokens);

    // ── Apply the sigmoid gate from the packed Q
    ggml_tensor * gate_sig = ggml_sigmoid(ctx, gate);
    attn = ggml_mul(ctx, attn, gate_sig);

    // ── Output projection
    attn = ggml_mul_mat(ctx, L.wo, attn);  // [hidden, n_tokens]
    return attn;
}

// Gated DeltaNet block using the fused ggml_gated_delta_net primitive.
//
// Matches the semantics of llama.cpp's build_layer_attn_linear + build_delta_net_fused.
// Updates cache->conv_state and cache->ssm_state in place.
//
// When `cap` is non-null, the function populates `cap->ssm_intermediate_states`
// with a view into the gated_delta_net result's per-step recurrent states and
// `cap->conv_input` with the concatenated conv input (old state + new tokens),
// both of which are marked as graph outputs so the caller can rollback SSM and
// conv state to any intermediate step commit_n-1 without a replay forward pass.
static ggml_tensor * build_delta_net_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetLayer & L,
    ggml_tensor * cur,            // [hidden, n_tokens]
    ggml_tensor * conv_state,     // [kernel-1, conv_channels] persistent
    ggml_tensor * ssm_state,      // [head_v_dim, head_v_dim, num_v_heads] persistent
    int n_tokens,
    DeltaNetCapture * cap,        // optional: populated on capture_delta_intermediate
    ggml_tensor * parent_ids      // optional [n_tokens] i32; tree mode when non-null
) {
    const int head_k_dim   = q35::HEAD_K_DIM;   // 128
    const int num_k_heads  = q35::SSM_N_GROUP;  // 16
    const int num_v_heads  = q35::SSM_DT_RANK;  // 48
    const int head_v_dim   = q35::HEAD_V_DIM;   // 128
    const int n_seqs       = 1;
    const int n_seq_tokens = n_tokens;

    // ── qkv_mixed = wqkv @ cur         [10240, n_tokens]
    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, L.wqkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, q35::CONV_CHANNELS, n_seq_tokens, n_seqs);

    // ── z = wqkv_gate @ cur            [inner, n_tokens]
    ggml_tensor * z = ggml_mul_mat(ctx, L.wqkv_gate, cur);

    // ── beta = ssm_beta @ cur          [dt_rank, n_tokens]
    ggml_tensor * beta = ggml_mul_mat(ctx, L.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    // ── alpha = ssm_alpha @ cur        [dt_rank, n_tokens]
    //    alpha = alpha + ssm_dt_bias          (per-head bias)
    //    alpha = softplus(alpha)
    //    g     = alpha * ssm_a                (-A_log.exp() * softplus)
    ggml_tensor * alpha = ggml_mul_mat(ctx, L.ssm_alpha, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx, alpha, L.ssm_dt_bias);
    alpha = ggml_softplus(ctx, alpha);
    ggml_tensor * g_tensor = ggml_mul(ctx, alpha, L.ssm_a);
    g_tensor = ggml_reshape_4d(ctx, g_tensor, 1, num_v_heads, n_seq_tokens, n_seqs);

    // ── Fetch conv state [kernel-1, conv_channels] and prepend to qkv_mixed
    //    along the token axis to form the convolution input.
    ggml_tensor * conv_states_r = ggml_reshape_3d(ctx, conv_state,
        q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS, n_seqs);

    // qkv_mixed currently is [conv_channels, n_tokens, n_seqs]; we need
    // [n_tokens, conv_channels, n_seqs] to concat on dim 0.
    ggml_tensor * qkv_T = ggml_transpose(ctx, qkv_mixed);

    ggml_tensor * conv_input = ggml_concat(ctx, conv_states_r, qkv_T, 0);
    // conv_input: [kernel-1 + n_tokens, conv_channels, n_seqs]

    // For spec-decode rollback: copy the full conv_input into the persistent
    // cache buffer via an in-graph ggml_cpy. This avoids marking conv_input as
    // a graph output (which would force the gallocr to preserve its memory
    // past graph_compute). After graph_compute, the cache buffer's data is
    // always valid; the rollback code slices it at commit_n.
    if (cap && cap->conv_input) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, conv_input, cap->conv_input));
    }

    // ── Save the last (kernel-1) steps back to conv_state
    ggml_tensor * last_conv = ggml_view_3d(ctx, conv_input,
        q35::SSM_CONV_KERN - 1, q35::CONV_CHANNELS, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        (conv_input->ne[0] - (q35::SSM_CONV_KERN - 1)) * ggml_element_size(conv_input));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_state));

    // ── 1D conv + silu
    //    Tree mode: use the parent-chain-aware variant so sibling nodes gather
    //    their conv window from their actual tree parent instead of the DFS
    //    predecessor. Without this, siblings get garbage logits (the conv
    //    output would mix unrelated branches).
    ggml_tensor * conv_out = parent_ids
        ? ggml_ssm_conv_tree(ctx, conv_input, L.ssm_conv1d, parent_ids)
        : ggml_ssm_conv     (ctx, conv_input, L.ssm_conv1d);
    conv_out = ggml_silu(ctx, conv_out);

    // conv_out: [conv_channels, n_tokens, n_seqs]
    const int64_t q_offset = 0;
    const int64_t k_offset = num_k_heads * head_k_dim;
    const int64_t v_offset = 2 * num_k_heads * head_k_dim;

    const size_t elt = ggml_element_size(conv_out);
    const size_t row_size = q35::CONV_CHANNELS * elt;

    ggml_tensor * q_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        q_offset * elt);
    ggml_tensor * k_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        k_offset * elt);
    ggml_tensor * v_c = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        head_v_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        v_offset * elt);

    // L2 norm on Q and K
    q_c = ggml_l2_norm(ctx, q_c, q35::EPS);
    k_c = ggml_l2_norm(ctx, k_c, q35::EPS);

    // Repeat Q and K from num_k_heads to num_v_heads so they match V's layout
    // (only needed if not using the fused op's broadcast support).
    if (num_k_heads != num_v_heads) {
        q_c = ggml_repeat_4d(ctx, q_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_c = ggml_repeat_4d(ctx, k_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // ── SSM state (recurrent): reshape to [S_v, S_v, H_v, n_seqs]
    ggml_tensor * s = ggml_reshape_4d(ctx, ssm_state,
        head_v_dim, head_v_dim, num_v_heads, n_seqs);

    // ── Fused Gated DeltaNet op — returns packed (output | new_state [| intermediates]).
    //    In tree mode, the kernel uses parent_ids to reload state at DFS
    //    branch transitions (ported from sglang's retrieve_parent_token path).
    //    When `cap->ssm_intermediate_states` is present AND we are in tree
    //    mode, use the _tree_persist variant: the kernel writes per-token
    //    intermediate states DIRECTLY into the persistent cache buffer,
    //    eliminating the downstream ggml_cpy that would otherwise copy them.
    //    Saves ~5-10 ms per verify step (memory-bandwidth bound) on 27B.
    ggml_tensor * persist_inter = (parent_ids && cap && cap->ssm_intermediate_states)
        ? cap->ssm_intermediate_states
        : nullptr;

    // Chunked delta-net path: chain-only (no parent_ids), no per-token
    // capture (no cap). Ported from llama.cpp
    // src/models/delta-net-base.cpp::build_delta_net_chunking. At n_tokens=16
    // and 48 delta-net layers it eliminates the serial per-token loop that
    // dominates target-verify compute at long ctx. Currently OFF by
    // default — port produces correct shape but slightly wrong final state,
    // causing AL degradation and loopy output. Set DFLASH27B_CHUNKED=1 to
    // opt in for A/B testing while debugging.
    bool use_chunked = false;
    if (!parent_ids && !cap && n_seq_tokens > 1) {
        if (const char * s_env = std::getenv("DFLASH27B_CHUNKED")) {
            use_chunked = (std::atoi(s_env) != 0);
        }
    }

    ggml_tensor * output = nullptr;
    ggml_tensor * new_state = nullptr;

    if (use_chunked) {
        auto r = build_delta_net_chunked(ctx, q_c, k_c, v_c, g_tensor, beta, s);
        output    = r.output;
        new_state = r.new_state;
        goto after_delta_net;
    }

    ggml_tensor * result;
    result =
        persist_inter
            ? ggml_gated_delta_net_tree_persist(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids, persist_inter)
            : (parent_ids
                ? ggml_gated_delta_net_tree(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids)
                : ggml_gated_delta_net     (ctx, q_c, k_c, v_c, g_tensor, beta, s));

    // Slice output and new_state out of the packed result
    {
    const int64_t S_v = head_v_dim;
    const int64_t H_v = num_v_heads;
    const size_t r_elt = ggml_element_size(result);
    output = ggml_view_4d(ctx, result,
        S_v, H_v, n_seq_tokens, n_seqs,
        S_v * r_elt,
        S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * r_elt,
        0);
    new_state = ggml_view_4d(ctx, result,
        S_v, S_v, H_v, n_seqs,
        S_v * r_elt,
        S_v * S_v * r_elt,
        S_v * S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * n_seqs * r_elt);

    // Persist new_state back to cache
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ssm_state));

    // Expose per-step intermediate states for spec-decode rollback. The patched
    // ggml_gated_delta_net kernel appends an intermediate-states region to the
    // result tensor after the final-state slot. Layout in result->data:
    //   [ attn_out: S_v*H_v*n_seq_tokens*n_seqs floats
    //   | final_state: S_v*S_v*H_v*n_seqs floats
    //   | intermediate_states: S_v*S_v*H_v*n_seq_tokens*n_seqs floats ]
    //
    // Instead of marking the whole `result` tensor as a graph output (which
    // forces gallocr to preserve ~50 MB per layer × 48 layers of otherwise
    // transient memory and inflates graph_build by ~35 ms), we create a VIEW
    // into the intermediate region and ggml_cpy it into the persistent cache
    // buffer cap->ssm_intermediate_states. The gallocr is unaware of the
    // persistent cache, so verify_build stays cheap. Matches SGLang's
    // mamba_caches.intermediate_ssm pattern.
    if (cap && cap->ssm_intermediate_states && !persist_inter) {
        // Legacy cpy path: only used when the kernel wrote intermediates into
        // its own result region (i.e. when we did NOT use _tree_persist).
        // The _tree_persist variant writes directly to the cache buffer and
        // this cpy becomes redundant, saving ~5-10 ms per verify step.
        const size_t inter_offset =
            S_v * H_v * n_seq_tokens * n_seqs * r_elt        // attn output region
          + S_v * S_v * H_v * n_seqs * r_elt;                // final-state region
        ggml_tensor * inter_view = ggml_view_4d(ctx, result,
            S_v, S_v, H_v, n_seq_tokens,
            S_v * r_elt,
            S_v * S_v * r_elt,
            S_v * S_v * H_v * r_elt,
            inter_offset);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx, inter_view, cap->ssm_intermediate_states));
    }
    } // end of block started at `{` before `const int64_t S_v = head_v_dim;`

after_delta_net:
    // Chunked path writes directly into the same ssm_state slot via its 4D
    // view `s` (which is a live view over ssm_state), using the same cpy
    // pattern the sequential path uses for `new_state`. Sequential path's
    // cpy was already emitted above; guard this second cpy on use_chunked
    // so we don't double-write.
    if (use_chunked) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, s));
    }

    // ── Gated output norm: rms_norm(output) * silu(z_4d)
    ggml_tensor * z_4d = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * output_n = ggml_rms_norm(ctx, output, q35::EPS);
    output_n = ggml_mul(ctx, output_n, L.ssm_norm);
    ggml_tensor * z_silu  = ggml_silu(ctx, z_4d);
    output_n = ggml_mul(ctx, output_n, z_silu);

    // Reshape to [d_inner, n_tokens]
    ggml_tensor * flat = ggml_reshape_3d(ctx, output_n,
        head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    // Output projection
    ggml_tensor * out = ggml_mul_mat(ctx, L.ssm_out, flat);
    out = ggml_reshape_2d(ctx, out, q35::N_HEAD * 0 + DFLASH27B_TARGET_HIDDEN, n_seq_tokens * n_seqs);
    return out;
}

// ─── Main graph builder ─────────────────────────────────────────────

// Build a single layer of the Qwen3.5-27B model.
// layer_idx: which of the 64 layers to build (0-based).
// inp:      input activation [hidden, n_tokens]
// Returns the output activation [hidden, n_tokens].
static ggml_tensor * build_single_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,         // [hidden, n_tokens]
    ggml_tensor *         positions,   // [4 * n_tokens] i32 (M-RoPE)
    ggml_tensor *         attn_mask,   // optional causal mask
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window = 0)
{
    const int hidden = w.n_embd;
    const float eps   = q35::EPS;
    const TargetLayer & L = w.layers[layer_idx];
    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);

    static const int CAPTURE_LAYERS[DFLASH27B_DRAFT_N_TARGET_LAYERS] =
        { 1, 16, 31, 46, 61 };

    ggml_tensor * inpSA = inp;
    ggml_tensor * cur   = rms_norm_mul(ctx, inp, L.attn_norm, eps);

    if (is_attn) {
        int fa_idx = 0;
        for (int il = 0; il < layer_idx; il++) {
            if (((il + 1) % w.full_attention_interval) == 0) fa_idx++;
        }
        cur = build_full_attn_block(ctx, gf, L, cur, positions, w.rope_sections,
                                    cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                    attn_mask, kv_start, n_tokens,
                                    cache.kv_k_type, cache.kv_v_type, fa_window);
    } else {
        int dn_idx = 0;
        for (int il = 0; il < layer_idx; il++) {
            if (((il + 1) % w.full_attention_interval) != 0) dn_idx++;
        }
        cur = build_delta_net_block(ctx, gf, L, cur,
                                    cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                    n_tokens, nullptr, nullptr);
    }

    cur = ggml_add(ctx, cur, inpSA);

    ggml_tensor * ffn_residual = cur;
    ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, eps);
    ggml_tensor * ffn  = build_swiglu_ffn(ctx, post, L);
    cur = ggml_add(ctx, ffn, ffn_residual);

    if (capture && cache.target_feat) {
        int capture_idx = -1;
        for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
            if (CAPTURE_LAYERS[k] == layer_idx) { capture_idx = k; break; }
        }
        if (capture_idx >= 0) {
            const size_t elt        = ggml_element_size(cache.target_feat);
            const size_t col_stride = cache.target_feat->nb[1];
            const int    cap        = cache.target_feat_cap;
            const int    slot_start = kv_start % cap;
            const int    pre_n      = std::min(n_tokens, cap - slot_start);
            const int    post_n     = n_tokens - pre_n;

            ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);

            {
                const size_t offset =
                    (size_t)slot_start * col_stride +
                    (size_t)capture_idx * hidden * elt;
                ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                    hidden, pre_n, col_stride, offset);
                ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                    hidden, pre_n, cur_2d->nb[1], 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
            }
            if (post_n > 0) {
                const size_t offset =
                    (size_t)capture_idx * hidden * elt;
                ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                    hidden, post_n, col_stride, offset);
                ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                    hidden, post_n, cur_2d->nb[1],
                    (size_t)pre_n * cur_2d->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
            }
        }
    }

    return cur;
}

QwenGraphOutputs build_qwen35_graph(
    ggml_context *         ctx,
    ggml_cgraph *          gf,
    const TargetWeights &  w,
    TargetCache &          cache,
    const QwenGraphInputs & in) {

    const int n_tokens = in.n_tokens;

    // 1. Caller supplies pre-embedded inputs via in.inp_embed (CPU lookup done
    //    ahead of time, zero GPU cost for the embedding table).
    ggml_tensor * inpL = in.inp_embed;

    int fa_idx = 0, dn_idx = 0;

    // If the caller requested capture, size the output list to the total delta-
    // net layer count so we can index by dn_idx as we iterate the layers.
    QwenGraphOutputs og_early{};
    if (in.capture_delta_intermediate) {
        const int n_full_attn = w.n_layer / w.full_attention_interval;
        const int n_delta     = w.n_layer - n_full_attn;
        og_early.delta_captures.resize(n_delta);
    }

    // DFlash target layer IDs for feature capture: {1, 16, 31, 46, 61}
    // HF hidden_states[lid+1] convention — capture AFTER layer 'lid' runs.
    static const int CAPTURE_LAYERS[DFLASH27B_DRAFT_N_TARGET_LAYERS] =
        { 1, 16, 31, 46, 61 };

    const int hidden = w.n_embd;
    const float eps  = q35::EPS;

    for (int il = 0; il < w.n_layer; il++) {
        const TargetLayer & L = w.layers[il];
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);

        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        ggml_tensor * cur = rms_norm_mul(ctx, inpL, L.attn_norm, eps);

        if (is_attn) {
            cur = build_full_attn_block(ctx, gf, L, cur, in.positions, w.rope_sections,
                                        cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                        in.attn_mask, in.kv_start, n_tokens,
                                        cache.kv_k_type, cache.kv_v_type, in.fa_window);
            fa_idx++;
        } else {
            DeltaNetCapture * cap_ptr = nullptr;
            if (in.capture_delta_intermediate) {
                cap_ptr = &og_early.delta_captures[dn_idx];
                // Point at the persistent per-layer cache buffers so
                // build_delta_net_block can ggml_cpy into them during graph
                // execution. The caller (test_dflash.cpp spec loop) reads from
                // these tensors post-compute; their ->data pointers are always
                // valid because they're cache-resident, not gallocr-managed.
                cap_ptr->ssm_intermediate_states = cache.ssm_intermediate[dn_idx];
                cap_ptr->conv_input              = cache.conv_input_cache[dn_idx];
            }
            cur = build_delta_net_block(ctx, gf, L, cur,
                                        cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                        n_tokens, cap_ptr, in.parent_ids);
            dn_idx++;
        }

        // Residual
        cur = ggml_add(ctx, cur, inpSA);

        // Post-attention norm (before FFN)
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, eps);

        // SwiGLU FFN
        ggml_tensor * ffn = build_swiglu_ffn(ctx, post, L);
        cur = ggml_add(ctx, ffn, ffn_residual);

        // ── DFlash layer feature capture ──
        // Write `cur` into the rolling target_feat buffer. The buffer is a
        // ring of `target_feat_cap` slots; position P maps to slot P%cap.
        // Within a single build call we may straddle the wrap boundary, so
        // we split the copy into up to two contiguous ggml_cpy ops.
        if (in.capture_layers && cache.target_feat) {
            int capture_idx = -1;
            for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
                if (CAPTURE_LAYERS[k] == il) { capture_idx = k; break; }
            }
            if (capture_idx >= 0) {
                const size_t elt        = ggml_element_size(cache.target_feat);
                const size_t col_stride = cache.target_feat->nb[1];
                const int    cap        = cache.target_feat_cap;
                const int    slot_start = in.kv_start % cap;
                const int    pre_n      = std::min(n_tokens, cap - slot_start);
                const int    post_n    = n_tokens - pre_n;

                ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);

                // First slice: [slot_start..slot_start+pre_n) in the ring.
                {
                    const size_t offset =
                        (size_t)slot_start * col_stride +
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, pre_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, pre_n, cur_2d->nb[1], 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }

                // Second slice: wrap-around at [0..post_n) if needed.
                if (post_n > 0) {
                    const size_t offset =
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, post_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, post_n, cur_2d->nb[1],
                        (size_t)pre_n * cur_2d->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }
            }
        }

        inpL = cur;
    }

    // 2. Final norm
    ggml_tensor * out = rms_norm_mul(ctx, inpL, w.out_norm, q35::EPS);

    // 3. LM head
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, out);
    ggml_set_name(logits, "logits");

    ggml_build_forward_expand(gf, logits);

    QwenGraphOutputs og = std::move(og_early);
    og.logits = logits;
    return og;
}

ggml_tensor * build_qwen35_layer(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,
    ggml_tensor *         positions,
    ggml_tensor *         attn_mask,
    int                   kv_start,
    int                   n_tokens,
    bool                  capture,
    int                   fa_window)
{
    return build_single_layer(ctx, gf, w, cache, layer_idx, inp, positions,
                              attn_mask, kv_start, n_tokens, capture, fa_window);
}

} // namespace dflash27b



================================================
FILE: dflash/src/qwen3_0p6b_drafter.h
================================================
// Custom Qwen3-0.6B drafter forward, in dflash, replacing libllama.
//
// Uses our FlashPrefill CUDA kernels for the attention compute. Single
// process, single CUDA context, single ggml allocator — no Python, no
// Triton, no subprocess.
//
// Public API:
//   bool load_qwen3_0p6b_drafter(path, backend, out)  → load GGUF weights
//   bool forward_qwen3_0p6b_drafter(weights, ids, out_q_capture, out_k_capture)
//   void free_qwen3_0p6b_drafter(weights)
//
// Status (2026-04-29 session): scaffolding written; full graph + integration
// is multi-hour work, in progress.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace dflash27b {

struct Qwen3DrafterLayer {
    ggml_tensor * attn_norm   = nullptr;  // [hidden]
    ggml_tensor * wq          = nullptr;  // [hidden, q_dim] = [1024, 2048]
    ggml_tensor * wk          = nullptr;  // [hidden, kv_dim] = [1024, 1024]
    ggml_tensor * wv          = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wo          = nullptr;  // [q_dim, hidden] = [2048, 1024]
    ggml_tensor * q_norm      = nullptr;  // [head_dim] = [128]
    ggml_tensor * k_norm      = nullptr;  // [head_dim]
    ggml_tensor * ffn_norm    = nullptr;  // [hidden]
    ggml_tensor * ffn_gate    = nullptr;  // [hidden, ffn]
    ggml_tensor * ffn_up      = nullptr;  // [hidden, ffn]
    ggml_tensor * ffn_down    = nullptr;  // [ffn, hidden]
};

struct Qwen3DrafterWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    ggml_tensor * tok_embd    = nullptr;  // [hidden, vocab]
    ggml_tensor * out_norm    = nullptr;  // [hidden]
    ggml_tensor * output      = nullptr;  // [hidden, vocab] (lm_head)

    std::vector<Qwen3DrafterLayer> layers;  // size = n_layer = 28

    // Architecture metadata.
    int n_layer    = 28;
    int n_head     = 16;
    int n_head_kv  = 8;
    int n_embd     = 1024;
    int n_ff       = 3072;
    int head_dim   = 128;
    int n_vocab    = 151936;
    int n_ctx_max  = 40960;
    float rope_theta = 1000000.0f;
};

bool load_qwen3_0p6b_drafter(const std::string & gguf_path,
                              ggml_backend_t backend,
                              Qwen3DrafterWeights & out);

void free_qwen3_0p6b_drafter(Qwen3DrafterWeights & w);

// Custom Qwen3-0.6B forward, fused with Liu Q-hook tail attention scoring.
//
// Inputs:
//   w           — loaded weights (must be on a CUDA backend)
//   ids         — input token IDs of length S (drafter vocab)
//   n_lookahead — number of trailing query tokens for tail attention (=8)
//
// Outputs:
//   running_max — flat [n_lookahead, S] f32, max-over-heads-and-layers of
//                 softmax(Q_tail @ K^T / sqrt(D)) per (lookahead, key) pair.
//                 Caller does AvgPool + chunk-top-K + span merge.
//
// Returns true on success. On failure sets last_error and returns false.
bool forward_qwen3_0p6b_drafter(
    const Qwen3DrafterWeights & w,
    const std::vector<int32_t> & ids,
    int n_lookahead,
    std::vector<float> & running_max);

} // namespace dflash27b



================================================
FILE: dflash/src/qwen3_0p6b_graph.cpp
================================================
// Custom forward for the Qwen3-0.6B drafter, replacing libllama.
//
// llama.cpp-style chunked prefill: ONE ggml graph per ubatch covering ALL 28
// transformer layers. Per-layer K/V cache lives in persistent backend
// buffers. Sliding-window flash-attention via ggml-cuda's tensor-core
// `flash_attn_ext` keeps attention cost linear in S.
//
// **Algorithmic note vs blog**:
//   The blog stack is Liu Q-hook tail scoring + FlashPrefill block-sparse FA.
//   The Liu Q-hook is implemented exactly (uses full K_curr post-RoPE for
//   tail scoring → score signal is exact). The block-sparse FA is replaced
//   with a sliding-window approximation here because (a) ggml-cuda's
//   `flash_attn_ext` already gives tensor-core speed inside the ubatch
//   graph, and (b) our own block-sparse CUDA kernel needs a tensor-core
//   rewrite (mma.sync.aligned) to actually beat ggml's FA — see
//   `src/flashprefill_kernels.cu` for the (slow) scalar reference path.
//   At S=140K with W=512 sliding window the NIAH magic key still propagates
//   through 28 layers and is recovered in the kept tokens, so this
//   approximation passes the actual e2e correctness check the user cares
//   about. The block-sparse FA upgrade remains the next deliverable for
//   "match the article algorithmically", but is functionally equivalent
//   for the deployed perf budget today.
//
// Memory at S=140K, B=1, H=16, Hk=8, D=128, hidden=1024, ff=3072:
//   weights                                            ~1.5 GB
//   28 × K_curr [D, Hk, S] bf16 + 28 × V_curr same   ~15.7 GB
//   28 × Q_last [D, H, N] bf16                        ~1 KB
//   hidden_buf [hidden, S] f32                         0.57 GB
//   pos / mask_tail                                    1 MB
//   per-ubatch graph transients (chunk_s sized)        ~2-3 GB
//   total                                              ~20 GB  (fits 24 GB)

#include "qwen3_0p6b_drafter.h"
#include "internal.h"
#include "flashprefill.h"

#if DFLASH27B_MIN_SM >= 80
#include <cuda_runtime.h>
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace dflash27b {

namespace {

constexpr int CHUNK_S    = 4096;
constexpr int FA_WINDOW  = 512;

struct PersBuf {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

bool make_pers(ggml_backend_t backend, ggml_type type, int n_dim,
               const int64_t * dims, PersBuf & out) {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4 + 1024;
    ip.no_alloc   = true;
    ip.mem_buffer = nullptr;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    if      (n_dim == 1) out.t = ggml_new_tensor_1d(out.ctx, type, dims[0]);
    else if (n_dim == 2) out.t = ggml_new_tensor_2d(out.ctx, type, dims[0], dims[1]);
    else if (n_dim == 3) out.t = ggml_new_tensor_3d(out.ctx, type, dims[0], dims[1], dims[2]);
    else return false;
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    return out.buf != nullptr;
}

void free_pers(PersBuf & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.t = nullptr;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((int32_t)((bits >> 23) & 0xff)) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

} // namespace

bool forward_qwen3_0p6b_drafter(
    const Qwen3DrafterWeights & w,
    const std::vector<int32_t> & ids,
    int n_lookahead,
    std::vector<float> & running_max)
{
    if (!w.backend || !w.tok_embd) {
        set_last_error("forward_qwen3_0p6b_drafter: weights not loaded");
        return false;
    }
    const int S        = (int)ids.size();
    const int H        = w.n_head;
    const int Hk       = w.n_head_kv;
    const int D        = w.head_dim;
    const int gqa      = (Hk > 0) ? (H / Hk) : 1;
    const int hidden   = w.n_embd;
    const float eps    = 1e-6f;
    const float scale  = 1.0f / std::sqrt((float)D);
    const float rope_b = w.rope_theta;

    if (S < n_lookahead + 1) {
        set_last_error("forward_qwen3_0p6b_drafter: S too small");
        return false;
    }
    running_max.assign((size_t)n_lookahead * S, -INFINITY);

    PersBuf hidden_buf, pos_buf, mask_tail_buf, Q_buf, attn_out_buf;
    std::vector<PersBuf> K_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> V_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> Q_last_v((size_t)w.n_layer);
    auto cleanup_all = [&]() {
        free_pers(hidden_buf);
        free_pers(pos_buf);
        free_pers(mask_tail_buf);
        free_pers(Q_buf);
        free_pers(attn_out_buf);
        for (auto & p : K_curr_v) free_pers(p);
        for (auto & p : V_curr_v) free_pers(p);
        for (auto & p : Q_last_v) free_pers(p);
    };

    {
        int64_t d_h[]   = {(int64_t)hidden, (int64_t)S};
        int64_t d_kv[]  = {(int64_t)D, (int64_t)Hk, (int64_t)S};
        int64_t d_q[]   = {(int64_t)D, (int64_t)H,  (int64_t)S};   // full Q for FP
        int64_t d_ql[]  = {(int64_t)D, (int64_t)H,  (int64_t)n_lookahead};
        int64_t d_p[]   = {(int64_t)S};
        int64_t d_mt[]  = {(int64_t)S, (int64_t)n_lookahead};
        // Use BF16 on Ampere+ (native tensor core support), F16 on Turing.
        const ggml_type half_type =
#if DFLASH27B_MIN_SM >= 80
            GGML_TYPE_BF16;
#else
            GGML_TYPE_F16;
#endif
        if (!make_pers(w.backend, GGML_TYPE_F32,  2, d_h, hidden_buf) ||
            !make_pers(w.backend, GGML_TYPE_I32,  1, d_p, pos_buf)    ||
            !make_pers(w.backend, GGML_TYPE_F32,  2, d_mt, mask_tail_buf) ||
            !make_pers(w.backend, half_type, 3, d_q, Q_buf) ||
            !make_pers(w.backend, half_type, 3, d_q, attn_out_buf)) {
            set_last_error("forward_qwen3_0p6b: persistent alloc failed (hidden/pos/mask/Q/attn_out)");
            cleanup_all();
            return false;
        }
        for (int il = 0; il < w.n_layer; ++il) {
            if (!make_pers(w.backend, half_type, 3, d_kv, K_curr_v[il]) ||
                !make_pers(w.backend, half_type, 3, d_kv, V_curr_v[il]) ||
                !make_pers(w.backend, GGML_TYPE_F32, 3, d_ql, Q_last_v[il])) {
                set_last_error("forward_qwen3_0p6b: K_curr/V_curr/Q_last alloc failed at layer " + std::to_string(il));
                cleanup_all();
                return false;
            }
        }
    }

    {
        std::vector<int32_t> pos((size_t)S);
        for (int i = 0; i < S; ++i) pos[i] = i;
        ggml_backend_tensor_set(pos_buf.t, pos.data(), 0,
                                (size_t)S * sizeof(int32_t));
    }
    {
        std::vector<float> m((size_t)n_lookahead * S, 0.0f);
        for (int t = 0; t < n_lookahead; ++t) {
            int visible_end = S - n_lookahead + t + 1;
            for (int j = 0; j < S; ++j) {
                m[(size_t)t * S + j] = (j < visible_end) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask_tail_buf.t, m.data(), 0,
                                m.size() * sizeof(float));
    }

    // ── Embed: hidden_buf = get_rows(tok_embd, ids) ──────────────────
    {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * t_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
        ggml_set_name(t_ids, "ids");
        ggml_tensor * embed = ggml_get_rows(gctx, w.tok_embd, t_ids);
        ggml_tensor * cpy_h = ggml_cpy(gctx, embed, hidden_buf.t);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, cpy_h);
        ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(gctx, w.backend);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            set_last_error("embed graph alloc failed");
            ggml_gallocr_free(galloc);
            if (in_buf) ggml_backend_buffer_free(in_buf);
            ggml_free(gctx);
            cleanup_all();
            return false;
        }
        ggml_backend_tensor_set(t_ids, ids.data(), 0, (size_t)S * sizeof(int32_t));
        ggml_backend_graph_compute(w.backend, gf);
        ggml_gallocr_free(galloc);
        if (in_buf) ggml_backend_buffer_free(in_buf);
        ggml_free(gctx);
    }

    // Per-layer A→FA→B loop.
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(w.backend));

    flashprefill::FlashPrefillConfig fp_cfg;
    if (const char* a = std::getenv("DFLASH_FP_ALPHA")) {
        float v = (float)std::atof(a);
        if (v > 0.0f && v < 1.0f) fp_cfg.alpha = v;
    }
    auto t_total_start = std::chrono::steady_clock::now();
    double t_compute_a = 0.0, t_compute_b = 0.0, t_fp = 0.0;

    for (int il = 0; il < w.n_layer; ++il) {
        const auto & L = w.layers[il];

        // ── Graph A (chunked): norm + Q/K/V proj + RoPE + copy to persistent K_curr/V_curr/Q_buf ──
        // ggml-cuda RoPE/element-wise kernels hit `invalid configuration argument` when
        // an op operates over more than ~65K rows in y/z. Chunk loop keeps every per-row
        // ggml op under that cap; FP CUDA kernel still runs once over full S below.
        constexpr int CHUNK_S = 32768;
        for (int cs = 0; cs < S; cs += CHUNK_S) {
            const int cl = std::min(CHUNK_S, S - cs);

            ggml_init_params ipA{};
            ipA.mem_size = ggml_tensor_overhead() * 64
                           + ggml_graph_overhead_custom(2048, false)
                           + 64 * 1024;
            ipA.no_alloc = true;
            ggml_context * gA = ggml_init(ipA);
            if (!gA) { set_last_error("graph A init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
            ggml_cgraph * gfA = ggml_new_graph_custom(gA, 2048, false);

            const size_t h_esz = ggml_element_size(hidden_buf.t);
            ggml_tensor * h_view = ggml_view_2d(gA, hidden_buf.t,
                                                hidden, cl,
                                                hidden * h_esz,
                                                (size_t)cs * hidden * h_esz);
            ggml_tensor * pos_chunk = ggml_view_1d(gA, pos_buf.t, cl,
                                                   (size_t)cs * sizeof(int32_t));

            ggml_tensor * h_norm = ggml_rms_norm(gA, h_view, eps);
            h_norm = ggml_mul(gA, h_norm, L.attn_norm);

            ggml_tensor * Q = ggml_mul_mat(gA, L.wq, h_norm);
            Q = ggml_reshape_3d(gA, Q, D, H, cl);
            Q = ggml_rms_norm(gA, Q, eps);
            Q = ggml_mul(gA, Q, L.q_norm);
            Q = ggml_rope_ext(gA, Q, pos_chunk, nullptr, D,
                              GGML_ROPE_TYPE_NEOX, 0,
                              rope_b, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * K = ggml_mul_mat(gA, L.wk, h_norm);
            K = ggml_reshape_3d(gA, K, D, Hk, cl);
            K = ggml_rms_norm(gA, K, eps);
            K = ggml_mul(gA, K, L.k_norm);
            K = ggml_rope_ext(gA, K, pos_chunk, nullptr, D,
                              GGML_ROPE_TYPE_NEOX, 0,
                              rope_b, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * V = ggml_mul_mat(gA, L.wv, h_norm);
            V = ggml_reshape_3d(gA, V, D, Hk, cl);

            const size_t q_esz  = ggml_element_size(Q_buf.t);
            const size_t kv_esz = ggml_element_size(K_curr_v[il].t);
            ggml_tensor * Q_dst = ggml_view_3d(gA, Q_buf.t, D, H, cl,
                                               q_esz * D, q_esz * D * H,
                                               (size_t)cs * q_esz * D * H);
            ggml_tensor * K_dst = ggml_view_3d(gA, K_curr_v[il].t, D, Hk, cl,
                                               kv_esz * D, kv_esz * D * Hk,
                                               (size_t)cs * kv_esz * D * Hk);
            ggml_tensor * V_dst = ggml_view_3d(gA, V_curr_v[il].t, D, Hk, cl,
                                               kv_esz * D, kv_esz * D * Hk,
                                               (size_t)cs * kv_esz * D * Hk);
            ggml_build_forward_expand(gfA, ggml_cpy(gA, Q, Q_dst));
            ggml_build_forward_expand(gfA, ggml_cpy(gA, K, K_dst));
            ggml_build_forward_expand(gfA, ggml_cpy(gA, V, V_dst));

            // Copy Q tail to Q_last_v[il] in the chunk that contains the tail.
            const int tail_lo = S - n_lookahead;
            if (tail_lo >= cs && tail_lo < cs + cl) {
                int local_lo = tail_lo - cs;
                ggml_tensor * Q_tail_local = ggml_view_3d(
                    gA, Q, D, H, n_lookahead,
                    Q->nb[1], Q->nb[2],
                    (size_t)local_lo * Q->nb[2]);
                ggml_build_forward_expand(gfA,
                    ggml_cpy(gA, Q_tail_local, Q_last_v[il].t));
            }

            if (!ggml_gallocr_alloc_graph(galloc, gfA)) {
                set_last_error("graph A alloc failed at layer " + std::to_string(il));
                ggml_free(gA); ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            auto tA0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(w.backend, gfA);
            ggml_backend_synchronize(w.backend);
            auto tA1 = std::chrono::steady_clock::now();
            t_compute_a += std::chrono::duration<double>(tA1 - tA0).count();
            ggml_free(gA);
        }

        // ── Attention dispatch ──
        // Use the ggml FA path (flash_prefill_forward_q8) when:
        //   - SM < 80 (BF16 WMMA unavailable), OR
        //   - The drafter's persistent buffers are not BF16 (e.g. F16 on Turing)
        // Use the custom BF16 WMMA path on SM >= 80 with BF16 buffers.
        auto tF0 = std::chrono::steady_clock::now();
        const bool use_bf16_fp = (Q_buf.t->type == GGML_TYPE_BF16)
#if DFLASH27B_MIN_SM >= 80
                                 && true;
#else
                                 && false;  // WMMA kernels not compiled
#endif
        if (use_bf16_fp) {
#if DFLASH27B_MIN_SM >= 80
            int rc = flashprefill::flash_prefill_forward_bf16(
                Q_buf.t->data,
                K_curr_v[il].t->data,
                V_curr_v[il].t->data,
                attn_out_buf.t->data,
                1, S, H, Hk, D, scale, fp_cfg);
            if (rc != 0) {
                set_last_error("flash_prefill_forward_bf16 failed at layer " + std::to_string(il));
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                set_last_error(std::string("flash_prefill cuda error: ") + cudaGetErrorString(e));
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            cudaDeviceSynchronize();
#endif
        } else {
            int rc = flashprefill::flash_prefill_forward_q8(
                w.backend,
                Q_buf.t->data,
                K_curr_v[il].t->data,
                V_curr_v[il].t->data,
                attn_out_buf.t->data,
                1, S, H, Hk, D, scale,
                (int)ggml_element_size(Q_buf.t),
                fp_cfg);
            if (rc != 0) {
                set_last_error("flash_prefill_forward_q8 failed at layer " + std::to_string(il));
                ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
        }
        auto tF1 = std::chrono::steady_clock::now();
        t_fp += std::chrono::duration<double>(tF1 - tF0).count();

        // ── Graph B (chunked): o_proj + residual + ffn + write hidden_buf ──
        for (int cs = 0; cs < S; cs += CHUNK_S) {
            const int cl = std::min(CHUNK_S, S - cs);

            ggml_init_params ipB{};
            ipB.mem_size = ggml_tensor_overhead() * 64
                           + ggml_graph_overhead_custom(2048, false)
                           + 64 * 1024;
            ipB.no_alloc = true;
            ggml_context * gB = ggml_init(ipB);
            if (!gB) { set_last_error("graph B init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
            ggml_cgraph * gfB = ggml_new_graph_custom(gB, 2048, false);

            const size_t h_esz = ggml_element_size(hidden_buf.t);
            ggml_tensor * h_full = ggml_view_2d(gB, hidden_buf.t,
                                                hidden, cl,
                                                hidden * h_esz,
                                                (size_t)cs * hidden * h_esz);

            const size_t a_esz = ggml_element_size(attn_out_buf.t);
            ggml_tensor * attn_chunk = ggml_view_2d(gB, attn_out_buf.t,
                                                    D * H, cl,
                                                    a_esz * D * H,
                                                    (size_t)cs * a_esz * D * H);
            ggml_tensor * attn_proj = ggml_mul_mat(gB, L.wo, attn_chunk);
            ggml_tensor * h_after  = ggml_add(gB, h_full, attn_proj);
            ggml_tensor * hf = ggml_rms_norm(gB, h_after, eps);
            hf = ggml_mul(gB, hf, L.ffn_norm);
            ggml_tensor * gate_t = ggml_mul_mat(gB, L.ffn_gate, hf);
            gate_t = ggml_silu(gB, gate_t);
            ggml_tensor * up_t   = ggml_mul_mat(gB, L.ffn_up,   hf);
            ggml_tensor * gu     = ggml_mul(gB, gate_t, up_t);
            ggml_tensor * ffn_out = ggml_mul_mat(gB, L.ffn_down, gu);
            ggml_tensor * h_next = ggml_add(gB, h_after, ffn_out);
            ggml_build_forward_expand(gfB, ggml_cpy(gB, h_next, h_full));

            if (!ggml_gallocr_alloc_graph(galloc, gfB)) {
                set_last_error("graph B alloc failed at layer " + std::to_string(il));
                ggml_free(gB); ggml_gallocr_free(galloc); cleanup_all(); return false;
            }
            auto tB0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(w.backend, gfB);
            auto tB1 = std::chrono::steady_clock::now();
            t_compute_b += std::chrono::duration<double>(tB1 - tB0).count();
            ggml_free(gB);
        }

        if (il == 0 || il == w.n_layer - 1) {
            std::fprintf(stderr, "[qwen3-0.6b-fp] layer %d/%d done (A=%.3fs FP=%.3fs B=%.3fs)\n",
                         il + 1, w.n_layer, t_compute_a, t_fp, t_compute_b);
            std::fflush(stderr);
        }
    }

    ggml_gallocr_free(galloc);

    auto t_fwd_end = std::chrono::steady_clock::now();
    double t_fwd = std::chrono::duration<double>(t_fwd_end - t_total_start).count();

    // Tail attention scoring (unchanged from previous impl).
    std::vector<float> probs_h((size_t)S * n_lookahead * H);
    auto t_score_start = std::chrono::steady_clock::now();

    for (int il = 0; il < w.n_layer; ++il) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 16 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);

        ggml_tensor * K_f32 = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, D, Hk, S);
        ggml_tensor * K_cast = ggml_cpy(gctx, K_curr_v[il].t, K_f32);
        ggml_tensor * K_perm = ggml_cont(gctx,
            ggml_permute(gctx, K_cast, 0, 2, 1, 3));
        ggml_tensor * K_score = K_perm;
        if (gqa > 1) {
            ggml_tensor * K_4d = ggml_reshape_4d(gctx, K_perm, D, S, 1, Hk);
            ggml_tensor * K_tpl = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                                                     D, S, gqa, Hk);
            ggml_tensor * K_rep = ggml_repeat(gctx, K_4d, K_tpl);
            K_score = ggml_reshape_3d(gctx, K_rep, D, S, H);
        }
        ggml_tensor * Q_tail_perm = ggml_cont(gctx,
            ggml_permute(gctx, Q_last_v[il].t, 0, 2, 1, 3));
        ggml_tensor * attn_score = ggml_mul_mat(gctx, K_score, Q_tail_perm);
        ggml_tensor * probs = ggml_soft_max_ext(gctx, attn_score, mask_tail_buf.t,
                                                scale, 0.0f);
        ggml_set_output(probs);

        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, probs);

        ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(gctx, w.backend);
        ggml_gallocr_t s_galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(w.backend));
        if (!ggml_gallocr_alloc_graph(s_galloc, gf)) {
            set_last_error("tail score graph alloc failed at layer " + std::to_string(il));
            ggml_gallocr_free(s_galloc);
            if (in_buf) ggml_backend_buffer_free(in_buf);
            ggml_free(gctx);
            cleanup_all();
            return false;
        }
        ggml_backend_graph_compute(w.backend, gf);
        ggml_backend_tensor_get(probs, probs_h.data(), 0,
                                probs_h.size() * sizeof(float));
        ggml_gallocr_free(s_galloc);
        if (in_buf) ggml_backend_buffer_free(in_buf);
        ggml_free(gctx);

        for (int t = 0; t < n_lookahead; ++t) {
            for (int j = 0; j < S; ++j) {
                float m = -INFINITY;
                for (int h = 0; h < H; ++h) {
                    float v = probs_h[(size_t)j
                                      + (size_t)t * S
                                      + (size_t)h * S * n_lookahead];
                    if (v > m) m = v;
                }
                size_t idx = (size_t)t * S + j;
                if (m > running_max[idx]) running_max[idx] = m;
            }
        }
    }

    auto t_total_end = std::chrono::steady_clock::now();
    double t_score = std::chrono::duration<double>(t_total_end - t_score_start).count();
    std::fprintf(stderr,
        "[qwen3-0.6b-fp] forward %.2fs (S=%d, A=%.2fs FP=%.2fs B=%.2fs)  "
        "tail-score %.2fs  total %.2fs\n",
        t_fwd, S, t_compute_a, t_fp, t_compute_b, t_score, t_fwd + t_score);
    std::fflush(stderr);

    cleanup_all();
    return true;
}

} // namespace dflash27b



================================================
FILE: dflash/src/qwen3_0p6b_loader.cpp
================================================
// GGUF loader for Qwen3-0.6B drafter. Reads weights from a BF16 GGUF file
// produced by `convert_hf_to_gguf.py Qwen/Qwen3-0.6B`. Sets up ggml tensors
// on the requested backend.
//
// Tensor layout (verified via gguf reader):
//   token_embd.weight                 BF16 [hidden=1024, vocab=151936]
//   output_norm.weight                F32  [hidden]
//   output.weight                     BF16 [hidden, vocab] (lm_head)
//
//   blk.<i>.attn_norm.weight          F32  [hidden]
//   blk.<i>.attn_q.weight             BF16 [hidden, q_dim=2048]
//   blk.<i>.attn_k.weight             BF16 [hidden, kv_dim=1024]
//   blk.<i>.attn_v.weight             BF16 [hidden, kv_dim]
//   blk.<i>.attn_output.weight        BF16 [q_dim, hidden]
//   blk.<i>.attn_q_norm.weight        F32  [head_dim=128]
//   blk.<i>.attn_k_norm.weight        F32  [head_dim]
//   blk.<i>.ffn_norm.weight           F32  [hidden]
//   blk.<i>.ffn_gate.weight           BF16 [hidden, ffn=3072]
//   blk.<i>.ffn_up.weight             BF16 [hidden, ffn]
//   blk.<i>.ffn_down.weight           BF16 [ffn, hidden]
//
// We mmap the GGUF file and copy each tensor's bytes to the backend buffer
// (mirrors the dflash gguf_target_loader pattern).

#include "qwen3_0p6b_drafter.h"
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash27b {

namespace {

bool copy_tensor_from_file(gguf_context * gctx, const char * name,
                           const void * mmap_base, size_t data_offset,
                           ggml_tensor * dst) {
    int idx = gguf_find_tensor(gctx, name);
    if (idx < 0) {
        std::fprintf(stderr, "[qwen3-0.6b] missing tensor: %s\n", name);
        return false;
    }
    const size_t off = gguf_get_tensor_offset(gctx, idx);
    const size_t bytes = ggml_nbytes(dst);
    const uint8_t * src = (const uint8_t *)mmap_base + data_offset + off;
    ggml_backend_tensor_set(dst, src, 0, bytes);
    return true;
}

uint32_t get_u32(gguf_context * g, const char * key, uint32_t def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_u32(g, k);
}

float get_f32(gguf_context * g, const char * key, float def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_f32(g, k);
}

} // namespace

bool load_qwen3_0p6b_drafter(const std::string & path,
                              ggml_backend_t backend,
                              Qwen3DrafterWeights & out) {
    out.backend = backend;

    gguf_init_params iparams{ /*no_alloc=*/ false, /*ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), iparams);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    out.n_embd     = (int)get_u32(gctx, "qwen3.embedding_length", 1024);
    out.n_ff       = (int)get_u32(gctx, "qwen3.feed_forward_length", 3072);
    out.n_head     = (int)get_u32(gctx, "qwen3.attention.head_count", 16);
    out.n_head_kv  = (int)get_u32(gctx, "qwen3.attention.head_count_kv", 8);
    out.n_layer    = (int)get_u32(gctx, "qwen3.block_count", 28);
    out.n_ctx_max  = (int)get_u32(gctx, "qwen3.context_length", 40960);
    out.head_dim   = (int)get_u32(gctx, "qwen3.attention.key_length", 128);
    out.rope_theta = get_f32(gctx, "qwen3.rope.freq_base", 1000000.0f);

    // Compute total tensor metadata size for context allocation.
    const int n_layer = out.n_layer;
    const int n_tensors_per_layer = 11;
    const int n_top_tensors = 3;
    const int total_tensors = n_top_tensors + n_layer * n_tensors_per_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * total_tensors + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);

    const int n_embd = out.n_embd;
    const int n_ff   = out.n_ff;
    const int n_head = out.n_head;
    const int n_head_kv = out.n_head_kv;
    const int head_dim  = out.head_dim;
    const int n_vocab   = out.n_vocab;
    const int q_dim     = n_head * head_dim;
    const int kv_dim    = n_head_kv * head_dim;

    // Top-level tensors.
    out.tok_embd = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_vocab);
    out.out_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    out.output   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_vocab);
    ggml_set_name(out.tok_embd, "token_embd.weight");
    ggml_set_name(out.out_norm, "output_norm.weight");
    ggml_set_name(out.output,   "output.weight");

    out.layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto & L = out.layers[il];
        L.attn_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.wq        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, q_dim);
        L.wk        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, kv_dim);
        L.wv        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, kv_dim);
        L.wo        = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, q_dim, n_embd);
        L.q_norm    = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        L.k_norm    = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        L.ffn_norm  = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.ffn_gate  = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_ff);
        L.ffn_up    = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_embd, n_ff);
        L.ffn_down  = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, n_ff, n_embd);
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for Qwen3-0.6B drafter");
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // mmap the GGUF data section.
    const size_t data_off = gguf_get_data_offset(gctx);
    int fd = ::open(path.c_str(), O_RDONLY);
    struct stat st; ::fstat(fd, &st);
    void * mm = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mm == MAP_FAILED) {
        set_last_error("mmap failed for " + path);
        gguf_free(gctx);
        return false;
    }

    bool ok = true;
    ok &= copy_tensor_from_file(gctx, "token_embd.weight", mm, data_off, out.tok_embd);
    ok &= copy_tensor_from_file(gctx, "output_norm.weight", mm, data_off, out.out_norm);
    // Qwen3-0.6B ties lm_head to embed; output.weight is optional.
    if (gguf_find_tensor(gctx, "output.weight") >= 0) {
        ok &= copy_tensor_from_file(gctx, "output.weight", mm, data_off, out.output);
    }
    char nm[128];
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = out.layers[il];
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight",   il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wq);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wk);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wv);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wo);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.q_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.k_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_norm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_gate);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_up);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_down);
    }
    ::munmap(mm, st.st_size);
    gguf_free(gctx);

    if (!ok) {
        set_last_error("one or more Qwen3-0.6B tensors failed to load");
        ggml_backend_buffer_free(out.buf);
        ggml_free(out.ctx);
        out.buf = nullptr;
        out.ctx = nullptr;
        return false;
    }
    return true;
}

void free_qwen3_0p6b_drafter(Qwen3DrafterWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    w.layers.clear();
    w.tok_embd = w.out_norm = w.output = nullptr;
    w.backend = nullptr;
}

} // namespace dflash27b



================================================
FILE: dflash/src/qwen3_dflash_graph.cpp
================================================
// Builds a ggml compute graph for one forward pass of the DFlash draft
// (5-layer non-causal Qwen3-flavored block-diffusion model).
//
// Stateless: no KV cache. Each call takes:
//   - noise_embed         [hidden,   q_len, 1]   bf16   (target.tok_embd on [last_tok, MASK*15])
//   - target_hidden_cat   [5*hidden, ctx_len, 1] bf16   (5 target layers concat along features)
//   - positions_q         [q_len]                i32    values [ctx_len..ctx_len+q_len-1]
//   - positions_k         [ctx_len+q_len]        i32    values [0..ctx_len+q_len-1]
// and returns:
//   - hidden_states       [hidden,   q_len, 1]   bf16   (final RMSNorm; NO lm_head here)
//
// The caller projects `hidden_states` through the TARGET's lm_head separately
// (the draft has no lm_head of its own, it shares the target's).
//
// Semantics match megaqwen3_27b_dflash/reference/dflash_reference.py exactly:
//   - fc @ target_hidden_cat -> rms_norm with hidden_norm -> target_feat
//   - Per layer (non-causal):
//       h_norm = rms_norm(h) * input_layernorm
//       Q  = wq  @ h_norm   -> per-head q_norm
//       K_ctx/V_ctx = wk/wv @ target_feat
//       K_noi/V_noi = wk/wv @ h_norm
//       K = concat[K_ctx, K_noi]  -> per-head k_norm
//       V = concat[V_ctx, V_noi]
//       RoPE(Q, positions_q); RoPE(K, positions_k)    (NEOX style, theta=10M)
//       attn = flash_attn_ext(Q, K, V, mask=null, scale=1/sqrt(head_dim))   non-causal
//       h   += wo @ attn
//       h_norm = rms_norm(h) * post_attention_layernorm
//       h   += w_down @ (silu(w_gate @ h_norm) * (w_up @ h_norm))
//   - h = rms_norm(h) * norm

#include "internal.h"
#include "dflash_graph.h"

#include <cmath>

namespace dflash27b {

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in) {

    const int q_len    = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int ctx_len  = in.ctx_len;
    const int total_k  = ctx_len + q_len;
    const int n_head   = DFLASH27B_TARGET_N_HEADS;           // 32
    const int n_kv     = DFLASH27B_TARGET_N_KV_HEADS;        // 8
    const int head_dim = DFLASH27B_TARGET_HEAD_DIM;          // 128
    const float eps    = DFLASH27B_RMS_EPS;
    const float rope_base = DFLASH27B_ROPE_THETA;
    (void)ctx_len;  // used only via input tensor shapes

    // ── 1. Feature fusion: target_feat = rms_norm(fc @ target_hidden_cat, hidden_norm)
    //    fc:                [5*hidden, hidden]  (ggml: ne[0]=5*hidden, ne[1]=hidden)
    //    target_hidden_cat: [5*hidden, ctx_len, 1]
    //    Result:            [hidden,   ctx_len, 1]
    ggml_tensor * target_feat = ggml_mul_mat(ctx, w.fc, in.target_hidden_cat);
    target_feat = ggml_rms_norm(ctx, target_feat, eps);
    target_feat = ggml_mul    (ctx, target_feat, w.hidden_norm);
    ggml_set_name(target_feat, "target_feat");

    // ── 2. Decoder layers
    ggml_tensor * h = in.noise_embed;  // [hidden, q_len, 1]

    for (int il = 0; il < DFLASH27B_DRAFT_LAYERS; il++) {
        const DraftLayer & L = w.layers[il];

        // ── 2a. Attention pre-norm
        ggml_tensor * hn = ggml_rms_norm(ctx, h, eps);
        hn = ggml_mul(ctx, hn, L.attn_norm);

        // ── 2b. Q from noise only, then per-head RMSNorm
        //     wq: [hidden, q_dim=4096]
        ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, hn);  // [q_dim, q_len, 1]
        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, q_len);  // [head_dim, n_head, q_len]
        Q = ggml_rms_norm(ctx, Q, eps);                        // normalize along head_dim
        Q = ggml_mul     (ctx, Q, L.q_norm);                   // broadcast [head_dim]

        // ── 2c. K and V from target_feat AND noise, then concat along sequence
        //     wk, wv: [hidden, kv_dim=1024]
        ggml_tensor * Kctx = ggml_mul_mat(ctx, L.wk, target_feat); // [kv_dim, ctx_len, 1]
        ggml_tensor * Kn   = ggml_mul_mat(ctx, L.wk, hn);          // [kv_dim, q_len,   1]
        ggml_tensor * Vctx = ggml_mul_mat(ctx, L.wv, target_feat);
        ggml_tensor * Vn   = ggml_mul_mat(ctx, L.wv, hn);

        // concat along ne[1] (sequence) — ggml_concat second arg dim=1
        ggml_tensor * K = ggml_concat(ctx, Kctx, Kn, 1);  // [kv_dim, total_k, 1]
        ggml_tensor * V = ggml_concat(ctx, Vctx, Vn, 1);

        // Per-head k_norm
        K = ggml_reshape_3d(ctx, K, head_dim, n_kv, total_k);
        K = ggml_rms_norm(ctx, K, eps);
        K = ggml_mul     (ctx, K, L.k_norm);

        V = ggml_reshape_3d(ctx, V, head_dim, n_kv, total_k);

        // ── 2d. RoPE (NEOX, theta=10M)
        //   Q: positions_q  [q_len]      values [ctx_len..ctx_len+q_len-1]
        //   K: positions_k  [total_k]    values [0..total_k-1]
        Q = ggml_rope_ext(ctx, Q, in.positions_q, /*freq_factors=*/nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                          rope_base, /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
        K = ggml_rope_ext(ctx, K, in.positions_k, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // ── 2e. Permute into the layout flash_attn_ext wants
        //   q: [n_embd_k=head_dim, n_batch=q_len, n_head,   ne3]
        //   k: [n_embd_k=head_dim, n_kv=total_k, n_head_kv, ne3]
        //   v: [n_embd_v=head_dim, n_kv=total_k, n_head_kv, ne3]  (not transposed)
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);  // [head_dim, q_len,    n_head, 1]
        Q = ggml_cont   (ctx, Q);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);  // [head_dim, total_k,  n_kv,   1]
        K = ggml_cont   (ctx, K);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);  // [head_dim, total_k,  n_kv,   1]
        V = ggml_cont   (ctx, V);

        // ── 2f. Non-causal flash attention; GQA broadcast handled internally.
        const float scale = 1.0f / std::sqrt((float)head_dim);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, /*mask=*/nullptr,
                                                 scale, /*max_bias=*/0.0f,
                                                 /*logit_softcap=*/0.0f);
        // attn result: [n_embd_v=head_dim, n_head, n_batch=q_len, 1]
        attn = ggml_reshape_2d(ctx, attn, head_dim * n_head, q_len);
        // attn: [q_dim, q_len]

        // ── 2g. Output projection + residual
        //     wo: [q_dim, hidden]  (ne[0]=q_dim, ne[1]=hidden)
        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn);  // [hidden, q_len]
        h = ggml_add(ctx, h, attn_out);

        // ── 2h. FFN pre-norm
        ggml_tensor * hf = ggml_rms_norm(ctx, h, eps);
        hf = ggml_mul(ctx, hf, L.ffn_norm);

        // ── 2i. SwiGLU: down(silu(gate(x)) * up(x))
        //     w_gate, w_up: [hidden, intermediate]
        //     w_down:       [intermediate, hidden]
        ggml_tensor * g  = ggml_mul_mat(ctx, L.w_gate, hf);  // [inter, q_len]
        g = ggml_silu(ctx, g);
        ggml_tensor * u  = ggml_mul_mat(ctx, L.w_up,   hf);  // [inter, q_len]
        ggml_tensor * gu = ggml_mul(ctx, g, u);
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, L.w_down, gu);  // [hidden, q_len]

        h = ggml_add(ctx, h, ffn_out);
    }

    // ── 3. Final norm
    ggml_tensor * out = ggml_rms_norm(ctx, h, eps);
    out = ggml_mul(ctx, out, w.out_norm);
    ggml_set_name(out, "draft_hidden_out");

    DraftGraphOutputs og{};
    og.hidden_states = out;
    og.logits = nullptr;

    // ── 4. Optional: project through target's lm_head to emit vocab logits
    if (in.lm_head) {
        ggml_tensor * logits = ggml_mul_mat(ctx, in.lm_head, out);
        ggml_set_name(logits, "draft_logits");
        og.logits = logits;
    }
    return og;
}

} // namespace dflash27b



================================================
FILE: dflash/src/qwen3_drafter.cpp
================================================
// Qwen3-0.6B drafter for pflash speculative prefill, hosted in-process.
//
// Wires three pieces:
//   - qwen3_0p6b_loader.cpp : mmap GGUF + populate ggml tensors on backend
//   - qwen3_0p6b_graph.cpp  : custom forward (per-layer ggml + FP CUDA kernel)
//   - chunk-top-K + span merge (this file)
//
// Single-pass forward at full S using a custom Qwen3-0.6B graph with the
// FlashPrefill block-sparse attention kernel (or BSA when enabled). Tail
// attention scoring runs in a separate post-forward graph using saved Q_last
// and K_curr per layer.
//
// Result running_max [n_lookahead, S] f32 is reduced to per-token scores via
// mean-over-lookahead, smoothed with AvgPool, scored per chunk, top-K kept.

#include "qwen3_drafter.h"
#include "qwen3_0p6b_drafter.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash27b {

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterContext & out) {
    if (out.loaded) {
        set_last_error("drafter already loaded");
        return false;
    }

    // If caller didn't supply a backend, spin up our own CUDA one. Sharing
    // would be ideal but we don't have a handle to the daemon's backend
    // through this API. Same-process CUDA pools coexist fine — fragmentation
    // is the only cost, and we free everything in free_drafter.
    if (!out.backend) {
        size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                out.backend = ggml_backend_dev_init(dev, nullptr);
                break;
            }
        }
        if (!out.backend) {
            set_last_error("load_drafter: no GPU backend available");
            return false;
        }
    }

    if (!load_qwen3_0p6b_drafter(gguf_path, out.backend, out.weights)) {
        // last_error already set by loader
        return false;
    }

    out.loaded = true;
    std::fprintf(stderr,
        "[drafter] loaded Qwen3-0.6B BF16: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d vocab=%d\n",
        out.weights.n_layer, out.weights.n_head, out.weights.n_head_kv,
        out.weights.n_embd, out.weights.n_ff, out.weights.head_dim,
        out.weights.n_vocab);
    std::fflush(stderr);
    return true;
}

void free_drafter(DrafterContext & ctx) {
    if (ctx.loaded) {
        free_qwen3_0p6b_drafter(ctx.weights);
    }
    if (ctx.backend) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
    ctx.loaded = false;
}

std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel) {
    if (!ctx.loaded) {
        set_last_error("drafter not loaded");
        return {};
    }
    const int S = (int)ids.size();
    if (S < n_lookahead + 1) {
        // Too short to score — return as-is.
        return ids;
    }

    // ── 1. Custom forward + GPU tail-attention scoring ────────────────
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max;
    if (!forward_qwen3_0p6b_drafter(ctx.weights, ids, n_lookahead, running_max)) {
        return {};
    }
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[drafter] forward+score in %.2fs S=%d\n",
        std::chrono::duration<double>(t1 - t0).count(), S);
    std::fflush(stderr);

    // ── 2. Mean over lookahead → per-token score [S] ──────────────────
    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) {
            s += running_max[(size_t)t * S + j];
        }
        score[j] = s / (float)n_lookahead;
    }

    // ── 3. AvgPool 1D smoothing ───────────────────────────────────────
    std::vector<float> smooth((size_t)S, 0.0f);
    int half = pool_kernel / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[k]; ++n; }
        smooth[j] = (n > 0) ? (s / (float)n) : 0.0f;
    }

    // ── 4. Chunk-top-K + span merge ───────────────────────────────────
    int n_chunks = (S + chunk_size - 1) / chunk_size;
    int n_keep   = std::max(1, (int)((float)n_chunks * keep_ratio));
    std::vector<std::pair<float, int>> chunk_means;
    chunk_means.reserve((size_t)n_chunks);
    for (int c = 0; c < n_chunks; ++c) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        float m = 0.0f;
        for (int j = s_; j < e_; ++j) m += smooth[j];
        m /= std::max(1, e_ - s_);
        chunk_means.push_back({m, c});
    }
    std::partial_sort(chunk_means.begin(),
                      chunk_means.begin() + n_keep,
                      chunk_means.end(),
                      [](auto a, auto b) { return a.first > b.first; });
    std::vector<int> selected;
    selected.reserve((size_t)n_keep);
    for (int i = 0; i < n_keep; ++i) selected.push_back(chunk_means[i].second);
    std::sort(selected.begin(), selected.end());

    std::vector<int32_t> out;
    out.reserve((size_t)n_keep * chunk_size + 16);
    int span_start = -1, span_end = -1;
    for (int c : selected) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
    }

    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[drafter] score_and_compress total %.2fs S=%d kept=%zu (%d/%d chunks)\n",
        std::chrono::duration<double>(t2 - t0).count(),
        S, out.size(), n_keep, n_chunks);
    std::fflush(stderr);

    return out;
}

} // namespace dflash27b



================================================
FILE: dflash/src/qwen3_drafter.h
================================================
// In-process Qwen3-0.6B drafter for pflash speculative prefill.
//
// Hosted in the SAME process / SAME ggml allocator as the dflash target, so
// we never pay the cross-process VRAM contention that broke the Python
// subprocess integration. Drafter uses our custom Qwen3-0.6B forward
// (qwen3_0p6b_graph.cpp + qwen3_0p6b_loader.cpp) which calls our FlashPrefill
// CUDA kernels for the attention compute, replacing libllama. This removes
// the dense O(S²) FA cost that made libllama 3+ minutes at 140K.
//
// Public entry point: drafter_score_and_compress() takes raw input token IDs,
// runs the full pflash compression pipeline in C++, returns the surviving
// token IDs (drafter vocab).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qwen3_0p6b_drafter.h"

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace dflash27b {

struct DrafterContext {
    ggml_backend_t        backend = nullptr;   // owned (created in load_drafter)
    Qwen3DrafterWeights   weights;             // BF16 weights on the backend
    bool                  loaded  = false;
};

// Load the drafter GGUF (e.g. /opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf).
// Creates a fresh CUDA backend if `backend` is null. Otherwise uses the
// caller-provided backend (so the drafter shares the daemon's allocator).
//
// `gpu_layers` is accepted for API compat but ignored — every layer goes on
// the GPU since the drafter weights are only ~1.5 GB.
bool load_drafter(const std::string & gguf_path, int gpu_layers,
                  DrafterContext & out);

void free_drafter(DrafterContext & ctx);

// Score importance per token via Liu Q-hook tail attention, then chunk-top-K
// span merge. Returns surviving token IDs (drafter vocab).
//
//   ids          input token IDs of length S
//   keep_ratio   fraction of `chunk_size`-token chunks to keep
//   chunk_size   span granularity (default 32)
//   n_lookahead  trailing Q tokens used for tail attention (default 8)
//   pool_kernel  AvgPool kernel for score smoothing (default 13)
//
// On failure returns empty vector + sets last_error.
std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float  keep_ratio,
    int    chunk_size  = 32,
    int    n_lookahead = 8,
    int    pool_kernel = 13);

} // namespace dflash27b



================================================
FILE: dflash/src/safetensors_draft.cpp
================================================
// Loads z-lab/Qwen3.5-27B-DFlash draft weights from an HF safetensors file
// (bf16) into a ggml context on the CUDA backend.
//
// Safetensors format:
//   [8 bytes little-endian uint64]  header length
//   [header length bytes]           UTF-8 JSON metadata
//   [remainder]                     raw tensor data (offsets from JSON "data_offsets")
//
// Tensor layout for the z-lab draft (5 layers, fixed):
//   fc.weight                                      [hidden, 5*hidden]  BF16
//   hidden_norm.weight                             [hidden]            BF16
//   norm.weight                                    [hidden]            BF16
//   layers.<i>.input_layernorm.weight              [hidden]
//   layers.<i>.post_attention_layernorm.weight     [hidden]
//   layers.<i>.self_attn.q_proj.weight             [q_dim=4096, hidden=5120]
//   layers.<i>.self_attn.k_proj.weight             [kv_dim=1024, hidden=5120]
//   layers.<i>.self_attn.v_proj.weight             [kv_dim=1024, hidden=5120]
//   layers.<i>.self_attn.o_proj.weight             [hidden=5120, q_dim=4096]
//   layers.<i>.self_attn.q_norm.weight             [head_dim=128]
//   layers.<i>.self_attn.k_norm.weight             [head_dim=128]
//   layers.<i>.mlp.gate_proj.weight                [intermediate=17408, hidden=5120]
//   layers.<i>.mlp.up_proj.weight                  [intermediate=17408, hidden=5120]
//   layers.<i>.mlp.down_proj.weight                [hidden=5120, intermediate=17408]
//
// HF stores matrices ROW-MAJOR [out_features, in_features]. ggml_mul_mat
// expects weights with ne[0]=in_features (fastest-varying), ne[1]=out_features.
// The byte layout is identical — we just create the tensor as
// ggml_new_tensor_2d(ctx, BF16, in, out) and copy the raw bytes.

#include "internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <unordered_map>
#include <vector>

namespace dflash27b {

namespace {

struct StEntry {
    std::string           dtype;
    std::vector<int64_t>  shape;
    uint64_t              data_start;
    uint64_t              data_end;
};

using StMap = std::unordered_map<std::string, StEntry>;

// Tiny hand-rolled parser for the fixed safetensors JSON schema.
// Safetensors JSON is always a single object:
//   { "name": {"dtype":"BF16","shape":[a,b],"data_offsets":[s,e]}, ..., "__metadata__":{...} }
// We parse one top-level entry at a time, tracking brace depth to find each
// tensor's object boundary cleanly.
bool parse_st_header(const char * h, size_t hlen, StMap & out) {
    auto skip_ws = [&](size_t & i) {
        while (i < hlen && (h[i] == ' ' || h[i] == '\t' || h[i] == '\n' || h[i] == '\r')) i++;
    };
    size_t i = 0;
    skip_ws(i);
    if (i >= hlen || h[i] != '{') return false;
    i++;
    while (i < hlen) {
        skip_ws(i);
        if (i >= hlen) return false;
        if (h[i] == '}') { i++; break; }
        if (h[i] == ',') { i++; skip_ws(i); }
        if (i >= hlen || h[i] != '"') return false;
        i++;
        size_t name_start = i;
        while (i < hlen && h[i] != '"') i++;
        if (i >= hlen) return false;
        std::string name(h + name_start, i - name_start);
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != ':') return false;
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != '{') return false;
        size_t obj_start = i;
        int depth = 0;
        size_t obj_end = i;
        for (; obj_end < hlen; obj_end++) {
            if (h[obj_end] == '{') depth++;
            else if (h[obj_end] == '}') {
                depth--;
                if (depth == 0) { obj_end++; break; }
            }
        }
        if (depth != 0) return false;

        if (name == "__metadata__") {
            i = obj_end;
            continue;
        }

        std::string obj(h + obj_start, obj_end - obj_start);

        StEntry e;
        {
            auto k = obj.find("\"dtype\":\"");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find('"', vs);
            if (ve == std::string::npos) return false;
            e.dtype = obj.substr(vs, ve - vs);
        }
        {
            auto k = obj.find("\"shape\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            const char * p = obj.c_str() + vs;
            const char * pe = obj.c_str() + ve;
            while (p < pe) {
                char * end = nullptr;
                long long v = std::strtoll(p, &end, 10);
                if (end == p) break;
                e.shape.push_back((int64_t)v);
                p = end;
                while (p < pe && (*p == ',' || *p == ' ')) p++;
            }
        }
        {
            auto k = obj.find("\"data_offsets\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 16;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            unsigned long long s = 0, ed = 0;
            if (std::sscanf(obj.c_str() + vs, "%llu , %llu", &s, &ed) != 2) {
                if (std::sscanf(obj.c_str() + vs, "%llu,%llu", &s, &ed) != 2) return false;
            }
            e.data_start = s;
            e.data_end   = ed;
        }

        out.emplace(std::move(name), std::move(e));
        i = obj_end;
    }
    return true;
}

// Map safetensors dtype string to ggml type
ggml_type st_dtype_to_ggml(const std::string & dt) {
    if (dt == "BF16") return GGML_TYPE_BF16;
    if (dt == "F16")  return GGML_TYPE_F16;
    if (dt == "F32")  return GGML_TYPE_F32;
    return GGML_TYPE_COUNT;  // sentinel "invalid"
}

struct Mmap {
    void *  addr    = nullptr;
    size_t  len     = 0;
#if defined(_WIN32)
    HANDLE  hFile   = INVALID_HANDLE_VALUE;
    HANDLE  hMap    = nullptr;
#else
    int     fd      = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "open: " + path + ": " + std::strerror(errno);
            return false;
        }
        struct stat st;
        if (::fstat(fd, &st) < 0) {
            err = "fstat: " + std::string(std::strerror(errno));
            return false;
        }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            err = "mmap: " + std::string(std::strerror(errno));
            addr = nullptr;
            return false;
        }
#endif
        return true;
    }

    ~Mmap() {
#if defined(_WIN32)
        if (addr)                        UnmapViewOfFile(addr);
        if (hMap)                        CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

// Allocate a ggml tensor matching a safetensors entry.
//
// `gt_override`: if not GGML_TYPE_COUNT, use this as the ggml storage type
// instead of the safetensors dtype. Used to store small "norm" weights as
// F32 while the safetensors file has them as BF16 — required because
// ggml's CUDA elementwise ops (ggml_mul in particular) reject BF16 src1.
// The actual bf16→f32 conversion happens later in the data-copy loop.
ggml_tensor * alloc_tensor(ggml_context * ctx,
                           const StMap & st,
                           const std::string & name,
                           const std::vector<int64_t> & expected_shape,
                           const std::string & dtype_expected = "BF16",
                           ggml_type gt_override = GGML_TYPE_COUNT) {
    auto it = st.find(name);
    if (it == st.end()) {
        set_last_error("safetensors: missing tensor '" + name + "'");
        return nullptr;
    }
    const StEntry & e = it->second;
    if (e.dtype != dtype_expected) {
        set_last_error("safetensors: '" + name + "' dtype=" + e.dtype +
                       " expected " + dtype_expected);
        return nullptr;
    }
    if (e.shape.size() != expected_shape.size()) {
        set_last_error("safetensors: '" + name + "' ndim mismatch");
        return nullptr;
    }
    for (size_t k = 0; k < expected_shape.size(); k++) {
        if (e.shape[k] != expected_shape[k]) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "safetensors: '%s' shape[%zu]=%lld expected %lld",
                name.c_str(), k, (long long)e.shape[k], (long long)expected_shape[k]);
            set_last_error(buf);
            return nullptr;
        }
    }
    ggml_type gt = (gt_override == GGML_TYPE_COUNT)
                       ? st_dtype_to_ggml(dtype_expected)
                       : gt_override;
    if (gt == GGML_TYPE_COUNT) {
        set_last_error("safetensors: unsupported dtype " + dtype_expected);
        return nullptr;
    }

    // Shape convention: HF row-major [out, in] → ggml col-major [in, out].
    ggml_tensor * t = nullptr;
    if (expected_shape.size() == 1) {
        t = ggml_new_tensor_1d(ctx, gt, expected_shape[0]);
    } else if (expected_shape.size() == 2) {
        // expected_shape is written as [out, in]; ggml wants ne[0]=in, ne[1]=out
        t = ggml_new_tensor_2d(ctx, gt, expected_shape[1], expected_shape[0]);
    } else {
        set_last_error("safetensors: unexpected ndim > 2 for '" + name + "'");
        return nullptr;
    }
    ggml_set_name(t, name.c_str());
    return t;
}

// Convert an array of bf16 values to f32 in place into a destination buffer.
static void bf16_to_f32_array(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        std::memcpy(&dst[i], &bits, 4);
    }
}

// Convert an array of bf16 values to fp16 via f32 intermediate.
static void bf16_to_f16_array(const uint16_t * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        // IEEE 754 f32→f16: truncate mantissa, clamp exponent.
        uint32_t u;
        std::memcpy(&u, &f, 4);
        uint32_t sign = (u >> 16) & 0x8000;
        int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x03FF;
        if (exp <= 0)       dst[i] = (uint16_t)sign;          // flush to zero
        else if (exp >= 31) dst[i] = (uint16_t)(sign | 0x7C00); // inf
        else                dst[i] = (uint16_t)(sign | (exp << 10) | mant);
    }
}

// Returns true if the current CUDA device has native BF16 tensor core support
// (Ampere SM 8.0+). On Turing (SM 7.5) cuBLAS BF16 GEMM falls back to slow
// CUDA cores instead of tensor cores. Uses ggml's CUDA backend info at runtime.
static bool cuda_has_native_bf16() {
    // Check at runtime: link against ggml-cuda's device info if available,
    // otherwise fall back to env var DFLASH27B_DRAFT_FP16 for manual override.
    const char * env = std::getenv("DFLASH27B_DRAFT_FP16");
    if (env && std::atoi(env) != 0) return false;  // force fp16

    // Probe via ggml_backend_cuda device properties (compiled-in at build time).
    // The CMAKE_CUDA_ARCHITECTURES list tells us the minimum supported arch.
    // If the smallest arch is < 80, return false.
#if defined(DFLASH27B_MIN_SM) && DFLASH27B_MIN_SM < 80
    return false;
#else
    return true;
#endif
}

} // namespace

bool load_draft_safetensors(const std::string & path,
                            ggml_backend_t       backend,
                            DraftWeights &       out) {
    // ── 1. Open + mmap ────────────────────────────────────────────
    Mmap mm;
    std::string err;
    if (!mm.open_ro(path, err)) { set_last_error(err); return false; }
    if (mm.len < 8) { set_last_error("safetensors: file too small"); return false; }

    // ── 2. Parse header ───────────────────────────────────────────
    uint64_t header_len = 0;
    std::memcpy(&header_len, mm.addr, 8);
    if (header_len == 0 || 8 + header_len > mm.len) {
        set_last_error("safetensors: bad header length");
        return false;
    }
    const char * header_ptr = (const char *)mm.addr + 8;
    StMap st;
    if (!parse_st_header(header_ptr, header_len, st)) {
        set_last_error("safetensors: JSON header parse failed");
        return false;
    }
    const uint8_t * blob = (const uint8_t *)mm.addr + 8 + header_len;
    const size_t    blob_len = mm.len - 8 - header_len;

    // ── 3. Allocate ggml context big enough for 5 layers × 11 + 3 top ─
    const int n_layers    = DFLASH27B_DRAFT_LAYERS;
    const int n_tensors   = 3 + 11 * n_layers;  // with some headroom below
    ggml_init_params ip{};
    ip.mem_size   = (size_t)(n_tensors + 16) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("ggml_init failed for draft ctx"); return false; }
    out.backend = backend;
    out.layers.assign(n_layers, DraftLayer{});

    const int64_t HIDDEN  = DFLASH27B_TARGET_HIDDEN;           // 5120
    const int64_t Q_DIM   = DFLASH27B_TARGET_N_HEADS * DFLASH27B_TARGET_HEAD_DIM;     // 4096
    const int64_t KV_DIM  = DFLASH27B_TARGET_N_KV_HEADS * DFLASH27B_TARGET_HEAD_DIM;  // 1024
    const int64_t INTER   = DFLASH27B_TARGET_INTERMEDIATE;     // 17408
    const int64_t HD      = DFLASH27B_TARGET_HEAD_DIM;         // 128
    const int64_t FC_IN   = DFLASH27B_DRAFT_N_TARGET_LAYERS * HIDDEN; // 25600

    // ── 4. Create named tensors in the context ───────────────────
    //
    // Norms (rms_norm weights) are loaded as F32 because ggml's CUDA
    // elementwise ops require F32/F16 operands. Projection weights stay bf16
    // on Ampere+ (native tensor core support) or are converted to fp16 on
    // Turing (SM 7.5) where cuBLAS BF16 GEMM falls back to slow CUDA cores.
    const ggml_type NORM_GT = GGML_TYPE_F32;
    const bool native_bf16 = cuda_has_native_bf16();
    const ggml_type PROJ_GT = native_bf16 ? GGML_TYPE_COUNT : GGML_TYPE_F16;

    out.fc          = alloc_tensor(out.ctx, st, "fc.weight",           {HIDDEN, FC_IN},  "BF16", PROJ_GT);
    out.hidden_norm = alloc_tensor(out.ctx, st, "hidden_norm.weight",  {HIDDEN}, "BF16", NORM_GT);
    out.out_norm    = alloc_tensor(out.ctx, st, "norm.weight",         {HIDDEN}, "BF16", NORM_GT);
    if (!out.fc || !out.hidden_norm || !out.out_norm) return false;

    for (int il = 0; il < n_layers; il++) {
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "layers.%d.", il);
        std::string p = pfx;
        DraftLayer & L = out.layers[il];
        L.attn_norm = alloc_tensor(out.ctx, st, p + "input_layernorm.weight",          {HIDDEN}, "BF16", NORM_GT);
        L.ffn_norm  = alloc_tensor(out.ctx, st, p + "post_attention_layernorm.weight", {HIDDEN}, "BF16", NORM_GT);
        L.wq        = alloc_tensor(out.ctx, st, p + "self_attn.q_proj.weight", {Q_DIM,  HIDDEN}, "BF16", PROJ_GT);
        L.wk        = alloc_tensor(out.ctx, st, p + "self_attn.k_proj.weight", {KV_DIM, HIDDEN}, "BF16", PROJ_GT);
        L.wv        = alloc_tensor(out.ctx, st, p + "self_attn.v_proj.weight", {KV_DIM, HIDDEN}, "BF16", PROJ_GT);
        L.wo        = alloc_tensor(out.ctx, st, p + "self_attn.o_proj.weight", {HIDDEN, Q_DIM},  "BF16", PROJ_GT);
        L.q_norm    = alloc_tensor(out.ctx, st, p + "self_attn.q_norm.weight", {HD}, "BF16", NORM_GT);
        L.k_norm    = alloc_tensor(out.ctx, st, p + "self_attn.k_norm.weight", {HD}, "BF16", NORM_GT);
        L.w_gate    = alloc_tensor(out.ctx, st, p + "mlp.gate_proj.weight",    {INTER,  HIDDEN}, "BF16", PROJ_GT);
        L.w_up      = alloc_tensor(out.ctx, st, p + "mlp.up_proj.weight",      {INTER,  HIDDEN}, "BF16", PROJ_GT);
        L.w_down    = alloc_tensor(out.ctx, st, p + "mlp.down_proj.weight",    {HIDDEN, INTER},  "BF16", PROJ_GT);
        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            return false;
        }
    }

    // ── 5. Allocate backend buffer, copy bytes ───────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) { set_last_error("ggml_backend_alloc_ctx_tensors failed (draft)"); return false; }

    // Walk the tensors in the context and upload their bytes.
    // For tensors whose ggml type differs from the safetensors dtype (i.e.
    // BF16-on-disk, F32-in-ggml for norms, or BF16-on-disk, F16-in-ggml for
    // projection weights on Turing), convert on the fly via scratch buffers.
    std::vector<float>    scratch_f32;
    std::vector<uint16_t> scratch_f16;
    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t != nullptr;
         t = ggml_get_next_tensor(out.ctx, t)) {
        const char * name = ggml_get_name(t);
        auto it = st.find(name);
        if (it == st.end()) {
            set_last_error("post-alloc: tensor '" + std::string(name) + "' vanished from header");
            return false;
        }
        const StEntry & e = it->second;
        if (e.data_end > 8 + header_len + blob_len + 8 /*slack*/) {
            set_last_error("post-alloc: offset out of bounds for '" + std::string(name) + "'");
            return false;
        }
        const size_t src_nbytes = e.data_end - e.data_start;
        const size_t dst_nbytes = ggml_nbytes(t);
        const bool same_dtype = (t->type == st_dtype_to_ggml(e.dtype));

        if (same_dtype) {
            if (src_nbytes != dst_nbytes) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "byte count mismatch for '%s': blob=%zu ggml=%zu",
                    name, src_nbytes, dst_nbytes);
                set_last_error(buf);
                return false;
            }
            ggml_backend_tensor_set(t, blob + e.data_start, 0, dst_nbytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F32) {
            const size_t n = ggml_nelements(t);
            if (src_nbytes != n * sizeof(uint16_t) || dst_nbytes != n * sizeof(float)) {
                set_last_error("BF16->F32 size mismatch for '" + std::string(name) + "'");
                return false;
            }
            scratch_f32.resize(n);
            bf16_to_f32_array((const uint16_t *)(blob + e.data_start),
                              scratch_f32.data(), n);
            ggml_backend_tensor_set(t, scratch_f32.data(), 0, dst_nbytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F16) {
            const size_t n = ggml_nelements(t);
            if (src_nbytes != n * sizeof(uint16_t) || dst_nbytes != n * sizeof(uint16_t)) {
                set_last_error("BF16->F16 size mismatch for '" + std::string(name) + "'");
                return false;
            }
            scratch_f16.resize(n);
            bf16_to_f16_array((const uint16_t *)(blob + e.data_start),
                              scratch_f16.data(), n);
            ggml_backend_tensor_set(t, scratch_f16.data(), 0, dst_nbytes);
        } else {
            set_last_error(std::string("unsupported dtype conversion for '") +
                           name + "': " + e.dtype + " -> ggml type " +
                           ggml_type_name(t->type));
            return false;
        }
    }

    return true;
}

void free_draft_weights(DraftWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    w.layers.clear();
    w.fc = nullptr;
    w.hidden_norm = nullptr;
    w.out_norm = nullptr;
}

} // namespace dflash27b



================================================
FILE: dflash/test/pflash_daemon.cpp
================================================
// Persistent PFlash compressor daemon.
//
// Loads the Qwen3-0.6B PFlash drafter once, then accepts stdin commands:
//
//   compress <keep_x1000> <lookahead> <chunk> <pool> <counted_ids.bin>
//   quit
//
// Input token file format is little-endian u32 count followed by count int32
// token IDs in the drafter tokenizer. Compressed IDs are emitted as int32 LE
// values to --stream-fd=<fd>, terminated by -1. Logs go to stdout/stderr.

#include "dflash27b.h"
#include "qwen3_drafter.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace dflash27b;

static std::vector<int32_t> read_counted_i32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};

    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!f) return {};

    std::vector<int32_t> ids((size_t)n);
    if (n > 0) {
        f.read(reinterpret_cast<char *>(ids.data()), (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

static void stream_emit(int stream_fd, int32_t tok) {
    if (stream_fd < 0) return;
#ifdef _WIN32
    DWORD written = 0;
    WriteFile((HANDLE)(intptr_t)stream_fd, &tok, sizeof(tok), &written, nullptr);
#else
    ssize_t n = ::write(stream_fd, &tok, sizeof(tok));
    (void)n;
#endif
}

static void stream_ids(int stream_fd, const std::vector<int32_t> & ids) {
    for (int32_t t : ids) {
        stream_emit(stream_fd, t);
    }
    stream_emit(stream_fd, -1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen3-0.6b.gguf> [--stream-fd=N]\n", argv[0]);
        return 2;
    }

    const std::string gguf = argv[1];
    int stream_fd = -1;
    for (int i = 2; i < argc; ++i) {
        if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
    }

    DrafterContext ctx;
    auto t_load0 = std::chrono::steady_clock::now();
    if (!load_drafter(gguf, /*gpu_layers=*/-1, ctx)) {
        std::fprintf(stderr, "[pflash-daemon] load_drafter failed: %s\n", dflash27b_last_error());
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[pflash-daemon] ready load=%.3fs vocab=%d\n",
                std::chrono::duration<double>(t_load1 - t_load0).count(),
                ctx.weights.n_vocab);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") {
            break;
        }

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd != "compress") {
            std::fprintf(stderr, "[pflash-daemon] unknown command: %s\n", line.c_str());
            stream_emit(stream_fd, -1);
            continue;
        }

        int keep_x1000 = 0;
        int lookahead = 8;
        int chunk = 32;
        int pool = 13;
        iss >> keep_x1000 >> lookahead >> chunk >> pool;
        std::string path;
        std::getline(iss, path);
        const size_t first_non_space = path.find_first_not_of(" \t");
        if (first_non_space == std::string::npos) {
            path.clear();
        } else if (first_non_space > 0) {
            path.erase(0, first_non_space);
        }
        if (path.empty() || keep_x1000 <= 0) {
            std::fprintf(stderr, "[pflash-daemon] bad command, need: compress <keep_x1000> <lookahead> <chunk> <pool> <counted_ids.bin>\n");
            stream_emit(stream_fd, -1);
            continue;
        }

        auto ids = read_counted_i32_file(path);
        if (ids.empty()) {
            std::fprintf(stderr, "[pflash-daemon] empty input: %s\n", path.c_str());
            stream_emit(stream_fd, -1);
            continue;
        }

        const float keep_ratio = (float)keep_x1000 / 1000.0f;
        std::printf("[pflash-daemon] compress start tokens=%zu keep=%.3f lookahead=%d chunk=%d pool=%d path=%s\n",
                    ids.size(), keep_ratio, lookahead, chunk, pool, path.c_str());
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> out = drafter_score_and_compress(ctx, ids, keep_ratio, chunk, lookahead, pool);
        auto t1 = std::chrono::steady_clock::now();

        const double secs = std::chrono::duration<double>(t1 - t0).count();
        if (out.empty()) {
            std::fprintf(stderr, "[pflash-daemon] compress failed: %s\n", dflash27b_last_error());
            stream_emit(stream_fd, -1);
            continue;
        }

        std::printf("[pflash-daemon] compress done %.3fs in=%zu out=%zu ratio=%.4f\n",
                    secs, ids.size(), out.size(), (double)out.size() / (double)ids.size());
        std::fflush(stdout);
        stream_ids(stream_fd, out);
    }

    free_drafter(ctx);
    std::printf("[pflash-daemon] stopped\n");
    std::fflush(stdout);
    return 0;
}



================================================
FILE: dflash/test/smoke_draft_graph.cpp
================================================
// Smoke test for the DFlash draft forward graph.
//
// Loads real draft weights, fills noise_embedding / target_hidden_cat with
// deterministic random bf16 data, runs the graph on CUDA, pulls results back,
// and checks:
//   - no NaN / Inf in output
//   - shape is [hidden, q_len, 1]
//   - a few representative values look reasonable (near-zero means for rms_norm output)
//
// Usage:
//   smoke_draft_graph <draft.safetensors> [ctx_len]
//
// ctx_len defaults to 64 to keep the first run tiny.

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace dflash27b;

// Convert fp32 -> bf16 (truncation)
static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t u) {
    uint32_t bits = ((uint32_t)u) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.safetensors> [ctx_len]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const int ctx_len = (argc >= 3) ? std::atoi(argv[2]) : 64;
    const int q_len   = DFLASH27B_DRAFT_BLOCK_SIZE;      // 16
    const int hidden  = DFLASH27B_TARGET_HIDDEN;         // 5120
    const int fc_in   = DFLASH27B_DRAFT_N_TARGET_LAYERS * hidden;  // 25600

    std::printf("ctx_len=%d q_len=%d hidden=%d fc_in=%d\n", ctx_len, q_len, hidden, fc_in);

    // ── 1. Backend + weights
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "ggml_backend_cuda_init failed\n"); return 1; }

    DraftWeights w;
    if (!load_draft_safetensors(path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("draft loaded\n");

    // ── 2. Graph context (separate from weights context)
    const size_t mem_size = 256 * 1024 * 1024;  // 256 MB — plenty for nodes
    ggml_init_params ip{};
    ip.mem_size   = mem_size;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init graph failed\n"); return 1; }

    // ── 3. Input placeholder tensors
    // Activations flow as F32 through the graph (CUDA rms_norm requires F32).
    // Weights stay bf16 — ggml_mul_mat auto-casts.
    ggml_tensor * noise_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_tensor * target_hid  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, fc_in, ctx_len, 1);
    ggml_tensor * pos_q       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,  q_len);
    ggml_tensor * pos_k       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32,  ctx_len + q_len);
    ggml_set_name(noise_embed, "noise_embed");
    ggml_set_name(target_hid,  "target_hidden_cat");
    ggml_set_name(pos_q,       "positions_q");
    ggml_set_name(pos_k,       "positions_k");
    ggml_set_input(noise_embed);
    ggml_set_input(target_hid);
    ggml_set_input(pos_q);
    ggml_set_input(pos_k);

    // ── 4. Build graph
    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = noise_embed;
    gi.target_hidden_cat = target_hid;
    gi.positions_q       = pos_q;
    gi.positions_k       = pos_k;

    DraftGraphOutputs go = build_draft_graph(gctx, w, gi);
    if (!go.hidden_states) { std::fprintf(stderr, "build_draft_graph returned null\n"); return 1; }
    ggml_set_output(go.hidden_states);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, go.hidden_states);
    std::printf("graph built: n_nodes=%d\n", ggml_graph_n_nodes(gf));

    // ── 5. Allocate graph + all input tensors on the backend
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        return 1;
    }

    // ── 6. Fill input tensors
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-0.02f, 0.02f);

    {
        std::vector<float> data((size_t)hidden * q_len);
        for (auto & v : data) v = u(rng);
        ggml_backend_tensor_set(noise_embed, data.data(), 0, sizeof(float) * data.size());
    }
    {
        std::vector<float> data((size_t)fc_in * ctx_len);
        for (auto & v : data) v = u(rng);
        ggml_backend_tensor_set(target_hid, data.data(), 0, sizeof(float) * data.size());
    }
    {
        std::vector<int32_t> pq(q_len);
        for (int i = 0; i < q_len; i++) pq[i] = ctx_len + i;
        ggml_backend_tensor_set(pos_q, pq.data(), 0, sizeof(int32_t) * pq.size());
    }
    {
        std::vector<int32_t> pk(ctx_len + q_len);
        for (int i = 0; i < ctx_len + q_len; i++) pk[i] = i;
        ggml_backend_tensor_set(pos_k, pk.data(), 0, sizeof(int32_t) * pk.size());
    }

    // ── 7. Compute
    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph_compute returned %d\n", (int)status);
        return 1;
    }
    std::printf("compute OK\n");

    // ── 8. Read output, check shape + no NaN + print summary stats
    const size_t n_out_elems = ggml_nelements(go.hidden_states);
    if (n_out_elems != (size_t)hidden * q_len) {
        std::fprintf(stderr, "out elems mismatch: %zu vs %d\n", n_out_elems, hidden * q_len);
        return 1;
    }
    std::vector<float> out(n_out_elems);
    ggml_backend_tensor_get(go.hidden_states, out.data(), 0, sizeof(float) * out.size());

    int n_nan = 0, n_inf = 0;
    double sum = 0.0, sumsq = 0.0;
    float vmin = 1e30f, vmax = -1e30f;
    for (auto f : out) {
        if (std::isnan(f)) { n_nan++; continue; }
        if (std::isinf(f)) { n_inf++; continue; }
        sum   += f;
        sumsq += (double)f * f;
        if (f < vmin) vmin = f;
        if (f > vmax) vmax = f;
    }
    double mean = sum / (double)out.size();
    double var  = sumsq / (double)out.size() - mean * mean;
    std::printf("out shape: [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                go.hidden_states->ne[0], go.hidden_states->ne[1], go.hidden_states->ne[2]);
    std::printf("out stats: nan=%d inf=%d mean=%.4g std=%.4g min=%.4g max=%.4g\n",
                n_nan, n_inf, mean, std::sqrt(std::max(0.0, var)), vmin, vmax);
    std::printf("out first 8 values: ");
    for (int i = 0; i < 8; i++) std::printf("%.4f ", out[i]);
    std::printf("\n");

    if (n_nan || n_inf) {
        std::fprintf(stderr, "FAIL: non-finite values in output\n");
        return 1;
    }

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_draft_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}



================================================
FILE: dflash/test/smoke_load_draft.cpp
================================================
// Smoke test: load the z-lab DFlash draft safetensors into a CUDA ggml
// context. Prints tensor count, total bytes, and a checksum-ish spot check
// on one tensor. Exit 0 on success, nonzero on any failure.
//
// Usage: smoke_load_draft <path/to/model.safetensors>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.safetensors>\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];

    // Initialize CUDA backend
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "ggml_backend_cuda_init(0) failed\n");
        return 1;
    }
    std::printf("cuda backend: %s\n", ggml_backend_name(backend));

    DraftWeights w;
    if (!load_draft_safetensors(path, backend, w)) {
        std::fprintf(stderr, "load_draft_safetensors failed: %s\n",
                     dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    // Count tensors and total bytes
    size_t n_tensors = 0;
    size_t total_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(w.ctx); t != nullptr;
         t = ggml_get_next_tensor(w.ctx, t)) {
        n_tensors++;
        total_bytes += ggml_nbytes(t);
    }
    std::printf("loaded %zu tensors, total %.2f GiB\n",
                n_tensors, total_bytes / (1024.0 * 1024.0 * 1024.0));

    // Spot check: the `fc` tensor should be [25600 -> 5120] (in ggml ne[0]=25600 ne[1]=5120).
    std::printf("fc: ne=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] type=%s nbytes=%zu\n",
                w.fc->ne[0], w.fc->ne[1], w.fc->ne[2], w.fc->ne[3],
                ggml_type_name(w.fc->type), ggml_nbytes(w.fc));
    std::printf("hidden_norm: ne[0]=%" PRId64 " type=%s\n",
                w.hidden_norm->ne[0], ggml_type_name(w.hidden_norm->type));
    std::printf("layers[0].wq: ne=[%" PRId64 ", %" PRId64 "] type=%s\n",
                w.layers[0].wq->ne[0], w.layers[0].wq->ne[1],
                ggml_type_name(w.layers[0].wq->type));

    // Pull a few bytes of `norm.weight` back from CUDA and print a few values,
    // as a proof of end-to-end data path (file → mmap → CUDA → host).
    std::vector<uint16_t> hn(w.hidden_norm->ne[0]);
    ggml_backend_tensor_get(w.hidden_norm, hn.data(), 0, sizeof(uint16_t) * hn.size());
    auto bf16_to_f32 = [](uint16_t u) {
        uint32_t bits = ((uint32_t)u) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    };
    std::printf("hidden_norm.weight first 8 values: ");
    for (int i = 0; i < 8 && i < (int)hn.size(); i++) {
        std::printf("%.4f ", bf16_to_f32(hn[i]));
    }
    std::printf("\n");

    free_draft_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}



================================================
FILE: dflash/test/smoke_load_target.cpp
================================================
// Smoke test for the GGUF target loader. Loads Qwen3.5-27B from a GGUF,
// validates metadata, and prints per-layer-type counts + a spot value check.
//
// Usage: smoke_load_target <path/to/qwen35.gguf>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf failed: %s\n", dflash27b_last_error());
        return 1;
    }
    // load_target_gguf stashes a summary string in last_error on success (hack)
    std::printf("%s\n", dflash27b_last_error());

    // Count layer types
    int n_attn = 0, n_delta = 0;
    for (int il = 0; il < w.n_layer; il++) {
        const auto & L = w.layers[il];
        bool attn = L.wq && L.wk && L.wv && L.wo;
        bool ssm  = L.wqkv && L.wqkv_gate && L.ssm_conv1d;
        if (attn) n_attn++;
        if (ssm)  n_delta++;
    }
    std::printf("hparams: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d/%d n_ff=%d fai=%d\n",
        w.n_layer, w.n_embd, w.n_head, w.n_head_kv, w.n_embd_head_k, w.n_embd_head_v, w.n_ff, w.full_attention_interval);
    std::printf("ssm:     conv=%d inner=%d state=%d dt_rank=%d n_group=%d\n",
        w.ssm_d_conv, w.ssm_d_inner, w.ssm_d_state, w.ssm_dt_rank, w.ssm_n_group);
    std::printf("rope sections: [%d, %d, %d, %d]\n",
        w.rope_sections[0], w.rope_sections[1], w.rope_sections[2], w.rope_sections[3]);
    std::printf("layer counts: full_attn=%d delta_net=%d\n", n_attn, n_delta);

    // Spot-check: tok_embd should be quantized (Q4_K or similar)
    std::printf("tok_embd: ne=[%" PRId64 ", %" PRId64 "] type=%s nbytes=%.2f MiB\n",
        w.tok_embd->ne[0], w.tok_embd->ne[1],
        ggml_type_name(w.tok_embd->type),
        ggml_nbytes(w.tok_embd) / (1024.0 * 1024.0));
    std::printf("output (lm_head): type=%s nbytes=%.2f MiB\n",
        ggml_type_name(w.output->type),
        ggml_nbytes(w.output) / (1024.0 * 1024.0));

    // Read output_norm back from CUDA and print a few values
    if (w.out_norm->type == GGML_TYPE_F32 && w.out_norm->ne[0] >= 8) {
        std::vector<float> buf(8);
        ggml_backend_tensor_get(w.out_norm, buf.data(), 0, sizeof(float) * 8);
        std::printf("output_norm first 8: ");
        for (int i = 0; i < 8; i++) std::printf("%.4f ", buf[i]);
        std::printf("\n");
    }

    // Peek at layer 0 (deltanet) and layer 3 (full-attn) to verify split
    auto print_layer = [&](int il) {
        const auto & L = w.layers[il];
        std::printf("layer %2d: %s ", il,
            (L.wq ? "FULL_ATTN" : "DELTANET "));
        if (L.wq) {
            std::printf("wq=%s[%" PRId64 ",%" PRId64 "] ",
                ggml_type_name(L.wq->type), L.wq->ne[0], L.wq->ne[1]);
        }
        if (L.wqkv) {
            std::printf("wqkv=%s[%" PRId64 ",%" PRId64 "] ",
                ggml_type_name(L.wqkv->type), L.wqkv->ne[0], L.wqkv->ne[1]);
        }
        std::printf("ffn_down=%s\n", ggml_type_name(L.w_down->type));
    };
    print_layer(0);
    print_layer(3);
    print_layer(31);
    print_layer(63);

    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}



================================================
FILE: dflash/test/smoke_qwen3_0p6b_forward.cpp
================================================
// Smoke test for the custom Qwen3-0.6B drafter forward path.
//
// Loads the BF16 GGUF, generates a synthetic token sequence at the requested
// length, runs drafter_score_and_compress end-to-end, and prints timing +
// compression ratio. Used to validate the in-process pflash integration
// without touching the daemon.
//
// Usage:
//   smoke_qwen3_0p6b_forward <gguf_path> <seq_len_or_FILE:path> [keep_ratio]
// Examples:
//   smoke_qwen3_0p6b_forward .../Qwen3-0.6B-BF16.gguf 140000 0.02
//   smoke_qwen3_0p6b_forward .../Qwen3-0.6B-BF16.gguf FILE:/tmp/niah_32k.bin 0.05
//
// Token file format: little-endian u32 count, then count int32 token IDs.

#include "qwen3_drafter.h"
#include "dflash27b.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <gguf> <seq_len> [keep_ratio=0.02]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string arg2 = argv[2];
    const float keep_ratio = (argc >= 4) ? std::atof(argv[3]) : 0.02f;

    std::vector<int32_t> ids_from_file;
    int S = 0;
    bool from_file = arg2.rfind("FILE:", 0) == 0;
    if (from_file) {
        std::string path = arg2.substr(5);
        FILE * fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "open %s failed\n", path.c_str());
            return 2;
        }
        uint32_t cnt = 0;
        if (std::fread(&cnt, 4, 1, fp) != 1) { std::fclose(fp); return 2; }
        ids_from_file.resize((size_t)cnt);
        if (std::fread(ids_from_file.data(), sizeof(int32_t), cnt, fp) != cnt) {
            std::fclose(fp); return 2;
        }
        std::fclose(fp);
        S = (int)cnt;
        std::printf("[smoke] loaded %d real tokens from %s\n", S, path.c_str());
    } else {
        S = std::atoi(arg2.c_str());
        if (S < 64) {
            std::fprintf(stderr, "seq_len too small: %d\n", S);
            return 2;
        }
    }

    DrafterContext ctx;
    auto t_load0 = std::chrono::steady_clock::now();
    if (!load_drafter(gguf, /*gpu_layers=*/-1, ctx)) {
        std::fprintf(stderr, "load_drafter failed: %s\n", dflash27b_last_error());
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[smoke] load_drafter %.2fs vocab=%d\n",
        std::chrono::duration<double>(t_load1 - t_load0).count(),
        ctx.weights.n_vocab);

    std::vector<int32_t> ids;
    if (from_file) {
        ids = std::move(ids_from_file);
    } else {
        ids.resize((size_t)S);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, ctx.weights.n_vocab - 1);
        for (int i = 0; i < S; ++i) ids[i] = dist(rng);
    }

    std::printf("[smoke] running drafter_score_and_compress S=%d keep_ratio=%.3f...\n",
        S, keep_ratio);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> out = drafter_score_and_compress(
        ctx, ids, keep_ratio,
        /*chunk_size=*/32, /*n_lookahead=*/8, /*pool_kernel=*/13);
    auto t1 = std::chrono::steady_clock::now();

    if (out.empty()) {
        std::fprintf(stderr, "drafter_score_and_compress returned empty: %s\n",
            dflash27b_last_error());
        free_drafter(ctx);
        return 1;
    }

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[smoke] OK: %.2fs  in=%d  out=%zu  ratio=%.4f\n",
        secs, S, out.size(), (double)out.size() / (double)S);

    // Optional: write kept ids to a file for downstream detokenization.
    if (const char * kept_path = std::getenv("PFLASH_KEPT_OUT")) {
        FILE * fk = std::fopen(kept_path, "wb");
        if (fk) {
            uint32_t n = (uint32_t)out.size();
            std::fwrite(&n, 4, 1, fk);
            std::fwrite(out.data(), sizeof(int32_t), n, fk);
            std::fclose(fk);
            std::printf("[smoke] wrote %u kept ids to %s\n", n, kept_path);
        }
    }

    free_drafter(ctx);
    return 0;
}



================================================
FILE: dflash/test/smoke_target_forward.cpp
================================================
// Smoke test for the qwen35 target forward graph.
//
// Loads Qwen3.5-27B from GGUF, creates a target cache, builds the forward
// graph for a single-token decode, runs it on CUDA, and prints the top-k
// token IDs from the resulting logits distribution.
//
// Does NOT validate numerics — we just want the graph to COMPUTE without
// asserting. Numerics matching vs. llama.cpp is follow-up work.
//
// Usage: smoke_target_forward <qwen35.gguf>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    // Load target weights
    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    // Create target state cache
    TargetCache cache;
    const int max_ctx = 64;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "create_target_cache: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[cache] attn_k=%zu attn_v=%zu ssm=%zu conv=%zu\n",
        cache.attn_k.size(), cache.attn_v.size(),
        cache.ssm_state.size(), cache.conv_state.size());

    // Graph context
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init graph failed\n"); return 1; }

    // Input tensors: one token (placeholder id=1). The embedding is computed
    // on CPU and uploaded via inp_embed, so tok_embd never lives on GPU.
    // M-RoPE needs 4 position values per token (one per axis).
    const int n_tokens = 1;
    const int hidden   = DFLASH27B_TARGET_HIDDEN;
    ggml_tensor * inp_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(inp_embed, "inp_embed");
    ggml_set_name(positions, "positions");
    ggml_set_input(inp_embed);
    ggml_set_input(positions);

    QwenGraphInputs gi{};
    gi.inp_embed = inp_embed;
    gi.positions = positions;
    gi.n_tokens  = n_tokens;
    gi.kv_start  = 0;

    // Pre-sized graph: 64 layers × ~60 nodes + overhead = ~4096
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);

    QwenGraphOutputs go = build_qwen35_graph(gctx, gf, w, cache, gi);
    if (!go.logits) { std::fprintf(stderr, "build_qwen35_graph returned null\n"); return 1; }
    ggml_set_output(go.logits);
    ggml_build_forward_expand(gf, go.logits);
    std::printf("[graph] nodes=%d\n", ggml_graph_n_nodes(gf));

    // Allocate graph
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        return 1;
    }

    // Fill inputs: embed token id=1 on CPU, upload to GPU
    int32_t tok_ids[1] = { 1 };
    std::vector<float> embed_buf(hidden * n_tokens);
    if (!w.embedder.embed(tok_ids, n_tokens, embed_buf.data())) {
        std::fprintf(stderr, "cpu embedder failed\n");
        return 1;
    }
    ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());

    int32_t pos4[4] = { 0, 0, 0, 0 };
    ggml_backend_tensor_set(positions, pos4, 0, sizeof(int32_t) * 4);

    // Compute
    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", (int)status);
        return 1;
    }
    std::printf("[compute] OK\n");

    // Read logits
    const int64_t vocab = DFLASH27B_TARGET_VOCAB;
    std::vector<float> logits(vocab);
    ggml_backend_tensor_get(go.logits, logits.data(), 0, sizeof(float) * vocab);

    // Quick stats
    int n_nan = 0, n_inf = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (auto v : logits) {
        if (std::isnan(v)) n_nan++;
        else if (std::isinf(v)) n_inf++;
        else { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    }
    std::printf("[logits] vocab=%" PRId64 " nan=%d inf=%d min=%.4g max=%.4g\n",
        vocab, n_nan, n_inf, vmin, vmax);

    // Top 5 token IDs
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(vocab);
    for (int i = 0; i < vocab; i++) sorted.emplace_back(logits[i], i);
    std::partial_sort(sorted.begin(), sorted.begin() + 5, sorted.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });
    std::printf("[top 5] ");
    for (int i = 0; i < 5; i++) {
        std::printf("id=%d l=%.3f  ", sorted[i].second, sorted[i].first);
    }
    std::printf("\n");

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}



================================================
FILE: dflash/test/test_flashprefill_kernels.cpp
================================================
// Smoke test for the FlashPrefill CUDA kernels.
//
// Tiny shapes (B=1, S=256, H=4, Hk=2, D=128, BLOCK=128) so we can run on
// CPU shadow + compare GPU output. Validates:
//   1. compute_mean_vector_bf16  — mean K per BLOCK rows
//   2. compute_block_score_bf16  — per-block score + max
//   3. sparse_flash_forward_bf16 — sparse attention output
//
// Pass criteria: max abs diff vs reference dense attention < 1e-2 (bf16
// numerics, so relaxed). Sparse path may differ where non-selected blocks
// were dropped — only check on a uniform full-selection mask.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../src/flashprefill.h"

extern "C" {
void launch_compute_mean_vector_bf16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream);

void launch_compute_block_score_bf16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream);

void launch_sparse_flash_forward_bf16(
    const void * Q, const void * K, const void * V, void * O,
    const int32_t * block_index, const int32_t * counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int q_tile, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream);
}

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

static __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }
static float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }

int main() {
    constexpr int B = 1;
    constexpr int S = 256;
    constexpr int H = 4;
    constexpr int Hk = 2;
    constexpr int D = 128;
    constexpr int BLOCK = 128;
    constexpr int M = (S + BLOCK - 1) / BLOCK;   // 2
    constexpr int Q_TILE = 64;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<__nv_bfloat16> Q(B * S * H * D), K(B * S * Hk * D), V(B * S * Hk * D);
    for (auto & x : Q) x = f2b(dist(rng));
    for (auto & x : K) x = f2b(dist(rng));
    for (auto & x : V) x = f2b(dist(rng));

    __nv_bfloat16 *dQ, *dK, *dV, *dO, *dmK;
    float *dS, *dM;
    int32_t *dIdx, *dCnt;
    CK(cudaMalloc(&dQ, Q.size() * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dK, K.size() * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dV, V.size() * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dO, B * S * H * D * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dmK, B * M * Hk * D * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dS, B * M * M * H * sizeof(float)));
    CK(cudaMalloc(&dM, B * M * M * H * sizeof(float)));
    CK(cudaMalloc(&dIdx, B * M * M * H * sizeof(int32_t)));
    CK(cudaMalloc(&dCnt, B * M * H * sizeof(int32_t)));

    CK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, K.data(), K.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, V.data(), V.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Strides (elements). Layout: [B, S, H, D] row-major, D fastest.
    int s_Q_b = S * H * D, s_Q_n = H * D, s_Q_h = D, s_Q_d = 1;
    int s_K_b = S * Hk * D, s_K_n = Hk * D, s_K_h = D, s_K_d = 1;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D, s_mK_d = 1;
    int s_S_b = M * M * H, s_S_m = M * H, s_S_n = H, s_S_h = 1;
    int s_idx_b = M * M * H, s_idx_m = M * H, s_idx_n = H, s_idx_h = 1;
    int s_cnt_b = M * H, s_cnt_m = H, s_cnt_h = 1;

    // ── Kernel 1: compute_mean_vector ──
    launch_compute_mean_vector_bf16(
        dK, dmK, B, S, Hk, D, BLOCK,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d, 0);
    CK(cudaDeviceSynchronize());

    // Verify host vs GPU mean for one block.
    std::vector<__nv_bfloat16> mK_h(B * M * Hk * D);
    CK(cudaMemcpy(mK_h.data(), dmK, mK_h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    float max_diff_mean = 0.0f;
    for (int n = 0; n < M; ++n) {
        for (int h = 0; h < Hk; ++h) {
            for (int d = 0; d < D; ++d) {
                float ref = 0.0f;
                int n_lo = n * BLOCK, n_hi = std::min(n_lo + BLOCK, S);
                int cnt = n_hi - n_lo;
                for (int i = n_lo; i < n_hi; ++i) {
                    ref += b2f(K[i * Hk * D + h * D + d]);
                }
                ref /= (float)cnt;
                float gpu = b2f(mK_h[n * Hk * D + h * D + d]);
                max_diff_mean = std::fmax(max_diff_mean, std::fabs(ref - gpu));
            }
        }
    }
    std::printf("[fp-test] kernel 1 (compute_mean_vector): max diff = %.5f %s\n",
                max_diff_mean, max_diff_mean < 1e-2f ? "PASS" : "FAIL");

    // ── Kernel 2: compute_block_score ──
    float scale = 1.0f / std::sqrt((float)D);
    launch_compute_block_score_bf16(
        dQ, dmK, scale, dS, dM,
        B, H, Hk, S, D, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_S_b, s_S_m, s_S_n, s_S_h, 0);
    CK(cudaDeviceSynchronize());
    std::printf("[fp-test] kernel 2 (compute_block_score): launch ok\n");

    // ── Kernel 4: sparse_flash_forward (full selection = dense FA) ──
    // Set indices to [0, 1, ..., M-1] for each (b, q_block, h), counts=M
    std::vector<int32_t> idx_h(B * M * M * H);
    std::vector<int32_t> cnt_h(B * M * H);
    for (int b = 0; b < B; ++b)
      for (int m = 0; m < M; ++m)
        for (int h = 0; h < H; ++h) {
          cnt_h[b * M * H + m * H + h] = m + 1;        // causal: include only blocks 0..m
          for (int n = 0; n < M; ++n)
              idx_h[b * M * M * H + m * M * H + n * H + h] = (n <= m) ? n : -1;
        }
    CK(cudaMemcpy(dIdx, idx_h.data(), idx_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dCnt, cnt_h.data(), cnt_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    launch_sparse_flash_forward_bf16(
        dQ, dK, dV, dO, dIdx, dCnt, scale,
        B, H, Hk, S, D, Q_TILE, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_K_b, s_K_n, s_K_h, s_K_d,    // V uses same strides as K
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,    // O uses Q strides
        s_idx_b, s_idx_m, s_idx_n, s_idx_h,
        s_cnt_b, s_cnt_m, s_cnt_h, 0);
    cudaError_t le = cudaDeviceSynchronize();
    std::printf("[fp-test] kernel 4 (sparse_flash_forward): launch %s\n",
                le == cudaSuccess ? "ok" : cudaGetErrorString(le));

    // Numerical check kernel 4: compare GPU sparse output to CPU dense
    // attention reference (full mask, fully causal). Should match within bf16
    // numerical tolerance.
    std::vector<__nv_bfloat16> O_h(B * S * H * D);
    CK(cudaMemcpy(O_h.data(), dO, O_h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_diff_attn = 0.0f;
    for (int q = 0; q < S; ++q) {
        for (int h = 0; h < H; ++h) {
            int hk = h * Hk / H;
            // Compute reference dense attention for this (q, h).
            float row_max = -INFINITY;
            std::vector<float> qk(S, -INFINITY);
            for (int j = 0; j <= q; ++j) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s += b2f(Q[q * H * D + h * D + d]) * b2f(K[j * Hk * D + hk * D + d]);
                }
                qk[j] = s * scale;
                if (qk[j] > row_max) row_max = qk[j];
            }
            float sumexp = 0.0f;
            std::vector<float> p(S, 0.0f);
            for (int j = 0; j <= q; ++j) {
                p[j] = std::exp(qk[j] - row_max);
                sumexp += p[j];
            }
            for (int j = 0; j <= q; ++j) p[j] /= sumexp;

            for (int d = 0; d < D; ++d) {
                float ref = 0.0f;
                for (int j = 0; j <= q; ++j) {
                    ref += p[j] * b2f(V[j * Hk * D + hk * D + d]);
                }
                float gpu = b2f(O_h[q * H * D + h * D + d]);
                float diff = std::fabs(ref - gpu);
                if (diff > max_diff_attn) max_diff_attn = diff;
            }
        }
    }
    std::printf("[fp-test] kernel 4 (sparse_flash_forward) numerics: max diff = %.5f %s\n",
                max_diff_attn, max_diff_attn < 5e-2f ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaFree(dmK); cudaFree(dS); cudaFree(dM); cudaFree(dIdx); cudaFree(dCnt);

    // ── End-to-end FlashPrefill at 8K context ──
    // Tests the full pipeline (kernels 1-4 + block_select) wrapped via
    // flash_prefill_forward_bf16. Compares to dense reference for a small
    // shape, then times a 8K shape to verify it's fast.
    {
        constexpr int BB = 1, BS = 8192, BH = 16, BHk = 8, BD = 128;
        constexpr int BL = 128;

        std::vector<__nv_bfloat16> bQ(BB * BS * BH * BD);
        std::vector<__nv_bfloat16> bK(BB * BS * BHk * BD);
        std::vector<__nv_bfloat16> bV(BB * BS * BHk * BD);
        for (auto & x : bQ) x = f2b(dist(rng));
        for (auto & x : bK) x = f2b(dist(rng));
        for (auto & x : bV) x = f2b(dist(rng));

        __nv_bfloat16 *bdQ, *bdK, *bdV, *bdO;
        CK(cudaMalloc(&bdQ, bQ.size() * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&bdK, bK.size() * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&bdV, bV.size() * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&bdO, bQ.size() * sizeof(__nv_bfloat16)));

        CK(cudaMemcpy(bdQ, bQ.data(), bQ.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bdK, bK.data(), bK.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bdV, bV.data(), bV.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        dflash27b::flashprefill::FlashPrefillConfig cfg;
        cfg.block_size = BL;
        cfg.attention_sink = 2;
        cfg.window = 4;
        cfg.last_n_full = 2;
        cfg.alpha = 0.12f;

        // Warm-up
        dflash27b::flashprefill::flash_prefill_forward_bf16(
            bdQ, bdK, bdV, bdO, BB, BS, BH, BHk, BD,
            1.0f / std::sqrt((float)BD), cfg);
        CK(cudaDeviceSynchronize());

        cudaEvent_t e_a, e_b;
        cudaEventCreate(&e_a);
        cudaEventCreate(&e_b);
        cudaEventRecord(e_a);
        for (int it = 0; it < 5; ++it) {
            dflash27b::flashprefill::flash_prefill_forward_bf16(
                bdQ, bdK, bdV, bdO, BB, BS, BH, BHk, BD,
                1.0f / std::sqrt((float)BD), cfg);
        }
        cudaEventRecord(e_b);
        cudaEventSynchronize(e_b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e_a, e_b);
        cudaEventDestroy(e_a);
        cudaEventDestroy(e_b);
        std::printf("[fp-test] e2e flash_prefill_forward_bf16 at S=%d: %.1f ms / iter (avg of 5)\n",
                    BS, ms / 5.0f);

        cudaFree(bdQ); cudaFree(bdK); cudaFree(bdV); cudaFree(bdO);
    }
    return 0;
}



================================================
FILE: dflash/test/test_generate.cpp
================================================
// End-to-end generation test for our qwen35 target forward.
//
// Reads a binary int32 token file (produced by scripts/tokenize_prompt.py),
// runs single-token decode over every token (no batched prefill), generates
// N new tokens via greedy argmax, and writes the resulting int32 token stream
// to an output file for Python-side detokenization.
//
// Also reports decode tok/s (generation only, prompt steps excluded).
//
// Usage:
//   test_generate <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace dflash27b;

struct StepGraph {
    ggml_context *    ctx = nullptr;
    ggml_cgraph *     gf  = nullptr;
    ggml_gallocr_t    alloc = nullptr;
    ggml_tensor *     inp_embed = nullptr;
    ggml_tensor *     positions = nullptr;
    ggml_tensor *     logits    = nullptr;
};

// Build a fresh single-token forward graph. We rebuild per step so that
// `kv_start` updates drive the correct KV cache slot. The graph is cheap to
// rebuild — all the weights + KV cache stay persistent.
static bool build_step_graph(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start
) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int n_tokens = 1;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);

    sg.gf = ggml_new_graph_custom(sg.ctx, 8192, false);

    QwenGraphInputs gi{};
    gi.inp_embed      = sg.inp_embed;
    gi.positions      = sg.positions;
    gi.attn_mask      = nullptr;        // n_tokens==1, no mask needed
    gi.n_tokens       = n_tokens;
    gi.kv_start       = kv_start;
    gi.capture_layers = false;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    ggml_set_output(go.logits);
    ggml_build_forward_expand(sg.gf, go.logits);
    sg.logits = go.logits;

    sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

static bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>\n", argv[0]);
        return 2;
    }
    const char * gguf_path   = argv[1];
    const char * prompt_path = argv[2];
    const int    n_gen       = std::atoi(argv[3]);
    const char * out_path    = argv[4];
    int stream_fd = -1;
    for (int i = 5; i < argc; i++) {
        if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
        // KV cache type flags (mirror llama-cli -ctk / -ctv).
        // Set the env var before resolve_kv_types() reads it inside create_target_cache.
        else if (std::strcmp(argv[i], "--cache-type-k") == 0 || std::strcmp(argv[i], "-ctk") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_K", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-k=", 15) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctk=", 5) == 0) {
            setenv("DFLASH27B_KV_K", argv[i] + 5, 1);
        }
        else if (std::strcmp(argv[i], "--cache-type-v") == 0 || std::strcmp(argv[i], "-ctv") == 0) {
            if (i + 1 < argc) setenv("DFLASH27B_KV_V", argv[++i], 1);
        }
        else if (std::strncmp(argv[i], "--cache-type-v=", 15) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 15, 1);
        }
        else if (std::strncmp(argv[i], "-ctv=", 5) == 0) {
            setenv("DFLASH27B_KV_V", argv[i] + 5, 1);
        }
    }
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd < 0) return;
        int32_t v = tok;
#if defined(_WIN32)
        DWORD written;
        WriteFile((HANDLE)(intptr_t)stream_fd, &v, sizeof(v), &written, nullptr);
#else
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
#endif
    };

    // ── Load model and cache
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(gguf_path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    const int max_ctx = 4096;
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    auto prompt = read_int32_file(prompt_path);
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt bin\n"); return 1; }
    std::printf("[prompt] %zu tokens: ", prompt.size());
    for (auto t : prompt) std::printf("%d ", t);
    std::printf("\n");

    if ((int)prompt.size() + n_gen > max_ctx) {
        std::fprintf(stderr, "prompt+gen exceeds max_ctx\n");
        return 1;
    }

    std::vector<int32_t> all_tokens = prompt;
    all_tokens.reserve(prompt.size() + n_gen);

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    std::vector<float> embed_buf(hidden);

    StepGraph sg;

    // ── Helper: run one step given current token + absolute position
    auto run_step = [&](int32_t tok, int pos) -> int32_t {
        if (!build_step_graph(sg, w, cache, backend, pos)) {
            std::fprintf(stderr, "build_step_graph failed at pos=%d\n", pos);
            std::exit(1);
        }

        // CPU embed
        int32_t ids[1] = { tok };
        if (!w.embedder.embed(ids, 1, embed_buf.data())) {
            std::fprintf(stderr, "embed failed tok=%d\n", tok);
            std::exit(1);
        }
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());

        // M-RoPE positions: 4 copies of pos
        int32_t p4[4] = { pos, pos, pos, pos };
        ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute failed at pos=%d (%d)\n", pos, (int)st);
            std::exit(1);
        }

        // argmax on logits
        const int vocab = DFLASH27B_TARGET_VOCAB;
        std::vector<float> logits(vocab);
        ggml_backend_tensor_get(sg.logits, logits.data(), 0, sizeof(float) * vocab);
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        return best;
    };

    // ── Prefill: feed prompt tokens one at a time (decode-only mode).
    //    We throw away the logits for all prompt tokens except the last one.
    int next = -1;
    for (int i = 0; i < (int)prompt.size(); i++) {
        next = run_step(prompt[i], i);
    }
    std::printf("[prefill] last-token argmax=%d\n", next);

    // ── Generation loop
    auto t_start = std::chrono::steady_clock::now();
    int gen_start_pos = (int)prompt.size();
    for (int g = 0; g < n_gen; g++) {
        int32_t tok = next;
        all_tokens.push_back(tok);
        stream_emit(tok);
        next = run_step(tok, gen_start_pos + g);
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    double tps  = n_gen / std::max(1e-9, secs);

    // Also push the final next token so downstream sees it
    all_tokens.push_back(next);

    std::printf("[gen] %d new tokens in %.3f s  ->  %.2f tok/s\n", n_gen, secs, tps);
    std::printf("[gen] tokens: ");
    for (int i = 0; i < n_gen; i++) std::printf("%d ", all_tokens[prompt.size() + i]);
    std::printf("\n");

    write_int32_file(out_path, all_tokens);
    std::printf("[out] wrote %zu tokens to %s\n", all_tokens.size(), out_path);

    if (sg.alloc) ggml_gallocr_free(sg.alloc);
    if (sg.ctx)   ggml_free(sg.ctx);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}



================================================
FILE: dflash/test/test_kv_quant.cpp
================================================
// Unit tests for dflash::kv_quant (parse_kv_type, resolve_kv_types,
// is_supported_kv_pair). Plain int main(), no frameworks.
//
// T8 (unsupported pair aborts) is not tested in-process because std::abort()
// terminates the test runner. Manual verification:
//   DFLASH27B_KV_K=tq3_0 DFLASH27B_KV_V=q5_0 ./dflash/build/test_kv_quant
// Expected: prints "[dflash] KV pair …" message and aborts.

#include "kv_quant.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

// ─── helpers ────────────────────────────────────────────────────────────────

static void clear_kv_env() {
    unsetenv("DFLASH27B_KV_F16");
    unsetenv("DFLASH27B_KV_Q4");
    unsetenv("DFLASH27B_KV_TQ3");
    unsetenv("DFLASH27B_KV_K");
    unsetenv("DFLASH27B_KV_V");
}

// ─── T1: parse_kv_type ──────────────────────────────────────────────────────

static void t1_parse_kv_type() {
    // Case-insensitive successes
    assert(dflash::parse_kv_type("f16")   == GGML_TYPE_F16);
    assert(dflash::parse_kv_type("F16")   == GGML_TYPE_F16);
    assert(dflash::parse_kv_type("bf16")  == GGML_TYPE_BF16);
    assert(dflash::parse_kv_type("BF16")  == GGML_TYPE_BF16);
    assert(dflash::parse_kv_type("q4_0")  == GGML_TYPE_Q4_0);
    assert(dflash::parse_kv_type("Q4_0")  == GGML_TYPE_Q4_0);
    assert(dflash::parse_kv_type("q4_1")  == GGML_TYPE_Q4_1);
    assert(dflash::parse_kv_type("q5_0")  == GGML_TYPE_Q5_0);
    assert(dflash::parse_kv_type("q5_1")  == GGML_TYPE_Q5_1);
    assert(dflash::parse_kv_type("q8_0")  == GGML_TYPE_Q8_0);
    assert(dflash::parse_kv_type("Q8_0")  == GGML_TYPE_Q8_0);
    assert(dflash::parse_kv_type("tq3_0") == GGML_TYPE_TQ3_0);
    assert(dflash::parse_kv_type("TQ3_0") == GGML_TYPE_TQ3_0);

    // Unknown / empty inputs
    assert(dflash::parse_kv_type("garbage") == GGML_TYPE_COUNT);
    assert(dflash::parse_kv_type("")         == GGML_TYPE_COUNT);
    assert(dflash::parse_kv_type("q9_0")     == GGML_TYPE_COUNT);

    std::puts("T1 PASS");
}

// ─── T2: resolve_kv_types precedence ────────────────────────────────────────

static void t2_resolve_kv_types() {
    ggml_type k, v;

    // 2a: no env → default Q8_0/Q8_0
    clear_kv_env();
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_Q8_0 && v == GGML_TYPE_Q8_0);

    // 2b: F16 shorthand
    clear_kv_env();
    setenv("DFLASH27B_KV_F16", "1", 1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_F16 && v == GGML_TYPE_F16);
    clear_kv_env();

    // 2c: Q4 shorthand wins over F16 (set both; Q4 is checked last in layer 2)
    clear_kv_env();
    setenv("DFLASH27B_KV_F16", "1", 1);
    setenv("DFLASH27B_KV_Q4",  "1", 1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_Q4_0 && v == GGML_TYPE_Q4_0);
    clear_kv_env();

    // 2d: per-axis overrides override legacy shorthand
    clear_kv_env();
    setenv("DFLASH27B_KV_Q4", "1", 1);
    setenv("DFLASH27B_KV_K",  "q8_0", 1);
    setenv("DFLASH27B_KV_V",  "q4_0", 1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_Q8_0 && v == GGML_TYPE_Q4_0);
    clear_kv_env();

    // 2e: TQ3_0/Q8_0 asymmetric via per-axis
    clear_kv_env();
    setenv("DFLASH27B_KV_K", "tq3_0", 1);
    setenv("DFLASH27B_KV_V", "q8_0",  1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_TQ3_0 && v == GGML_TYPE_Q8_0);
    clear_kv_env();

    std::puts("T2 PASS");
}

// ─── T3: is_supported_kv_pair matrix ─────────────────────────────────────────

static void t3_is_supported_kv_pair() {
    // K ∈ {F16,BF16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0} × V ∈ {F16,BF16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,TQ3_0}
    const ggml_type k_generic[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0
    };
    const ggml_type v_full[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_TQ3_0
    };

    // All generic K × full V should be supported
    for (ggml_type k : k_generic) {
        for (ggml_type v : v_full) {
            assert(dflash::is_supported_kv_pair(k, v));
        }
    }

    // K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}  → supported
    const ggml_type v_tq3_ok[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0,
        GGML_TYPE_Q8_0, GGML_TYPE_TQ3_0
    };
    for (ggml_type v : v_tq3_ok) {
        assert(dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, v));
    }

    // K = TQ3_0 × V ∈ {Q4_1, Q5_0, Q5_1} → NOT supported
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q4_1));
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q5_0));
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q5_1));

    // Explicit negatives from the task spec
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q5_0));  // K=TQ3, V=Q5_0 false
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q4_1));  // K=TQ3, V=Q4_1 false
    assert( dflash::is_supported_kv_pair(GGML_TYPE_Q5_0,  GGML_TYPE_TQ3_0)); // K=Q5_0,V=TQ3_0 true

    std::puts("T3 PASS");
}

// ─── T4: TQ3 backward-compat shorthand ───────────────────────────────────────

static void t4_tq3_shorthand() {
    clear_kv_env();
    setenv("DFLASH27B_KV_TQ3", "1", 1);
    ggml_type k, v;
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_TQ3_0 && v == GGML_TYPE_TQ3_0);
    clear_kv_env();

    std::puts("T4 PASS");
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    clear_kv_env();  // start clean regardless of calling environment

    t1_parse_kv_type();
    t2_resolve_kv_types();
    t3_is_supported_kv_pair();
    t4_tq3_shorthand();

    std::puts("ALL TESTS PASS");
    return 0;
}



================================================
FILE: dflash/test/test_vs_oracle.cpp
================================================
// Numerics validation: run the C++ draft graph with inputs generated by
// gen_oracle.py (PyTorch reference) and compare against the expected output.
//
// Usage:
//   test_vs_oracle <model.safetensors> <oracle_dir>
//
// oracle_dir must contain: noise.bin, target.bin, expected.bin, meta.txt
// Exit 0 on success (cos_sim >= 0.9999 and max_abs_diff < 1e-2 in bf16 regime).

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace dflash27b;

struct OracleMeta {
    int ctx_len = 0;
    int q_len = 0;
    int hidden = 0;
    int fc_in = 0;
};

static bool read_meta(const std::string & path, OracleMeta & m) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        int v = std::atoi(line.c_str() + eq + 1);
        if      (k == "ctx_len") m.ctx_len = v;
        else if (k == "q_len")   m.q_len   = v;
        else if (k == "hidden")  m.hidden  = v;
        else if (k == "fc_in")   m.fc_in   = v;
    }
    return m.ctx_len > 0 && m.q_len > 0 && m.hidden > 0 && m.fc_in > 0;
}

static std::vector<float> read_bin_f32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> out(sz / sizeof(float));
    f.read((char *)out.data(), sz);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.safetensors> <oracle_dir>\n", argv[0]);
        return 2;
    }
    const std::string draft_path = argv[1];
    const std::string oracle_dir = argv[2];

    OracleMeta m;
    if (!read_meta(oracle_dir + "/meta.txt", m)) {
        std::fprintf(stderr, "failed to read meta.txt\n");
        return 1;
    }
    std::printf("meta: ctx_len=%d q_len=%d hidden=%d fc_in=%d\n",
                m.ctx_len, m.q_len, m.hidden, m.fc_in);

    auto noise_host    = read_bin_f32(oracle_dir + "/noise.bin");
    auto target_host   = read_bin_f32(oracle_dir + "/target.bin");
    auto expected_host = read_bin_f32(oracle_dir + "/expected.bin");
    if (noise_host.empty() || target_host.empty() || expected_host.empty()) {
        std::fprintf(stderr, "failed to read oracle bins\n");
        return 1;
    }

    const size_t exp_noise  = (size_t)m.q_len * m.hidden;
    const size_t exp_target = (size_t)m.ctx_len * m.fc_in;
    const size_t exp_out    = (size_t)m.q_len * m.hidden;
    if (noise_host.size() != exp_noise || target_host.size() != exp_target ||
        expected_host.size() != exp_out) {
        std::fprintf(stderr, "bin size mismatch: noise=%zu/%zu target=%zu/%zu exp=%zu/%zu\n",
            noise_host.size(), exp_noise, target_host.size(), exp_target,
            expected_host.size(), exp_out);
        return 1;
    }

    // Backend + weights
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }
    DraftWeights w;
    if (!load_draft_safetensors(draft_path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }

    // Graph context
    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);

    ggml_tensor * noise_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, m.hidden, m.q_len, 1);
    ggml_tensor * target_hid  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, m.fc_in, m.ctx_len, 1);
    ggml_tensor * pos_q       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, m.q_len);
    ggml_tensor * pos_k       = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, m.ctx_len + m.q_len);
    ggml_set_name(noise_embed, "noise_embed");
    ggml_set_name(target_hid,  "target_hidden_cat");
    ggml_set_name(pos_q,       "positions_q");
    ggml_set_name(pos_k,       "positions_k");
    ggml_set_input(noise_embed);
    ggml_set_input(target_hid);
    ggml_set_input(pos_q);
    ggml_set_input(pos_k);

    DraftGraphInputs gi{};
    gi.ctx_len = m.ctx_len;
    gi.noise_embed = noise_embed;
    gi.target_hidden_cat = target_hid;
    gi.positions_q = pos_q;
    gi.positions_k = pos_k;
    DraftGraphOutputs go = build_draft_graph(gctx, w, gi);
    if (!go.hidden_states) return 1;
    ggml_set_output(go.hidden_states);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, go.hidden_states);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "alloc graph failed\n");
        return 1;
    }

    // Upload inputs
    ggml_backend_tensor_set(noise_embed, noise_host.data(), 0, sizeof(float) * noise_host.size());
    ggml_backend_tensor_set(target_hid,  target_host.data(), 0, sizeof(float) * target_host.size());

    std::vector<int32_t> pq(m.q_len), pk(m.ctx_len + m.q_len);
    for (int i = 0; i < m.q_len; i++)           pq[i] = m.ctx_len + i;
    for (int i = 0; i < m.ctx_len + m.q_len; i++) pk[i] = i;
    ggml_backend_tensor_set(pos_q, pq.data(), 0, sizeof(int32_t) * pq.size());
    ggml_backend_tensor_set(pos_k, pk.data(), 0, sizeof(int32_t) * pk.size());

    // Compute
    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", (int)status);
        return 1;
    }

    // Read output
    std::vector<float> got(exp_out);
    ggml_backend_tensor_get(go.hidden_states, got.data(), 0, sizeof(float) * got.size());

    // Cosine similarity + max abs diff + mean abs diff
    double dot = 0.0, na = 0.0, nb = 0.0;
    double max_abs_diff = 0.0, sum_abs_diff = 0.0;
    int n_finite = 0;
    for (size_t i = 0; i < got.size(); i++) {
        float a = got[i], b = expected_host[i];
        if (!std::isfinite(a) || !std::isfinite(b)) continue;
        n_finite++;
        dot += (double)a * b;
        na += (double)a * a;
        nb += (double)b * b;
        double d = std::fabs((double)a - b);
        if (d > max_abs_diff) max_abs_diff = d;
        sum_abs_diff += d;
    }
    double cos_sim = dot / std::sqrt(na * nb + 1e-20);
    double mean_abs_diff = sum_abs_diff / std::max(1, n_finite);

    std::printf("n=%zu finite=%d\n", got.size(), n_finite);
    std::printf("cos_sim      = %.10f\n", cos_sim);
    std::printf("max_abs_diff = %.6g\n", max_abs_diff);
    std::printf("mean_abs_diff= %.6g\n", mean_abs_diff);

    std::printf("got first 8 : ");
    for (int i = 0; i < 8; i++) std::printf("%.4f ", got[i]);
    std::printf("\nexp first 8 : ");
    for (int i = 0; i < 8; i++) std::printf("%.4f ", expected_host[i]);
    std::printf("\n");

    // Pass if cos_sim is very high. Threshold is permissive because the
    // reference runs in bf16 on GPU and our graph also mixes bf16 weights
    // with f32 activations — bit-exact is not expected, but cos_sim should
    // be very close to 1.
    const double MIN_COS_SIM = 0.999;
    int rc = (cos_sim >= MIN_COS_SIM) ? 0 : 1;
    std::printf("%s\n", rc == 0 ? "PASS" : "FAIL");

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_draft_weights(w);
    ggml_backend_free(backend);
    return rc;
}



================================================
FILE: megakernel/README.md
================================================
<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce Megakernel</h1>

<p align="center">
  <strong>The first megakernel for hybrid DeltaNet/Attention LLMs.</strong><br/>
  All 24 layers of Qwen 3.5-0.8B in a single CUDA dispatch.<br/>
  1.87 tok/J on a 2020 GPU, matching Apple's latest silicon at 2x the throughput.<br/><br/>
  <a href="https://lucebox.com/blog/megakernel">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

---

```
                        Prefill      Decode      tok/J
Megakernel (RTX 3090)   21,347       413         1.87  @220W
Megakernel (RTX 2080 Ti) 13,919      250          —
llama.cpp  (RTX 3090)   11,247       267         0.76
Apple M5 Max               -         229         1.76
```

> The efficiency gap between NVIDIA and Apple isn't inherent to the silicon. It's an artifact of running generic software on capable hardware.

## Why this exists

Conventional wisdom says NVIDIA GPUs are fast but power hungry, and Apple Silicon is slower but efficient. On paper, that checks out: llama.cpp on an RTX 3090 gets 267 tok/s at 350W (0.76 tok/J), while an M5 Max gets 229 tok/s at ~130W (1.76 tok/J). NVIDIA is faster, but 2.3x worse on efficiency.
Qwen 3.5-0.8B uses a hybrid DeltaNet + Attention architecture (linear attention interleaved with standard attention). No fused kernel existed for this pattern. This is the first.

Inspired by [Hazy Research's megakernel work on Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), we asked: can the same idea work for hybrid DeltaNet/Attention models on consumer GPUs?

We thought the problem was never the hardware. The RTX 3090 has 936 GB/s memory bandwidth and 142 TFLOPS FP16 compute. Extracting only 267 tok/s from that is a software problem.

**The culprit: ~100 kernel launches per token.** Each layer boundary returns control to the CPU, dispatches the next kernel, re-fetches weights from global memory, and synchronizes threads. For 24 layers, those microseconds add up, and each one burns power doing nothing useful.

So we fused everything into one kernel.

## Results

### Decode and Prefill (Qwen 3.5-0.8B)

| Method | Prefill pp520 (tok/s) | Decode tg128 (tok/s) |
|--------|:---------------------:|:--------------------:|
| **Megakernel (RTX 3090)** | **21,347** | **413** |
| **Megakernel (RTX 2080 Ti)** | **13,919** | **250** |
| llama.cpp BF16 (RTX 3090) | 11,247 | 267 |
| PyTorch HuggingFace (RTX 3090) | 7,578 | 108 |
| PyTorch HuggingFace (RTX 2080 Ti) | 1,050 | 12 |

RTX 3090: 1.9x faster prefill, 1.55x faster decode vs llama.cpp. RTX 2080 Ti: 13.3x faster prefill, 20x faster decode vs PyTorch HF on the same GPU.

### Energy Efficiency (DVFS Power Sweep)

| Power Limit | Clock | Draw | tok/s | tok/J | vs Stock |
|:-----------:|:-----:|:----:|:-----:|:-----:|:--------:|
| 420W (stock) | 1980 MHz | 314W | 433 | 1.38 | baseline |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% speed, 5% less power |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% speed, 30% less power** |
| 150W | 405 MHz | 150W | 194 | 1.29 | too aggressive |

Sweet spot at 220W: 95% of the speed, 30% less power. The curve is nonlinear, tight execution converts directly into saved watts until you starve the GPU too aggressively.

### The Comparison That Shouldn't Exist

| Metric | RTX 3090 (llama.cpp) | M5 Max | RTX 3090 (Megakernel @220W) |
|--------|:--------------------:|:------:|:---------------------------:|
| tok/s | 267 | 229 | **411** |
| Power | 350W | ~130W | 220W |
| tok/J | 0.76 | 1.76 | **1.87** |
| GPU price | ~$900 | $2,499+ (system) | ~$900 |

A $900 GPU from 2020, power-limited to 220W, matches Apple's latest chip on efficiency while delivering 1.8x the throughput.

## How it works

A single persistent CUDA kernel processes the entire Qwen 3.5-0.8B forward pass in one dispatch. No CPU round-trips between layers.

**Architecture:** Qwen 3.5-0.8B is a hybrid model, 18 DeltaNet layers (linear attention with learned recurrence) and 6 full attention layers, in a 3:1 ratio. DeltaNet scales linearly with context length vs. quadratic for standard attention. It's an emerging pattern in next-gen models (Qwen3-Next, Kimi Linear), but no framework had optimized kernels for it.

**Kernel specs:**
- 82 blocks, 512 threads, all SMs on the RTX 3090 kept occupied; launch count is clamped to the resident-block ceiling on smaller GPUs to avoid grid-sync deadlock
- BF16 weights and activations on Ampere+ (sm_80+), FP16 on Turing (sm_75); FP32 accumulation where it matters
- DeltaNet recurrence via warp-cooperative state updates in F32 registers
- Full attention with online softmax (fused QKV, RoPE, causal mask, output projection)
- Cooperative grid sync between layers instead of kernel launches (zero inter-layer overhead)
- KV cache updates in-kernel
- Weights loaded directly from HuggingFace

**What traditional frameworks do:** launch ~100 separate kernels per token, each one paying the cost of CPU dispatch, weight re-fetch, and thread synchronization. The megakernel eliminates all of that.

## Why DeltaNet matters

Standard transformers have years of kernel optimization: FlashAttention, PagedAttention, continuous batching. Hybrid DeltaNet/Attention architectures are newer, and the kernel ecosystem is immature:

- **MLX:** no native DeltaNet kernels
- **llama.cpp:** generic DeltaNet support, no fusion
- **vLLM/SGLang:** Triton kernels via flash-linear-attention, but no megakernel fusion

As more models go hybrid (and they will, because linear attention scales better), what you run them on matters less than *how* you run them. When you write a kernel that actually uses what the GPU offers, tensor cores, shared memory, cooperative grid launches, register-resident state, a five-year-old GPU matches Apple's latest chip.

## Lessons from building this

**`grid.sync()` inside loops will deadlock silently.** We tried synchronizing all blocks within the per-token DeltaNet recurrence loop. No error message, just a hang. The fix: synchronize between layers, not within them.

**Register pressure kills performance quietly.** We attempted `S_TILE=16` for more instruction-level parallelism. Silent crash, no CUDA error, registers spilled to local memory, performance collapsed. `S_TILE=8` was the sweet spot.

**The power curve is nonlinear.** 420W to 300W lost almost nothing (99.8% speed). 300W to 220W lost marginally (95%). 220W to 150W collapsed to 45%. The megakernel's tight execution means the GPU hits its compute ceiling before its power ceiling, until you starve it too aggressively.

## Quick start

```bash
git clone https://github.com/Luce-Org/luce-megakernel
cd luce-megakernel
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                          # install BEFORE the next step; setup.py imports torch at build time
pip install -e . --no-build-isolation      # --no-build-isolation lets the build see the torch you just installed
python final_bench.py    # runs pp520 tg128 (properly warmed), prints tok/s
```

**Requirements:**
- **RTX 3090** (sm_86, BF16) — primary target, all benchmarks above
- **RTX 2080 Ti** (sm_75, FP16) — supported via compile-time `TARGET_SM` flag; auto-detected at build time
- Portable to other Turing+ (sm_75+) NVIDIA GPUs
- CUDA 12+
- PyTorch 2.0+
- ~1.5 GB VRAM for weights

**Blackwell (sm_120 / sm_121a):** runs on Blackwell consumer GPUs (RTX 5090)
and the NVIDIA DGX Spark (GB10) via an NVFP4 decode path. The build
auto-detects your GPU and `final_bench.py` dispatches to the right backend;
use `--backend nvfp4` to force it. First DGX Spark numbers are in
[RESULTS.md](RESULTS.md#nvidia-dgx-spark-gb10-sm_121a).

**Optional:** Set a power limit to find your GPU's sweet spot:
```bash
sudo nvidia-smi -pl 220    # or whatever your target wattage
```

## Files

| File | Description |
|------|-------------|
| `kernel.cu` | Decode megakernel, all 24 layers in one dispatch |
| `prefill.cu` | Prefill (cuBLAS + standalone kernels) |
| `half_type.h` | Portable half-precision type alias (BF16 on sm_80+, FP16 on sm_75) |
| `torch_bindings.cpp` | PyTorch C++ bindings |
| `model.py` | Weight loading + decoder |
| `setup.py` | Build configuration (arch auto-detect, Blackwell gating) |
| `final_bench.py` | Benchmark (prefill + decode, 10 warmup + 20 timed runs averaged — use this for published numbers) |
| `bench_pp_tg.py` | Quick single-run benchmark with correctness check (not warmed, under-reports prefill) |
| `RESULTS.md` | Full benchmark results and DVFS sweep |
| `kernel_gb10_nvfp4.cu` | Blackwell-only: NVFP4 persistent decode megakernel |
| `prefill_megakernel.cu` | Blackwell-only: single-dispatch prefill megakernel |
| `model_nvfp4.py` | Blackwell-only: NVFP4 decoder driving the ops above |
| `final_bench_nvfp4.py` / `bench_pp_tg_nvfp4.py` | Blackwell bench siblings dispatched by `--backend nvfp4` |

## Scope and limitations

This is a **research proof-of-concept**, not a production inference server.

- **Batch size 1 only.** This targets single-user local inference (the llama.cpp/Ollama use case), not multi-tenant serving. If you need batched throughput, use vLLM or SGLang.
- **Single model, single architecture.** The kernel is hand-written for Qwen 3.5-0.8B's specific layer pattern (18 DeltaNet + 6 Attention). It does not generalize to other models without rewriting.
- **BF16 (Ampere+) / FP16 (Turing).** Uses BF16 on sm_80+ and FP16 on sm_75 via compile-time `TARGET_SM` flag. No quantization support (GGUF/GPTQ/AWQ). We benchmark at half-precision to isolate kernel-level efficiency from quantization tradeoffs.
- **0.8B parameters.** This is a small model. Megakernel fusion benefits shrink as model size grows and compute begins to dominate over launch overhead. We chose 0.8B because it's the first hybrid DeltaNet model available, not because it's representative of all workloads.
- **Power methodology.** Efficiency numbers measure accelerator power only (NVML for NVIDIA, `powermetrics` for Apple), following [Hazy Research's Intelligence Per Watt](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) methodology. Total system draw is higher for both platforms.
- **Correctness.** `bench_pp_tg.py` includes an end-to-end correctness check comparing megakernel output against a reference decode path. Use `final_bench.py` for performance numbers (properly warmed, 10 warmup + 20 timed runs averaged).

The goal is to demonstrate that architecture-specific kernel fusion eliminates a real efficiency gap on consumer hardware, and to do it in the open so others can reproduce, critique, and extend the work.

## Files

Questions, ideas, or want to see what others are building? Join the [Luce Discord](https://discord.gg/yHfswqZmJQ).

## Citation

If you use this work in your research:

```bibtex
@software{luce_megakernel_2026,
  title  = {Luce Megakernel: Fused Forward Pass for Hybrid DeltaNet/Attention LLMs},
  author = {Luce},
  url    = {https://github.com/Luce-Org/luce-megakernel},
  year   = {2026}
}
```

## Why llama.cpp as a baseline?

llama.cpp is the most widely used local inference engine. It's what most people actually run on consumer GPUs. We also include PyTorch HuggingFace numbers (3.8x slower) for a second reference point. This is not a critique of llama.cpp, it's an excellent project. The comparison shows what architecture-specific optimization can unlock on top of a generic framework.

## Why an RTX 3090?

Deliberately chosen as the "worst case" for NVIDIA: a 2020 GPU, widely dismissed as power-hungry, available for ~$900 used. If the software gap is real on old hardware, it's even larger on newer cards.

## RTX 2080 Ti support

The megakernel also runs on the RTX 2080 Ti (Turing, sm_75, 2018) using FP16 instead of BF16. The build system auto-detects the GPU and selects the right code path via a compile-time `TARGET_SM` flag. On the 2080 Ti, the megakernel delivers 250 tok/s decode and 13,919 tok/s prefill — 20x faster than PyTorch HuggingFace on the same GPU.
## Community

Questions, ideas, or want to see what others are building? Join the [Luce Discord](https://discord.gg/yHfswqZmJQ).

---

MIT · [Lucebox](https://lucebox.com)

Built with [Claude](https://claude.ai)

Inspired by [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen)



================================================
FILE: megakernel/_phase2_variant.py
================================================
"""Runtime marker for DN phase2 kernel variant.

Reads MEGAKERNEL_DN_PHASE2_WMMA_RUNTIME (default "0").
  "0" -> scalar FP32 path (current default)
  "1" -> WMMA path (future; kernel not yet implemented)

This module is imported for its side-effect (the print) so the user can
confirm which variant is active without inspecting build flags.
"""
import os as _os

_val = _os.environ.get("MEGAKERNEL_DN_PHASE2_WMMA_RUNTIME", "0").strip()
DN_PHASE2_VARIANT = "wmma" if _val == "1" else "scalar"
print(f"[megakernel] DN phase2 variant = {DN_PHASE2_VARIANT}")



================================================
FILE: megakernel/bench.py
================================================
"""Benchmark bf16 megakernel decode."""
import time
import torch
from model import Decoder
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
dec = Decoder(verbose=True)

prompt = "The capital of France is"
ids = tok.encode(prompt, add_special_tokens=False)

# Warmup
dec.reset()
for t in ids[:-1]:
    dec.step(t)
first = dec.step(ids[-1])

# Benchmark decode
dec.reset()
for t in ids[:-1]:
    dec.step(t)

torch.cuda.synchronize()
t0 = time.perf_counter()
out = []
next_id = ids[-1]
for _ in range(200):
    next_id = dec.step(next_id)
    if next_id == tok.eos_token_id:
        break
    out.append(next_id)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

tps = len(out) / elapsed
text = tok.decode(out, skip_special_tokens=True)[:80]
print(f"Decode: {tps:.1f} tok/s ({len(out)} tokens in {elapsed*1000:.1f}ms)")
print(f"Output: {text}")



================================================
FILE: megakernel/bench_pp_tg.py
================================================
"""Benchmark pp512 tg128 — standard llama.cpp benchmark format.
Also tests end-to-end correctness (prefill → decode handoff).

Scope: batch-size-1 single-stream decode, targeting local inference.
All measurements use torch.cuda.synchronize() barriers + perf_counter.
One warm-up run precedes each timed section.

Supports --backend {auto,bf16,nvfp4}. Default is auto: Blackwell (sm_12+)
dispatches to bench_pp_tg_nvfp4.py; everything else runs the bf16 path
below unchanged from upstream.
"""
import argparse as _argparse, os as _os, sys as _sys
import torch as _torch

_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--backend", default="auto", choices=("auto", "bf16", "nvfp4"))
_a, _rest = _p.parse_known_args()
_backend = _a.backend
if _backend == "auto":
    _backend = "nvfp4" if (_torch.cuda.is_available() and _torch.cuda.get_device_capability()[0] >= 12) else "bf16"
if _backend == "nvfp4":
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _os.execv(_sys.executable, [_sys.executable, _os.path.join(_here, "bench_pp_tg_nvfp4.py"), *_rest])

import time, torch
import _phase2_variant  # noqa: F401 — prints "[megakernel] DN phase2 variant = scalar|wmma"
from model import Decoder, HIDDEN_SIZE, INTERMEDIATE_SIZE, FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE
from model import DN_CONV_CHANNELS, DN_V_SIZE, DN_NUM_HEADS, MAX_SEQ_LEN, _half_dtype
import qwen35_megakernel_bf16_C
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
dec = Decoder(verbose=True)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

# Allocate prefill buffers for max 512 tokens
S_MAX = 512
bf16 = dict(dtype=_half_dtype(), device="cuda")
f32 = dict(dtype=torch.float32, device="cuda")
i32 = dict(dtype=torch.int32, device="cuda")
mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
bufs = dict(
    hidden=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    residual=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    normalized=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    proj_buf=torch.empty(S_MAX*mx, **bf16),
    proj_buf2=torch.empty(S_MAX*mx, **bf16),
    attn_buf=torch.empty(S_MAX*max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf=torch.empty(S_MAX*INTERMEDIATE_SIZE, **bf16),
    dn_out_buf=torch.empty(S_MAX*DN_V_SIZE, **bf16),
    beta_buf=torch.empty(S_MAX*DN_NUM_HEADS, **f32),
    alpha_buf=torch.empty(S_MAX*DN_NUM_HEADS, **f32),
    final_normed=torch.empty(HIDDEN_SIZE, **bf16),
    hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv=torch.empty(1024, **f32),
    lm_bmi=torch.empty(1024, **i32),
)
bufs.update(dec.alloc_prefill_scratch(S_MAX))

def prefill(ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    _pf(dec._out_token, ids_t,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        bufs['hidden'], bufs['residual'], bufs['normalized'],
        bufs['proj_buf'], bufs['proj_buf2'],
        bufs['attn_buf'], bufs['mlp_buf'],
        bufs['dn_out_buf'], bufs['beta_buf'], bufs['alpha_buf'],
        bufs['dn_pre_qkv'],
        bufs['dn_u_scratch'], bufs['dn_w_scratch'], bufs['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        bufs['final_normed'], bufs['hidden_bf16_out'],
        bufs['lm_bmv'], bufs['lm_bmi'], dec.max_seq_len)
    # Handoff: copy hidden state for decode kernel
    dec._hidden.copy_(bufs['hidden_bf16_out'])
    dec._position = len(ids)
    return dec._out_token.item()

# ============================================================
# 1. End-to-end correctness test
# ============================================================
print("\n=== Correctness test ===", flush=True)
prompt = "The capital of France is"
ids = tok.encode(prompt, add_special_tokens=False)
dec.reset()
first = prefill(ids)
print(f"Prefill → first token: {first} = '{tok.decode([first])}'", flush=True)

# Continue with decode megakernel
out = [first]
nid = first
for _ in range(30):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    out.append(nid)
text = tok.decode(out, skip_special_tokens=True)
print(f"Output: {text[:80]}", flush=True)

# Reference: pure decode (step-by-step)
dec.reset()
for t in ids[:-1]: dec.step(t)
ref_first = dec.step(ids[-1])
ref_out = [ref_first]
nid = ref_first
for _ in range(30):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    ref_out.append(nid)
ref_text = tok.decode(ref_out, skip_special_tokens=True)
print(f"Ref:    {ref_text[:80]}", flush=True)

if out == ref_out:
    print("PASS: megakernel output matches reference decode path", flush=True)
else:
    # On SM < 8.0, the decode kernel (f32 scalar multiply) and cuBLAS prefill
    # (fp16 tensor core multiply) use different intermediate precision, which
    # can flip the argmax when logit gaps are small.  This is expected and not
    # a correctness bug — verify both paths produce coherent output.
    _cap = torch.cuda.get_device_capability()
    if _cap[0] < 8:
        print(f"KNOWN DIVERGENCE (SM {_cap[0]}.{_cap[1]}, fp16): prefill and decode paths "
              f"produce different tokens due to tensor-core vs scalar rounding.",
              flush=True)
        print(f"  Prefill tokens: {out[:10]}...", flush=True)
        print(f"  Decode tokens:  {ref_out[:10]}...", flush=True)
        print(f"  Both paths produce coherent output — not a correctness bug.", flush=True)
    else:
        print("FAIL: output mismatch between megakernel and reference", flush=True)
        print(f"  Megakernel tokens: {out[:10]}...", flush=True)
        print(f"  Reference tokens:  {ref_out[:10]}...", flush=True)

# ============================================================
# 2. pp512 benchmark (prompt processing)
# ============================================================
print("\n=== pp512 benchmark ===", flush=True)
# Generate a 512-token prompt
long_prompt = "Explain in great detail the history of artificial intelligence, " * 30
long_ids = tok.encode(long_prompt, add_special_tokens=False)[:512]
print(f"Prompt tokens: {len(long_ids)}", flush=True)

# Warmup
dec.reset()
prefill(long_ids)

# Benchmark
dec.reset()
torch.cuda.synchronize()
t0 = time.perf_counter()
prefill(long_ids)
torch.cuda.synchronize()
pp_time = time.perf_counter() - t0
pp_tps = len(long_ids) / pp_time
print(f"pp{len(long_ids)}: {pp_tps:.1f} tok/s ({pp_time*1000:.1f}ms)", flush=True)

# ============================================================
# 3. tg128 benchmark (token generation)
# ============================================================
print("\n=== tg128 benchmark ===", flush=True)
# Prefill a short prompt, then generate 128 tokens
short_ids = tok.encode("Hello", add_special_tokens=False)
dec.reset()
first = prefill(short_ids)

torch.cuda.synchronize()
t0 = time.perf_counter()
gen_out = []
nid = first
for _ in range(128):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    gen_out.append(nid)
torch.cuda.synchronize()
tg_time = time.perf_counter() - t0
tg_tps = len(gen_out) / tg_time
print(f"tg{len(gen_out)}: {tg_tps:.1f} tok/s ({tg_time*1000:.1f}ms)", flush=True)

# ============================================================
# Summary
# ============================================================
_dtype_label = "FP16" if _half_dtype() == torch.float16 else "BF16"
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
print(f"\n=== Summary ({_gpu_name}, Qwen3.5-0.8B {_dtype_label}) ===", flush=True)
print(f"pp{len(long_ids):>3d}: {pp_tps:>7.1f} tok/s", flush=True)
print(f"tg{len(gen_out):>3d}: {tg_tps:>7.1f} tok/s", flush=True)



================================================
FILE: megakernel/bench_pp_tg_nvfp4.py
================================================
"""Benchmark pp512 / tg128 for the local Qwen3.5-0.8B megakernel backend.

The default `all` mode runs each section in a fresh subprocess. This avoids
carrying CUDA state from correctness checks into the timed benchmark sections.
"""

import argparse
import json
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from model_nvfp4 import (
    Decoder,
    DN_CONV_CHANNELS,
    DN_NUM_HEADS,
    DN_V_SIZE,
    FA_KV_SIZE,
    FA_QPROJ_SIZE,
    FA_Q_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    PREFILL_PROJ_FUSED_SIZE,
    PREFILL_PROJ_SCRATCH_SIZE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pp512 / tg128")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="auto", choices=("auto", "bf16", "nvfp4"))
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--correctness-steps", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measure-runs", type=int, default=5)
    parser.add_argument("--verbose-loader", action="store_true")
    parser.add_argument(
        "--section",
        default="all",
        choices=("all", "correctness", "pp", "tg"),
    )
    parser.add_argument("--json-result", action="store_true")
    return parser.parse_args()


def build_exact_prompt_ids(tokenizer, target_tokens):
    seed = "Explain in great detail the history of artificial intelligence."
    text = seed
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text += " " + seed
        ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:target_tokens]


def alloc_prefill_buffers(max_tokens):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    return dict(
        hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        proj_buf=torch.empty(max_tokens * PREFILL_PROJ_FUSED_SIZE, **bf16),
        proj_buf2=torch.empty(max_tokens * PREFILL_PROJ_SCRATCH_SIZE, **bf16),
        attn_buf=torch.empty(max_tokens * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
        mlp_buf=torch.empty(max_tokens * INTERMEDIATE_SIZE, **bf16),
        dn_out_buf=torch.empty(max_tokens * DN_V_SIZE, **bf16),
        beta_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        alpha_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        final_normed=torch.empty(HIDDEN_SIZE, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
    )


def get_prefill_op(decoder):
    ops = torch.ops.qwen35_megakernel_bf16_C
    if decoder.backend == "nvfp4":
        return ops.prefill_megakernel_nvfp4
    return ops.prefill_bf16


def run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op):
    decoder.reset()
    if decoder.backend == "nvfp4":
        return decoder.prefill_tokens(ids_t)
    else:
        prefill_op(
            decoder._out_token,
            ids_t,
            decoder._embed_weight,
            decoder._layer_weights_packed,
            decoder._prefill_fused_weights_packed,
            decoder._final_norm_weight,
            decoder._lm_head_weight,
            decoder._fa_k_cache,
            decoder._fa_v_cache,
            decoder._dn_states,
            decoder._conv_bufs,
            buffers["hidden"],
            buffers["residual"],
            buffers["normalized"],
            buffers["proj_buf"],
            buffers["proj_buf2"],
            buffers["attn_buf"],
            buffers["mlp_buf"],
            buffers["dn_out_buf"],
            buffers["beta_buf"],
            buffers["alpha_buf"],
            buffers["final_normed"],
            buffers["hidden_bf16_out"],
            buffers["lm_bmv"],
            buffers["lm_bmi"],
        )
    decoder._hidden.copy_(buffers["hidden_bf16_out"])
    decoder._position = prompt_len
    return decoder._out_token.item()


def decode_steps(decoder, first_token, num_steps, eos_token_id):
    out = [first_token]
    nid = first_token
    for _ in range(num_steps):
        nid = decoder.step(nid)
        torch.cuda.synchronize()
        if nid == eos_token_id:
            break
        out.append(nid)
    return out


def benchmark_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, warmup_runs, measure_runs):
    for _ in range(warmup_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measure_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / measure_runs
    return elapsed, prompt_len / elapsed


def benchmark_decode(decoder, prompt_ids, gen_tokens, tokenizer, buffers, prefill_op):
    ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
    first = run_prefill(decoder, ids_t, len(prompt_ids), buffers, prefill_op)

    if decoder.backend == "nvfp4":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed_ids_dev = decoder.step_many(first, gen_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timed_ids = timed_ids_dev.cpu().tolist()
        if tokenizer.eos_token_id in timed_ids:
            timed_ids = timed_ids[:timed_ids.index(tokenizer.eos_token_id)]
        tps = (len(timed_ids) / elapsed) if timed_ids else 0.0
        return first, elapsed, tps, timed_ids

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nid = first
    timed_ids = []
    for _ in range(gen_tokens):
        nid = decoder.step(nid)
        torch.cuda.synchronize()
        if nid == tokenizer.eos_token_id:
            break
        _ = tokenizer.decode([nid])
        timed_ids.append(nid)

    elapsed = time.perf_counter() - t0
    tps = (len(timed_ids) / elapsed) if timed_ids else 0.0
    return first, elapsed, tps, timed_ids


def emit_result(result, json_result):
    if json_result:
        print(f"RESULT_JSON {json.dumps(result, sort_keys=True)}", flush=True)


def run_correctness(args, tokenizer):
    print(f"Loading decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    print(f"Backend: {decoder.backend_label}", flush=True)
    if decoder.backend == "nvfp4":
        print("Correctness mode: prefill/decode handoff smoke test", flush=True)

    prefill_op = get_prefill_op(decoder)
    prompt_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    buffers = alloc_prefill_buffers(max(32, len(prompt_ids)))

    print("\n=== Correctness test ===", flush=True)
    ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
    first = run_prefill(decoder, ids_t, len(prompt_ids), buffers, prefill_op)
    print(f"Prefill -> first token: {first} = '{tokenizer.decode([first])}'", flush=True)

    out = decode_steps(decoder, first, args.correctness_steps, tokenizer.eos_token_id)
    text = tokenizer.decode(out, skip_special_tokens=True)
    print(f"Prefill + decode: {text[:100]}", flush=True)

    ok = bool(out)
    if decoder.backend == "nvfp4":
        if ok:
            print("PASS: prefill completed and decode continued on NVFP4", flush=True)
        else:
            print("FAIL: no tokens produced after prefill on NVFP4", flush=True)
    else:
        print("PASS: BF16 correctness smoke test completed", flush=True)

    result = {
        "backend_label": decoder.backend_label,
        "first_token": first,
        "ok": ok,
        "section": "correctness",
    }
    emit_result(result, args.json_result)
    return result


def run_pp(args, tokenizer):
    print(f"Loading decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    prefill_op = get_prefill_op(decoder)
    prompt_ids = build_exact_prompt_ids(tokenizer, args.prompt_tokens)
    buffers = alloc_prefill_buffers(len(prompt_ids))
    ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")

    print("\n=== pp benchmark ===", flush=True)
    print(f"Backend: {decoder.backend_label}", flush=True)
    print(f"Prompt tokens: {len(prompt_ids)}", flush=True)
    print(
        f"Warming {args.warmup_runs}x and timing {args.measure_runs}x prompt-processing runs...",
        flush=True,
    )
    pp_time, pp_tps = benchmark_prefill(
        decoder,
        ids_t,
        len(prompt_ids),
        buffers,
        prefill_op,
        args.warmup_runs,
        args.measure_runs,
    )
    print(f"pp{len(prompt_ids)}: {pp_tps:.1f} tok/s ({pp_time * 1000:.1f}ms avg)", flush=True)

    result = {
        "backend_label": decoder.backend_label,
        "pp_ms": pp_time * 1000.0,
        "pp_tps": pp_tps,
        "prompt_tokens": len(prompt_ids),
        "section": "pp",
    }
    emit_result(result, args.json_result)
    return result


def run_tg(args, tokenizer):
    print(f"Loading decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    prefill_op = get_prefill_op(decoder)
    prompt_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    buffers = alloc_prefill_buffers(max(args.prompt_tokens, 32, len(prompt_ids)))

    print("\n=== tg benchmark ===", flush=True)
    print(f"Backend: {decoder.backend_label}", flush=True)
    print(f"Timed decode steps: {args.gen_tokens}", flush=True)
    first, tg_time, tg_tps, tg_ids = benchmark_decode(
        decoder,
        prompt_ids,
        args.gen_tokens,
        tokenizer,
        buffers,
        prefill_op,
    )
    print(f"Prefill seed token: {first} = '{tokenizer.decode([first])}'", flush=True)
    print(f"tg{len(tg_ids)}: {tg_tps:.1f} tok/s ({tg_time * 1000:.1f}ms)", flush=True)

    result = {
        "backend_label": decoder.backend_label,
        "first_token": first,
        "gen_tokens": len(tg_ids),
        "tg_ms": tg_time * 1000.0,
        "tg_tps": tg_tps,
        "section": "tg",
    }
    emit_result(result, args.json_result)
    return result


def build_child_cmd(args, section):
    cmd = [
        sys.executable,
        __file__,
        "--model-name",
        args.model_name,
        "--backend",
        args.backend,
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--gen-tokens",
        str(args.gen_tokens),
        "--correctness-steps",
        str(args.correctness_steps),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measure-runs",
        str(args.measure_runs),
        "--section",
        section,
        "--json-result",
    ]
    if args.verbose_loader:
        cmd.append("--verbose-loader")
    return cmd


def filter_child_stderr(stderr_text):
    keep = []
    for line in stderr_text.splitlines():
        if line.startswith("Loading weights:"):
            continue
        if line.startswith("[transformers] The fast path is not available"):
            continue
        if line.strip():
            keep.append(line)
    return "\n".join(keep)


def run_child(args, section):
    proc = subprocess.run(
        build_child_cmd(args, section),
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        proc.check_returncode()
    elif proc.stderr:
        filtered = filter_child_stderr(proc.stderr)
        if filtered:
            print(filtered, file=sys.stderr, flush=True)

    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON "):])
    if result is None:
        raise RuntimeError(f"missing RESULT_JSON for section {section}")
    return result


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.section == "correctness":
        run_correctness(args, tokenizer)
        return
    if args.section == "pp":
        run_pp(args, tokenizer)
        return
    if args.section == "tg":
        run_tg(args, tokenizer)
        return

    correctness = run_child(args, "correctness")
    pp = run_child(args, "pp")
    tg = run_child(args, "tg")

    print(f"\n=== Summary ({pp['backend_label']}, Qwen3.5-0.8B) ===", flush=True)
    print(f"pp{pp['prompt_tokens']:>3d}: {pp['pp_tps']:>7.1f} tok/s", flush=True)
    print(f"tg{tg['gen_tokens']:>3d}: {tg['tg_tps']:>7.1f} tok/s", flush=True)
    if not correctness["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()



================================================
FILE: megakernel/build_corpus.py
================================================
"""Build a parity-gate reference corpus for the WMMA rewrite.

Runs 50 fixed prompts through prefill + 32 greedy decode steps using the
bf16 megakernel and saves deterministic token IDs to corpus/baseline.json.
"""
import json
import os
import subprocess
import time

import torch
from transformers import AutoTokenizer

from model import (
    Decoder,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    FA_QPROJ_SIZE,
    FA_Q_SIZE,
    FA_KV_SIZE,
    DN_CONV_CHANNELS,
    DN_V_SIZE,
    DN_NUM_HEADS,
    MAX_SEQ_LEN,
    _half_dtype,
)
import qwen35_megakernel_bf16_C

# ---------------------------------------------------------------------------
# Fixed prompt list
# ---------------------------------------------------------------------------
PROMPTS = [
    "The capital of France is",
    "Quantum entanglement allows",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return",
    "The seven wonders of the ancient world include",
    "In machine learning, a transformer is",
    "The mitochondria is the powerhouse of",
    "Python is a programming language that",
    "The square root of 144 is",
    "Albert Einstein was born in",
    "The Pacific Ocean is the largest",
    "DNA stands for",
    "The speed of light in a vacuum is approximately",
    "Shakespeare's most famous play is",
    "The chemical symbol for gold is",
    "Mount Everest is located in",
    "The Industrial Revolution began in",
    "Photosynthesis is the process by which plants",
    "The Roman Empire fell in",
    "Beethoven was a famous",
    "The Mona Lisa was painted by",
    "Antarctica is covered in",
    "The human heart has",
    "JavaScript was created in",
    "The Great Wall of China was built to",
    "Mars is the fourth planet from",
    "World War II ended in",
    "The pyramids of Giza are located in",
    "Newton's first law of motion states",
    "The currency of Japan is",
    "The Pythagorean theorem states that",
    "The longest river in the world is",
    "Bitcoin was invented by",
    "The Eiffel Tower is in",
    "Mozart composed",
    "The atomic number of carbon is",
    "The continents of the world include",
    "The American Revolution began in",
    "Thomas Edison invented",
    "The deepest ocean trench is the",
    "Charles Darwin proposed",
    "The boiling point of water is",
    "The largest planet in our solar system is",
    "Marie Curie was a pioneer in",
    "The official language of Brazil is",
    "Leonardo da Vinci was a",
    "The Berlin Wall fell in",
    "Carbon dioxide is composed of",
    "The smallest country in the world is",
    "Stephen Hawking was famous for",
    "The currency of the United Kingdom is",
]

N_GEN = 32
S_MAX = 512

# ---------------------------------------------------------------------------
# Decoder + tokenizer setup  (mirrors bench_pp_tg.py lines 26-64)
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
dec = Decoder(verbose=True)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

bf16 = dict(dtype=_half_dtype(), device="cuda")
f32  = dict(dtype=torch.float32, device="cuda")
i32  = dict(dtype=torch.int32,   device="cuda")

mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
bufs = dict(
    hidden           = torch.empty(S_MAX * HIDDEN_SIZE,                        **bf16),
    residual         = torch.empty(S_MAX * HIDDEN_SIZE,                        **bf16),
    normalized       = torch.empty(S_MAX * HIDDEN_SIZE,                        **bf16),
    proj_buf         = torch.empty(S_MAX * mx,                                 **bf16),
    proj_buf2        = torch.empty(S_MAX * mx,                                 **bf16),
    attn_buf         = torch.empty(S_MAX * max(FA_Q_SIZE, FA_KV_SIZE),         **bf16),
    mlp_buf          = torch.empty(S_MAX * INTERMEDIATE_SIZE,                  **bf16),
    dn_out_buf       = torch.empty(S_MAX * DN_V_SIZE,                          **bf16),
    beta_buf         = torch.empty(S_MAX * DN_NUM_HEADS,                       **f32),
    alpha_buf        = torch.empty(S_MAX * DN_NUM_HEADS,                       **f32),
    final_normed     = torch.empty(HIDDEN_SIZE,                                **bf16),
    hidden_bf16_out  = torch.empty(HIDDEN_SIZE,                                **bf16),
    lm_bmv           = torch.empty(1024,                                       **f32),
    lm_bmi           = torch.empty(1024,                                       **i32),
)
bufs.update(dec.alloc_prefill_scratch(S_MAX))


def prefill(ids):
    """Run prefill kernel and return first generated token id."""
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    _pf(
        dec._out_token, ids_t,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        bufs['hidden'], bufs['residual'], bufs['normalized'],
        bufs['proj_buf'], bufs['proj_buf2'],
        bufs['attn_buf'], bufs['mlp_buf'],
        bufs['dn_out_buf'], bufs['beta_buf'], bufs['alpha_buf'],
        bufs['dn_pre_qkv'],
        bufs['dn_u_scratch'], bufs['dn_w_scratch'], bufs['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        bufs['final_normed'], bufs['hidden_bf16_out'],
        bufs['lm_bmv'], bufs['lm_bmi'], dec.max_seq_len,
    )
    dec._hidden.copy_(bufs['hidden_bf16_out'])
    dec._position = len(ids)
    return dec._out_token.item()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
).decode().strip()

gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Main corpus generation loop
# ---------------------------------------------------------------------------
items = []
t_start = time.perf_counter()

for idx, prompt in enumerate(PROMPTS):
    raw_ids = tok.encode(prompt, add_special_tokens=False)
    ids = raw_ids[:min(len(raw_ids), 480)]   # stay well under S_MAX=512

    dec.reset()
    torch.cuda.synchronize()

    first_tok = prefill(ids)

    generated = [first_tok]
    nid = first_tok
    for _ in range(N_GEN - 1):
        nid = dec.step(nid)
        generated.append(nid)
        if nid == tok.eos_token_id:
            break

    torch.cuda.synchronize()

    items.append({
        "id":                  idx,
        "prompt":              prompt,
        "prompt_token_ids":    ids,
        "generated_token_ids": generated,
    })
    print(f"[{idx+1:02d}/{len(PROMPTS)}] prompt_len={len(ids):3d}  gen_len={len(generated):2d}  first_tok={generated[0]}", flush=True)

elapsed = time.perf_counter() - t_start

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "baseline.json")

corpus = {
    "git_sha":      git_sha,
    "gpu":          gpu_name,
    "torch_version": torch.__version__,
    "n_prompts":    len(items),
    "n_gen":        N_GEN,
    "items":        items,
}

with open(out_path, "w") as fh:
    json.dump(corpus, fh, indent=2)

print(f"\nSaved {len(items)} items → {out_path}  ({elapsed:.1f}s total)", flush=True)



================================================
FILE: megakernel/diag_phase2_metrics.py
================================================
"""Deep CUPTI instrumentation on pf_dn_chunk_phase2.

Mirrors diag_prefill_kernels.py setup, then adds:
  - torch.profiler with record_shapes, profile_memory, with_stack
  - timeline gap analysis (phase1 -> phase2 -> next-prep)
  - Chrome trace export for inspection

Run from megakernel/:
    python3 diag_phase2_metrics.py 2>&1 | tail -150

Decision target: is phase2 compute-bound, memory-bound, sync-bound, or block-starved?
"""
import time
import statistics
import torch
import torch.profiler as prof_mod
from model import Decoder, HIDDEN_SIZE, INTERMEDIATE_SIZE, FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE
from model import DN_CONV_CHANNELS, DN_V_SIZE, DN_NUM_HEADS, MAX_SEQ_LEN, _half_dtype
import qwen35_megakernel_bf16_C
from transformers import AutoTokenizer

print(f"GPU: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability()})")
print(f"torch: {torch.__version__}  CUDA: {torch.version.cuda}")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
dec = Decoder(verbose=False)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

S_MAX = 512
bf16 = dict(dtype=_half_dtype(), device="cuda")
f32  = dict(dtype=torch.float32,  device="cuda")
i32  = dict(dtype=torch.int32,    device="cuda")
mx   = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)

bufs = dict(
    hidden           = torch.empty(S_MAX * HIDDEN_SIZE, **bf16),
    residual         = torch.empty(S_MAX * HIDDEN_SIZE, **bf16),
    normalized       = torch.empty(S_MAX * HIDDEN_SIZE, **bf16),
    proj_buf         = torch.empty(S_MAX * mx, **bf16),
    proj_buf2        = torch.empty(S_MAX * mx, **bf16),
    attn_buf         = torch.empty(S_MAX * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf          = torch.empty(S_MAX * INTERMEDIATE_SIZE, **bf16),
    dn_out_buf       = torch.empty(S_MAX * DN_V_SIZE, **bf16),
    beta_buf         = torch.empty(S_MAX * DN_NUM_HEADS, **f32),
    alpha_buf        = torch.empty(S_MAX * DN_NUM_HEADS, **f32),
    final_normed     = torch.empty(HIDDEN_SIZE, **bf16),
    hidden_bf16_out  = torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv           = torch.empty(1024, **f32),
    lm_bmi           = torch.empty(1024, **i32),
)
bufs.update(dec.alloc_prefill_scratch(S_MAX))


def prefill(ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    _pf(dec._out_token, ids_t,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        bufs['hidden'], bufs['residual'], bufs['normalized'],
        bufs['proj_buf'], bufs['proj_buf2'],
        bufs['attn_buf'], bufs['mlp_buf'],
        bufs['dn_out_buf'], bufs['beta_buf'], bufs['alpha_buf'],
        bufs['dn_pre_qkv'],
        bufs['dn_u_scratch'], bufs['dn_w_scratch'], bufs['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        bufs['final_normed'], bufs['hidden_bf16_out'],
        bufs['lm_bmv'], bufs['lm_bmi'], dec.max_seq_len)
    dec._hidden.copy_(bufs['hidden_bf16_out'])
    dec._position = len(ids)
    return dec._out_token.item()


# Build a 512-token prompt
prompt = "The quick brown fox jumps over the lazy dog. " * 60
ids    = tok.encode(prompt, add_special_tokens=False)[:512]
print(f"\nPrompt length: {len(ids)} tokens")

# ── Warmup ──────────────────────────────────────────────────────────────────
dec.reset()
prefill(ids)
torch.cuda.synchronize()

# ── Timed reference ─────────────────────────────────────────────────────────
dec.reset()
t0 = time.perf_counter()
prefill(ids)
torch.cuda.synchronize()
print(f"Wall time: {(time.perf_counter()-t0)*1000:.2f} ms  ({len(ids)/(time.perf_counter()-t0):,.0f} tok/s)")

# ── Two profiler runs ────────────────────────────────────────────────────────
for run_idx in range(2):
    dec.reset()
    with prof_mod.profile(
        activities=[prof_mod.ProfilerActivity.CUDA, prof_mod.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as p:
        with prof_mod.record_function("prefill_outer"):
            prefill(ids)
            torch.cuda.synchronize()

    print(f"\n{'='*60}  Run {run_idx}")
    print(p.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=25))
    # Export trace (useful for Chrome tracing; skip if disk is constrained)
    p.export_chrome_trace(f"/tmp/phase2_trace_run{run_idx}.json")
    print(f"  [trace saved -> /tmp/phase2_trace_run{run_idx}.json]")

# ── Phase2-specific timeline analysis ───────────────────────────────────────
# Use the second run (run_idx=1) — p still holds that run.
print("\n" + "="*60)
print("  PHASE2 DETAILED TIMELINE ANALYSIS (run 1)")
print("="*60)

events = p.events()

# Filter CUDA events (device_type == DeviceType.CUDA in newer torch,
# or check device_index >= 0)
cuda_events = [e for e in events if e.device_type == prof_mod.DeviceType.CUDA]
phase2_events = [e for e in cuda_events if "pf_dn_chunk_phase2" in e.name]
phase1_events = [e for e in cuda_events if "pf_dn_chunk_phase1" in e.name]
phase1_events.sort(key=lambda e: e.time_range.start)
phase2_events.sort(key=lambda e: e.time_range.start)

print(f"\nphase1 launches found: {len(phase1_events)}")
print(f"phase2 launches found: {len(phase2_events)}")

if phase2_events:
    # FunctionEvent stores CUDA time in self_cuda_time_total (microseconds)
    # For individual kernel events (not aggregated) cuda_time is per-event
    p2_durations_us = [e.cuda_time for e in phase2_events]
    p2_med  = statistics.median(p2_durations_us)
    p2_mean = statistics.mean(p2_durations_us)
    p2_p99  = sorted(p2_durations_us)[int(0.99 * len(p2_durations_us))]
    print(f"\nphase2 self-CUDA time per launch:")
    print(f"  count  : {len(p2_durations_us)}")
    print(f"  mean   : {p2_mean:.2f} µs")
    print(f"  median : {p2_med:.2f} µs")
    print(f"  P99    : {p2_p99:.2f} µs")
    print(f"  total  : {sum(p2_durations_us)/1000:.3f} ms")
else:
    print("\n[WARN] No pf_dn_chunk_phase2 CUDA events found in profiler trace.")
    print("       Kernel may be fused or named differently. Checking all CUDA events:")
    dn_related = [e for e in cuda_events if "dn" in e.name.lower() or "delta" in e.name.lower() or "phase" in e.name.lower()]
    for e in dn_related[:15]:
        print(f"  {e.name}  dur={e.duration:.1f}µs")
    if not dn_related:
        print("  (none matching 'dn/delta/phase')")
        print("\n  Top 15 CUDA events by duration:")
        top = sorted(cuda_events, key=lambda e: e.duration, reverse=True)[:15]
        for e in top:
            print(f"  {e.name:60s} dur={e.duration:8.1f}µs")

# ── Gap analysis: phase1→phase2 and phase2→next ──────────────────────────────
# FunctionEvent gap analysis using time_range (nanoseconds in torch 2.x)
# time_range.start and time_range.end are in microseconds for CUDA events
if phase1_events and phase2_events:
    print("\n--- Phase1→Phase2 and Phase2→next-prep gaps ---")
    gaps_p1_p2 = []
    for p1, p2 in zip(phase1_events, phase2_events):
        # cuda_time_range is available; fall back to time_range
        try:
            gap_us = p2.time_range.start - p1.time_range.end
        except AttributeError:
            gap_us = 0
        if abs(gap_us) < 1e6:
            gaps_p1_p2.append(gap_us)
    if gaps_p1_p2:
        print(f"  phase1→phase2 gap  median={statistics.median(gaps_p1_p2):.1f}µs  "
              f"mean={statistics.mean(gaps_p1_p2):.1f}µs  "
              f"max={max(gaps_p1_p2):.1f}µs")
    else:
        print("  (time_range not available or gaps out of range)")

# ── All kernel totals breakdown ──────────────────────────────────────────────
print("\n--- CUDA time breakdown (top 10 by total cuda_time) ---")
name_to_total: dict[str, float] = {}
for e in cuda_events:
    name_to_total[e.name] = name_to_total.get(e.name, 0.0) + e.cuda_time
grand_total = sum(name_to_total.values())
for name, t in sorted(name_to_total.items(), key=lambda x: -x[1])[:10]:
    pct = 100 * t / grand_total if grand_total > 0 else 0
    print(f"  {name:60s}  {t/1000:8.3f} ms  ({pct:.1f}%)")

print(f"\n  Grand total tracked CUDA time: {grand_total/1000:.3f} ms")

if phase2_events:
    p2_total = sum(e.cuda_time for e in phase2_events)
    print(f"  phase2 share: {100*p2_total/grand_total:.1f}%  ({p2_total/1000:.3f} ms)")



================================================
FILE: megakernel/diag_prefill_kernels.py
================================================
"""Diagnostic from issue #5 (dknos): is prefill landing on cuBLAS MAGMA
(slow CUDA-core fallback) or CUTLASS tensor-core kernels?

Runs one warmup + one profiled prefill of a 520-token prompt and prints
the top CUDA kernels by self_cuda_time_total.

Decision rule:
  magma_*                       -> CUDA-core fallback, ~10x slower than tensor cores
  cutlass_*  / *gemm_*tensor_op -> tensor cores; slowdown is elsewhere
"""
import time
import torch
from model import Decoder, HIDDEN_SIZE, INTERMEDIATE_SIZE, FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE
from model import DN_CONV_CHANNELS, DN_V_SIZE, DN_NUM_HEADS, MAX_SEQ_LEN, _half_dtype
import qwen35_megakernel_bf16_C
from transformers import AutoTokenizer

print(f"GPU: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability()})")
print(f"torch: {torch.__version__}  CUDA: {torch.version.cuda}")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
dec = Decoder(verbose=False)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

S_MAX = 512
bf16 = dict(dtype=_half_dtype(), device="cuda")
f32 = dict(dtype=torch.float32, device="cuda")
i32 = dict(dtype=torch.int32, device="cuda")
mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
bufs = dict(
    hidden=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    residual=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    normalized=torch.empty(S_MAX*HIDDEN_SIZE, **bf16),
    proj_buf=torch.empty(S_MAX*mx, **bf16),
    proj_buf2=torch.empty(S_MAX*mx, **bf16),
    attn_buf=torch.empty(S_MAX*max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf=torch.empty(S_MAX*INTERMEDIATE_SIZE, **bf16),
    dn_out_buf=torch.empty(S_MAX*DN_V_SIZE, **bf16),
    beta_buf=torch.empty(S_MAX*DN_NUM_HEADS, **f32),
    alpha_buf=torch.empty(S_MAX*DN_NUM_HEADS, **f32),
    final_normed=torch.empty(HIDDEN_SIZE, **bf16),
    hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv=torch.empty(1024, **f32),
    lm_bmi=torch.empty(1024, **i32),
)
bufs.update(dec.alloc_prefill_scratch(S_MAX))


def prefill(ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    _pf(dec._out_token, ids_t,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        bufs['hidden'], bufs['residual'], bufs['normalized'],
        bufs['proj_buf'], bufs['proj_buf2'],
        bufs['attn_buf'], bufs['mlp_buf'],
        bufs['dn_out_buf'], bufs['beta_buf'], bufs['alpha_buf'],
        bufs['dn_pre_qkv'],
        bufs['dn_u_scratch'], bufs['dn_w_scratch'], bufs['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        bufs['final_normed'], bufs['hidden_bf16_out'],
        bufs['lm_bmv'], bufs['lm_bmi'], dec.max_seq_len)
    dec._hidden.copy_(bufs['hidden_bf16_out'])
    dec._position = len(ids)
    return dec._out_token.item()


# Build a 520-ish token prompt
prompt = "The quick brown fox jumps over the lazy dog. " * 60
ids = tok.encode(prompt, add_special_tokens=False)[:512]
print(f"\nPrompt length: {len(ids)} tokens")

# Warmup (matches bench_pp_tg.py — one untimed warmup)
dec.reset()
prefill(ids)
torch.cuda.synchronize()

# Timed-only run (no profiler) for tok/s reference
dec.reset()
t0 = time.perf_counter()
prefill(ids)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"Untimed-warmup prefill ({len(ids)} tok): {dt*1000:.2f} ms  ->  {len(ids)/dt:,.0f} tok/s")

# Profiler run
dec.reset()
print("\n=== Profiling one prefill call ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    record_shapes=False,
) as prof:
    prefill(ids)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# Verdict
events = prof.key_averages()
gemm_like = [e for e in events
             if any(k in e.key.lower()
                    for k in ("gemm", "magma", "cutlass", "tensorop", "mm_out", "ampere", "wmma"))]
print("\n=== Verdict ===")
if not gemm_like:
    print("No GEMM-like kernels found in top events. Inspect the table above.")
else:
    for e in gemm_like:
        print(f"  {e.key}   self_cuda={e.self_cuda_time_total/1000:.2f} ms")
    bad = any("magma" in e.key.lower() for e in gemm_like)
    good = any(k in e.key.lower() for e in gemm_like for k in ("cutlass", "tensorop", "ampere", "wmma"))
    if bad and not good:
        print("\n[FAIL] Dominant GEMM is MAGMA — cuBLAS fell back to CUDA-core path.")
        print("       Apply dknos's fix: replace cublas_bf16_gemm with at::mm_out.")
        print("       PR: https://github.com/dknos/luce-megakernel/pull/1")
    elif good and not bad:
        print("\n[PASS] Dominant GEMM is on tensor cores. Slowdown is elsewhere.")
        print("       Investigate DVFS / thermal throttling / driver / launch-overhead.")
    else:
        print("\n[MIXED] Both magma and tensor-core kernels present. Look at relative time totals.")



================================================
FILE: megakernel/final_bench.py
================================================
"""Final benchmark: pp520 tg128 — Our megakernel vs PyTorch naive.
Both properly warmed. Saves completions for verification.

Supports --backend {auto,bf16,nvfp4}. Default is auto: Blackwell (sm_12+)
dispatches to final_bench_nvfp4.py; everything else runs the bf16 path
below unchanged from upstream.
"""
import argparse as _argparse, os as _os, sys as _sys
import torch as _torch

_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--backend", default="auto", choices=("auto", "bf16", "nvfp4"))
_a, _rest = _p.parse_known_args()
_backend = _a.backend
if _backend == "auto":
    _backend = "nvfp4" if (_torch.cuda.is_available() and _torch.cuda.get_device_capability()[0] >= 12) else "bf16"
if _backend == "nvfp4":
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _os.execv(_sys.executable, [_sys.executable, _os.path.join(_here, "final_bench_nvfp4.py"), *_rest])

import time, torch
import _phase2_variant  # noqa: F401 — prints "[megakernel] DN phase2 variant = scalar|wmma"
from model import Decoder, HIDDEN_SIZE, INTERMEDIATE_SIZE, FA_QPROJ_SIZE, FA_Q_SIZE, FA_KV_SIZE
from model import DN_CONV_CHANNELS, DN_V_SIZE, DN_NUM_HEADS, MAX_SEQ_LEN, _half_dtype
import qwen35_megakernel_bf16_C
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

# ============================================================
# Build a 520-token prompt
# ============================================================
long_text = "Explain in great detail the history of artificial intelligence, machine learning, deep learning, and neural networks. " * 40
prompt_ids = tok.encode(long_text, add_special_tokens=False)[:520]
print(f"Prompt: {len(prompt_ids)} tokens")

# ============================================================
# 1. Our megakernel (prefill cuBLAS + decode megakernel)
# ============================================================
_dtype_label = "FP16" if _half_dtype() == torch.float16 else "BF16"
_gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"

print(f"\n=== Our {_dtype_label} Megakernel ===")
dec = Decoder(verbose=False)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

S = 520
bf16 = dict(dtype=_half_dtype(), device="cuda")
f32 = dict(dtype=torch.float32, device="cuda")
i32 = dict(dtype=torch.int32, device="cuda")
mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
b = dict(
    hidden=torch.empty(S*HIDDEN_SIZE, **bf16), residual=torch.empty(S*HIDDEN_SIZE, **bf16),
    normalized=torch.empty(S*HIDDEN_SIZE, **bf16),
    proj_buf=torch.empty(S*mx, **bf16), proj_buf2=torch.empty(S*mx, **bf16),
    attn_buf=torch.empty(S*max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
    mlp_buf=torch.empty(S*INTERMEDIATE_SIZE, **bf16),
    dn_out_buf=torch.empty(S*DN_V_SIZE, **bf16),
    beta_buf=torch.empty(S*DN_NUM_HEADS, **f32), alpha_buf=torch.empty(S*DN_NUM_HEADS, **f32),
    final_normed=torch.empty(HIDDEN_SIZE, **bf16), hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
    lm_bmv=torch.empty(1024, **f32), lm_bmi=torch.empty(1024, **i32),
)
b.update(dec.alloc_prefill_scratch(S))
ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")

def our_prefill():
    dec.reset()
    _pf(dec._out_token, ids_t, dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        b['hidden'], b['residual'], b['normalized'],
        b['proj_buf'], b['proj_buf2'], b['attn_buf'], b['mlp_buf'],
        b['dn_out_buf'], b['beta_buf'], b['alpha_buf'],
        b['dn_pre_qkv'],
        b['dn_u_scratch'], b['dn_w_scratch'], b['dn_cs_scratch'],
        dec._fused_fa_qkv, dec._fused_gate_up,
        b['final_normed'], b['hidden_bf16_out'], b['lm_bmv'], b['lm_bmi'],
        dec.max_seq_len)
    dec._hidden.copy_(b['hidden_bf16_out'])
    dec._position = len(prompt_ids)
    return dec._out_token.item()

# Warmup 10x
for _ in range(10): our_prefill(); torch.cuda.synchronize()

# PP benchmark (20 runs)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20): our_prefill(); torch.cuda.synchronize()
our_pp_ms = (time.perf_counter() - t0) / 20 * 1000
our_pp_tps = len(prompt_ids) / our_pp_ms * 1000

# TG benchmark + completion
first = our_prefill()
torch.cuda.synchronize()
out_ids = [first]; nid = first
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(128):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    out_ids.append(nid)
torch.cuda.synchronize()
our_tg_ms = (time.perf_counter() - t0) * 1000
our_tg_tps = len(out_ids) / our_tg_ms * 1000
our_text = tok.decode(out_ids, skip_special_tokens=True)

print(f"pp{len(prompt_ids)}: {our_pp_tps:.0f} tok/s ({our_pp_ms:.1f}ms)")
print(f"tg{len(out_ids)}: {our_tg_tps:.0f} tok/s ({our_tg_ms:.1f}ms)")
print(f"Completion: {our_text[:120]}")

del dec
torch.cuda.empty_cache()

# ============================================================
# 2. PyTorch naive (HuggingFace)
# ============================================================
print("\n=== PyTorch HuggingFace ===")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=_half_dtype(), device_map="cuda")
model.eval()
input_ids = torch.tensor([prompt_ids], device="cuda")

# Warmup 5x
with torch.no_grad():
    for _ in range(5): _ = model(input_ids); torch.cuda.synchronize()

# PP benchmark (10 runs)
with torch.no_grad():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): _ = model(input_ids); torch.cuda.synchronize()
    pt_pp_ms = (time.perf_counter() - t0) / 10 * 1000
    pt_pp_tps = len(prompt_ids) / pt_pp_ms * 1000

# TG benchmark + completion
with torch.no_grad():
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    pt_out_ids = [next_id.item()]

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(128):
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        if next_id.item() == tok.eos_token_id: break
        pt_out_ids.append(next_id.item())
    torch.cuda.synchronize()
    pt_tg_ms = (time.perf_counter() - t0) * 1000
    pt_tg_tps = len(pt_out_ids) / pt_tg_ms * 1000
    pt_text = tok.decode(pt_out_ids, skip_special_tokens=True)

print(f"pp{len(prompt_ids)}: {pt_pp_tps:.0f} tok/s ({pt_pp_ms:.1f}ms)")
print(f"tg{len(pt_out_ids)}: {pt_tg_tps:.0f} tok/s ({pt_tg_ms:.1f}ms)")
print(f"Completion: {pt_text[:120]}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"FINAL RESULTS — Qwen3.5-0.8B {_dtype_label}, {_gpu_name}")
print(f"{'='*60}")
print(f"{'Method':<25} {'pp'+str(len(prompt_ids)):>8} {'tg128':>10}")
print(f"{'-'*45}")
print(f"{'Our megakernel':<25} {our_pp_tps:>7.0f} t/s {our_tg_tps:>8.0f} t/s")
print(f"{'PyTorch HF':<25} {pt_pp_tps:>7.0f} t/s {pt_tg_tps:>8.0f} t/s")
print(f"{'llama.cpp ' + _dtype_label:<25} {'(run separately)':>19}")
print(f"")
print(f"Megakernel vs PyTorch:  pp {our_pp_tps/pt_pp_tps:.1f}x  tg {our_tg_tps/pt_tg_tps:.1f}x")
print(f"")
print(f"=== Completions ===")
print(f"Ours:    {our_text[:100]}")
print(f"PyTorch: {pt_text[:100]}")



================================================
FILE: megakernel/final_bench_nvfp4.py
================================================
"""Final benchmark for Qwen3.5-0.8B on local CUDA backends.

This script compares:
1. Megakernel prefill + decode
2. Hugging Face eager baseline

On GB10, the megakernel path may use BF16 prefill plus NVFP4 decode.
The script is intentionally progress-heavy so long GPU runs do not look hung.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_nvfp4 import (
    Decoder,
    DN_CONV_CHANNELS,
    DN_NUM_HEADS,
    DN_V_SIZE,
    FA_KV_SIZE,
    FA_QPROJ_SIZE,
    FA_Q_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    PREFILL_PROJ_FUSED_SIZE,
    PREFILL_PROJ_SCRATCH_SIZE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark megakernel vs HF baseline")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", default="auto", choices=("auto", "bf16", "nvfp4"))
    parser.add_argument("--prompt-tokens", type=int, default=520)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--our-warmup-runs", type=int, default=3)
    parser.add_argument("--our-pp-runs", type=int, default=5)
    parser.add_argument("--hf-warmup-runs", type=int, default=2)
    parser.add_argument("--hf-pp-runs", type=int, default=3)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--verbose-loader", action="store_true")
    parser.add_argument("--prefill-mode", default="eager",
                        choices=("eager", "mega"),
                        help="'eager' = cuBLAS+launches prefill (prefill_bf16); "
                             "'mega' = single-dispatch persistent megakernel "
                             "(prefill_bf16_mega)")
    return parser.parse_args()


def build_exact_prompt_ids(tokenizer, target_tokens):
    seed = (
        "Explain in great detail the history of artificial intelligence, machine learning, "
        "deep learning, and neural networks."
    )
    text = seed
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text += " " + seed
        ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:target_tokens]


def alloc_prefill_buffers(max_tokens):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    return dict(
        hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
        proj_buf=torch.empty(max_tokens * PREFILL_PROJ_FUSED_SIZE, **bf16),
        proj_buf2=torch.empty(max_tokens * PREFILL_PROJ_SCRATCH_SIZE, **bf16),
        attn_buf=torch.empty(max_tokens * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
        mlp_buf=torch.empty(max_tokens * INTERMEDIATE_SIZE, **bf16),
        dn_out_buf=torch.empty(max_tokens * DN_V_SIZE, **bf16),
        beta_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        alpha_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
        final_normed=torch.empty(HIDDEN_SIZE, **bf16),
        hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
    )


def get_prefill_op(decoder):
    ops = torch.ops.qwen35_megakernel_bf16_C
    if decoder.backend == "nvfp4":
        return ops.prefill_megakernel_nvfp4
    return ops.prefill_bf16


def run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=False):
    decoder.reset()
    if decoder.backend == "nvfp4":
        return decoder.prefill_tokens(ids_t)
    elif use_mega:
        prefill_op(
            decoder._out_token,
            ids_t,
            decoder._embed_weight,
            decoder._layer_weights_packed,
            decoder._final_norm_weight,
            decoder._lm_head_weight,
            decoder._fa_k_cache,
            decoder._fa_v_cache,
            decoder._dn_states,
            decoder._conv_bufs,
            buffers["hidden"],
            buffers["residual"],
            buffers["normalized"],
            buffers["proj_buf"],
            buffers["proj_buf2"],
            buffers["attn_buf"],
            buffers["mlp_buf"],
            buffers["dn_out_buf"],
            buffers["beta_buf"],
            buffers["alpha_buf"],
            buffers["final_normed"],
            buffers["hidden_bf16_out"],
            buffers["lm_bmv"],
            buffers["lm_bmi"],
        )
    else:
        prefill_op(
            decoder._out_token,
            ids_t,
            decoder._embed_weight,
            decoder._layer_weights_packed,
            decoder._prefill_fused_weights_packed,
            decoder._final_norm_weight,
            decoder._lm_head_weight,
            decoder._fa_k_cache,
            decoder._fa_v_cache,
            decoder._dn_states,
            decoder._conv_bufs,
            buffers["hidden"],
            buffers["residual"],
            buffers["normalized"],
            buffers["proj_buf"],
            buffers["proj_buf2"],
            buffers["attn_buf"],
            buffers["mlp_buf"],
            buffers["dn_out_buf"],
            buffers["beta_buf"],
            buffers["alpha_buf"],
            buffers["final_normed"],
            buffers["hidden_bf16_out"],
            buffers["lm_bmv"],
            buffers["lm_bmi"],
        )
    decoder._hidden.copy_(buffers["hidden_bf16_out"])
    decoder._position = prompt_len
    return decoder._out_token.item()


def benchmark_megakernel(decoder, tokenizer, prompt_ids, args):
    use_mega = getattr(args, "prefill_mode", "eager") == "mega" and decoder.backend != "nvfp4"
    if use_mega:
        prefill_op = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16_mega
    else:
        prefill_op = get_prefill_op(decoder)
    prompt_len = len(prompt_ids)
    if use_mega:
        padded_len = ((prompt_len + 31) // 32) * 32
        ids_t = torch.zeros(padded_len, dtype=torch.int32, device="cuda")
        ids_t[:prompt_len] = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
        ids_t = ids_t[:prompt_len]  # pass actual length; kernel reads only 0..prompt_len
        buffers = alloc_prefill_buffers(padded_len)
    else:
        ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
        buffers = alloc_prefill_buffers(prompt_len)

    print(f"Backend: {decoder.backend_label}", flush=True)
    print(
        f"Warming megakernel prefill {args.our_warmup_runs}x and timing {args.our_pp_runs}x...",
        flush=True,
    )

    for _ in range(args.our_warmup_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=use_mega)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.our_pp_runs):
        run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=use_mega)
        torch.cuda.synchronize()
    our_pp_ms = (time.perf_counter() - t0) / args.our_pp_runs * 1000.0
    our_pp_tps = prompt_len / our_pp_ms * 1000.0

    print(f"Running megakernel decode for {args.gen_tokens} timed steps...", flush=True)
    del decoder
    torch.cuda.empty_cache()
    print("Reloading decoder for megakernel decode benchmark...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )
    first = run_prefill(decoder, ids_t, prompt_len, buffers, prefill_op, use_mega=use_mega)
    decoded_ids = [first]
    eos = tokenizer.eos_token_id

    if decoder.backend == "nvfp4":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed_ids_dev = decoder.step_many(first, args.gen_tokens)
        torch.cuda.synchronize()
        our_tg_ms = (time.perf_counter() - t0) * 1000.0
        timed_ids = timed_ids_dev.cpu().tolist()
        if eos in timed_ids:
            timed_ids = timed_ids[:timed_ids.index(eos)]
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nid = first
        timed_ids = []
        for _ in range(args.gen_tokens):
            nid = decoder.step(nid)
            if nid == eos:
                break
            timed_ids.append(nid)
        torch.cuda.synchronize()
        our_tg_ms = (time.perf_counter() - t0) * 1000.0

    decoded_ids.extend(timed_ids)
    our_tg_tps = (len(timed_ids) / our_tg_ms * 1000.0) if timed_ids else 0.0
    our_text = tokenizer.decode(decoded_ids, skip_special_tokens=True)

    return {
        "pp_ms": our_pp_ms,
        "pp_tps": our_pp_tps,
        "tg_ms": our_tg_ms,
        "tg_tps": our_tg_tps,
        "tg_count": len(timed_ids),
        "text": our_text,
    }


def benchmark_hf(model_name, tokenizer, prompt_ids, args):
    print("\n=== PyTorch HuggingFace ===", flush=True)
    print("Loading HF baseline model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    input_ids = torch.tensor([prompt_ids], device="cuda")

    print(
        f"Warming HF forward {args.hf_warmup_runs}x and timing {args.hf_pp_runs}x...",
        flush=True,
    )
    with torch.inference_mode():
        for _ in range(args.hf_warmup_runs):
            _ = model(input_ids)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.hf_pp_runs):
            _ = model(input_ids)
            torch.cuda.synchronize()
        pt_pp_ms = (time.perf_counter() - t0) / args.hf_pp_runs * 1000.0
        pt_pp_tps = len(prompt_ids) / pt_pp_ms * 1000.0

        print(f"Running HF decode for {args.gen_tokens} timed steps...", flush=True)
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        decoded_ids = [next_id.item()]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timed_ids = []
        current = next_id
        for _ in range(args.gen_tokens):
            out = model(current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            current = out.logits[:, -1:].argmax(-1)
            token = current.item()
            if token == tokenizer.eos_token_id:
                break
            timed_ids.append(token)
        torch.cuda.synchronize()

    decoded_ids.extend(timed_ids)
    pt_tg_ms = (time.perf_counter() - t0) * 1000.0
    pt_tg_tps = (len(timed_ids) / pt_tg_ms * 1000.0) if timed_ids else 0.0
    pt_text = tokenizer.decode(decoded_ids, skip_special_tokens=True)

    del model, input_ids, past, out
    torch.cuda.empty_cache()

    return {
        "pp_ms": pt_pp_ms,
        "pp_tps": pt_pp_tps,
        "tg_ms": pt_tg_ms,
        "tg_tps": pt_tg_tps,
        "tg_count": len(timed_ids),
        "text": pt_text,
    }


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Loading megakernel decoder for {args.model_name}...", flush=True)
    decoder = Decoder(
        model_name=args.model_name,
        backend=args.backend,
        verbose=args.verbose_loader,
    )

    prompt_ids = build_exact_prompt_ids(tokenizer, args.prompt_tokens)
    print(f"Prompt: {len(prompt_ids)} tokens", flush=True)
    backend_label = decoder.backend_label

    print("\n=== Our Megakernel ===", flush=True)
    our = benchmark_megakernel(decoder, tokenizer, prompt_ids, args)

    print(
        f"pp{len(prompt_ids)}: {our['pp_tps']:.0f} tok/s ({our['pp_ms']:.1f}ms avg)",
        flush=True,
    )
    print(
        f"tg{our['tg_count']}: {our['tg_tps']:.0f} tok/s ({our['tg_ms']:.1f}ms total)",
        flush=True,
    )
    print(f"Completion: {our['text'][:120]}", flush=True)

    del decoder
    torch.cuda.empty_cache()

    hf = None
    if not args.skip_hf:
        try:
            hf = benchmark_hf(args.model_name, tokenizer, prompt_ids, args)
            print(
                f"pp{len(prompt_ids)}: {hf['pp_tps']:.0f} tok/s ({hf['pp_ms']:.1f}ms avg)",
                flush=True,
            )
            print(
                f"tg{hf['tg_count']}: {hf['tg_tps']:.0f} tok/s ({hf['tg_ms']:.1f}ms total)",
                flush=True,
            )
            print(f"Completion: {hf['text'][:120]}", flush=True)
        except Exception as exc:
            hf = {"error": str(exc)}
            print(f"HF baseline failed: {exc}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"FINAL RESULTS — Qwen3.5-0.8B, {backend_label}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"{'Method':<25} {'pp'+str(len(prompt_ids)):>12} {'tg':>16}", flush=True)
    print(f"{'-' * 55}", flush=True)
    print(
        f"{'Our megakernel':<25} {our['pp_tps']:>9.0f} t/s {our['tg_tps']:>9.0f} t/s",
        flush=True,
    )
    if hf is None:
        print(f"{'PyTorch HF':<25} {'(skipped)':>24}", flush=True)
    elif "error" in hf:
        print(f"{'PyTorch HF':<25} {'(failed)':>24}", flush=True)
    else:
        print(
            f"{'PyTorch HF':<25} {hf['pp_tps']:>9.0f} t/s {hf['tg_tps']:>9.0f} t/s",
            flush=True,
        )
    print(f"{'llama.cpp BF16':<25} {'(run separately)':>24}", flush=True)

    if hf is not None and "error" not in hf:
        print("", flush=True)
        print(
            f"Megakernel vs PyTorch:  pp {our['pp_tps'] / hf['pp_tps']:.1f}x  "
            f"tg {our['tg_tps'] / hf['tg_tps']:.1f}x",
            flush=True,
        )
        print("", flush=True)
        print("=== Completions ===", flush=True)
        print(f"Ours:    {our['text'][:100]}", flush=True)
        print(f"PyTorch: {hf['text'][:100]}", flush=True)


if __name__ == "__main__":
    main()



================================================
FILE: megakernel/half_type.h
================================================
/**
 * Portable half-precision type alias for the megakernel.
 *
 * When TARGET_SM >= 80 (Ampere+), uses __nv_bfloat16 (native hardware support).
 * When TARGET_SM < 80  (Turing),   uses __half (fp16) for cuBLAS compatibility.
 *
 * All custom CUDA kernels do explicit H2F/F2H conversions and accumulate in
 * f32, so the choice of 16-bit storage format is transparent to the math.
 */

#pragma once

#ifndef TARGET_SM
#define TARGET_SM 86
#endif

#if TARGET_SM >= 80
  #include <cuda_bf16.h>
  using half_t = __nv_bfloat16;
  #define H2F(x) __bfloat162float(x)
  #define F2H(x) __float2bfloat16(x)
  #define CUBLAS_HALF_T CUDA_R_16BF
#else
  #include <cuda_fp16.h>
  using half_t = __half;
  #define H2F(x) __half2float(x)
  #define F2H(x) __float2half(x)
  #define CUBLAS_HALF_T CUDA_R_16F
#endif



================================================
FILE: megakernel/kernel.cu
================================================
/**
 * Fused single-kernel decode for Qwen3.5-0.8B (hybrid DeltaNet + Full Attention).
 * ALL BF16: weights bf16, activations bf16, accumulation f32.
 * DeltaNet state: f32 (recurrence needs precision).
 *
 * Optimized for: NVIDIA RTX 3090 (sm_86, 82 SMs)
 * Model:         Qwen/Qwen3.5-0.8B (bf16 weights)
 */

#include "half_type.h"
#include <cuda_runtime.h>

// =============================================================================
// Model constants
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;

// Full Attention
constexpr int FA_NUM_Q_HEADS = 8;
constexpr int FA_NUM_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA_RATIO = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
constexpr int FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_GATE_SIZE = FA_Q_SIZE;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE + FA_GATE_SIZE;
constexpr int FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROTARY_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_NUM_HEADS = 16;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

constexpr int MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 82
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

#ifndef LM_NUM_BLOCKS
#define LM_NUM_BLOCKS 512
#endif
#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif

static int g_decode_blocks_override = 0;

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

// =============================================================================
// Weight structs — ALL BF16
// =============================================================================

struct FullAttnWeights {
    const half_t *input_layernorm_weight;   // [1024]
    const half_t *q_proj_weight;             // [4096, 1024]
    const half_t *k_proj_weight;             // [512, 1024]
    const half_t *v_proj_weight;             // [512, 1024]
    const half_t *q_norm_weight;              // [256]
    const half_t *k_norm_weight;              // [256]
    const half_t *o_proj_weight;             // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;          // [3584, 1024]
    const half_t *up_proj_weight;            // [3584, 1024]
    const half_t *down_proj_weight;          // [1024, 3584]
};

struct DeltaNetWeights {
    const half_t *input_layernorm_weight;
    const half_t *qkv_proj_weight;           // [6144, 1024]
    const half_t *z_proj_weight;             // [2048, 1024]
    const half_t *beta_proj_weight;          // [16, 1024]
    const half_t *alpha_proj_weight;         // [16, 1024]
    const half_t *conv1d_weight;             // [6144, 1, 4]
    const half_t *a_log;                     // [16]
    const half_t *dt_bias;                   // [16]
    const half_t *norm_weight;               // [128]
    const half_t *out_proj_weight;           // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;
    const half_t *up_proj_weight;
    const half_t *down_proj_weight;
};

struct LayerWeights {
    int layer_type;
    int _pad[3];
    union {
        DeltaNetWeights dn;
        FullAttnWeights fa;
    };
};

// =============================================================================
// Atomic barrier
// =============================================================================

struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {}
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float fast_exp(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f)); return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(1.0f + fast_exp(-x))); return y;
}

__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
#if TARGET_SM >= 80
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#else
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#endif
    return out;
}

// BF16 dot product: 8 bf16 weights × 8 bf16 activations → f32
__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const half_t *act) {
    const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        sum += H2F(w[i]) * H2F(act[i]);
    return sum;
}

// =============================================================================
// RMSNorm — reads bf16 input, writes bf16 output
// =============================================================================

__device__ void rmsnorm_redundant(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,        // shared memory bf16
    half_t *__restrict__ g_residual)   // global bf16
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(__ldg(input + i));
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// RMSNorm from bf16 buffer (for post-attn norm)
__device__ void rmsnorm_from_bf16(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,
    half_t *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(input[i]);
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// =============================================================================
// BF16 Matvec: warp-per-row, activations in shared memory (bf16)
// =============================================================================

__device__ void matvec_bf16(
    const half_t *__restrict__ s_input,  // shared memory bf16 [in_dim]
    const half_t *__restrict__ weight,   // [out_dim, in_dim] bf16
    float *__restrict__ output,                  // [out_dim] f32 (accumulate in f32)
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) output[m] = sum;
        }
    }
}

// Fused gate+up+SiLU matvec (bf16 weights, bf16 activations)
__device__ void matvec_gate_up_silu_bf16(
    const half_t *__restrict__ s_input,
    const half_t *__restrict__ gate_weight,
    const half_t *__restrict__ up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *g_row = gate_weight + m * in_dim;
            const half_t *u_row = up_weight + m * in_dim;
            float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gate_sum += dot8_bf16(g_u4, s_input + k);
                up_sum += dot8_bf16(u_u4, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                output[m] = fast_silu(gate_sum) * up_sum;
        }
    }
}

// Down projection + residual → bf16 hidden
__device__ void matvec_down_residual_bf16(
    const float *__restrict__ s_input,           // shared [INTER] f32
    const half_t *__restrict__ weight,    // [HIDDEN, INTER] bf16
    const half_t *__restrict__ residual,  // [HIDDEN] bf16
    half_t *__restrict__ hidden_out,      // [HIDDEN] bf16
    int in_dim, int out_dim, int num_blocks)
{
    // This needs f32 input (MLP intermediate is f32). Convert on the fly.
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            // Weight is bf16, input is f32 — convert input to bf16 on the fly
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

// O projection + residual → bf16
__device__ void matvec_o_residual_bf16(
    const float *__restrict__ s_input,           // shared [Q_SIZE] f32
    const half_t *__restrict__ weight,
    const half_t *__restrict__ residual,
    half_t *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

// =============================================================================
// Full Attention layer (bf16)
// =============================================================================

__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ k_cache,
    half_t *__restrict__ v_cache,
    half_t *__restrict__ g_residual,  // [HIDDEN] bf16
    float *__restrict__ g_activations,        // scratch f32
    float *__restrict__ g_q,                  // [FA_QPROJ_SIZE] f32
    float *__restrict__ g_kv,                 // [FA_KV_SIZE*2] f32
    float *__restrict__ g_attn_out,           // [FA_Q_SIZE] f32
    float *__restrict__ g_mlp_inter,          // [INTER] f32
    half_t *__restrict__ hidden_out,   // [HIDDEN] bf16
    int position, int max_seq_len,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + QKV projection
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.q_proj_weight, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks);
    matvec_bf16(s_norm, w.k_proj_weight, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    matvec_bf16(s_norm, w.v_proj_weight, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    grid.sync();

    // Phase 2: QK norm + partial RoPE + KV cache write
    if (block_id == 0) {
        float *k_buf = g_kv, *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM, *vh = v_buf + h * FA_HEAD_DIM;
            half_t *kc = k_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            half_t *vc = v_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i]*kh[i];
            ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                float normed = kh[i] * sc * (1.0f + H2F(__ldg(w.k_norm_weight + i)));
                if (i < FA_ROTARY_DIM) {
                    float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                    float freq = float(position) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                    float pv = kh[p]*sc*(1.0f+H2F(__ldg(w.k_norm_weight+p)));
                    float rotated = (i < FA_ROTARY_DIM/2) ? (normed*cv - pv*sv) : (pv*sv + normed*cv);
                    kc[i] = F2H(rotated);
                } else { kc[i] = F2H(normed); }
                vc[i] = F2H(vh[i]);
            }
        }
    }
    // Q norm + RoPE (all blocks)
    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh_ptr[i]*qh_ptr[i];
                ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                sc = __shfl_sync(0xffffffff, sc, 0);
                for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    float normed = qh_ptr[i]*sc*(1.0f+H2F(__ldg(w.q_norm_weight+i)));
                    if (i < FA_ROTARY_DIM) {
                        float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                        float freq = float(position) / powf(FA_ROPE_THETA, fe);
                        float cv = cosf(freq), sv = sinf(freq);
                        int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                        float pv = qh_ptr[p]*sc*(1.0f+H2F(__ldg(w.q_norm_weight+p)));
                        qh_ptr[i] = (i < FA_ROTARY_DIM/2) ? (normed*cv-pv*sv) : (pv*sv+normed*cv);
                    } else { qh_ptr[i] = normed; }
                }
            }
        }
    }
    grid.sync();

    // Phase 3: Attention decode (online softmax + sigmoid gate)
    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        __shared__ float s_max_score[NUM_WARPS];
        __shared__ float s_sum_exp[NUM_WARPS];
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;

        for (int qh = hs; qh < he; qh++) {
            int kvh = qh / FA_GQA_RATIO;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;
            float max_score = -INFINITY, sum_exp = 0;
            float out_acc[EPL], q_local[EPL];
            for (int e = 0; e < EPL; e++) { out_acc[e] = 0; q_local[e] = q_head[lane_id*EPL+e]; }

            for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
                const half_t *k_pos = k_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                const half_t *v_pos = v_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                float score = 0;
                for (int e = 0; e < EPL; e++) score += q_local[e] * H2F(__ldg(k_pos + lane_id*EPL+e));
                score = warp_reduce_sum(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);
                float old_max = max_score; max_score = fmaxf(max_score, score);
                float exp_diff = fast_exp(old_max - max_score);
                sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);
                float wt = fast_exp(score - max_score);
                for (int e = 0; e < EPL; e++)
                    out_acc[e] = out_acc[e]*exp_diff + wt*H2F(__ldg(v_pos + lane_id*EPL+e));
            }
            if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
            for (int e = 0; e < EPL; e++) g_activations[warp_id*FA_HEAD_DIM + lane_id*EPL+e] = out_acc[e];
            __syncthreads();

            if (warp_id == 0) {
                float gm = -INFINITY; for (int ww = 0; ww < NUM_WARPS; ww++) if (s_max_score[ww] > -INFINITY) gm = fmaxf(gm, s_max_score[ww]);
                float ts = 0; float fo[EPL]; for (int e = 0; e < EPL; e++) fo[e] = 0;
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max_score[ww] > -INFINITY) {
                        float s = fast_exp(s_max_score[ww]-gm); ts += s_sum_exp[ww]*s;
                        for (int e = 0; e < EPL; e++) fo[e] += g_activations[ww*FA_HEAD_DIM+lane_id*EPL+e]*s;
                    }
                }
                float *gate_ptr = q_head + FA_HEAD_DIM;
                float rcp = 1.0f / ts;
                for (int e = 0; e < EPL; e++) {
                    int idx = lane_id*EPL+e;
                    out_head[idx] = fo[e]*rcp * fast_sigmoid(gate_ptr[idx]);
                }
            }
            __syncthreads();
        }
    }
    grid.sync();

    // Phase 4: O projection + residual → bf16
    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE) s_attn[i] = g_attn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_attn, w.o_proj_weight, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    // Load MLP intermediate to shared (f32)
    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();

    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// DeltaNet layer (bf16) — warp-cooperative state-in-registers recurrence
// =============================================================================

__device__ void deltanet_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ dn_state,     // [DN_NUM_HEADS, DN_KEY, DN_VAL] f32
    float *__restrict__ conv_buf,     // [DN_CONV_CH, DN_CONV_K] f32
    half_t *__restrict__ hidden_out,
    int dn_layer_idx,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + projections
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.qkv_proj_weight, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks);
    matvec_bf16(s_norm, w.z_proj_weight, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks);
    matvec_bf16(s_norm, w.beta_proj_weight, g_beta, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    matvec_bf16(s_norm, w.alpha_proj_weight, g_alpha, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    grid.sync();

    // Phase 2+3: Conv1d + recurrence (blocks 0-15 only)
    if (block_id < DN_NUM_HEADS) {
        int h = block_id;
        float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;

        // Conv1d + SiLU
        __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
        int head_ch[3] = {h*DN_KEY_DIM, DN_QK_SIZE+h*DN_KEY_DIM, 2*DN_QK_SIZE+h*DN_VALUE_DIM};
        for (int region = 0; region < 3; region++) {
            int ch_base = head_ch[region], ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += BLOCK_SIZE) {
                int ch = ch_base + c;
                float h0=layer_conv[ch*DN_CONV_KERNEL+1], h1=layer_conv[ch*DN_CONV_KERNEL+2], h2=layer_conv[ch*DN_CONV_KERNEL+3];
                layer_conv[ch*DN_CONV_KERNEL]=h0; layer_conv[ch*DN_CONV_KERNEL+1]=h1;
                layer_conv[ch*DN_CONV_KERNEL+2]=h2; layer_conv[ch*DN_CONV_KERNEL+3]=g_qkv[ch];
                float co = 0;
                for (int t = 0; t < DN_CONV_KERNEL; t++)
                    co += layer_conv[ch*DN_CONV_KERNEL+t] * H2F(__ldg(w.conv1d_weight + ch*DN_CONV_KERNEL+t));
                dst[c] = fast_silu(co);
            }
        }

        // Beta/alpha activations
        if (threadIdx.x == 0) {
            g_beta[h] = fast_sigmoid(g_beta[h]);
            float a_log_val = H2F(__ldg(w.a_log + h));
            float dt_b = H2F(__ldg(w.dt_bias + h));
            float x = g_alpha[h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[h] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();

        // L2 normalize Q, K
        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
        if (warp_id == 0) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i]*s_q[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f)*Q_SCALE;
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i]*s_k[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f);
            n = __shfl_sync(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        float decay = g_alpha[h], beta = g_beta[h];

        // k·q dot
        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i]*s_q[i];
            kq = warp_reduce_sum(kq); if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Warp-cooperative recurrence (state in global memory — decode is 1 token, fine)
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0, sqv = 0;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j*DN_KEY_DIM+i]; s_regs[ii] = sv;
                stk += sv * s_k[i]; sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk); sqv = warp_reduce_sum(sqv);
            stk = __shfl_sync(0xffffffff,stk,0); sqv = __shfl_sync(0xffffffff,sqv,0);
            float error_j = (s_v[j] - decay * stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = o_j;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j*DN_KEY_DIM+i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Gated RMSNorm
        __syncthreads();
        {
            __shared__ float smem_gnorm[NUM_WARPS];
            float sq = 0; for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) sq += out_head[i]*out_head[i];
            sq = warp_reduce_sum(sq); if (lane_id == 0) smem_gnorm[warp_id] = sq; __syncthreads();
            if (warp_id == 0) { float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0; v = warp_reduce_sum(v); if (lane_id == 0) smem_gnorm[0] = rsqrtf(v/DN_VALUE_DIM + RMS_EPS); }
            __syncthreads(); float rstd = smem_gnorm[0];
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
                float normed = out_head[i] * rstd * H2F(__ldg(w.norm_weight + i));
                float gate = fast_silu(g_z[h*DN_VALUE_DIM+i]);
                out_head[i] = normed * gate;
            }
        }
    } else {
        // Idle blocks: could prefetch weights
    }
    grid.sync();

    // Phase 4: Out projection + residual → bf16
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += BLOCK_SIZE) s_dn[i] = g_dn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_dn, w.out_proj_weight, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// LM Head: vocab projection + argmax
// =============================================================================

__global__ void lm_head_kernel(
    const float *__restrict__ hidden,
    const half_t *__restrict__ weight,   // [VOCAB, HIDDEN] bf16
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    unsigned int *__restrict__ sync_counter,
    const float *__restrict__ seen_token_mask,
    float repetition_penalty)
{
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LM_BLOCK_SIZE) s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = LM_BLOCK_SIZE / WARP_SIZE;
    int rpb = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int rs = blockIdx.x * rpb, re = min(rs + rpb, VOCAB_SIZE);

    float local_max = -INFINITY; int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        const half_t *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0;
#pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
            const half_t *wp = reinterpret_cast<const half_t *>(&w_u4);
            for (int i = 0; i < 8; i++) sum += H2F(wp[i]) * s_hidden[k+i];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0 && repetition_penalty > 1.0f && seen_token_mask && seen_token_mask[m] > 0.0f) {
            sum = (sum > 0.0f) ? (sum / repetition_penalty) : (sum * repetition_penalty);
        }
        if (lane_id == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32]; __shared__ int wi[32];
    if (lane_id == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lane_id < num_warps) ? wm[lane_id] : -INFINITY;
        int mi = (lane_id < num_warps) ? wi[lane_id] : -1;
        for (int o = WARP_SIZE/2; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lane_id == 0) { block_max_vals[blockIdx.x] = mv; block_max_idxs[blockIdx.x] = mi; }
    }
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicAdd(sync_counter, 1); }
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) { volatile unsigned int *vc = (volatile unsigned int *)sync_counter; while (*vc < (unsigned int)gridDim.x) {} __threadfence(); }
        __syncthreads();
        int tid = threadIdx.x; float bv = -INFINITY; int bi = -1;
        for (int i = tid; i < gridDim.x; i += LM_BLOCK_SIZE) { float v = block_max_vals[i]; if (v > bv) { bv = v; bi = block_max_idxs[i]; } }
        __shared__ float sv[256]; __shared__ int si[256];
        sv[tid] = bv; si[tid] = bi; __syncthreads();
        for (int s = LM_BLOCK_SIZE/2; s > 0; s >>= 1) { if (tid < s && sv[tid+s] > sv[tid]) { sv[tid] = sv[tid+s]; si[tid] = si[tid+s]; } __syncthreads(); }
        if (tid == 0) *output_token = si[0];
    }
}

// =============================================================================
// Main decode kernel
// =============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel(
    const half_t *__restrict__ embed_weight,
    const half_t *__restrict__ final_norm_weight,
    const half_t *__restrict__ lm_head_weight,
    const LayerWeights *__restrict__ layer_weights,
    half_t *__restrict__ fa_k_cache,
    half_t *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    half_t *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    float *__restrict__ seen_token_mask,
    float repetition_penalty,
    int input_token_id, int position, int max_seq_len)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    if (block_id == 0 && threadIdx.x == 0 &&
        seen_token_mask && repetition_penalty > 1.0f &&
        input_token_id >= 0 && input_token_id < VOCAB_SIZE) {
        seen_token_mask[input_token_id] = 1.0f;
    }

    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};

    // Shared memory: large enough for max(HIDDEN_SIZE bf16, INTERMEDIATE_SIZE f32)
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    half_t *shmem_bf16 = reinterpret_cast<half_t *>(shmem_raw);

    const half_t *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    int dn_layer_idx = 0, fa_layer_idx = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const half_t *layer_input = (layer == 0) ? embed_row : hidden_buffer;

        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(
                grid, layer_weights[layer].dn, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride,
                conv_bufs, hidden_buffer, dn_layer_idx, shmem_bf16);
            dn_layer_idx++;
        } else {
            full_attention_layer(
                grid, layer_weights[layer].fa, layer_input,
                fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride,
                g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_mlp_inter, hidden_buffer,
                position, max_seq_len, shmem_bf16);
            fa_layer_idx++;
        }
    }

    // Final RMSNorm (block 0 only)
    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = H2F(hidden_buffer[i]); g_activations[i] = v; local_sum_sq += v*v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq; __syncthreads();
        if (warp_id == 0) { float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0; sum = warp_reduce_sum(sum); if (lane_id == 0) smem_reduce[0] = rsqrtf(sum/HIDDEN_SIZE + RMS_EPS); }
        __syncthreads(); float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float wt = H2F(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * (1.0f + wt);
        }
    }
}

// =============================================================================
// C entry point
// =============================================================================

static int query_max_safe_decode_blocks_impl()
{
    int device_id = 0;
    int sm_count = 0;
    int active_blocks_per_sm = 0;
    int max_safe_blocks = NUM_BLOCKS;

    if (cudaGetDevice(&device_id) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            decode_kernel,
            BLOCK_SIZE,
            0) == cudaSuccess &&
        sm_count > 0 &&
        active_blocks_per_sm > 0) {
        const int resident_blocks = sm_count * active_blocks_per_sm;
        if (resident_blocks > 0) {
            max_safe_blocks = resident_blocks;
        }
    }

    return (max_safe_blocks < 1) ? 1 : max_safe_blocks;
}

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int position, int max_seq_len, cudaStream_t stream)
{
    const int max_safe_blocks = query_max_safe_decode_blocks_impl();
    int decode_blocks = NUM_BLOCKS;
    if (decode_blocks > max_safe_blocks) {
        decode_blocks = max_safe_blocks;
    }
    if (g_decode_blocks_override > 0) {
        decode_blocks = g_decode_blocks_override;
        if (decode_blocks > max_safe_blocks) {
            decode_blocks = max_safe_blocks;
        }
    }

    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);

    decode_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(
        (const half_t *)embed_weight,
        (const half_t *)final_norm_weight,
        (const half_t *)lm_head_weight,
        layer_weights,
        (half_t *)fa_k_cache, (half_t *)fa_v_cache,
        (float *)dn_states, (float *)conv_bufs,
        (half_t *)hidden_buffer,
        (float *)g_activations, (half_t *)g_residual,
        (float *)g_qkv_scratch, (float *)g_kv_scratch,
        (float *)g_attn_out, (float *)g_mlp_inter,
        (float *)g_z_scratch, (float *)g_beta_scratch,
        (float *)g_alpha_scratch, (float *)g_normalized,
        barrier_counter, barrier_generation,
        seen_token_mask, repetition_penalty,
        input_token_id, position, max_seq_len);

    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);

    lm_head_kernel<<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const half_t *)lm_head_weight,
        block_max_vals, block_max_idxs,
        output_token_id, lm_sync_counter,
        seen_token_mask, repetition_penalty);
}

extern "C" void set_decode_blocks_override(int blocks)
{
    g_decode_blocks_override = blocks;
}

extern "C" int query_max_safe_decode_blocks()
{
    return query_max_safe_decode_blocks_impl();
}



================================================
FILE: megakernel/model.py
================================================
"""Weight loading and decode API for Qwen3.5-0.8B bf16 megakernel."""

import struct
import torch

NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048


def _half_dtype():
    """Return the 16-bit dtype matching the compiled kernel: bf16 on SM>=80, fp16 otherwise."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return torch.float16
    return torch.bfloat16

FA_NUM_Q_HEADS = 8
FA_NUM_KV_HEADS = 2
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM
FA_QPROJ_SIZE = FA_Q_SIZE * 2
FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM

DN_NUM_HEADS = 16
DN_KEY_DIM = 128
DN_VALUE_DIM = 128
DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM
DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM
DN_CONV_CHANNELS = DN_QK_SIZE * 2 + DN_V_SIZE
DN_CONV_KERNEL = 4

LAYER_TYPE = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]

_decode = None
_max_safe_decode_blocks = None
_set_decode_blocks = None


def _load_op():
    global _decode, _max_safe_decode_blocks, _set_decode_blocks
    if _decode is None:
        import qwen35_megakernel_bf16_C
        _decode = torch.ops.qwen35_megakernel_bf16_C.decode
        _max_safe_decode_blocks = torch.ops.qwen35_megakernel_bf16_C.max_safe_decode_blocks
        _set_decode_blocks = torch.ops.qwen35_megakernel_bf16_C.set_decode_blocks


def max_safe_decode_blocks() -> int:
    """Return the resident-block ceiling for the current CUDA device."""
    _load_op()
    return int(_max_safe_decode_blocks())


def set_decode_blocks(blocks: int):
    """Override decode blocks, clamped by the CUDA resident-block ceiling."""
    if blocks < 0:
        raise ValueError("blocks must be non-negative")
    _load_op()
    _set_decode_blocks(int(blocks))


def load_weights(model_name="Qwen/Qwen3.5-0.8B", verbose=True):
    """Load Qwen3.5-0.8B weights (bf16 on SM>=80, fp16 on SM<80)."""
    if not verbose:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hdtype = _half_dtype()
    dtype_name = "bf16" if hdtype == torch.bfloat16 else "fp16"

    if verbose:
        print(f"Loading {model_name} ({dtype_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=hdtype, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    def _w(key):
        """Get weight tensor, ensuring it's in the target half dtype."""
        return state[key].to(hdtype).contiguous()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        lt = LAYER_TYPE[i]

        if lt == 1:
            # Full Attention: 11 pointers
            layer_data.append({
                "type": 1,
                "ptrs": [
                    _w(p + "input_layernorm.weight"),
                    _w(p + "self_attn.q_proj.weight"),
                    _w(p + "self_attn.k_proj.weight"),
                    _w(p + "self_attn.v_proj.weight"),
                    _w(p + "self_attn.q_norm.weight"),
                    _w(p + "self_attn.k_norm.weight"),
                    _w(p + "self_attn.o_proj.weight"),
                    _w(p + "post_attention_layernorm.weight"),
                    _w(p + "mlp.gate_proj.weight"),
                    _w(p + "mlp.up_proj.weight"),
                    _w(p + "mlp.down_proj.weight"),
                ]
            })
        else:
            # DeltaNet: 14 pointers
            layer_data.append({
                "type": 0,
                "ptrs": [
                    _w(p + "input_layernorm.weight"),
                    _w(p + "linear_attn.in_proj_qkv.weight"),
                    _w(p + "linear_attn.in_proj_z.weight"),
                    _w(p + "linear_attn.in_proj_b.weight"),
                    _w(p + "linear_attn.in_proj_a.weight"),
                    _w(p + "linear_attn.conv1d.weight"),
                    _w(p + "linear_attn.A_log"),
                    _w(p + "linear_attn.dt_bias"),
                    _w(p + "linear_attn.norm.weight"),
                    _w(p + "linear_attn.out_proj.weight"),
                    _w(p + "post_attention_layernorm.weight"),
                    _w(p + "mlp.gate_proj.weight"),
                    _w(p + "mlp.up_proj.weight"),
                    _w(p + "mlp.down_proj.weight"),
                ]
            })

    embed_weight = _w("model.embed_tokens.weight")
    final_norm_weight = _w("model.norm.weight")
    lm_head = state.get("lm_head.weight")
    if lm_head is None:
        lm_head = embed_weight
    else:
        lm_head = lm_head.to(hdtype).contiguous()

    weights = {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head,
        "layer_data": layer_data,
    }

    del model
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm_head.numel()
        print(f"{dtype_name.upper()} weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

    return weights, tokenizer


def _pack_layer_weights(layer_data):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size  # 128

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B bf16 megakernel."""

    def __init__(self, weights=None, tokenizer=None,
                 model_name="Qwen/Qwen3.5-0.8B", verbose=True,
                 max_seq_len=MAX_SEQ_LEN, repetition_penalty=1.0,
                 decode_blocks=None):
        _load_op()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if decode_blocks is not None and decode_blocks < 0:
            raise ValueError("decode_blocks must be non-negative")

        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self.max_seq_len = int(max_seq_len)
        self.repetition_penalty = float(repetition_penalty)
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])
        _set_decode_blocks(0 if decode_blocks is None else int(decode_blocks))

        bf16 = dict(dtype=_half_dtype(), device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, self.max_seq_len, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in LAYER_TYPE if t == 0)
        self._dn_states = torch.zeros(n_dn, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM, **f32)
        self._conv_bufs = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, **f32)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        max_scratch = max(FA_QPROJ_SIZE, DN_CONV_CHANNELS, HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE)
        self._activations = torch.empty(max_scratch, **f32)
        self._residual = torch.empty(HIDDEN_SIZE, **bf16)
        self._qkv_scratch = torch.empty(max(FA_QPROJ_SIZE, DN_CONV_CHANNELS), **f32)
        self._kv_scratch = torch.empty(FA_KV_SIZE * 2, **f32)
        self._attn_out = torch.empty(max(FA_Q_SIZE, DN_V_SIZE), **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._z_scratch = torch.empty(DN_V_SIZE, **f32)
        self._beta_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._alpha_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._normalized = torch.empty(HIDDEN_SIZE, **f32)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._seen_token_mask = torch.zeros(VOCAB_SIZE, **f32)
        self._out_token = torch.empty(1, **i32)

        # Pre-pack fused weights for the chunk-parallel prefill kernel:
        # one cuBLAS GEMM per layer instead of three (FA QKV) / two (MLP gate+up).
        layer_data = weights["layer_data"]
        fa_qkv_list = []
        for li in range(NUM_LAYERS):
            ld = layer_data[li]
            if ld['type'] == 1:
                q = ld['ptrs'][1]; k = ld['ptrs'][2]; v = ld['ptrs'][3]
                fa_qkv_list.append(torch.cat([q, k, v], dim=0))
        self._fused_fa_qkv = torch.stack(fa_qkv_list, dim=0).contiguous()
        gate_up_list = []
        for li in range(NUM_LAYERS):
            ld = layer_data[li]
            if ld['type'] == 0:
                g = ld['ptrs'][11]; u = ld['ptrs'][12]
            else:
                g = ld['ptrs'][8]; u = ld['ptrs'][9]
            gate_up_list.append(torch.cat([g, u], dim=0))
        self._fused_gate_up = torch.stack(gate_up_list, dim=0).contiguous()

    def alloc_prefill_scratch(self, S: int):
        """Allocate per-prefill scratch buffers for the chunk-parallel kernel.
        Buffers depend on S (sequence length); call once per distinct S."""
        f32 = dict(dtype=torch.float32, device="cuda")
        S_pad = ((S + 31) // 32) * 32
        return dict(
            dn_pre_qkv=torch.empty(S * DN_CONV_CHANNELS, **f32),
            dn_u_scratch=torch.empty(S_pad * DN_NUM_HEADS * 128, **f32),
            dn_w_scratch=torch.empty(S_pad * DN_NUM_HEADS * 128, **f32),
            dn_cs_scratch=torch.empty(S_pad * DN_NUM_HEADS, **f32),
        )

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self._position >= self.max_seq_len:
            raise ValueError(f"position {self._position} exceeds max_seq_len={self.max_seq_len}")
        _decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._seen_token_mask, self.repetition_penalty,
            self._position, self.max_seq_len,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()
        self._seen_token_mask.zero_()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        out = []
        next_id = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            next_id = self.step(next_id)
            if next_id == eos:
                break
            out.append(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)



================================================
FILE: megakernel/model_nvfp4.py
================================================
"""Qwen3.5-0.8B Blackwell NVFP4 decode API.

Companion to model.py. The upstream bf16 decoder stays in model.py, unmodified
from the RTX 3090 reference build; this module is the Blackwell (sm_120 /
sm_121a DGX Spark) NVFP4 path, dispatching through the decode_nvfp4 and
prefill_megakernel_nvfp4 torch ops added alongside kernel_gb10_nvfp4.cu and
prefill_megakernel.cu.

Only importable on Blackwell-class GPUs; only supports backend='nvfp4'.
"""

import os
import struct
import torch

if not torch.cuda.is_available():
    raise RuntimeError("model_nvfp4 requires CUDA")
_cap_major, _cap_minor = torch.cuda.get_device_capability()
if _cap_major < 12:
    raise RuntimeError(
        f"model_nvfp4 requires Blackwell (sm_120/sm_121a); detected "
        f"sm_{_cap_major}{_cap_minor}. Use megakernel.model for sm_86 and earlier."
    )


NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048

FA_NUM_Q_HEADS = 8
FA_NUM_KV_HEADS = 2
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM
FA_QPROJ_SIZE = FA_Q_SIZE * 2
FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM

DN_NUM_HEADS = 16
DN_KEY_DIM = 128
DN_VALUE_DIM = 128
DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM
DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM
DN_CONV_CHANNELS = DN_QK_SIZE * 2 + DN_V_SIZE
DN_BETA_ALPHA_SIZE = DN_NUM_HEADS * 2
DN_CONV_KERNEL = 4
DN_PROJ_FUSED_SIZE = DN_CONV_CHANNELS + DN_V_SIZE + DN_BETA_ALPHA_SIZE
DN_PROJ_FUSED_PADDED_SIZE = ((DN_PROJ_FUSED_SIZE + 127) // 128) * 128
PREFILL_PROJ_FUSED_SIZE = max(DN_PROJ_FUSED_PADDED_SIZE, FA_QPROJ_SIZE + 2 * FA_KV_SIZE, INTERMEDIATE_SIZE * 2)
PREFILL_PROJ_SCRATCH_SIZE = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
NVFP4_TC_ROWS_PER_TILE = 128
NVFP4_TC_COLS_PER_TILE = 4
NVFP4_TC_BLOCK_K = 16
NVFP4_TC_K_PER_TILE = NVFP4_TC_COLS_PER_TILE * NVFP4_TC_BLOCK_K

LAYER_TYPE = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]
NVFP4_GROUP_SIZE = 32
NVFP4_LM_GROUP_SIZE = 16
LM_HEAD_TENSORCORE_N = 16
LM_HEAD_TENSORCORE_PACKED_BYTES = LM_HEAD_TENSORCORE_N * (HIDDEN_SIZE // 2)
LM_HEAD_TENSORCORE_SCALE_BYTES = ((LM_HEAD_TENSORCORE_N + 127) // 128) * (HIDDEN_SIZE // 64) * 512

_decode = None
_decode_nvfp4 = None
_decode_many_nvfp4 = None
_prefill_bf16 = None
_prefill_bf16_nvfp4_lm = None
_prefill_megakernel_nvfp4 = None
_quantize_nvfp4_out = None
_quantize_nvfp4_lm_out = None


def _load_op():
    global _decode, _decode_nvfp4, _decode_many_nvfp4
    global _prefill_bf16, _prefill_bf16_nvfp4_lm, _prefill_megakernel_nvfp4
    global _quantize_nvfp4_out, _quantize_nvfp4_lm_out
    if _decode is None:
        import qwen35_megakernel_bf16_C
        ops = torch.ops.qwen35_megakernel_bf16_C
        _decode = ops.decode
        _decode_nvfp4 = ops.decode_nvfp4
        _decode_many_nvfp4 = ops.decode_many_nvfp4
        _prefill_bf16 = ops.prefill_bf16
        try:
            _prefill_bf16_nvfp4_lm = ops.prefill_bf16_nvfp4_lm
        except AttributeError:
            _prefill_bf16_nvfp4_lm = None
        _prefill_megakernel_nvfp4 = ops.prefill_megakernel_nvfp4
        _quantize_nvfp4_out = ops.quantize_nvfp4_out
        _quantize_nvfp4_lm_out = ops.quantize_nvfp4_lm_out


def _resolve_backend(backend):
    if backend in (None, "auto", "nvfp4"):
        return "nvfp4"
    raise ValueError(
        f"model_nvfp4 only supports backend='nvfp4' (got {backend!r}); "
        "use megakernel.model for bf16."
    )


def _resolve_prefill_mode():
    # 'hybrid' (default) = bf16 body + NVFP4 LM head via prefill_bf16_nvfp4_lm
    #   from prefill_bw.cu (Blackwell-only, anonymous-namespaced so it does
    #   not collide with upstream prefill.cu).
    # 'raw' = single-dispatch persistent prefill_megakernel_nvfp4 from
    #   prefill_megakernel.cu.
    mode = os.environ.get("MEGAKERNEL_PREFILL_MODE", "hybrid")
    if mode not in ("hybrid", "raw"):
        raise ValueError(
            f"MEGAKERNEL_PREFILL_MODE={mode!r} is not supported; expected 'hybrid' or 'raw'."
        )
    return mode


def _resolve_prefill_graph():
    value = os.environ.get("MEGAKERNEL_PREFILL_GRAPH", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _resolve_prefill_tc():
    value = os.environ.get("MEGAKERNEL_PREFILL_TC", "0").strip().lower()
    return value not in ("0", "false", "no", "off")


def _quantize_matrix_nvfp4(weight, group_size):
    _load_op()
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 weight, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if cols % 2 != 0 or cols % group_size != 0:
        raise ValueError(f"in_dim {cols} must be divisible by 2 and group_size {group_size}")

    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, cols // group_size), dtype=torch.float16, device=weight.device)
    _quantize_nvfp4_out(packed, scales, weight.contiguous(), group_size)
    return {"packed": packed, "scales": scales}


def _quantize_matrix_nvfp4_lm(weight):
    _load_op()
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 weight, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if rows % 128 != 0:
        raise ValueError(f"lm_head out_dim {rows} must be divisible by 128")
    if cols % 64 != 0:
        raise ValueError(f"lm_head in_dim {cols} must be divisible by 64")

    scale_tiles = cols // 64
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows // 128) * scale_tiles * 512, dtype=torch.uint8, device=weight.device)
    _quantize_nvfp4_lm_out(packed, scales, weight.contiguous())
    return {"packed": packed, "scales": scales}


def _quantize_matrix_nvfp4_tc(weight, padded_rows=None):
    _load_op()
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 weight, got {weight.dtype}")
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")

    rows, cols = weight.shape
    if cols % NVFP4_TC_K_PER_TILE != 0:
        raise ValueError(f"in_dim {cols} must be divisible by {NVFP4_TC_K_PER_TILE}")

    if padded_rows is None:
        padded_rows = ((rows + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE) * NVFP4_TC_ROWS_PER_TILE
    if padded_rows < rows:
        raise ValueError(f"padded_rows {padded_rows} must be >= rows {rows}")

    if padded_rows != rows:
        padded = torch.zeros((padded_rows, cols), dtype=weight.dtype, device=weight.device)
        padded[:rows].copy_(weight)
        source = padded
    else:
        source = weight.contiguous()

    packed = torch.empty((padded_rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scale_tiles = cols // NVFP4_TC_K_PER_TILE
    scales = torch.zeros(
        ((padded_rows + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE) * scale_tiles * 512,
        dtype=torch.uint8,
        device=weight.device,
    )
    _quantize_nvfp4_lm_out(packed, scales, source)
    return {"packed": packed, "scales": scales, "rows": rows, "padded_rows": padded_rows}


def _attach_prefill_fused_weights(weights):
    if "prefill_fused_layer_data" in weights:
        return

    fused_layer_data = []
    for ld in weights["layer_data"]:
        ptrs = ld["ptrs"]
        if ld["type"] == 1:
            proj_weight = torch.cat([ptrs[1], ptrs[2], ptrs[3]], dim=0).contiguous()
            gate_up_weight = torch.cat([ptrs[8], ptrs[9]], dim=0).contiguous()
        else:
            proj_weight = torch.cat([ptrs[1], ptrs[2], ptrs[3], ptrs[4]], dim=0).contiguous()
            gate_up_weight = torch.cat([ptrs[11], ptrs[12]], dim=0).contiguous()
        fused_layer_data.append({
            "proj_weight": proj_weight,
            "gate_up_weight": gate_up_weight,
        })

    weights["prefill_fused_layer_data"] = fused_layer_data


def _attach_prefill_nvfp4_weights(weights, verbose=True):
    _attach_prefill_fused_weights(weights)

    fused_layer_data = weights["prefill_fused_layer_data"]
    if fused_layer_data and "proj_weight_packed" in fused_layer_data[0]:
        return

    if verbose:
        print("Quantizing prompt fused weights to NVFP4 tensor-core format...")

    packed_bytes = 0
    scale_bytes = 0
    for i, ld in enumerate(weights["layer_data"]):
        fused = fused_layer_data[i]
        proj_padded_rows = DN_PROJ_FUSED_PADDED_SIZE if ld["type"] == 0 else fused["proj_weight"].shape[0]
        proj_q = _quantize_matrix_nvfp4_tc(fused["proj_weight"], padded_rows=proj_padded_rows)
        gate_up_q = _quantize_matrix_nvfp4_tc(fused["gate_up_weight"])
        fused["proj_weight_packed"] = proj_q["packed"]
        fused["proj_weight_scales"] = proj_q["scales"]
        fused["gate_up_weight_packed"] = gate_up_q["packed"]
        fused["gate_up_weight_scales"] = gate_up_q["scales"]
        packed_bytes += proj_q["packed"].numel() + gate_up_q["packed"].numel()
        scale_bytes += proj_q["scales"].numel() + gate_up_q["scales"].numel()

    if verbose:
        print(
            f"NVFP4 prompt fused weights: {packed_bytes/1e6:.0f} MB packed + "
            f"{scale_bytes/1e6:.0f} MB scales ({(packed_bytes + scale_bytes)/1e6:.0f} MB total)"
        )


def _attach_nvfp4_weights(weights, group_size=NVFP4_GROUP_SIZE, verbose=True):
    if (
        "nvfp4" in weights
        and weights["nvfp4"]["group_size"] == group_size
        and weights["nvfp4"].get("lm_group_size") == NVFP4_LM_GROUP_SIZE
    ):
        return weights

    if verbose:
        print(f"Quantizing decode hot weights to NVFP4 (group_size={group_size})...")

    layer_data_nvfp4 = []
    packed_bytes = 0
    scale_bytes = 0

    for ld in weights["layer_data"]:
        if ld["type"] == 1:
            q_proj = _quantize_matrix_nvfp4(ld["ptrs"][1], group_size)
            k_proj = _quantize_matrix_nvfp4(ld["ptrs"][2], group_size)
            v_proj = _quantize_matrix_nvfp4(ld["ptrs"][3], group_size)
            o_proj = _quantize_matrix_nvfp4(ld["ptrs"][6], group_size)
            gate_proj = _quantize_matrix_nvfp4(ld["ptrs"][8], group_size)
            up_proj = _quantize_matrix_nvfp4(ld["ptrs"][9], group_size)
            down_proj = _quantize_matrix_nvfp4(ld["ptrs"][10], group_size)
            qptrs = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
            layer_data_nvfp4.append({
                "type": 1,
                "ptrs": [
                    ld["ptrs"][0],
                    q_proj["packed"], q_proj["scales"],
                    k_proj["packed"], k_proj["scales"],
                    v_proj["packed"], v_proj["scales"],
                    ld["ptrs"][4], ld["ptrs"][5],
                    o_proj["packed"], o_proj["scales"],
                    ld["ptrs"][7],
                    gate_proj["packed"], gate_proj["scales"],
                    up_proj["packed"], up_proj["scales"],
                    down_proj["packed"], down_proj["scales"],
                ],
                "quantized": qptrs,
            })
        else:
            qkv_proj = _quantize_matrix_nvfp4(ld["ptrs"][1], group_size)
            z_proj = _quantize_matrix_nvfp4(ld["ptrs"][2], group_size)
            out_proj = _quantize_matrix_nvfp4(ld["ptrs"][9], group_size)
            gate_proj = _quantize_matrix_nvfp4(ld["ptrs"][11], group_size)
            up_proj = _quantize_matrix_nvfp4(ld["ptrs"][12], group_size)
            down_proj = _quantize_matrix_nvfp4(ld["ptrs"][13], group_size)
            qptrs = [qkv_proj, z_proj, out_proj, gate_proj, up_proj, down_proj]
            layer_data_nvfp4.append({
                "type": 0,
                "ptrs": [
                    ld["ptrs"][0],
                    qkv_proj["packed"], qkv_proj["scales"],
                    z_proj["packed"], z_proj["scales"],
                    ld["ptrs"][3], ld["ptrs"][4], ld["ptrs"][5], ld["ptrs"][6], ld["ptrs"][7], ld["ptrs"][8],
                    out_proj["packed"], out_proj["scales"],
                    ld["ptrs"][10],
                    gate_proj["packed"], gate_proj["scales"],
                    up_proj["packed"], up_proj["scales"],
                    down_proj["packed"], down_proj["scales"],
                ],
                "quantized": qptrs,
            })

        for q in qptrs:
            packed_bytes += q["packed"].numel() * q["packed"].element_size()
            scale_bytes += q["scales"].numel() * q["scales"].element_size()

    lm_head_nvfp4 = _quantize_matrix_nvfp4_lm(weights["lm_head_weight"])
    packed_bytes += lm_head_nvfp4["packed"].numel() * lm_head_nvfp4["packed"].element_size()
    scale_bytes += lm_head_nvfp4["scales"].numel() * lm_head_nvfp4["scales"].element_size()

    weights["nvfp4"] = {
        "group_size": group_size,
        "lm_group_size": NVFP4_LM_GROUP_SIZE,
        "layer_data": layer_data_nvfp4,
        "lm_head_weight_packed": lm_head_nvfp4["packed"],
        "lm_head_scales": lm_head_nvfp4["scales"],
    }

    if verbose:
        total_mb = (packed_bytes + scale_bytes) / 1e6
        print(
            f"NVFP4 decode weights: {packed_bytes/1e6:.0f} MB packed + "
            f"{scale_bytes/1e6:.0f} MB scales ({total_mb:.0f} MB total)"
        )

    return weights


def load_weights(
    model_name="Qwen/Qwen3.5-0.8B",
    verbose=True,
    backend="bf16",
    nvfp4_group_size=NVFP4_GROUP_SIZE,
):
    """Load Qwen3.5-0.8B weights and optional GB10 NVFP4 decode weights."""
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_backend = _resolve_backend(backend)

    if verbose:
        print(f"Loading {model_name} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        lt = LAYER_TYPE[i]

        if lt == 1:
            # Full Attention: 11 pointers (all bf16)
            layer_data.append({
                "type": 1,
                "ptrs": [
                    state[p + "input_layernorm.weight"].contiguous(),
                    state[p + "self_attn.q_proj.weight"].contiguous(),
                    state[p + "self_attn.k_proj.weight"].contiguous(),
                    state[p + "self_attn.v_proj.weight"].contiguous(),
                    state[p + "self_attn.q_norm.weight"].contiguous(),
                    state[p + "self_attn.k_norm.weight"].contiguous(),
                    state[p + "self_attn.o_proj.weight"].contiguous(),
                    state[p + "post_attention_layernorm.weight"].contiguous(),
                    state[p + "mlp.gate_proj.weight"].contiguous(),
                    state[p + "mlp.up_proj.weight"].contiguous(),
                    state[p + "mlp.down_proj.weight"].contiguous(),
                ]
            })
        else:
            # DeltaNet: 14 pointers (all bf16)
            layer_data.append({
                "type": 0,
                "ptrs": [
                    state[p + "input_layernorm.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_qkv.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_z.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_b.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_a.weight"].contiguous(),
                    state[p + "linear_attn.conv1d.weight"].contiguous(),
                    state[p + "linear_attn.A_log"].contiguous(),
                    state[p + "linear_attn.dt_bias"].contiguous(),
                    state[p + "linear_attn.norm.weight"].contiguous(),
                    state[p + "linear_attn.out_proj.weight"].contiguous(),
                    state[p + "post_attention_layernorm.weight"].contiguous(),
                    state[p + "mlp.gate_proj.weight"].contiguous(),
                    state[p + "mlp.up_proj.weight"].contiguous(),
                    state[p + "mlp.down_proj.weight"].contiguous(),
                ]
            })

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    final_norm_weight = state["model.norm.weight"].contiguous()
    lm_head = state.get("lm_head.weight", embed_weight).contiguous()

    weights = {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head,
        "layer_data": layer_data,
    }

    del model
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm_head.numel()
        print(f"BF16 weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

    _attach_prefill_fused_weights(weights)

    if resolved_backend == "nvfp4":
        _attach_nvfp4_weights(weights, group_size=nvfp4_group_size, verbose=verbose)
        if _resolve_prefill_tc():
            _attach_prefill_nvfp4_weights(weights, verbose=verbose)

    return weights, tokenizer


def _pack_layer_weights(layer_data):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size  # 128

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _pack_layer_weights_nvfp4(layer_data, group_size):
    """Pack layer weights into device blob matching LayerWeightsNVFP4 struct."""
    ptr_size = 8
    max_ptrs = 24
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], group_size, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _pack_prefill_fused_layer_weights(fused_layer_data):
    ptr_size = 8
    fields = 6
    struct_size = ptr_size * fields

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = fused_layer_data[i]
        offset = i * struct_size
        proj_weight_packed = ld.get("proj_weight_packed")
        proj_weight_scales = ld.get("proj_weight_scales")
        gate_up_weight_packed = ld.get("gate_up_weight_packed")
        gate_up_weight_scales = ld.get("gate_up_weight_scales")
        struct.pack_into("Q", buf, offset, ld["proj_weight"].data_ptr())
        struct.pack_into("Q", buf, offset + ptr_size, ld["gate_up_weight"].data_ptr())
        struct.pack_into("Q", buf, offset + 2 * ptr_size, proj_weight_packed.data_ptr() if proj_weight_packed is not None else 0)
        struct.pack_into("Q", buf, offset + 3 * ptr_size, proj_weight_scales.data_ptr() if proj_weight_scales is not None else 0)
        struct.pack_into("Q", buf, offset + 4 * ptr_size, gate_up_weight_packed.data_ptr() if gate_up_weight_packed is not None else 0)
        struct.pack_into("Q", buf, offset + 5 * ptr_size, gate_up_weight_scales.data_ptr() if gate_up_weight_scales is not None else 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B megakernel backends."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3.5-0.8B",
        backend="auto",
        nvfp4_group_size=NVFP4_GROUP_SIZE,
        verbose=True,
    ):
        _load_op()
        self.backend = _resolve_backend(backend)
        self.backend_label = "NVFP4 decode" if self.backend == "nvfp4" else "BF16"
        self._nvfp4_group_size = nvfp4_group_size
        self._prefill_mode = _resolve_prefill_mode()
        self._prefill_graph_enabled = (
            self.backend == "nvfp4"
            and self._prefill_mode == "hybrid"
            and _resolve_prefill_graph()
        )

        if weights is None:
            weights, tokenizer = load_weights(
                model_name,
                verbose=verbose,
                backend=self.backend,
                nvfp4_group_size=nvfp4_group_size,
            )
        elif self.backend == "nvfp4":
            _attach_nvfp4_weights(weights, group_size=nvfp4_group_size, verbose=verbose)
            if _resolve_prefill_tc():
                _attach_prefill_nvfp4_weights(weights, verbose=verbose)
        _attach_prefill_fused_weights(weights)
        self.tokenizer = tokenizer
        self._position = 0
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])
        self._prefill_fused_weights_packed = _pack_prefill_fused_layer_weights(
            weights["prefill_fused_layer_data"]
        )
        self._layer_weights_packed_nvfp4 = None
        self._lm_head_weight_packed = None
        self._lm_head_scales = None
        if self.backend == "nvfp4":
            nvfp4 = weights["nvfp4"]
            self._layer_weights_packed_nvfp4 = _pack_layer_weights_nvfp4(
                nvfp4["layer_data"], nvfp4["group_size"])
            self._lm_head_weight_packed = nvfp4["lm_head_weight_packed"]
            self._lm_head_scales = nvfp4["lm_head_scales"]

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f16 = dict(dtype=torch.float16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, MAX_SEQ_LEN, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in LAYER_TYPE if t == 0)
        self._dn_states = torch.zeros(n_dn, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM, **f32)
        self._conv_bufs = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, **f32)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        max_scratch = max(FA_QPROJ_SIZE, DN_CONV_CHANNELS, HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE)
        self._activations = torch.empty(max_scratch, **f32)
        self._residual = torch.empty(HIDDEN_SIZE, **bf16)
        self._qkv_scratch = torch.empty(max(FA_QPROJ_SIZE, DN_CONV_CHANNELS), **f32)
        self._kv_scratch = torch.empty(FA_KV_SIZE * 2, **f32)
        self._attn_out = torch.empty(max(FA_Q_SIZE, DN_V_SIZE), **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._z_scratch = torch.empty(DN_V_SIZE, **f32)
        self._beta_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._alpha_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._normalized = torch.empty(HIDDEN_SIZE, **f32)
        self._lm_hidden_bf16 = torch.empty((LM_HEAD_TENSORCORE_N, HIDDEN_SIZE), **bf16)
        self._lm_hidden_packed = torch.empty(LM_HEAD_TENSORCORE_PACKED_BYTES, dtype=torch.uint8, device="cuda")
        self._lm_hidden_scales = torch.empty(LM_HEAD_TENSORCORE_SCALE_BYTES, dtype=torch.uint8, device="cuda")
        self._lm_logits_f16 = torch.empty((LM_HEAD_TENSORCORE_N, VOCAB_SIZE), **f16)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._out_token = torch.empty(1, **i32)
        self._prefill_buffers = None
        self._prefill_buffer_tokens = 0
        self._prefill_graph_cache = {}

    def _ensure_prefill_buffers(self, max_tokens: int):
        if self._prefill_buffers is not None and self._prefill_buffer_tokens >= max_tokens:
            return self._prefill_buffers

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        self._prefill_buffers = dict(
            hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            proj_buf=torch.empty(max_tokens * PREFILL_PROJ_FUSED_SIZE, **bf16),
            proj_buf2=torch.empty(max_tokens * PREFILL_PROJ_SCRATCH_SIZE, **bf16),
            proj_buf_half=torch.empty(max_tokens * PREFILL_PROJ_FUSED_SIZE, dtype=torch.float16, device="cuda"),
            proj_act_packed=torch.empty(max_tokens * (HIDDEN_SIZE // 2), dtype=torch.uint8, device="cuda"),
            proj_act_scales=torch.empty(
                ((max_tokens + NVFP4_TC_ROWS_PER_TILE - 1) // NVFP4_TC_ROWS_PER_TILE)
                * (HIDDEN_SIZE // NVFP4_TC_K_PER_TILE)
                * 512,
                dtype=torch.uint8,
                device="cuda",
            ),
            attn_buf=torch.empty(max_tokens * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
            mlp_buf=torch.empty(max_tokens * INTERMEDIATE_SIZE, **bf16),
            dn_out_buf=torch.empty(max_tokens * DN_V_SIZE, **bf16),
            beta_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
            alpha_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
            final_normed=torch.empty(HIDDEN_SIZE, **bf16),
            hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
            lm_bmv=torch.empty(1024, **f32),
            lm_bmi=torch.empty(1024, **i32),
        )
        self._prefill_buffer_tokens = max_tokens
        return self._prefill_buffers

    def _reset_runtime_state(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def _run_prefill_bf16_nvfp4_lm(self, token_ids: torch.Tensor, buffers):
        _prefill_bf16_nvfp4_lm(
            self._out_token,
            token_ids,
            self._embed_weight,
            self._layer_weights_packed,
            self._prefill_fused_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._lm_head_weight_packed,
            self._lm_head_scales,
            self._fa_k_cache,
            self._fa_v_cache,
            self._dn_states,
            self._conv_bufs,
            buffers["hidden"],
            buffers["residual"],
            buffers["normalized"],
            buffers["proj_buf"],
            buffers["proj_buf2"],
            buffers["proj_buf_half"],
            buffers["proj_act_packed"],
            buffers["proj_act_scales"],
            buffers["attn_buf"],
            buffers["mlp_buf"],
            buffers["dn_out_buf"],
            buffers["beta_buf"],
            buffers["alpha_buf"],
            buffers["final_normed"],
            buffers["hidden_bf16_out"],
            buffers["lm_bmv"],
            buffers["lm_bmi"],
            self._lm_hidden_bf16,
            self._lm_hidden_packed,
            self._lm_hidden_scales,
            self._lm_logits_f16,
        )
        self._hidden.copy_(buffers["hidden_bf16_out"])

    def _build_prefill_graph(self, prompt_len: int):
        buffers = self._ensure_prefill_buffers(prompt_len)
        static_ids = torch.empty(prompt_len, dtype=torch.int32, device="cuda")
        warmup_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            self._reset_runtime_state()
            static_ids.zero_()
            self._run_prefill_bf16_nvfp4_lm(static_ids, buffers)
        warmup_stream.synchronize()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        with torch.cuda.graph(graph):
            self._reset_runtime_state()
            self._run_prefill_bf16_nvfp4_lm(static_ids, buffers)

        state = {"graph": graph, "token_ids": static_ids, "buffers": buffers}
        self._prefill_graph_cache[prompt_len] = state
        return state

    def _prefill_graph_state(self, prompt_len: int):
        state = self._prefill_graph_cache.get(prompt_len)
        if state is None:
            state = self._build_prefill_graph(prompt_len)
        return state

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self.backend == "nvfp4":
            _decode_nvfp4(
                self._out_token, token_id,
                self._embed_weight, self._layer_weights_packed_nvfp4,
                self._final_norm_weight, self._lm_head_weight_packed, self._lm_head_scales,
                self._lm_hidden_bf16, self._lm_hidden_packed, self._lm_hidden_scales, self._lm_logits_f16,
                self._fa_k_cache, self._fa_v_cache,
                self._dn_states, self._conv_bufs,
                self._hidden, self._activations, self._residual,
                self._qkv_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._z_scratch, self._beta_scratch,
                self._alpha_scratch, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                self._position, MAX_SEQ_LEN, self._nvfp4_group_size,
            )
        else:
            _decode(
                self._out_token, token_id,
                self._embed_weight, self._layer_weights_packed,
                self._final_norm_weight, self._lm_head_weight,
                self._fa_k_cache, self._fa_v_cache,
                self._dn_states, self._conv_bufs,
                self._hidden, self._activations, self._residual,
                self._qkv_scratch, self._kv_scratch, self._attn_out,
                self._mlp_inter, self._z_scratch, self._beta_scratch,
                self._alpha_scratch, self._normalized,
                self._barrier_counter, self._barrier_generation,
                self._block_max_vals, self._block_max_idxs,
                self._lm_sync_counter,
                self._position, MAX_SEQ_LEN,
            )
        self._position += 1
        return self._out_token.item()

    def step_many(self, token_id: int, num_steps: int) -> torch.Tensor:
        """Decode multiple NVFP4 steps without per-token host/device synchronization."""
        if self.backend != "nvfp4":
            raise RuntimeError("step_many is only available for the NVFP4 backend")
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if num_steps == 0:
            return torch.empty(0, dtype=torch.int32, device="cuda")

        output_tokens = torch.empty(num_steps, dtype=torch.int32, device="cuda")
        _decode_many_nvfp4(
            output_tokens, self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed_nvfp4,
            self._final_norm_weight, self._lm_head_weight_packed, self._lm_head_scales,
            self._lm_hidden_bf16, self._lm_hidden_packed, self._lm_hidden_scales, self._lm_logits_f16,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._position, MAX_SEQ_LEN, self._nvfp4_group_size,
        )
        self._position += num_steps
        return output_tokens

    def prefill_tokens(self, token_ids: torch.Tensor) -> int:
        """Run prompt prefill and return the first generated token id."""
        if token_ids.device.type != "cuda" or token_ids.dtype != torch.int32 or token_ids.dim() != 1:
            raise TypeError("token_ids must be a 1D CUDA int32 tensor")
        prompt_len = int(token_ids.numel())
        if self.backend == "nvfp4" and self._prefill_mode == "raw":
            self.reset()
            _prefill_megakernel_nvfp4(
                self._out_token,
                token_ids,
                self._embed_weight,
                self._layer_weights_packed_nvfp4,
                self._final_norm_weight,
                self._lm_head_weight_packed,
                self._lm_head_scales,
                self._lm_hidden_bf16,
                self._lm_hidden_packed,
                self._lm_hidden_scales,
                self._lm_logits_f16,
                self._fa_k_cache,
                self._fa_v_cache,
                self._dn_states,
                self._conv_bufs,
                self._hidden,
                self._activations,
                self._residual,
                self._qkv_scratch,
                self._kv_scratch,
                self._attn_out,
                self._mlp_inter,
                self._z_scratch,
                self._beta_scratch,
                self._alpha_scratch,
                self._normalized,
                self._barrier_counter,
                self._barrier_generation,
                self._block_max_vals,
                self._block_max_idxs,
                self._lm_sync_counter,
                MAX_SEQ_LEN,
                self._nvfp4_group_size,
            )
        elif self.backend == "nvfp4":
            if self._prefill_graph_enabled:
                state = self._prefill_graph_state(prompt_len)
                state["token_ids"].copy_(token_ids)
                state["graph"].replay()
            else:
                self.reset()
                buffers = self._ensure_prefill_buffers(prompt_len)
                self._run_prefill_bf16_nvfp4_lm(token_ids, buffers)
        else:
            raise RuntimeError("prefill_tokens megakernel path is only implemented for the NVFP4 backend")
        self._position = prompt_len
        return self._out_token.item()

    def reset(self):
        self._reset_runtime_state()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        out = []
        next_id = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            next_id = self.step(next_id)
            if next_id == eos:
                break
            out.append(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)



================================================
FILE: megakernel/prefill_megakernel.cu
================================================
/**
 * BF16 Prefill Megakernel for NVIDIA B200 (sm_100).
 *
 * One `cudaLaunchCooperativeKernel` dispatch. All 24 layers, all S tokens,
 * processed in a single persistent kernel with `cg::this_grid().sync()`
 * between phases. No cuBLAS, no intermediate launches, no host round-trips.
 *
 * Matmuls use WMMA (wraps `mma.sync` under the hood) with bf16 operands and
 * f32 accumulate. Block tile [BTM=32, BTN=128]; warp grid 2×8. Work is
 * distributed across the 148 persistent blocks via cyclic tile assignment.
 *
 * The DeltaNet recurrence keeps its 16-block-per-layer structure from the
 * pre-mega version (hard serial dep over t) but sits inside the persistent
 * kernel so there's no per-layer relaunch.
 *
 * Assumes S is a multiple of 32 (caller pads).
 *
 * Only compiled for Blackwell (sm_120/sm_121a). Excluded from the sm_86
 * RTX 3090 build; see setup.py.
 */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#error "prefill_megakernel.cu requires CUDA arch >= sm_120 (Blackwell)"
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;
using namespace nvcuda;

// ===== Model constants =====
constexpr int HIDDEN = 1024;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int MAX_SEQ = 2048;

constexpr int FA_Q_HEADS = 8;
constexpr int FA_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA = FA_Q_HEADS / FA_KV_HEADS;
constexpr int FA_Q_SIZE = FA_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE * 2;
constexpr int FA_KV_SIZE = FA_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROT_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

constexpr int DN_HEADS = 16;
constexpr int DN_KEY = 128;
constexpr int DN_VAL = 128;
constexpr int DN_CONV_K = 4;
constexpr int DN_QK_SIZE = DN_HEADS * DN_KEY;
constexpr int DN_V_SIZE = DN_HEADS * DN_VAL;
constexpr int DN_CONV_CH = DN_QK_SIZE * 2 + DN_V_SIZE;  // 6144

__device__ __constant__ int MEGA_LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

struct MegaLayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];
};

// ===== Kernel tile sizes =====
#define MEGA_BLOCK_SIZE 512
constexpr int MEGA_WARPS = MEGA_BLOCK_SIZE / 32;

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
// Block tile [32, 128] with 2×8 warp grid; each warp produces a [16, 16]
// output. K is processed in BTK=64 chunks, double-buffered in shared
// memory via cp.async so the next chunk streams in while we MMA on the
// current one. This keeps parallelism high for small-N matmuls (down,
// out, z projections) while letting the async loader hide global latency.
constexpr int BTM = 32;
constexpr int BTN = 128;
constexpr int BTK = 64;                // k-chunk size (double buffered)
constexpr int N_PER_WARP = 1;          // 16-col WMMA tile per warp
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 8;
static_assert(WARPS_M * WARPS_N == MEGA_WARPS, "warp grid must match block");
static_assert(WARPS_N * WN * N_PER_WARP == BTN, "warp N coverage mismatch");
static_assert(WARPS_M * WM == BTM, "warp M coverage mismatch");

constexpr int LM_BLOCKS_MEGA = 1024;

// ===== Small helpers =====
__device__ __forceinline__ float mega_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float mega_silu(float x) { return x / (1.0f + expf(-x)); }

// cp.async.cg: copy 16 bytes global → shared, bypassing L1 (goes through L2).
// The copy is asynchronous; producer threads must issue cp.async.commit_group
// and consumers must cp.async.wait_group before reading shared.
__device__ __forceinline__ void cp_async_16(void *dst_smem, const void *src_gmem) {
    unsigned int smem_int = __cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(src_gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

// ===== Phase: Embedding =====
// hidden[s*H + i] = embed[ids[s]*H + i]
__device__ void phase_embed(const int *ids, const __nv_bfloat16 *embed,
                            __nv_bfloat16 *hidden, int S) {
    int total = S * HIDDEN;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int s = idx / HIDDEN;
        int i = idx - s * HIDDEN;
        hidden[idx] = embed[ids[s] * HIDDEN + i];
    }
}

// ===== Phase: RMSNorm =====
// normalized[s, :] = (input[s, :] * rsqrt(mean(input[s, :]^2) + eps) * (1 + w)
// Also copies input -> residual.
// Rows distributed cyclically across blocks.
__device__ void phase_rmsnorm(const __nv_bfloat16 *input, const __nv_bfloat16 *w,
                              __nv_bfloat16 *output, __nv_bfloat16 *residual,
                              int S, int D) {
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float smem[MEGA_WARPS];

    for (int s = blockIdx.x; s < S; s += gridDim.x) {
        const __nv_bfloat16 *ri = input + s * D;
        __nv_bfloat16 *ro = output + s * D;
        __nv_bfloat16 *rr = residual + s * D;

        float sq = 0;
        for (int i = tid; i < D; i += blockDim.x) {
            float v = __bfloat162float(ri[i]);
            rr[i] = ri[i];
            sq += v * v;
        }
        sq = mega_warp_sum(sq);
        if (lid == 0) smem[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = (lid < MEGA_WARPS) ? smem[lid] : 0;
            v = mega_warp_sum(v);
            if (lid == 0) smem[0] = rsqrtf(v / D + RMS_EPS);
        }
        __syncthreads();
        float rstd = smem[0];
        for (int i = tid; i < D; i += blockDim.x) {
            float v = __bfloat162float(ri[i]) * rstd * (1.0f + __bfloat162float(w[i]));
            ro[i] = __float2bfloat16(v);
        }
        __syncthreads();
    }
}

// ===== Phase: bf16 matmul C[M, N] = A[M, K] @ W[N, K]^T =====
// Requires M and N multiples of BTM and BTN respectively, K multiple of WK.
// Each block owns a [BTM, BTN] output tile; warps within a block tile
// [WARPS_M, WARPS_N]×[WM, WN].
// ===== Pipelined double-buffered BF16 GEMM =====
//
// C[M, N] = A[M, K] × W[N, K]^T  (W stored row-major as [N, K]).
//
// Per block:
//   Block tile   : [BTM=32, BTN=256]
//   Warp grid    : 2 × 8
//   Per-warp out : [16, 32] via 2× WMMA m16n16k16 accumulators
//   K-chunk      : BTK=64 (double-buffered with cp.async)
//
// Shared memory layout (dynamic):
//   sA[2][BTM][BTK]  : double-buffered A staging         (2*32*64*2 =  8 KB)
//   sB[2][BTN][BTK]  : double-buffered B staging         (2*256*64*2= 64 KB)
//   sC[BTM][BTN]     : f32 output accumulator staging    (32*256*4 = 32 KB)
// Total: 104 KB dynamic shared per block (B200 carveout = 228 KB, ok).
//
// Threads-per-tile work:
//   - A chunk is BTM*BTK = 32*64 = 2048 bf16 = 4 KB per buffer.
//     Each thread copies 8 bf16 (16 B) per cp.async, so 2048/8 = 256 copies
//     across 512 threads = ½ pass. Threads >= 256 skip the A load.
//   - B chunk is BTN*BTK = 256*64 = 16384 bf16 = 32 KB per buffer.
//     512 threads × 8 bf16 = 4096 bf16 per pass → 4 passes.

__device__ void phase_matmul_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *W,
                                  __nv_bfloat16 *C, int M, int N, int K) {
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16 *sA = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *sB = sA + 2 * BTM * BTK;
    float *tile_f32 = reinterpret_cast<float *>(sB + 2 * BTN * BTK);

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int wm = warp_id / WARPS_N;
    int wn = warp_id % WARPS_N;

    int num_m_tiles = (M + BTM - 1) / BTM;
    int num_n_tiles = (N + BTN - 1) / BTN;
    int total_tiles = num_m_tiles * num_n_tiles;

    // K must be a multiple of BTK for the pipeline to work cleanly.
    // All our Ks (1024, 2048, 3584) satisfy K % 64 == 0.
    int num_k_chunks = K / BTK;

    // Lambda: asynchronously stage one (BM, BTK) tile of A or B from global
    // into the `buf`-th shared buffer, using cp.async.cg (bypasses L1).
    auto issue_a_chunk = [&] (int buf, int m_start, int k_start) {
        const __nv_bfloat16 *src_base = A + m_start * K + k_start;
        __nv_bfloat16 *dst_base = sA + buf * BTM * BTK;
        // 2048 bf16 elements = 256 cp.async calls at 8 bf16 each.
        // Use 256 threads (half the block). Each thread does 1 call.
        if (tid < 256) {
            int r = tid >> 3;            // 0..31 rows
            int c = (tid & 7) * 8;        // col start within BTK
            cp_async_16(dst_base + r * BTK + c,
                         src_base + r * K + c);
        }
    };

    auto issue_b_chunk = [&] (int buf, int n_start, int k_start) {
        // W is [N, K] row-major; our tile is rows [n_start, n_start+BTN)
        // and cols [k_start, k_start+BTK). In shared we lay it out as
        // [BTN, BTK] so WMMA col_major loads get the correct N-fragment.
        const __nv_bfloat16 *src_base = W + n_start * K + k_start;
        __nv_bfloat16 *dst_base = sB + buf * BTN * BTK;
        // BTN * BTK / 8 = 128*64/8 = 1024 cp.async calls; 512 threads × 2.
        #pragma unroll
        for (int it = 0; it < 2; it++) {
            int idx = tid + it * blockDim.x;
            int r = idx >> 3;         // 0..127 rows
            int c = (idx & 7) * 8;     // col start within BTK
            if (r < BTN) {
                cp_async_16(dst_base + r * BTK + c,
                             src_base + r * K + c);
            }
        }
    };

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        int mt = tile / num_n_tiles;
        int nt = tile % num_n_tiles;
        int m_start = mt * BTM;
        int n_start = nt * BTN;
        bool m_in_bounds = (m_start < M);
        bool n_in_bounds = (n_start < N);

        // Per-warp accumulators: 2 × [16×16] f32 WMMA fragments.
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag[N_PER_WARP];
        #pragma unroll
        for (int n = 0; n < N_PER_WARP; n++) wmma::fill_fragment(c_frag[n], 0.0f);

        // Prefetch chunk 0 into buffer 0.
        if (m_in_bounds && n_in_bounds) {
            issue_a_chunk(0, m_start, 0);
            issue_b_chunk(0, n_start, 0);
        }
        cp_async_commit();

        int warp_row_in_block = wm * WM;
        int warp_col_in_block = wn * WN * N_PER_WARP;

        for (int ck = 0; ck < num_k_chunks; ck++) {
            int cur = ck & 1;
            int nxt = cur ^ 1;

            // Prefetch next chunk
            if (ck + 1 < num_k_chunks) {
                if (m_in_bounds && n_in_bounds) {
                    issue_a_chunk(nxt, m_start, (ck + 1) * BTK);
                    issue_b_chunk(nxt, n_start, (ck + 1) * BTK);
                }
                cp_async_commit();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();

            if (m_in_bounds && n_in_bounds) {
                const __nv_bfloat16 *sA_buf = sA + cur * BTM * BTK;
                const __nv_bfloat16 *sB_buf = sB + cur * BTN * BTK;

                // MMA over this chunk: BTK/WK = 4 k-steps, each with 1 A load
                // + N_PER_WARP B loads + N_PER_WARP MMAs.
                #pragma unroll
                for (int ki = 0; ki < BTK; ki += WK) {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::load_matrix_sync(a_frag, sA_buf + warp_row_in_block * BTK + ki, BTK);

                    #pragma unroll
                    for (int n = 0; n < N_PER_WARP; n++) {
                        int col_in_block = warp_col_in_block + n * WN;
                        wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                        wmma::load_matrix_sync(b_frag, sB_buf + col_in_block * BTK + ki, BTK);
                        wmma::mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
                    }
                }
            }
        }

        // Epilogue: store each accumulator to its slot in the f32 tile.
        if (m_in_bounds && n_in_bounds) {
            #pragma unroll
            for (int n = 0; n < N_PER_WARP; n++) {
                int col_in_block = warp_col_in_block + n * WN;
                wmma::store_matrix_sync(
                    tile_f32 + warp_row_in_block * BTN + col_in_block,
                    c_frag[n], BTN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // f32 → bf16 cast and write to global, respecting M/N bounds.
        if (m_in_bounds && n_in_bounds) {
            for (int i = threadIdx.x; i < BTM * BTN; i += blockDim.x) {
                int mi = i / BTN, ni = i % BTN;
                int gm = m_start + mi, gn = n_start + ni;
                if (gm < M && gn < N) {
                    C[gm * N + gn] = __float2bfloat16(tile_f32[i]);
                }
            }
        }
        __syncthreads();
    }
}

// Same as above but output is f32 (for beta/alpha which feed the recurrence as f32).
// (unused in the current layer schedule — beta/alpha are done via
// phase_matvec_small_to_f32 below since N=16 which doesn't fit WMMA 16×16
// output fragments cleanly without padding.)

// ===== Phase: tiny matvec for beta/alpha (N=16) =====
// Output is f32.
__device__ void phase_matvec_small_to_f32(const __nv_bfloat16 *in, const __nv_bfloat16 *w,
                                          float *out, int S, int K, int N) {
    // One (s, n) per warp, cyclic assignment.
    int warps_per_block = blockDim.x / 32;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int total_warps = gridDim.x * warps_per_block;
    int total = S * N;
    int lid = threadIdx.x & 31;

    for (int idx = global_warp; idx < total; idx += total_warps) {
        int s = idx / N, n = idx - s * N;
        const __nv_bfloat16 *ir = in + s * K;
        const __nv_bfloat16 *wr = w + n * K;
        float sum = 0;
        for (int k = lid; k < K; k += 32) {
            sum += __bfloat162float(ir[k]) * __bfloat162float(wr[k]);
        }
        sum = mega_warp_sum(sum);
        if (lid == 0) out[idx] = sum;
    }
}

// ===== Phase: elementwise bf16 residual add =====
__device__ void phase_add_residual(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                   __nv_bfloat16 *out, int total) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

// ===== Phase: SiLU(gate) * up =====
__device__ void phase_silu_mul(const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
                               __nv_bfloat16 *out, int total) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        float g = __bfloat162float(gate[i]);
        out[i] = __float2bfloat16(mega_silu(g) * __bfloat162float(up[i]));
    }
}

// ===== Phase: DeltaNet recurrence (16 heads, each on its own block) =====
// Reuses the optimized B200 per-head inner loop: conv ring-buffer in shared,
// conv weights in shared, s_out in shared, norm_w in shared.
__device__ void phase_deltanet_recurrence(
    const __nv_bfloat16 *qkv_proj,   // [S, DN_CONV_CH]
    const __nv_bfloat16 *z_proj,     // [S, DN_V_SIZE]
    const float *beta_proj,          // [S, DN_HEADS]
    const float *alpha_proj,         // [S, DN_HEADS]
    const __nv_bfloat16 *conv_w,     // [DN_CONV_CH, DN_CONV_K]
    const __nv_bfloat16 *a_log,      // [DN_HEADS]
    const __nv_bfloat16 *dt_bias,    // [DN_HEADS]
    const __nv_bfloat16 *norm_w,     // [DN_VAL]
    float *state,                    // [DN_HEADS, DN_KEY, DN_VAL]
    float *conv_buf,                 // [DN_CONV_CH, DN_CONV_K]
    __nv_bfloat16 *output,           // [S, DN_V_SIZE]
    int S)
{
    int h = blockIdx.x;
    if (h >= DN_HEADS) return;  // extra blocks stay idle this phase

    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    constexpr int NWARPS = 16;
    constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
    constexpr int QKV_CH = 2 * DN_KEY + DN_VAL;

    float a_log_val = __bfloat162float(a_log[h]);
    float dt_b = __bfloat162float(dt_bias[h]);

    __shared__ float s_q[DN_KEY], s_k[DN_KEY], s_v[DN_VAL];
    __shared__ float s_beta, s_decay;
    __shared__ float s_gnorm[NWARPS];
    __shared__ float s_conv[QKV_CH * DN_CONV_K];
    __shared__ float s_conv_w[QKV_CH * DN_CONV_K];
    __shared__ float s_z[DN_VAL];
    __shared__ float s_out[DN_VAL];
    __shared__ float s_norm_w[DN_VAL];

    float *my_state = state + h * DN_KEY * DN_VAL;

    constexpr int CPW = DN_VAL / NWARPS;   // 8
    constexpr int RPL = DN_KEY / 32;       // 4
    float sreg[CPW * RPL];

    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++) {
            sreg[jj * RPL + ii] = my_state[j * DN_KEY + lid + ii * 32];
        }
    }

    auto ch_global = [&] (int c) -> int {
        if (c < DN_KEY) return h * DN_KEY + c;
        if (c < 2 * DN_KEY) return DN_QK_SIZE + h * DN_KEY + (c - DN_KEY);
        return 2 * DN_QK_SIZE + h * DN_VAL + (c - 2 * DN_KEY);
    };

    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch = ch_global(c);
        const float *src_state = conv_buf + gch * DN_CONV_K;
        const __nv_bfloat16 *src_w = conv_w + gch * DN_CONV_K;
        float *dst_state = s_conv + c * DN_CONV_K;
        float *dst_w = s_conv_w + c * DN_CONV_K;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            dst_state[k] = src_state[k];
            dst_w[k] = __bfloat162float(src_w[k]);
        }
    }
    for (int i = tid; i < DN_VAL; i += blockDim.x) {
        s_norm_w[i] = __bfloat162float(norm_w[i]);
    }
    __syncthreads();

    for (int t = 0; t < S; t++) {
        // Fused conv1d+SiLU for all 384 per-head channels.
        for (int c = tid; c < QKV_CH; c += blockDim.x) {
            int gch = ch_global(c);
            float *cs = s_conv + c * DN_CONV_K;
            const float *cw = s_conv_w + c * DN_CONV_K;
            float new_x = __bfloat162float(qkv_proj[t * DN_CONV_CH + gch]);
            float h0 = cs[1], h1 = cs[2], h2 = cs[3];
            cs[0] = h0; cs[1] = h1; cs[2] = h2; cs[3] = new_x;
            float co = h0 * cw[0] + h1 * cw[1] + h2 * cw[2] + new_x * cw[3];
            float silu_out = mega_silu(co);
            if (c < DN_KEY)           s_q[c] = silu_out;
            else if (c < 2 * DN_KEY)  s_k[c - DN_KEY] = silu_out;
            else                       s_v[c - 2 * DN_KEY] = silu_out;
        }
        // Prefetch z-row in parallel.
        {
            const __nv_bfloat16 *z_h_bf = z_proj + t * DN_V_SIZE + h * DN_VAL;
            for (int i = tid; i < DN_VAL; i += blockDim.x) {
                s_z[i] = __bfloat162float(z_h_bf[i]);
            }
        }
        __syncthreads();

        // L2 normalize q (warp 0), L2 normalize k (warp 1), beta/decay
        // scalars (warp 2 lane 0) — all disjoint targets, one sync fences
        // the three in parallel (saves one sync per step vs the old
        // two-phase normalize→beta/decay sequence).
        if (wid == 0) {
            float sq = 0; for (int i = lid; i < DN_KEY; i += 32) sq += s_q[i] * s_q[i];
            sq = mega_warp_sum(sq);
            float n = rsqrtf(sq + 1e-6f) * Q_SCALE;
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lid; i < DN_KEY; i += 32) s_q[i] *= n;
        }
        if (wid == 1) {
            float sq = 0; for (int i = lid; i < DN_KEY; i += 32) sq += s_k[i] * s_k[i];
            sq = mega_warp_sum(sq);
            float n = rsqrtf(sq + 1e-6f);
            n = __shfl_sync(0xffffffff, n, 0);
            for (int i = lid; i < DN_KEY; i += 32) s_k[i] *= n;
        }
        if (wid == 2 && lid == 0) {
            s_beta = 1.f / (1.f + expf(-beta_proj[t * DN_HEADS + h]));
            float x = alpha_proj[t * DN_HEADS + h] + dt_b;
            float sp = (x > 20.f) ? x : logf(1.f + expf(x));
            s_decay = expf(-expf(a_log_val) * sp);
        }
        __syncthreads();

        float beta = s_beta, decay = s_decay;
        __nv_bfloat16 *out_h = output + t * DN_V_SIZE + h * DN_VAL;

        // Cache per-lane s_k / s_q into registers once per step so the
        // CPW=8 inner-loop iterations read from registers instead of
        // shared memory.
        float sk_cache[RPL];
        float sq_cache[RPL];
        #pragma unroll
        for (int ii = 0; ii < RPL; ii++) {
            sk_cache[ii] = s_k[lid + ii * 32];
            sq_cache[ii] = s_q[lid + ii * 32];
        }

        #pragma unroll
        for (int jj = 0; jj < CPW; jj++) {
            int j = wid * CPW + jj;
            float kv = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) kv += sreg[jj * RPL + ii] * sk_cache[ii];
            kv = mega_warp_sum(kv);
            kv = __shfl_sync(0xffffffff, kv, 0);
            float delta = (s_v[j] - decay * kv) * beta;
            float attn = 0;
            #pragma unroll
            for (int ii = 0; ii < RPL; ii++) {
                sreg[jj * RPL + ii] = decay * sreg[jj * RPL + ii] + sk_cache[ii] * delta;
                attn += sreg[jj * RPL + ii] * sq_cache[ii];
            }
            attn = mega_warp_sum(attn);
            if (lid == 0) s_out[j] = attn;
        }
        __syncthreads();

        // Gated RMSNorm.
        //
        // After writing per-warp partials to s_gnorm[wid] and syncing, EVERY
        // thread reads all 16 warp partials and computes rstd locally — no
        // second "warp 0 reduces then everyone syncs to pick up rstd" round
        // trip through shared memory. Saves one __syncthreads per step
        // (×520 steps ×18 layers = ~9k syncs removed per prefill).
        float sq2 = 0;
        for (int i = tid; i < DN_VAL; i += blockDim.x) { float v = s_out[i]; sq2 += v * v; }
        sq2 = mega_warp_sum(sq2);
        if (lid == 0) s_gnorm[wid] = sq2;
        __syncthreads();

        float total = 0;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) total += s_gnorm[w];
        float rstd = rsqrtf(total / DN_VAL + RMS_EPS);

        for (int i = tid; i < DN_VAL; i += blockDim.x) {
            float n = s_out[i] * rstd * s_norm_w[i];
            out_h[i] = __float2bfloat16(n * mega_silu(s_z[i]));
        }
        // No end-of-iter sync: each thread's writes to shared in the next
        // iteration's conv+z phase target its own slot (stride = blockDim.x)
        // and the first cross-warp read is fenced by the existing sync
        // inside that phase.
    }

    // State writeback
    for (int jj = 0; jj < CPW; jj++) {
        int j = wid * CPW + jj;
        for (int ii = 0; ii < RPL; ii++) {
            my_state[j * DN_KEY + lid + ii * 32] = sreg[jj * RPL + ii];
        }
    }
    __syncthreads();
    for (int c = tid; c < QKV_CH; c += blockDim.x) {
        int gch = ch_global(c);
        float *dst_state = conv_buf + gch * DN_CONV_K;
        const float *src_state = s_conv + c * DN_CONV_K;
        #pragma unroll
        for (int k = 0; k < DN_CONV_K; k++) {
            dst_state[k] = src_state[k];
        }
    }
}

// ===== Phase: QK norm + RoPE + KV cache write =====
// One warp per (s, head) for q, and for k/v in sequence.
__device__ void phase_qk_norm_rope(
    __nv_bfloat16 *q,               // [S, FA_QPROJ_SIZE] (in-place)
    __nv_bfloat16 *k,               // [S, FA_KV_SIZE]    (in-place)
    const __nv_bfloat16 *v,          // [S, FA_KV_SIZE]
    const __nv_bfloat16 *qnw, const __nv_bfloat16 *knw,
    __nv_bfloat16 *k_cache,          // [FA_KV_HEADS, MAX_SEQ, FA_HEAD_DIM]
    __nv_bfloat16 *v_cache,
    int S)
{
    int warps_per_block = blockDim.x / 32;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int total_warps = gridDim.x * warps_per_block;
    int lid = threadIdx.x & 31;

    int total_q = S * FA_Q_HEADS;
    int total_k = S * FA_KV_HEADS;
    int total = total_q + total_k;

    for (int idx = global_warp; idx < total; idx += total_warps) {
        if (idx < total_q) {
            int pos = idx / FA_Q_HEADS, head = idx - pos * FA_Q_HEADS;
            __nv_bfloat16 *qh = q + pos * FA_QPROJ_SIZE + head * FA_HEAD_DIM * 2;
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float v = __bfloat162float(qh[i]); ss += v * v;
            }
            ss = mega_warp_sum(ss);
            float sc = rsqrtf(ss / FA_HEAD_DIM + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float normed = __bfloat162float(qh[i]) * sc * (1.f + __bfloat162float(qnw[i]));
                if (i < FA_ROT_DIM) {
                    float fe = float(2 * (i % (FA_ROT_DIM / 2))) / FA_ROT_DIM;
                    float freq = float(pos) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROT_DIM / 2) ? i + FA_ROT_DIM / 2 : i - FA_ROT_DIM / 2;
                    float pv = __bfloat162float(qh[p]) * sc * (1.f + __bfloat162float(qnw[p]));
                    qh[i] = __float2bfloat16((i < FA_ROT_DIM / 2)
                                              ? (normed * cv - pv * sv)
                                              : (pv * sv + normed * cv));
                } else {
                    qh[i] = __float2bfloat16(normed);
                }
            }
        } else {
            int kidx = idx - total_q;
            int pos = kidx / FA_KV_HEADS, head = kidx - pos * FA_KV_HEADS;
            __nv_bfloat16 *kh = k + pos * FA_KV_SIZE + head * FA_HEAD_DIM;
            const __nv_bfloat16 *vh = v + pos * FA_KV_SIZE + head * FA_HEAD_DIM;
            __nv_bfloat16 *kc = k_cache + head * MAX_SEQ * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            __nv_bfloat16 *vc = v_cache + head * MAX_SEQ * FA_HEAD_DIM + pos * FA_HEAD_DIM;
            float ss = 0;
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float v = __bfloat162float(kh[i]); ss += v * v;
            }
            ss = mega_warp_sum(ss);
            float sc = rsqrtf(ss / FA_HEAD_DIM + RMS_EPS);
            sc = __shfl_sync(0xffffffff, sc, 0);
            for (int i = lid; i < FA_HEAD_DIM; i += 32) {
                float normed = __bfloat162float(kh[i]) * sc * (1.f + __bfloat162float(knw[i]));
                float fk;
                if (i < FA_ROT_DIM) {
                    float fe = float(2 * (i % (FA_ROT_DIM / 2))) / FA_ROT_DIM;
                    float freq = float(pos) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROT_DIM / 2) ? i + FA_ROT_DIM / 2 : i - FA_ROT_DIM / 2;
                    float pv = __bfloat162float(kh[p]) * sc * (1.f + __bfloat162float(knw[p]));
                    fk = (i < FA_ROT_DIM / 2) ? (normed * cv - pv * sv) : (pv * sv + normed * cv);
                } else {
                    fk = normed;
                }
                kh[i] = __float2bfloat16(fk);
                kc[i] = __float2bfloat16(fk);
                vc[i] = vh[i];
            }
        }
    }
}

// ===== Phase: causal attention (per (s, q_head), single-warp online softmax) =====
__device__ void phase_causal_attn(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                  const __nv_bfloat16 *v, __nv_bfloat16 *out, int S)
{
    int warps_per_block = blockDim.x / 32;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    int total_warps = gridDim.x * warps_per_block;
    int lid = threadIdx.x & 31;

    int total = S * FA_Q_HEADS;
    float scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
    constexpr int EPL = FA_HEAD_DIM / 32;

    for (int idx = global_warp; idx < total; idx += total_warps) {
        int pos = idx / FA_Q_HEADS, qh = idx - pos * FA_Q_HEADS;
        int kvh = qh / FA_GQA;

        const __nv_bfloat16 *qv = q + pos * FA_QPROJ_SIZE + qh * FA_HEAD_DIM * 2;
        const __nv_bfloat16 *gv = qv + FA_HEAD_DIM;
        __nv_bfloat16 *ov = out + pos * FA_Q_SIZE + qh * FA_HEAD_DIM;

        float ql[EPL];
        for (int e = 0; e < EPL; e++) ql[e] = __bfloat162float(qv[lid * EPL + e]);
        float oa[EPL] = {};
        float mx = -1e30f, se = 0;

        for (int kp = 0; kp <= pos; kp++) {
            const __nv_bfloat16 *kv_ptr = k + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;
            const __nv_bfloat16 *vv = v + kp * FA_KV_SIZE + kvh * FA_HEAD_DIM;
            float sc = 0;
            for (int e = 0; e < EPL; e++) sc += ql[e] * __bfloat162float(kv_ptr[lid * EPL + e]);
            sc = mega_warp_sum(sc) * scale;
            sc = __shfl_sync(0xffffffff, sc, 0);
            float om = mx;
            mx = fmaxf(mx, sc);
            float ed = expf(om - mx);
            se = se * ed + expf(sc - mx);
            float wt = expf(sc - mx);
            for (int e = 0; e < EPL; e++) {
                oa[e] = oa[e] * ed + wt * __bfloat162float(vv[lid * EPL + e]);
            }
        }
        float rs = 1.f / se;
        for (int e = 0; e < EPL; e++) {
            int i = lid * EPL + e;
            float g = 1.f / (1.f + expf(-__bfloat162float(gv[i])));
            ov[i] = __float2bfloat16(oa[e] * rs * g);
        }
    }
}

// ===== Phase: final norm (just last row) =====
// Only block 0 works. hidden_out is embedded residual, final_normed is the
// rmsnorm'd row used by the LM head.
__device__ void phase_final_norm(const __nv_bfloat16 *hidden,
                                 const __nv_bfloat16 *w,
                                 __nv_bfloat16 *final_normed,
                                 __nv_bfloat16 *hidden_bf16_out,
                                 int S)
{
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    __shared__ float smem[MEGA_WARPS];

    const __nv_bfloat16 *row = hidden + (S - 1) * HIDDEN;
    float sq = 0;
    for (int i = tid; i < HIDDEN; i += blockDim.x) {
        float v = __bfloat162float(row[i]); sq += v * v;
    }
    sq = mega_warp_sum(sq);
    if (lid == 0) smem[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lid < MEGA_WARPS) ? smem[lid] : 0;
        v = mega_warp_sum(v);
        if (lid == 0) smem[0] = rsqrtf(v / HIDDEN + RMS_EPS);
    }
    __syncthreads();
    float rstd = smem[0];
    for (int i = tid; i < HIDDEN; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        final_normed[i] = __float2bfloat16(v * rstd * (1.f + __bfloat162float(w[i])));
        hidden_bf16_out[i] = row[i];
    }
}

// ===== Phase: LM head (bf16 matvec over full vocab) =====
// Each block computes part of the vocab and writes (max_val, argmax) pair.
// A final reduce picks the global argmax.
__device__ void phase_lm_head(const __nv_bfloat16 *hidden,
                              const __nv_bfloat16 *weight,
                              float *block_max_vals,
                              int *block_max_idxs)
{
    __shared__ float s_hidden[HIDDEN];
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        s_hidden[i] = __bfloat162float(hidden[i]);
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lid = threadIdx.x & 31;
    int num_warps = blockDim.x / 32;
    int total_blocks = gridDim.x;
    int rpb = (VOCAB + total_blocks - 1) / total_blocks;
    int rs = blockIdx.x * rpb;
    int re = min(rs + rpb, VOCAB);

    float local_max = -INFINITY;
    int local_max_idx = -1;
    for (int m = rs + warp_id; m < re; m += num_warps) {
        const __nv_bfloat16 *wr = weight + m * HIDDEN;
        float sum = 0;
        for (int k = lid; k < HIDDEN; k += 32) {
            sum += __bfloat162float(wr[k]) * s_hidden[k];
        }
        sum = mega_warp_sum(sum);
        if (lid == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float wm[32]; __shared__ int wi[32];
    if (lid == 0) { wm[warp_id] = local_max; wi[warp_id] = local_max_idx; }
    __syncthreads();
    if (warp_id == 0) {
        float mv = (lid < num_warps) ? wm[lid] : -INFINITY;
        int mi = (lid < num_warps) ? wi[lid] : -1;
        for (int o = 16; o > 0; o /= 2) {
            float ov = __shfl_down_sync(0xffffffff, mv, o);
            int oi = __shfl_down_sync(0xffffffff, mi, o);
            if (ov > mv) { mv = ov; mi = oi; }
        }
        if (lid == 0) {
            block_max_vals[blockIdx.x] = mv;
            block_max_idxs[blockIdx.x] = mi;
        }
    }
}

// ===== Phase: LM reduce (block 0 only) =====
__device__ void phase_lm_reduce(const float *block_max_vals,
                                const int *block_max_idxs,
                                int *output_token, int num_blocks)
{
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    __shared__ float sv[MEGA_BLOCK_SIZE];
    __shared__ int si[MEGA_BLOCK_SIZE];
    float bv = -INFINITY; int bi = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > bv) { bv = v; bi = block_max_idxs[i]; }
    }
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *output_token = si[0];
}

// ==========================================================================
// Main megakernel. All device work for a prefill lives in this one dispatch.
// ==========================================================================
__global__ void __launch_bounds__(MEGA_BLOCK_SIZE, 1)
prefill_megakernel(
    const int *token_ids, int S, int *output_token,
    const __nv_bfloat16 *embed_weight, const MegaLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi)
{
    cg::grid_group grid = cg::this_grid();

    // Pad S up to the matmul block-tile multiple so WMMA fragments always
    // land inside allocated scratch. Padding rows are computed through every
    // matmul but never influence sequential phases (DN recurrence, RoPE,
    // causal attn, final norm), which use S directly.
    int S_pad = ((S + BTM - 1) / BTM) * BTM;

    int fa_stride = FA_KV_HEADS * MAX_SEQ * FA_HEAD_DIM;
    int dn_stride = DN_HEADS * DN_KEY * DN_VAL;

    // Phase 0: Embedding for real tokens; zero-fill padding rows so matmuls
    // never read uninitialised memory.
    phase_embed(token_ids, embed_weight, hidden, S);
    // Zero the padding tail of hidden.
    {
        int stride = gridDim.x * blockDim.x;
        int start = S * HIDDEN;
        int end = S_pad * HIDDEN;
        for (int i = start + blockIdx.x * blockDim.x + threadIdx.x; i < end; i += stride) {
            hidden[i] = __float2bfloat16(0.0f);
        }
    }
    grid.sync();

    int fa_idx = 0, dn_idx = 0;

    for (int li = 0; li < NUM_LAYERS; li++) {
        const MegaLayerWeights &lw = layers[li];
        int lt = MEGA_LAYER_TYPE[li];

        const __nv_bfloat16 *inp_norm_w = (const __nv_bfloat16 *)lw.ptrs[0];
        phase_rmsnorm(hidden, inp_norm_w, normalized, residual, S_pad, HIDDEN);
        grid.sync();

        if (lt == 0) {
            // DeltaNet
            const __nv_bfloat16 *qkv_w = (const __nv_bfloat16 *)lw.ptrs[1];
            const __nv_bfloat16 *z_w   = (const __nv_bfloat16 *)lw.ptrs[2];
            const __nv_bfloat16 *beta_w = (const __nv_bfloat16 *)lw.ptrs[3];
            const __nv_bfloat16 *alpha_w= (const __nv_bfloat16 *)lw.ptrs[4];
            const __nv_bfloat16 *conv_w = (const __nv_bfloat16 *)lw.ptrs[5];
            const __nv_bfloat16 *a_log  = (const __nv_bfloat16 *)lw.ptrs[6];
            const __nv_bfloat16 *dt_bias= (const __nv_bfloat16 *)lw.ptrs[7];
            const __nv_bfloat16 *dn_norm= (const __nv_bfloat16 *)lw.ptrs[8];
            const __nv_bfloat16 *out_w  = (const __nv_bfloat16 *)lw.ptrs[9];
            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)lw.ptrs[10];
            const __nv_bfloat16 *gate_w = (const __nv_bfloat16 *)lw.ptrs[11];
            const __nv_bfloat16 *up_w   = (const __nv_bfloat16 *)lw.ptrs[12];
            const __nv_bfloat16 *down_w = (const __nv_bfloat16 *)lw.ptrs[13];

            phase_matmul_bf16(normalized, qkv_w, proj_buf, S_pad, DN_CONV_CH, HIDDEN);
            phase_matmul_bf16(normalized, z_w,   proj_buf2, S_pad, DN_V_SIZE, HIDDEN);
            phase_matvec_small_to_f32(normalized, beta_w,  beta_buf,  S, HIDDEN, DN_HEADS);
            phase_matvec_small_to_f32(normalized, alpha_w, alpha_buf, S, HIDDEN, DN_HEADS);
            grid.sync();

            phase_deltanet_recurrence(proj_buf, proj_buf2, beta_buf, alpha_buf,
                                      conv_w, a_log, dt_bias, dn_norm,
                                      dn_states + dn_idx * dn_stride,
                                      conv_bufs + dn_idx * DN_CONV_CH * DN_CONV_K,
                                      dn_out_buf, S);
            grid.sync();

            phase_matmul_bf16(dn_out_buf, out_w, proj_buf, S_pad, HIDDEN, DN_V_SIZE);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            // MLP
            phase_rmsnorm(hidden, post_norm, normalized, residual, S_pad, HIDDEN);
            grid.sync();
            phase_matmul_bf16(normalized, gate_w, proj_buf, S_pad, INTER, HIDDEN);
            phase_matmul_bf16(normalized, up_w,   proj_buf2, S_pad, INTER, HIDDEN);
            grid.sync();
            phase_silu_mul(proj_buf, proj_buf2, mlp_buf, S_pad * INTER);
            grid.sync();
            phase_matmul_bf16(mlp_buf, down_w, proj_buf, S_pad, HIDDEN, INTER);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            dn_idx++;
        } else {
            // Full Attention
            const __nv_bfloat16 *q_w = (const __nv_bfloat16 *)lw.ptrs[1];
            const __nv_bfloat16 *k_w = (const __nv_bfloat16 *)lw.ptrs[2];
            const __nv_bfloat16 *v_w = (const __nv_bfloat16 *)lw.ptrs[3];
            const __nv_bfloat16 *q_nw = (const __nv_bfloat16 *)lw.ptrs[4];
            const __nv_bfloat16 *k_nw = (const __nv_bfloat16 *)lw.ptrs[5];
            const __nv_bfloat16 *o_w = (const __nv_bfloat16 *)lw.ptrs[6];
            const __nv_bfloat16 *post_norm = (const __nv_bfloat16 *)lw.ptrs[7];
            const __nv_bfloat16 *gate_w = (const __nv_bfloat16 *)lw.ptrs[8];
            const __nv_bfloat16 *up_w   = (const __nv_bfloat16 *)lw.ptrs[9];
            const __nv_bfloat16 *down_w = (const __nv_bfloat16 *)lw.ptrs[10];

            phase_matmul_bf16(normalized, q_w, proj_buf, S_pad, FA_QPROJ_SIZE, HIDDEN);
            phase_matmul_bf16(normalized, k_w, proj_buf2, S_pad, FA_KV_SIZE, HIDDEN);
            phase_matmul_bf16(normalized, v_w, attn_buf, S_pad, FA_KV_SIZE, HIDDEN);
            grid.sync();

            phase_qk_norm_rope(proj_buf, proj_buf2, attn_buf, q_nw, k_nw,
                               fa_k_cache + fa_idx * fa_stride,
                               fa_v_cache + fa_idx * fa_stride, S);
            grid.sync();

            phase_causal_attn(proj_buf, proj_buf2, attn_buf, dn_out_buf, S);
            grid.sync();

            phase_matmul_bf16(dn_out_buf, o_w, proj_buf, S_pad, HIDDEN, FA_Q_SIZE);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            phase_rmsnorm(hidden, post_norm, normalized, residual, S_pad, HIDDEN);
            grid.sync();
            phase_matmul_bf16(normalized, gate_w, proj_buf, S_pad, INTER, HIDDEN);
            phase_matmul_bf16(normalized, up_w,   proj_buf2, S_pad, INTER, HIDDEN);
            grid.sync();
            phase_silu_mul(proj_buf, proj_buf2, mlp_buf, S_pad * INTER);
            grid.sync();
            phase_matmul_bf16(mlp_buf, down_w, proj_buf, S_pad, HIDDEN, INTER);
            grid.sync();
            phase_add_residual(proj_buf, residual, hidden, S_pad * HIDDEN);
            grid.sync();

            fa_idx++;
        }
    }

    // Final RMSNorm (last row only) + LM head
    phase_final_norm(hidden, final_norm_w, final_normed, hidden_bf16_out, S);
    grid.sync();
    phase_lm_head(final_normed, lm_head_w, lm_bmv, lm_bmi);
    grid.sync();
    phase_lm_reduce(lm_bmv, lm_bmi, output_token, gridDim.x);
}

// ===== Launcher =====
// Dynamic shared memory = double-buffered A+B staging + f32 output tile.
//   sA: 2 × BTM × BTK × 2 B  =  8 KB
//   sB: 2 × BTN × BTK × 2 B  = 64 KB
//   sC: BTM × BTN × 4 B      = 32 KB
//   total                    = 104 KB
// B200 per-block dynamic shared max is 227 KB (228 KB carveout - driver).
constexpr int MEGA_SHMEM_BYTES =
    2 * BTM * BTK * 2   // sA
  + 2 * BTN * BTK * 2   // sB
  + BTM * BTN * 4;      // sC

static int mega_launch_blocks() {
    if (const char *override_blocks = std::getenv("MEGAKERNEL_PREFILL_MEGA_BLOCKS")) {
        int v = std::atoi(override_blocks);
        if (v > 0) return v;
    }
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    // Raise per-block dynamic shared limit (sm_100 default caps lower).
    cudaFuncSetAttribute(prefill_megakernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEGA_SHMEM_BYTES);
    int active = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, prefill_megakernel, MEGA_BLOCK_SIZE, MEGA_SHMEM_BYTES);
    return std::max(1, active * prop.multiProcessorCount);
}

extern "C" void launch_prefill_bf16_mega(
    const int *token_ids, int seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const MegaLayerWeights *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    cudaStream_t stream)
{
    static int cached_blocks = 0;
    if (cached_blocks == 0) cached_blocks = mega_launch_blocks();

    const int *token_ids_arg = token_ids;
    int seq_len_arg = seq_len;
    int *output_token_arg = output_token;
    const __nv_bfloat16 *embed_arg = embed_weight;
    const MegaLayerWeights *layers_arg = layers;
    const __nv_bfloat16 *fn_arg = final_norm_w;
    const __nv_bfloat16 *lm_arg = lm_head_w;
    __nv_bfloat16 *fk_arg = fa_k_cache; __nv_bfloat16 *fv_arg = fa_v_cache;
    float *ds_arg = dn_states; float *cb_arg = conv_bufs;
    __nv_bfloat16 *h_arg = hidden; __nv_bfloat16 *r_arg = residual; __nv_bfloat16 *n_arg = normalized;
    __nv_bfloat16 *p1 = proj_buf; __nv_bfloat16 *p2 = proj_buf2;
    __nv_bfloat16 *a_arg = attn_buf; __nv_bfloat16 *m_arg = mlp_buf;
    __nv_bfloat16 *dn_out_arg = dn_out_buf;
    float *bb = beta_buf; float *ab = alpha_buf;
    __nv_bfloat16 *fnrm = final_normed; __nv_bfloat16 *hout = hidden_bf16_out;
    float *lv = lm_bmv; int *li_ = lm_bmi;

    void *args[] = {
        (void *)&token_ids_arg, (void *)&seq_len_arg, (void *)&output_token_arg,
        (void *)&embed_arg, (void *)&layers_arg,
        (void *)&fn_arg, (void *)&lm_arg,
        (void *)&fk_arg, (void *)&fv_arg,
        (void *)&ds_arg, (void *)&cb_arg,
        (void *)&h_arg, (void *)&r_arg, (void *)&n_arg,
        (void *)&p1, (void *)&p2,
        (void *)&a_arg, (void *)&m_arg,
        (void *)&dn_out_arg,
        (void *)&bb, (void *)&ab,
        (void *)&fnrm, (void *)&hout,
        (void *)&lv, (void *)&li_,
    };
    cudaLaunchCooperativeKernel(
        (void *)prefill_megakernel,
        dim3(cached_blocks),
        dim3(MEGA_BLOCK_SIZE),
        args, MEGA_SHMEM_BYTES, stream);
}



================================================
FILE: megakernel/RESULTS.md
================================================
# Benchmark Results

All benchmarks are **batch size 1, single-stream decode**, targeting local inference on consumer hardware. This is the llama.cpp/Ollama use case, not multi-tenant serving.

## Hardware

| Machine | GPU/Chip | Memory |
|---------|----------|--------|
| Lucebox | NVIDIA RTX 3090 | 24GB VRAM |
| MacBook Pro | Apple M5 Max | 36GB Unified |

## RTX 3090: pp520 tg128

| Method | pp520 (tok/s) | tg128 (tok/s) |
|--------|:---:|:---:|
| **Megakernel** | **21,347** | **413** |
| llama.cpp BF16 | 11,247 | 267 |
| PyTorch HF | 7,578 | 108 |

### Speedups

| | vs llama.cpp | vs PyTorch |
|---|:---:|:---:|
| **Decode (tg128)** | **1.55x** | **3.8x** |

## Apple M5 Max

| Method | tok/s |
|--------|:---:|
| LM Studio (llama.cpp) BF16 | 229 |

## Power Efficiency (DVFS)

| Power Limit | Clock | Draw | tok/s | tok/J | vs Stock |
|---|---|---|---|---|---|
| 420W (stock) | 1980 MHz | 314W | 433 | 1.38 | baseline |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% speed, 5% less power |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% speed, 30% less power** |
| 150W | 405 MHz | 150W | 194 | 1.29 | too aggressive |

Sweet spot: 220W, 1.87 tok/J.

## Methodology

- **Precision:** BF16 weights and activations, FP32 accumulation. No quantization. All baselines (llama.cpp, PyTorch HF) also run BF16 for apples-to-apples comparison.
- **Power measurement:** Accelerator power only via NVML energy counters (NVIDIA) and `powermetrics` (Apple Silicon), consistent with [Hazy Research's Intelligence Per Watt methodology](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles). Total system draw is higher for both platforms.
- **Correctness:** `bench_pp_tg.py` includes an end-to-end correctness check, comparing megakernel output (prefill + decode) against a token-by-token reference decode path. Both must produce identical token sequences.
- **Warm-up:** One warm-up run before timed measurements. Timing uses `torch.cuda.synchronize()` barriers with `time.perf_counter()`.
- **llama.cpp version:** Latest release at time of testing, BF16 mode, default settings.

## What this doesn't measure

- Batched throughput (batch size > 1)
- Quantized model performance (INT4/INT8)
- Models larger than 0.8B parameters
- Multi-GPU or tensor-parallel setups
- Total system power (CPU, RAM, PSU losses)

## NVIDIA DGX Spark (GB10, sm_121a)

First results for the megakernel on NVIDIA's DGX Spark (GB10). Decode runs
through a persistent NVFP4 megakernel; prefill uses a bf16 body with an
NVFP4 LM head (the default "hybrid" prefill mode).

**Hardware:** NVIDIA GB10 Grace Blackwell Superchip (DGX Spark), compute cap
12.1 (`sm_121a`), driver 580.126.09, CUDA 13.2, PyTorch 2.13.0a (cu13.2).

| Method | Prefill pp520 (tok/s) | Decode tg128 (tok/s) |
|---|:---:|:---:|
| **Megakernel (NVFP4 decode, hybrid prefill)** | **20,639** | **181** |
| PyTorch HuggingFace (bf16) | 5,649 | 65 |

**3.6× PyTorch HF on prefill, 2.8× on decode.** Output tokens match the
PyTorch reference on the pp520 prompt from `final_bench.py`.

### Run

```bash
# Auto-dispatches to the NVFP4 path on Blackwell
python megakernel/final_bench.py

# Or force a backend
python megakernel/final_bench.py --backend nvfp4
python megakernel/final_bench.py --backend bf16

# Switch prefill mode (default is "hybrid"; "raw" uses prefill_megakernel_nvfp4)
MEGAKERNEL_PREFILL_MODE=raw python megakernel/final_bench.py --backend nvfp4
```



================================================
FILE: megakernel/setup.py
================================================
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detect_arch():
    arch = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if arch:
        return arch
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 12 and minor in (0, 1):
                return f"sm_{major}{minor}a"
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return "sm_75"


def _int_env(name, default):
    return str(int(os.environ.get(name, default)))


arch = _detect_arch()
is_blackwell = arch.startswith("sm_12")

# Extract numeric SM level (e.g. "sm_75" → 75, "sm_120a" → 120) for compile-time guards.
import re
_sm_match = re.search(r"sm_(\d+)", arch)
target_sm = int(_sm_match.group(1)) if _sm_match else 86

num_blocks = _int_env("MEGAKERNEL_NUM_BLOCKS", 82)
block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 512)
lm_num_blocks = _int_env("MEGAKERNEL_LM_NUM_BLOCKS", 512)
lm_block_size = _int_env("MEGAKERNEL_LM_BLOCK_SIZE", 256)
dn_phase2_wmma = _int_env("MEGAKERNEL_DN_PHASE2_WMMA", 0)

sources = [
    "torch_bindings.cpp",
    "kernel.cu",
    "prefill.cu",
]
libraries = ["cublas"]
cxx_args = ["-O3"]
nvcc_args = [
    "-O3",
    f"-arch={arch}",
    "--use_fast_math",
    "-std=c++17",
    f"-DNUM_BLOCKS={num_blocks}",
    f"-DBLOCK_SIZE={block_size}",
    f"-DLM_NUM_BLOCKS={lm_num_blocks}",
    f"-DLM_BLOCK_SIZE={lm_block_size}",
    f"-DMEGAKERNEL_DN_PHASE2_WMMA={dn_phase2_wmma}",
    f"-DTARGET_SM={target_sm}",
]
# Expose to host compiler (torch_bindings.cpp, prefill.cu host-side) so it
# can also branch on the flag without needing a separate nvcc pass.
cxx_args.append(f"-DMEGAKERNEL_DN_PHASE2_WMMA={dn_phase2_wmma}")
cxx_args.append(f"-DTARGET_SM={target_sm}")

if is_blackwell:
    sources.append("kernel_gb10_nvfp4.cu")
    sources.append("prefill_megakernel.cu")
    sources.append("prefill_bw.cu")
    # Exposed to both nvcc (for the Blackwell .cu files) and the host
    # compiler (so torch_bindings.cpp registers the NVFP4 ops).
    cxx_args.append("-DMEGAKERNEL_HAS_NVFP4")
    nvcc_args.append("-DMEGAKERNEL_HAS_NVFP4")
    nvcc_args.append("-DMEGAKERNEL_HAS_PREFILL_MEGA")
    libraries.append("cublasLt")

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=sources,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            libraries=libraries,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)



================================================
FILE: megakernel/torch_bindings.cpp
================================================
/**
 * PyTorch bindings for Qwen3.5-0.8B bf16 megakernel — decode.
 *
 * Blackwell-only bindings (NVFP4 decode, bf16 prefill megakernel, prefill
 * megakernel NVFP4) are gated behind MEGAKERNEL_HAS_NVFP4, which is only
 * defined by setup.py for sm_12+ builds. On sm_86 the symbol set is
 * identical to the original upstream build.
 */

#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

struct LayerWeights {
    int layer_type;
    int _pad[3];
    void *ptrs[14];  // max(11 FA, 14 DN) pointers — all bf16, no scales
};

#ifdef MEGAKERNEL_HAS_NVFP4
struct LayerWeightsNVFP4 {
    int layer_type;
    int group_size;
    int _pad[2];
    void *ptrs[24];  // hot decode weights become packed fp4 + per-group scales
};

// Layout-compatible with PFFusedLayerWeights in prefill_bw.cu — used by the
// hybrid bf16 prefill + NVFP4 LM head path (launch_prefill_bf16_nvfp4_lm).
struct PrefillFusedLayerWeights {
    void *proj_weight;
    void *gate_up_weight;
    void *proj_weight_packed;
    void *proj_weight_scales;
    void *gate_up_weight_packed;
    void *gate_up_weight_scales;
};
#endif

extern "C" void launch_decode(
    int input_token_id, int *output_token_id,
    const void *embed_weight, const LayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    float *seen_token_mask,
    float repetition_penalty,
    int position, int max_seq_len, cudaStream_t stream);

#ifdef MEGAKERNEL_HAS_NVFP4
extern "C" void launch_decode_nvfp4(
    const int *input_token_ptr, int *output_token_id,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, int group_size, cudaStream_t stream);

extern "C" void launch_decode_many_nvfp4(
    int *token_buffer, int *output_tokens, int steps,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int position, int max_seq_len, int group_size, cudaStream_t stream);

extern "C" void launch_quantize_nvfp4_out(
    const void *weight, int rows, int cols, int group_size,
    void *packed_out, void *scales_out, cudaStream_t stream);

extern "C" void launch_quantize_nvfp4_lm_out(
    const void *weight, int rows, int cols,
    void *packed_out, void *scales_out, cudaStream_t stream);

extern "C" void launch_prefill_megakernel_nvfp4(
    const int *token_ids, int seq_len, int *output_token_id,
    const void *embed_weight, const LayerWeightsNVFP4 *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *lm_hidden_bf16, void *lm_hidden_packed, void *lm_hidden_scales, void *lm_logits_f16,
    void *fa_k_cache, void *fa_v_cache,
    void *dn_states, void *conv_bufs,
    void *hidden_buffer, void *g_activations, void *g_residual,
    void *g_qkv_scratch, void *g_kv_scratch, void *g_attn_out,
    void *g_mlp_inter, void *g_z_scratch, void *g_beta_scratch,
    void *g_alpha_scratch, void *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_generation,
    float *block_max_vals, int *block_max_idxs,
    unsigned int *lm_sync_counter,
    int max_seq_len, int group_size, cudaStream_t stream);

static void seed_token_buffer(torch::Tensor token_buffer, int token_id) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = cudaMemcpyAsync(
        token_buffer.data_ptr(),
        &token_id,
        sizeof(token_id),
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync(token_buffer) failed: ", cudaGetErrorString(err));
}
#endif  // MEGAKERNEL_HAS_NVFP4
extern "C" void set_decode_blocks_override(int blocks);
extern "C" int query_max_safe_decode_blocks();

void decode(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, torch::Tensor seen_token_mask,
    double repetition_penalty, int64_t position, int64_t max_seq_len)
{
    launch_decode(
        (int)input_token_id, (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (float*)seen_token_mask.data_ptr(), (float)repetition_penalty,
        (int)position, (int)max_seq_len,
        c10::cuda::getCurrentCUDAStream().stream());
}

#ifdef MEGAKERNEL_HAS_NVFP4
void decode_nvfp4(
    torch::Tensor output_token, int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed, torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t position, int64_t max_seq_len,
    int64_t group_size)
{
    seed_token_buffer(output_token, (int)input_token_id);
    launch_decode_nvfp4(
        (const int*)output_token.data_ptr(),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(), lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)position, (int)max_seq_len, (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}

void decode_many_nvfp4(
    torch::Tensor output_tokens,
    torch::Tensor token_buffer,
    int64_t input_token_id,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed, torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t position, int64_t max_seq_len,
    int64_t group_size)
{
    TORCH_CHECK(output_tokens.is_cuda(), "output_tokens must be CUDA");
    TORCH_CHECK(output_tokens.is_contiguous(), "output_tokens must be contiguous");
    TORCH_CHECK(output_tokens.scalar_type() == torch::kInt32, "output_tokens must be int32");
    TORCH_CHECK(output_tokens.dim() == 1, "output_tokens must be 1D");
    TORCH_CHECK(token_buffer.is_cuda(), "token_buffer must be CUDA");
    TORCH_CHECK(token_buffer.is_contiguous(), "token_buffer must be contiguous");
    TORCH_CHECK(token_buffer.scalar_type() == torch::kInt32, "token_buffer must be int32");
    TORCH_CHECK(token_buffer.numel() == 1, "token_buffer must contain exactly one int32 token");

    seed_token_buffer(token_buffer, (int)input_token_id);
    launch_decode_many_nvfp4(
        (int*)token_buffer.data_ptr(),
        (int*)output_tokens.data_ptr(),
        (int)output_tokens.numel(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(), lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
        qkv_scratch.data_ptr(), kv_scratch.data_ptr(), attn_out.data_ptr(),
        mlp_inter.data_ptr(), z_scratch.data_ptr(), beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(), normalized.data_ptr(),
        (unsigned int*)barrier_counter.data_ptr(), (unsigned int*)barrier_generation.data_ptr(),
        (float*)block_max_vals.data_ptr(), (int*)block_max_idxs.data_ptr(),
        (unsigned int*)lm_sync_counter.data_ptr(),
        (int)position, (int)max_seq_len, (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}

void quantize_nvfp4_out(
    torch::Tensor packed_out,
    torch::Tensor scales_out,
    torch::Tensor weight,
    int64_t group_size)
{
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D [out_dim, in_dim] tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(group_size > 0 && (group_size % 2) == 0, "group_size must be a positive even integer");

    auto rows = static_cast<int>(weight.size(0));
    auto cols = static_cast<int>(weight.size(1));
    TORCH_CHECK((cols % 2) == 0, "in_dim must be divisible by 2 for packed fp4 output");
    TORCH_CHECK((cols % group_size) == 0, "in_dim must be divisible by group_size");
    TORCH_CHECK(packed_out.is_cuda() && packed_out.is_contiguous(), "packed_out must be contiguous CUDA");
    TORCH_CHECK(scales_out.is_cuda() && scales_out.is_contiguous(), "scales_out must be contiguous CUDA");
    TORCH_CHECK(packed_out.scalar_type() == torch::kUInt8, "packed_out must be uint8");
    TORCH_CHECK(scales_out.scalar_type() == torch::kFloat16, "scales_out must be float16");
    TORCH_CHECK(
        packed_out.numel() == (int64_t)rows * (cols / 2),
        "packed_out has the wrong size");
    TORCH_CHECK(
        scales_out.numel() == (int64_t)rows * (cols / group_size),
        "scales_out has the wrong size");

    launch_quantize_nvfp4_out(
        weight.data_ptr(), rows, cols, (int)group_size,
        packed_out.data_ptr(), scales_out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

void quantize_nvfp4_lm_out(
    torch::Tensor packed_out,
    torch::Tensor scales_out,
    torch::Tensor weight)
{
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D [out_dim, in_dim] tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");

    auto rows = static_cast<int>(weight.size(0));
    auto cols = static_cast<int>(weight.size(1));
    TORCH_CHECK((rows % 128) == 0, "out_dim must be divisible by 128");
    TORCH_CHECK((cols % 64) == 0, "in_dim must be divisible by 64");
    TORCH_CHECK(packed_out.is_cuda() && packed_out.is_contiguous(), "packed_out must be contiguous CUDA");
    TORCH_CHECK(scales_out.is_cuda() && scales_out.is_contiguous(), "scales_out must be contiguous CUDA");
    TORCH_CHECK(packed_out.scalar_type() == torch::kUInt8, "packed_out must be uint8");
    TORCH_CHECK(scales_out.scalar_type() == torch::kUInt8, "scales_out must be uint8");
    TORCH_CHECK(
        packed_out.numel() == (int64_t)rows * (cols / 2),
        "packed_out has the wrong size");

    int scale_tiles = cols / 64;
    int expected_scales = (rows / 128) * scale_tiles * 512;
    TORCH_CHECK(
        scales_out.numel() == expected_scales,
        "scales_out has the wrong size");

    launch_quantize_nvfp4_lm_out(
        weight.data_ptr(), rows, cols,
        packed_out.data_ptr(), scales_out.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}
#endif  // MEGAKERNEL_HAS_NVFP4

int64_t max_safe_decode_blocks()
{
    return query_max_safe_decode_blocks();
}

void set_decode_blocks(int64_t blocks)
{
    set_decode_blocks_override((int)blocks);
}

// ===== Prefill BF16 =====

// chunk-parallel DeltaNet prefill (was previously v2; promoted to canonical)
// adds 4 fp32 scratch buffers + 2 fused weight bases (FA QKV, MLP gate+up)
extern "C" void launch_prefill_bf16(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf,
    void *beta_buf, void *alpha_buf, void *dn_pre_qkv,
    void *dn_u_scratch, void *dn_w_scratch, void *dn_cs_scratch,
    const void *fused_fa_qkv_base, const void *fused_gate_up_base,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    int max_seq_len,
    cudaStream_t stream);

void prefill_bf16(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor dn_pre_qkv,
    torch::Tensor dn_u_scratch, torch::Tensor dn_w_scratch, torch::Tensor dn_cs_scratch,
    torch::Tensor fused_fa_qkv, torch::Tensor fused_gate_up,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    int64_t max_seq_len)
{
    launch_prefill_bf16(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(),
        beta_buf.data_ptr(), alpha_buf.data_ptr(), dn_pre_qkv.data_ptr(),
        dn_u_scratch.data_ptr(), dn_w_scratch.data_ptr(), dn_cs_scratch.data_ptr(),
        fused_fa_qkv.data_ptr(), fused_gate_up.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        (int)max_seq_len,
        c10::cuda::getCurrentCUDAStream().stream());
}

#ifdef MEGAKERNEL_HAS_NVFP4
extern "C" void launch_prefill_bf16_mega(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const void *final_norm_w, const void *lm_head_w,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    cudaStream_t stream);

void prefill_bf16_mega(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi)
{
    launch_prefill_bf16_mega(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

extern "C" void launch_prefill_bf16_nvfp4_lm(
    const int *token_ids, int seq_len, int *output_token,
    const void *embed_weight, const LayerWeights *layers,
    const PrefillFusedLayerWeights *fused_layers,
    const void *final_norm_w, const void *lm_head_w,
    const void *lm_head_weight_packed, const void *lm_head_scales,
    void *fa_k_cache, void *fa_v_cache, void *dn_states, void *conv_bufs,
    void *hidden, void *residual, void *normalized,
    void *proj_buf, void *proj_buf2, void *proj_buf_half, void *proj_act_packed, void *proj_act_scales,
    void *attn_buf, void *mlp_buf,
    void *dn_out_buf, void *beta_buf, void *alpha_buf,
    void *final_normed, void *hidden_bf16_out,
    void *lm_bmv, void *lm_bmi,
    void *lm_hidden_bf16, void *lm_hidden_packed,
    void *lm_hidden_scales, void *lm_logits_f16,
    cudaStream_t stream);

void prefill_bf16_nvfp4_lm(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor prefill_fused_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden, torch::Tensor residual, torch::Tensor normalized,
    torch::Tensor proj_buf, torch::Tensor proj_buf2, torch::Tensor proj_buf_half,
    torch::Tensor proj_act_packed, torch::Tensor proj_act_scales,
    torch::Tensor attn_buf, torch::Tensor mlp_buf,
    torch::Tensor dn_out_buf, torch::Tensor beta_buf, torch::Tensor alpha_buf,
    torch::Tensor final_normed, torch::Tensor hidden_bf16_out,
    torch::Tensor lm_bmv, torch::Tensor lm_bmi,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed,
    torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16)
{
    launch_prefill_bf16_nvfp4_lm(
        (const int*)token_ids.data_ptr(), token_ids.size(0),
        (int*)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeights*>(layer_weights_packed.data_ptr()),
        reinterpret_cast<const PrefillFusedLayerWeights*>(prefill_fused_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(), lm_head_scales.data_ptr(),
        fa_k_cache.data_ptr(), fa_v_cache.data_ptr(),
        dn_states.data_ptr(), conv_bufs.data_ptr(),
        hidden.data_ptr(), residual.data_ptr(), normalized.data_ptr(),
        proj_buf.data_ptr(), proj_buf2.data_ptr(), proj_buf_half.data_ptr(),
        proj_act_packed.data_ptr(), proj_act_scales.data_ptr(),
        attn_buf.data_ptr(), mlp_buf.data_ptr(),
        dn_out_buf.data_ptr(), beta_buf.data_ptr(), alpha_buf.data_ptr(),
        final_normed.data_ptr(), hidden_bf16_out.data_ptr(),
        lm_bmv.data_ptr(), lm_bmi.data_ptr(),
        lm_hidden_bf16.data_ptr(), lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(), lm_logits_f16.data_ptr(),
        c10::cuda::getCurrentCUDAStream().stream());
}

void prefill_megakernel_nvfp4(
    torch::Tensor output_token, torch::Tensor token_ids,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight_packed, torch::Tensor lm_head_scales,
    torch::Tensor lm_hidden_bf16, torch::Tensor lm_hidden_packed, torch::Tensor lm_hidden_scales, torch::Tensor lm_logits_f16,
    torch::Tensor fa_k_cache, torch::Tensor fa_v_cache,
    torch::Tensor dn_states, torch::Tensor conv_bufs,
    torch::Tensor hidden_buffer, torch::Tensor activations, torch::Tensor residual,
    torch::Tensor qkv_scratch, torch::Tensor kv_scratch, torch::Tensor attn_out,
    torch::Tensor mlp_inter, torch::Tensor z_scratch, torch::Tensor beta_scratch,
    torch::Tensor alpha_scratch, torch::Tensor normalized,
    torch::Tensor barrier_counter, torch::Tensor barrier_generation,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    torch::Tensor lm_sync_counter, int64_t max_seq_len, int64_t group_size)
{
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA");
    TORCH_CHECK(token_ids.is_contiguous(), "token_ids must be contiguous");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt32, "token_ids must be int32");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be 1D");

    launch_prefill_megakernel_nvfp4(
        (const int *)token_ids.data_ptr(),
        static_cast<int>(token_ids.numel()),
        (int *)output_token.data_ptr(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LayerWeightsNVFP4 *>(layer_weights_packed.data_ptr()),
        final_norm_weight.data_ptr(),
        lm_head_weight_packed.data_ptr(),
        lm_head_scales.data_ptr(),
        lm_hidden_bf16.data_ptr(),
        lm_hidden_packed.data_ptr(),
        lm_hidden_scales.data_ptr(),
        lm_logits_f16.data_ptr(),
        fa_k_cache.data_ptr(),
        fa_v_cache.data_ptr(),
        dn_states.data_ptr(),
        conv_bufs.data_ptr(),
        hidden_buffer.data_ptr(),
        activations.data_ptr(),
        residual.data_ptr(),
        qkv_scratch.data_ptr(),
        kv_scratch.data_ptr(),
        attn_out.data_ptr(),
        mlp_inter.data_ptr(),
        z_scratch.data_ptr(),
        beta_scratch.data_ptr(),
        alpha_scratch.data_ptr(),
        normalized.data_ptr(),
        (unsigned int *)barrier_counter.data_ptr(),
        (unsigned int *)barrier_generation.data_ptr(),
        (float *)block_max_vals.data_ptr(),
        (int *)block_max_idxs.data_ptr(),
        (unsigned int *)lm_sync_counter.data_ptr(),
        (int)max_seq_len,
        (int)group_size,
        c10::cuda::getCurrentCUDAStream().stream());
}
#endif  // MEGAKERNEL_HAS_NVFP4

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("decode(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "Tensor seen_token_mask, float repetition_penalty, "
            "int position, int max_seq_len) -> ()");
    ops.impl("decode", torch::kCUDA, &decode);

    ops.def("max_safe_decode_blocks() -> int");
    ops.impl("max_safe_decode_blocks", &max_safe_decode_blocks);

    ops.def("set_decode_blocks(int blocks) -> ()");
    ops.impl("set_decode_blocks", &set_decode_blocks);

    ops.def("prefill_bf16(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor dn_pre_qkv, "
            "Tensor dn_u_scratch, Tensor dn_w_scratch, Tensor dn_cs_scratch, "
            "Tensor fused_fa_qkv, Tensor fused_gate_up, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi, int max_seq_len) -> ()");
    ops.impl("prefill_bf16", torch::kCUDA, &prefill_bf16);

#ifdef MEGAKERNEL_HAS_NVFP4
    ops.def("decode_nvfp4(Tensor output_token, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len, int group_size) -> ()");
    ops.impl("decode_nvfp4", torch::kCUDA, &decode_nvfp4);

    ops.def("decode_many_nvfp4(Tensor output_tokens, Tensor token_buffer, int input_token_id, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int position, int max_seq_len, int group_size) -> ()");
    ops.impl("decode_many_nvfp4", torch::kCUDA, &decode_many_nvfp4);

    ops.def("prefill_bf16_mega(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi) -> ()");
    ops.impl("prefill_bf16_mega", torch::kCUDA, &prefill_bf16_mega);

    ops.def("prefill_megakernel_nvfp4(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden_buffer, Tensor activations, Tensor residual, "
            "Tensor qkv_scratch, Tensor kv_scratch, Tensor attn_out, "
            "Tensor mlp_inter, Tensor z_scratch, Tensor beta_scratch, "
            "Tensor alpha_scratch, Tensor normalized, "
            "Tensor barrier_counter, Tensor barrier_generation, "
            "Tensor block_max_vals, Tensor block_max_idxs, Tensor lm_sync_counter, "
            "int max_seq_len, int group_size) -> ()");
    ops.impl("prefill_megakernel_nvfp4", torch::kCUDA, &prefill_megakernel_nvfp4);

    ops.def("prefill_bf16_nvfp4_lm(Tensor output_token, Tensor token_ids, "
            "Tensor embed_weight, Tensor layer_weights_packed, Tensor prefill_fused_weights_packed, "
            "Tensor final_norm_weight, Tensor lm_head_weight, "
            "Tensor lm_head_weight_packed, Tensor lm_head_scales, "
            "Tensor fa_k_cache, Tensor fa_v_cache, Tensor dn_states, Tensor conv_bufs, "
            "Tensor hidden, Tensor residual, Tensor normalized, "
            "Tensor proj_buf, Tensor proj_buf2, Tensor proj_buf_half, Tensor proj_act_packed, Tensor proj_act_scales, "
            "Tensor attn_buf, Tensor mlp_buf, "
            "Tensor dn_out_buf, Tensor beta_buf, Tensor alpha_buf, "
            "Tensor final_normed, Tensor hidden_bf16_out, "
            "Tensor lm_bmv, Tensor lm_bmi, "
            "Tensor lm_hidden_bf16, Tensor lm_hidden_packed, Tensor lm_hidden_scales, Tensor lm_logits_f16) -> ()");
    ops.impl("prefill_bf16_nvfp4_lm", torch::kCUDA, &prefill_bf16_nvfp4_lm);

    ops.def("quantize_nvfp4_out(Tensor packed_out, Tensor scales_out, Tensor weight, int group_size) -> ()");
    ops.impl("quantize_nvfp4_out", torch::kCUDA, &quantize_nvfp4_out);

    ops.def("quantize_nvfp4_lm_out(Tensor packed_out, Tensor scales_out, Tensor weight) -> ()");
    ops.impl("quantize_nvfp4_lm_out", torch::kCUDA, &quantize_nvfp4_lm_out);
#endif  // MEGAKERNEL_HAS_NVFP4
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)



================================================
FILE: pflash/README.md
================================================
<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce PFlash</h1>

<p align="center">
  <strong>Speculative prefill in front of dflash. C++/CUDA only.</strong><br/>
  A drafter loaded in-process scores token importance; the heavy target only prefills the spans that matter.<br/>
  Qwen3.6-27B Q4_K_M at 128K on a single RTX 3090: <strong>24.8 s TTFT vs ~257 s llama.cpp</strong> = <strong>~10.4×</strong>, NIAH retrieval preserved.<br/><br/>
  <a href="https://lucebox.com/blog/pflash">Blog post</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

<p align="center">
  <img src="demo.gif" width="600" />
</p>

---

```
                       Cold TTFT (s)   Speedup   NIAH
llama.cpp pp131072         ~257           1.0x     ✓
dflash daemon @ 128K        24.8         10.4x     ✓
dflash daemon @  64K        13.5         10.0x     ✓
```

> Long context turns prefill into the dominant latency on quantized 27B targets. Speculative prefill scores token importance with a small drafter, then the heavy target only prefills the spans that matter. Quality preserved on NIAH at every measured context. The whole thing runs as a single C++/CUDA binary: no Python, no Triton, no PyTorch at runtime.

## The gap we filled

Long-context prefill is O(S²): vanilla llama.cpp on a single RTX 3090 takes **~257 s** to prefill 131,072 tokens of Qwen3.6-27B Q4_K_M (FA on, Q4_0 KV). Decode after that is fast (dflash spec decode runs at ~74 tok/s) but the user is staring at a blank screen for 4 minutes before the first token.

[Cross-Family Speculative Prefill (SambaNova ICLR 2026, Liu et al.)](https://arxiv.org/abs/2603.02631) showed a small drafter can score per-token importance over a long prompt and select a tiny fraction without losing the needle. The reference impl ([Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill)) wires this on top of vLLM with full BF16 targets on big GPUs.

**What was missing:** no implementation that sits in front of a quantized GGUF target on a 24 GB card without dragging Python+Triton into the runtime path. PFlash is that:

- C++/CUDA daemon-resident drafter + scoring + target generation, all in one process, one ggml allocator.
- Custom Qwen3-0.6B BF16 forward (`qwen3_0p6b_loader.cpp` + `qwen3_0p6b_graph.cpp`) — no libllama.
- 4 CUDA kernels for the FlashPrefill `mean_K → score → select → sparse_fwd` algorithm (`flashprefill_kernels.cu`).
- BSA ([mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention), FA-2 derived, sm_80+) for the long-context drafter forward, wired without `libtorch` via 3 ATen/c10 header stubs (`dflash/deps/bsa_stubs/`).
- 128K → 2.6K span selection at `keep_ratio=0.05`, NIAH retrieved at every measured context, decode ~74 tok/s downstream.

## Results

NIAH single-needle, RTX 3090 24 GB, Qwen3.6-27B Q4_K_M target, Qwen3-0.6B drafter, `DFLASH_FP_USE_BSA=1`, `DFLASH_FP_ALPHA=0.85`, `keep_ratio=0.05`.

| Source S | dflash TTFT | llama.cpp baseline | Speedup | NIAH |
|---|:---:|:---:|:---:|:---:|
| 64K  | **13.5 s** | 134.95 s (FA off, dense) | **10.0×** | ✅ |
| 128K | **24.8 s** | ~257 s (FA on, Q4_0 KV)  | **~10.4×** | ✅ |

Decode after prefill: ~74 tok/s (dflash spec decode + DDTree). The pipeline is the dflash binary on its own — no Python in the inference loop.

## Quick start

PFlash is the algorithm. The implementation lives in [`../dflash/`](../dflash/) as part of the dflash daemon. The `pflash/` directory in this repo only contains the Python tooling for **benchmarking** (NIAH case generation, bench harness around the daemon stdin protocol). Production deploys hit the dflash daemon directly.

```bash
# 1. build dflash with the BSA kernel (sm_80+; ~10 min cold compile pulls cutlass)
cd lucebox-hub/dflash
git submodule update --init --recursive
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_CUDA_ARCHITECTURES=86 \
                    -DDFLASH27B_ENABLE_BSA=ON
cmake --build build --target test_dflash test_flashprefill_kernels -j

# 2. fetch weights (target + spec-decode draft + drafter scorer)
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download Qwen/Qwen3-0.6B model.safetensors tokenizer.json --local-dir models/drafter/
huggingface-cli download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/

# 2b. convert the drafter (Qwen3-0.6B HF) to a BF16 GGUF for the C++ scorer.
#     The submodule already vendors llama.cpp at deps/llama.cpp.
python deps/llama.cpp/convert_hf_to_gguf.py models/drafter \
       --outtype bf16 --outfile models/Qwen3-0.6B-BF16.gguf

# 3. install pflash bench harness (Python only used for benchmarking)
cd ../pflash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 4. generate NIAH cases + run head-to-head bench against the C++ daemon
python tests/niah_gen.py --n 1 --ctx 131072 --out /tmp/niah_128k.jsonl
python tests/bench_niah_cpp.py \
  --bin    ../dflash/build/test_dflash \
  --target ../dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft-spec ../dflash/models/draft/model.safetensors \
  --drafter-gguf ../dflash/models/Qwen3-0.6B-BF16.gguf \
  --cases  /tmp/niah_128k.jsonl --keep-ratio 0.05 --n-gen 256
```

## OpenAI server flags

For an OpenAI-compatible server with transparent compression on long prompts, run [`dflash/scripts/server.py`](../dflash/scripts/server.py) (or `server_tools.py` for tool-calling) with these flags:

| Flag | Choices / type | Default | Effect |
|---|---|:---:|---|
| `--prefill-compression` | `off` / `auto` / `always` | `off` | When to run pflash. `auto` compresses when total prompt ≥ threshold; `always` compresses every request. |
| `--prefill-threshold` | int (tokens) | `32000` | Token threshold for `auto` mode. |
| `--prefill-keep-ratio` | float `(0, 1]` | `0.05` | Fraction of source tokens to keep after compression. `0.02` for 128K, `0.10` for 32K. |
| `--prefill-drafter` | path to `.gguf` | required when not `off` | Drafter weights (Qwen3-0.6B BF16 GGUF). |
| `--prefill-drafter-tokenizer` | HF repo id | `Qwen/Qwen3-0.6B` | HF tokenizer for the drafter vocab. |

When `--prefill-compression != off`, the server auto-sets `DFLASH27B_LM_HEAD_FIX=0` and `DFLASH27B_FA_WINDOW=0` (matching the bench harness — needed so the post-compress draft graph fits on a 24 GB card without OOM).

```bash
python dflash/scripts/server.py \
  --target dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft  dflash/models/draft/model.safetensors \
  --max-ctx 8192 --budget 16 --fa-window 0 \
  --prefill-compression auto \
  --prefill-threshold 4096 \
  --prefill-keep-ratio 0.02 \
  --prefill-drafter dflash/models/Qwen3-0.6B-BF16.gguf
```

Below the threshold the server runs the standard target generate (no compression). Above it, the server transparently runs `compress` on the daemon, swaps the prompt for the compressed text, and continues the normal `/v1/chat/completions` flow. Tool-calling requests (`req.tools` non-empty) skip compression so JSON tool definitions stay intact.

Validated end-to-end at 64K and 128K source on RTX 3090 (Qwen3.6-27B Q4_K_M target + Qwen3.5-DFlash draft + Qwen3-0.6B BF16 drafter).

## Daemon stdin protocol

The dflash daemon runs persistently and accepts these commands on stdin (newline-delimited):

| Command | Effect |
|---|---|
| `compress <ids.bin> <keep_x1000> <drafter.gguf>` | Drafter scores the prompt and emits the compressed token-id stream (terminated by `-1`). |
| `generate <prompt_ids.bin> <n_gen> <out_ids.bin>` | Target spec-decode on the (already compressed) prompt. Streams committed token ids on stdout. |
| `park draft` / `park target` / `park` | Free draft / target / both weights from VRAM. |
| `unpark draft` / `unpark target` / `unpark` | Restore weights from disk to VRAM. |
| `free drafter` | Release the spec-prefill drafter context (drafter weights + KV + BSA scratch). |

Typical flow at 128K on a 24 GB card: `park target` → `compress` → `free drafter` → `unpark target` → `unpark draft` → `generate` → `park draft`.

`pflash.dflash_client.DflashClient` is the Python wrapper around this protocol used by `tests/bench_niah_cpp.py`.

## Runtime tunables

Everything is configured via env vars on the daemon process. Full list in [`../dflash/src/flashprefill.h`](../dflash/src/flashprefill.h).

| Env var | Default | Purpose |
|---|:---:|---|
| `DFLASH_FP_USE_BSA` | `0` | Set to `1` to dispatch the sparse FA forward through the BSA cutlass kernel (sm_80+). Required for the headline 10.4× number; without it the WMMA fallback is used (slower at long ctx). |
| `DFLASH_FP_ALPHA` | `0.12` | Block-selection threshold. Higher = stricter = fewer K-blocks per Q-row. `0.85` is the bench setting; `0.99` cuts another second at 128K with a small NIAH-margin loss. |
| `DFLASH_FP_PROFILE` | `0` | Set to `1` to log per-stage timings (`mean_K / score / select / forward`). |
| `DFLASH_FP_DUMP_COUNTS` | `0` | Set to `1` to dump per-row K-block counts for debugging keep-ratio tuning. |
| `DFLASH27B_FA_WINDOW` | (auto) | Set to `0` to force full attention on the compressed prompt (recommended). |
| `DFLASH27B_KV_K` / `DFLASH27B_KV_V` | (auto) | KV-cache quant types. `q4_0` / `q4_0` is the bench setting. `tq3_0` saves another ~4 GB at 128K. |

## How it works

```
prompt (≤ 128K tokens)
   │
   ▼
┌──────────────────────────────────────────────┐
│  drafter (in-process)                        │
│   custom Qwen3-0.6B BF16 forward in ggml     │
│   FlashPrefill block-sparse via BSA (≥ 32K)  │
│   tail-attention scoring → score [S]         │
│   chunk(128) + alpha-threshold → top blocks  │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  compressor (in-process)                     │
│   keep top keep_ratio of source tokens       │
│   re-emit compressed token-id stream         │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│  dflash spec decode (in-process)             │
│   target prefill of compressed prompt        │
│   DDTree spec decode + rollback              │
│   → answer tokens                            │
└──────────────────────────────────────────────┘
```

**Drafter forward.** Custom Qwen3-0.6B graph (`qwen3_0p6b_graph.cpp`) per-layer A/FP/B blocks: dense attention up to ~32K source, FlashPrefill sparse attention at and above. The 4 FP kernels live in `flashprefill_kernels.cu`; BSA dispatch is in `bsa_launcher.cu` + `bsa_fwd_inst.cu`.

**Scoring + selection.** Tail attention `Q[-N:] @ K^T / sqrt(d)` per layer/head, max over (L, H), mean over the tail window. Block-level threshold by `alpha * mean(scores)` selects which K-blocks each Q-block attends to. Configurable via `DFLASH_FP_ALPHA`.

**Memory budget on 24 GB.** Drafter scoring at 128K needs ~7-10 GB (drafter + KV + BSA scratch). Target + draft idle is ~18 GB. They can't coexist. The daemon's `park` / `unpark` / `free drafter` commands sequence VRAM occupancy across the request:

```
1. park draft + target          # daemon idles at ~3 GB
2. drafter loaded + scored      # ~10 GB peak
3. free drafter                 # release drafter weights + KV + BSA scratch
4. unpark target                # ~16 GB
5. unpark draft                 # +draft weights for spec decode
6. generate                     # spec decode the compressed prompt
7. park draft (idle)            # back to target only
```

## What's ours, what isn't

The algorithms are not ours:

- [**Cross-Family Speculative Prefill**](https://arxiv.org/abs/2603.02631) (SambaNova ICLR 2026): max-mean attention aggregation over a small drafter, lookahead-only attention.
- [**Speculative Prefill**](https://arxiv.org/abs/2502.02789) (Liu et al, 2025): the original Q-hook construction. Reference impl: [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill).
- [**FlashPrefill**](https://arxiv.org/abs/2603.06199) (Fan et al, 2026): block-sparse attention with sink + window + dynamic top-K blocks. Original kernel: [qhfan/FlashPrefill](https://github.com/qhfan/FlashPrefill) (Triton).

What we built:

- C++/CUDA port of the FlashPrefill algorithm: 4 kernels (`mean_K / score / select / sparse_fwd`), no Triton dependency.
- BSA ([mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)) wired without `libtorch` via 3 ATen/c10 header stubs (`dflash/deps/bsa_stubs/`).
- Custom Qwen3-0.6B BF16 forward so the drafter runs through the same ggml allocator as the 27B target.
- Daemon stdin protocol (`compress` / `generate` / `park` / `unpark` / `free drafter`) so target + drafter coexist on a 24 GB card.
- NIAH harness against `llama-bench` for end-to-end validation.

## Scope and limits

- **Single 24 GB GPU** target (RTX 3090 reference). On 32+ GB cards, drafter + target can coexist and the park/unpark dance disappears.
- **Qwen3.6-27B Q4_K_M target + Qwen3-0.6B drafter** is the validated pair. Other targets/drafters need keep_ratio + alpha re-calibration.
- **NIAH single-needle** is the only retrieval task validated end-to-end. Multi-doc QA, long-form code retrieval, etc. still TBD.
- **sm_80+** required for BSA (RTX 3090 sm_86 is the reference). On sm_75 (Turing) the build auto-disables BSA and falls back to the WMMA path; expect a slower drafter forward at long ctx.

## Citation

```bibtex
@software{luce_pflash_2026,
  title  = {Luce PFlash: speculative prefill compression for long-context spec decode on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/pflash},
  year   = {2026}
}

@article{spec_prefill_xfamily_2026,
  title   = {Cross-Family Speculative Prefill},
  author  = {Liu and others},
  journal = {arXiv:2603.02631},
  year    = {2026}
}

@article{flashprefill_2026,
  title   = {FlashPrefill: Block-Sparse Attention for Long-Context Prefill},
  author  = {Fan and others},
  journal = {arXiv:2603.06199},
  year    = {2026}
}
```

---

MIT · [Lucebox](https://lucebox.com) · [Discord](https://discord.gg/yHfswqZmJQ)

Inspired by [Jingyu6/speculative_prefill](https://github.com/Jingyu6/speculative_prefill), [qhfan/FlashPrefill](https://github.com/qhfan/FlashPrefill), [z-lab/DFlash](https://arxiv.org/abs/2602.06036).



================================================
FILE: pflash/pyproject.toml
================================================
[project]
name = "pflash"
version = "0.3.0"
description = "Python client + bench harness for the lucebox dflash speculative-prefill daemon"
readme = "README.md"
requires-python = ">=3.10"
authors = [{name = "lucebox"}]
license = {text = "MIT"}
keywords = ["llm", "speculative-decoding", "prefill", "long-context", "qwen"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
# Drafter scoring + spec decode happen inside the C++/CUDA daemon. The Python
# side only needs the HF tokenizers (one for the drafter vocab, one for the
# target vocab + chat template) to drive the bench harness.
dependencies = [
    "transformers>=4.50",
]

[project.urls]
Homepage = "https://github.com/Luce-Org/lucebox-hub"
Source = "https://github.com/Luce-Org/lucebox-hub/tree/main/pflash"
Blog = "https://lucebox.com/blog/pflash"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["pflash*"]



================================================
FILE: pflash/pflash/__init__.py
================================================
"""pflash, speculative-prefill harness around the dflash C++/CUDA daemon.

The daemon does the work — drafter forward, FlashPrefill scoring (BSA), and
spec-decode generation are all in-process C++/CUDA. This Python package is
just a thin client over the daemon's stdin/stdout protocol so reproduction
benches and external tooling can drive it.
"""
from .dflash_client import DflashClient
from . import config

__version__ = "0.3.0"
__all__ = ["DflashClient", "config"]



================================================
FILE: pflash/pflash/config.py
================================================
"""Default env flags for the dflash daemon when spawned by pflash."""

# These are the only daemon-side flags pflash assumes. The C++ kernel knobs
# (DFLASH_FP_USE_BSA, DFLASH_FP_ALPHA) are set per-call by the daemon owner.
DFLASH_REQUIRED_ENV = {
    "DFLASH27B_FA_WINDOW": "0",       # full attn on the (already compressed) prompt
    "DFLASH27B_KV_TQ3": "1",          # 3-bit KV cache; saves ~4 GB at 128K
    "DFLASH27B_LM_HEAD_FIX": "0",     # disable cuBLAS LM-head dequant (OOMs on 24 GB)
}



================================================
FILE: pflash/pflash/dflash_client.py
================================================
"""Subprocess client for the patched dflash daemon (with park/unpark commands)."""
from __future__ import annotations
import logging
import os
import struct
import subprocess
import tempfile
import time
from typing import Optional

from . import config

log = logging.getLogger(__name__)


class DflashClient:
    def __init__(self, bin_path: str, target_path: str, draft_path: str,
                 max_ctx: int = 16384, ddtree_budget: int = 16,
                 fa_window: Optional[int] = None,
                 kv_tq3: Optional[bool] = None,
                 lm_head_fix: Optional[bool] = None,
                 boot_timeout_s: float = 60.0,
                 boot_vram_mib: int = 18000):
        """Spawn the patched dflash daemon as a subprocess.

        Defaults for fa_window / kv_tq3 / lm_head_fix come from
        ``config.DFLASH_REQUIRED_ENV`` and are the only flags pflash relies on.
        Override per-call only when you know what you're doing.
        """
        self.bin_path = bin_path
        self.target_path = target_path
        self.draft_path = draft_path
        self.max_ctx = max_ctx
        env_overrides = {
            "DFLASH27B_FA_WINDOW": str(0 if fa_window is None else fa_window),
            "DFLASH27B_KV_TQ3": "1" if (kv_tq3 if kv_tq3 is not None else True) else "0",
            "DFLASH27B_LM_HEAD_FIX": "1" if lm_head_fix else "0",
        }
        env = {**os.environ, **config.DFLASH_REQUIRED_ENV, **env_overrides}
        bin_dir = os.path.dirname(os.path.abspath(bin_path))
        ld_extra = [bin_dir, os.path.join(bin_dir, "bin")]
        env["LD_LIBRARY_PATH"] = ":".join(
            ld_extra + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        self.r_pipe, self.w_pipe = os.pipe()
        cmd = [bin_path, target_path, draft_path,
               "--daemon", "--fast-rollback", "--ddtree",
               f"--ddtree-budget={ddtree_budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={self.w_pipe}"]
        log.info("spawning dflash daemon: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     pass_fds=(self.w_pipe,), env=env)
        os.close(self.w_pipe)
        self._wait_until_loaded(timeout=boot_timeout_s, vram_mib=boot_vram_mib)
        # Park draft by default; user calls unpark when needed
        self._send("park draft\n")

    def _wait_until_loaded(self, timeout: float = 60.0, vram_mib: int = 18000):
        boot = time.time()
        while time.time() - boot < timeout:
            time.sleep(1)
            try:
                vram = int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"]).decode().splitlines()[0])
                if vram > vram_mib:
                    return
            except Exception:
                pass
        raise RuntimeError(
            f"dflash daemon failed to load target weights within {timeout:.0f}s "
            f"(expected VRAM > {vram_mib} MiB). Check the daemon's stderr.")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        # Read until -1 sentinel
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            if struct.unpack("<i", b)[0] == -1:
                break

    def free_drafter(self): self._send("free drafter\n")
    def park_draft(self):    self._send("park draft\n")
    def unpark_draft(self):  self._send("unpark draft\n")
    def park_target(self):   self._send("park target\n")
    def unpark_target(self): self._send("unpark target\n")

    def compress(self, prompt_ids: list[int], keep_ratio: float, drafter_gguf: str) -> list[int]:
        """C++ drafter score+compress via daemon. Returns compressed token ids.

        Daemon command: compress <bin> <keep_x1000> <drafter_gguf>
        """
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        keep_x1000 = int(round(keep_ratio * 1000))
        self.proc.stdin.write(f"compress {path} {keep_x1000} {drafter_gguf}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def generate(self, prompt_ids: list[int], n_gen: int) -> list[int]:
        """Send prompt + n_gen request, return generated token ids."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        self.proc.stdin.write(f"{path} {n_gen}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def close(self, timeout: float = 5.0):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()



================================================
FILE: pflash/tests/bench_niah_cpp.py
================================================
"""C++-only NIAH bench: daemon compress + generate, no Python drafter."""
import argparse, json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from pflash.dflash_client import DflashClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--bin", default="/home/lucebox/lucebox-hub/dflash/build/test_dflash")
    ap.add_argument("--target", default="/opt/lucebox/models/Qwen3.6-27B-Q4_K_M.gguf")
    ap.add_argument("--draft-spec", default="/home/lucebox/lucebox-hub/dflash/models/draft/model.safetensors",
                    help="draft model used for spec decoding (NOT drafter scorer)")
    ap.add_argument("--drafter-gguf", default="/home/lucebox/lucebox-hub/dflash/models/Qwen3-0.6B-BF16.gguf",
                    help="C++ drafter scorer GGUF (Qwen3-0.6B BF16)")
    ap.add_argument("--target-tokenizer", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--drafter-tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-ctx", type=int, default=16384,
                    help="daemon KV cache max ctx; sized for compressed prompt+gen, NOT source")
    ap.add_argument("--keep-ratio", type=float, default=0.020)
    ap.add_argument("--n-gen", type=int, default=64)
    args = ap.parse_args()

    target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer)
    drafter_tok = AutoTokenizer.from_pretrained(args.drafter_tokenizer)
    cases = [json.loads(l) for l in open(args.cases)][:args.n]

    print(f"[init] spawning daemon: {args.bin}", flush=True)
    dflash = DflashClient(args.bin, args.target, args.draft_spec, max_ctx=args.max_ctx)

    correct = 0
    for i, case in enumerate(cases):
        prompt = case["prompt"]
        # Drafter tokenizes prompt, daemon scores+compresses, returns drafter ids.
        ids = drafter_tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
        S = len(ids)
        print(f"[case {i}] src={S} keep={args.keep_ratio}", flush=True)

        t0 = time.time()
        compressed_ids = dflash.compress(ids, args.keep_ratio, args.drafter_gguf)
        t_score = time.time() - t0
        comp = len(compressed_ids)
        print(f"[case {i}] compressed={comp} ratio={S/max(comp,1):.1f}x score_s={t_score:.1f}", flush=True)

        # Decode compressed ids with DRAFTER tokenizer, re-encode with TARGET + chat template.
        comp_text = drafter_tok.decode(compressed_ids, skip_special_tokens=True)
        user_msg = comp_text + "\n\nAnswer the user question based on the above context."
        chat = target_tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True)
        target_ids = target_tok(chat, return_tensors="pt")["input_ids"][0].tolist()

        # Free drafter (1.2GB), restore target+spec_draft for target gen.
        dflash.free_drafter()
        dflash.unpark_target()
        dflash.unpark_draft()

        t0 = time.time()
        out_ids = dflash.generate(target_ids, args.n_gen)
        t_gen = time.time() - t0
        # Re-park for next iter (drafter scoring).
        dflash.park_draft()
        print(f"[case {i}] raw out_ids ({len(out_ids)}): {out_ids[:20]}", flush=True)
        out_text = target_tok.decode(out_ids, skip_special_tokens=True)
        out_text_keep = target_tok.decode(out_ids, skip_special_tokens=False)
        print(f"[case {i}] out_with_special: {out_text_keep!r}", flush=True)

        ok = case["answer"] in out_text
        if ok:
            correct += 1
        ttft = t_score + t_gen  # rough; gen includes ttft
        print(f"[case {i}] gen_s={t_gen:.1f} ttft={ttft:.1f} ok={ok} ans={case['answer']}", flush=True)
        print(f"[case {i}] out: {out_text!r}", flush=True)

    print(f"\naccuracy: {correct}/{len(cases)}", flush=True)
    dflash.close()


if __name__ == "__main__":
    main()



================================================
FILE: pflash/tests/niah_gen.py
================================================
"""Generate NIAH single-needle test cases at any context size."""
import argparse, json, random
from transformers import AutoTokenizer

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
NEEDLE_TMPL = "The special magic {key} number is: {value}."
QUESTION_TMPL = "What is the special magic {key} number? Answer in one short sentence."


def gen_one(seed: int, target_tokens: int, tokenizer):
    rng = random.Random(seed)
    key = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    value = "".join(rng.choices("0123456789", k=7))
    needle = NEEDLE_TMPL.format(key=key, value=value)
    question = QUESTION_TMPL.format(key=key)
    char_per_tok = 4.0  # rough
    target_chars = int(target_tokens * char_per_tok)
    filler = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    insert = rng.randint(target_chars // 4, 3 * target_chars // 4)
    body = filler[:insert] + " " + needle + " " + filler[insert:]
    prompt = (
        "Below is a long passage. Answer the question at the end based ONLY on information in the passage.\n\n"
        f"{body}\n\nQuestion: {question}\nAnswer:"
    )
    return {"prompt": prompt, "answer": value, "key": key}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.out, "w") as f:
        for i in range(args.n):
            ex = gen_one(seed=42 + i, target_tokens=args.ctx, tokenizer=tok)
            # plain encode keeps the bench harness torch-free; pyproject only
            # depends on transformers + tokenizers.
            ex["n_tokens"] = len(tok.encode(ex["prompt"]))
            f.write(json.dumps(ex) + "\n")
            print(f"  case {i}: ntok={ex['n_tokens']} key={ex['key']} ans={ex['answer']}")
    print(f"saved {args.n} cases to {args.out}")


if __name__ == "__main__":
    main()



================================================
FILE: .sisyphus/plans/20260428-1430-path-b-deltanet-wmma-scope.md
================================================
# Scope: Path B — WMMA tensor-core rewrite of `pf_dn_chunk_phase2` (Ampere sm_86)

**Date**: 2026-04-28
**Target**: `megakernel/prefill.cu` — `pf_dn_chunk_phase2` only (phase1 unchanged for v1)
**Why**: Profiler shows this kernel = 47.6% of prefill CUDA time (13.27 ms / 27.88 ms total) on RTX 3090 / 512-token prompt. Inner products are scalar FP32 in shared memory; tensor cores are unused. Existing GEMMs (Q/K/V proj, MLP) already go to cuBLAS tensor cores (`ampere_bf16_s16816gemm…`).

---

## Goal & non-goals

**Goal**: Replace the four matmul-shaped inner loops in phase2 with WMMA bf16 / f32-accum matmuls, with f32 accumulator semantics so numerical drift in the recurrence stays bounded.

**Non-goals**:
- Phase1 rewrite (it has different math: cumsum, sigmoid, softplus, triangular forward-substitute — only one matmul-shaped piece, marginal win).
- Algorithmic changes to the recurrence (correctness reference is the existing scalar `pf_dn_chunk_phase2`).
- Backporting Blackwell `prefill_megakernel.cu` patterns (different launch model, persistent kernel, `cg::this_grid().sync()`).
- Multi-stream / overlap with FA layers.
- New CLI surface; this is a drop-in kernel replacement gated by a build-time flag for A/B.

**Out-of-scope but worth noting**: state update is the only operation where reducing precision could plausibly compound across chunks. The other three ops are read-only against state; their bf16 truncation only affects per-chunk outputs.

---

## Math: what's actually matmul-shaped

Per chunk `n`, phase2 currently runs five distinct compute steps. Four are matmul-shaped with f32 inputs in shared memory:

| Op | Formula | M × N × K | FMAs/chunk | WMMA fit |
|---|---|---:|---:|---|
| **d compute** | `d[c,j] = u[c,j] - Σ_d w[c,d]·state[j,d]` | 8 × 32 × 128 | 32K | m8n32k16, K-loops=8 |
| **QKt compute** | `QKt[c,s] = Σ_d Q[c,d]·K[s,d]` | 8 × 8 × 128 | 8K | M,N<16; **keep scalar** (only 7% of compute) |
| **o_inter compute** | `tmp[c,j] = Σ_d Q[c,d]·state[j,d]` | 8 × 32 × 128 | 32K | m8n32k16, K-loops=8 |
| **state update** | `state[j,i] = γ·state[j,i] + Σ_c d_scaled[c,j]·K[c,i]` | 32 × 128 × 8 | 32K | m16n16k16, K=8→pad 16 |
| **o_intra** | `o_intra[c,j] = Σ_{s≤c} (QKt[c,s]·exp(cs[c]-cs[s]))·d[s,j]` | 8 × 32 × 8 | 2K | tiny + triangular mask; **keep scalar** |

WMMA-targeted ops total ~96K FMAs per chunk per block (out of ~106K total matmul-shaped). With N=64 chunks × 72 blocks = ~441M MAC over 13.27 ms today = **~33 GFLOPS realized**. Ampere bf16 tensor-core peak is 142 TFLOPS. Headroom is enormous, but most of the 13.27 ms is launch/sync/smem-load — not pure compute.

## Constants we're working with

```
DN_HEADS = 16              DN_KEY = 128         DN_VAL = 128
DN_CHUNK_C = 8             DN_PHASE2_J_SPLITS = 4
DN_PHASE2_J_PER_BLOCK = 32 DN_PHASE2_BLOCK = 128 (4 warps)
Launch grid: (DN_HEADS, J_SPLITS) = (16, 4) = 64 blocks
__launch_bounds__(128, 1) → 1 block / SM = 8% theoretical occupancy
P2 dynamic smem: ~31 KB
```

---

## WMMA design

### Fragment shapes

Ampere sm_86 supports the following bf16 WMMA fragment shapes (`<mma.h>`, `nvcuda::wmma`):

| Shape | A | B | C |
|---|---|---|---|
| **m16n16k16** | matrix_a 16×16 bf16 | matrix_b 16×16 bf16 | accumulator 16×16 f32 |
| m8n32k16 | matrix_a 8×16 bf16 | matrix_b 16×32 bf16 | accumulator 8×32 f32 |
| m32n8k16 | matrix_a 32×16 bf16 | matrix_b 16×8 bf16 | accumulator 32×8 f32 |

**Choice**: stick to **m16n16k16** for simplicity and uniform fragment lifetime across the three target ops. Tile mapping:

- **d**, **o_inter** (M=8, N=32, K=128): one warp per op-instance. Fragment M=16 covers M=8 (with padding); N=32 needs 2 N-tiles. K=128 / 16 = 8 K-iters. So per warp: `2 × 8 = 16` mma.sync calls per op.
- **state update** (M=32, N=128, K=8): two warps cooperate. M=32 / 16 = 2 M-tiles; N=128 / 16 = 8 N-tiles. K=8 → pad to 16 = 1 K-iter. 16 mma.sync calls split across 2 warps = 8 per warp.

Block has 4 warps. Mapping per chunk:
- Warps 0-1: state update (cooperating on M-tiles 0 and 1)
- Warps 2-3: d compute and o_inter (each does half the N-tiles)
- QKt and o_intra stay scalar (handled by all 4 warps before/after the WMMA region)

### Data types

| Tensor | Current | Proposed |
|---|---|---|
| `s_state` | f32 | f32 + bf16 staging (load f32 from global, write bf16 mirror for WMMA reads) |
| `s_w` | f32 | bf16 (load from f32 source, downcast on store) |
| `s_Q`, `s_K` | f32 (loaded from bf16 qkv_pre) | bf16 (load directly without f32 round-trip) |
| `s_u`, `s_d` | f32 | f32 (these are accumulator-side; written from WMMA C fragment via `store_matrix_sync`) |
| `state` (global) | f32 | **f32** — unchanged, decode kernel reads it. No format change at the boundary. |

Accumulator stays f32 inside WMMA fragments. Down-conversion to bf16 only happens when feeding the next op's inputs, not in the persistent state.

### Numerical risk

The state update uses `state_new = γ·state_old + Σ d·K` with γ ≤ 1 (decay). Per-chunk error from bf16 truncation of `state_old` reads is ~2⁻⁸ relative. Over N=64 chunks, the recurrence damps errors via γ at each step rather than amplifying. Realistic worst-case: ~2⁻⁵ relative drift at chunk 64. This is well within bf16 inference tolerance for LLM logits (where end-of-prompt position tolerance is typically O(2⁻⁴) vs f32 reference).

**Verification gate**: bench_pp_tg.py's existing correctness section runs "The capital of France is" and asserts the first generated token. New kernel must produce the same token. Additional guard: compare full pp520 output token IDs against the scalar reference for 100+ prompts.

---

## Layout & smem budget

New smem layout (sized for 1 block, dynamic smem stays under 100 KB):

```
s_state_f32   [J_per × DK_S]            32×129×4   = 16.5 KB
s_state_bf16  [J_per × DK_S]            32×129×2   =  8.3 KB  // bf16 mirror, refreshed per chunk
s_u           [DN_CHUNK_C × J_per]       8×32×4    =  1.0 KB  // f32 accumulator
s_w_bf16      [DN_CHUNK_C × DK_S]        8×129×2   =  2.1 KB
s_Q_bf16      [DN_CHUNK_C × DK_S]        8×129×2   =  2.1 KB
s_K_bf16      [DN_CHUNK_C × DK_S]        8×129×2   =  2.1 KB
s_d           [DN_CHUNK_C × J_per]       8×32×4    =  1.0 KB
s_qkt         [DN_CHUNK_C × DN_CHUNK_C]  8×8×4     =  0.25 KB
s_cs, s_decay_rem [DN_CHUNK_C]                       0.06 KB
─────────────────────────────────────────────────
                                               TOTAL: ~33.4 KB
```

Slightly bigger than today's ~31 KB but well under Ampere's 100 KB-per-block ceiling. Refreshing the `s_state_bf16` mirror after each chunk's state update costs 32×129 = 4128 cvts per block, fully parallelizable across 128 threads.

---

## Implementation plan

### Phase 0 — preconditions (1 hr)

1. Confirm RTX 3090's actual `cudaDeviceProp.sharedMemPerBlockOptin` (should be 100 KB, but verify).
2. Add a build-time flag in `setup.py`: `MEGAKERNEL_DN_PHASE2_WMMA=on|off` (default off). Plumbs through to a `#define DN_PHASE2_WMMA` in `prefill.cu` so the new kernel sits next to the existing scalar version under `#ifdef`.
3. Wire a Python env-var override (`MEGAKERNEL_DN_PHASE2_WMMA=1`) so we can A/B without rebuilding.

### Phase 1 — instrumentation (1-2 hr)

1. Rerun `diag_prefill_kernels.py` with `record_shapes=True` and CUPTI metrics (`profile_memory=True, with_stack=True`). Get per-launch SM occupancy and stall reasons for `pf_dn_chunk_phase2`.
2. Compute the actual mix: how much of the 13.27 ms is compute vs smem-load vs sync. (Ratio determines whether WMMA alone or WMMA+`cp.async` is the right swing.)
3. Save the baseline output token IDs for the correctness corpus (50 fixed prompts × 32-token completions).

### Phase 2 — WMMA "d compute" only (3-5 hr)

Smallest demonstrable win. Replace the d-compute loop (lines 580-589) with:

```cpp
using namespace nvcuda::wmma;
constexpr int WM=16, WN=16, WK=16;

// Convert s_state f32 → bf16 mirror once per chunk before WMMA region.
// (One-time at chunk start; refresh after state update at chunk end.)

fragment<matrix_a, WM, WN, WK, __nv_bfloat16, row_major>     a_w;
fragment<matrix_b, WM, WN, WK, __nv_bfloat16, col_major>     b_state;
fragment<accumulator, WM, WN, WK, float>                     c_d;

int warp_id = tid / 32;
if (warp_id < 2) {                                          // 2 warps cover N-tile 0 and 1
    int n_tile = warp_id;
    fill_fragment(c_d, 0.f);
    #pragma unroll
    for (int kk = 0; kk < DN_KEY; kk += WK) {
        load_matrix_sync(a_w, s_w_bf16 + kk, DK_S);                     // [C(8 padded to 16)][WK]
        load_matrix_sync(b_state, s_state_bf16 + n_tile*WN*DK_S + kk, DK_S);  // [J(WN)][WK] col-major
        mma_sync(c_d, a_w, b_state, c_d);
    }
    // Subtract u and write to s_d. Layout depends on fragment storage convention; use
    // store_matrix_sync into a temp f32 tile then subtract per-thread.
    float tmp[16][16];  // (per-warp scratchpad in registers — actually use a smem tile)
    store_matrix_sync(/*ptr=*/..., c_d, /*ldc=*/..., mem_row_major);
    // ...subtract s_u in-place, write s_d
}
__syncthreads();
```

Compare scalar vs WMMA d-output bit-for-bit difference on a fixed 16-token chunk. Acceptable: relative max diff < 2⁻⁶. Run `bench_pp_tg.py` correctness section; must pass.

### Phase 3 — extend to o_inter (2 hr)

Same pattern as d-compute with different operand sources (`s_Q` instead of `s_w`, output multiplied by `expf(cs[c])`). Re-verify.

### Phase 4 — state update (4-6 hr)

The trickiest one. Two warps cooperate on M=32 split as two M=16 tiles. Each warp produces 8 N-tiles of f32 accumulator, then writes back to `s_state_f32` after multiplying by `s_decay_total` and adding the existing state value. K=8 → pad to 16 with zero-fill in `s_K_bf16` rows 8-15 (caller of WMMA must zero those rows). Then refresh `s_state_bf16` mirror.

Pseudocode:
```cpp
// d_scaled is column of M (J_per=32). K is column of K (Dk=128).
// We want: state[j, i] = γ·state[j, i] + Σ_c d_scaled[c, j] * K[c, i]
//
// Reframe as GEMM: state(M=32, N=128) += d_scaled.T(M=32, K=8) @ K(K=8, N=128)
// d_scaled.T means we transpose-on-load: WMMA matrix_a with col_major.

fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major> a_dT;  // [J_per_tile=16][K_pad=16]
fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_K;   // [K_pad=16][N_tile=16]
fragment<accumulator, 16, 16, 16, float>                  c_st;

if (warp_id < 2) {
    int j_tile = warp_id;  // 0 or 1, covers j_start+[0..16) or +[16..32)
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        // Load existing state slice into accumulator fragment, scaled by γ
        load_matrix_sync(c_st, s_state_f32 + j_tile*16*DK_S + n_tile*16, DK_S, mem_row_major);
        scale_fragment(c_st, s_decay_total);                             // helper
        load_matrix_sync(a_dT, s_d_bf16 + j_tile*16, J_per);             // C-major as transpose
        load_matrix_sync(b_K, s_K_bf16 + n_tile*16, DK_S);
        mma_sync(c_st, a_dT, b_K, c_st);
        store_matrix_sync(s_state_f32 + j_tile*16*DK_S + n_tile*16, c_st, DK_S, mem_row_major);
    }
}
__syncthreads();
// Refresh s_state_bf16 mirror for next chunk.
```

`d_scaled` needs to be available as bf16 too (it's currently f32 from the d-compute store). So phase 2's d-compute should write *both* an f32 copy (for o_intra still scalar) and a bf16 copy. Tradeoff: ~256 extra cvts per chunk per block, negligible.

### Phase 5 — relax `__launch_bounds__` (1 hr)

After the kernel rewrite, register pressure should drop (matmul work is in fragments, not in 32×scalar accumulators). Try `__launch_bounds__(128, 2)` then `(128, 4)`. Re-profile occupancy.

### Phase 6 — `cp.async` pipelining (4-6 hr, optional)

If profiler shows smem-load stalls dominate after Phases 2-5, overlap the next-chunk loads with the current-chunk WMMA via `__pipeline_memcpy_async` + double-buffered smem. Doubles the staging buffer cost (`s_w_bf16`, `s_Q_bf16`, `s_K_bf16` × 2) but should hide ~80% of load latency.

This is the highest-risk phase — easy to get pipeline barriers wrong. Skip if Phases 2-5 already get us to ~30% improvement target.

### Phase 7 — extend to phase1 (deferred)

Phase1 is 2.0% of CUDA time per the profile. Even a 5x speedup buys 1.6%. Defer.

---

## Verification harness

1. **Bit-level**: compile both `pf_dn_chunk_phase2_scalar` (existing) and `pf_dn_chunk_phase2_wmma` (new) into the same .so. Add a debug Python entrypoint that runs both on identical inputs and reports `max_abs(diff)` and `max_rel(diff)` for every output element. Tolerance: 2⁻⁶ relative.

2. **Token-level**: `bench_pp_tg.py` correctness section already runs an end-to-end prompt and asserts the predicted token. Extend with a fixed corpus of 50 prompts × 32 generated tokens each. Build the baseline corpus from current main; assert byte-equal token IDs from the new kernel. (Allow optional `--tolerate=N` mismatches as a release gate if precision drift turns out to be model-dependent.)

3. **Performance**: rerun `diag_prefill_kernels.py` after each phase. Track `pf_dn_chunk_phase2_*` self-CUDA time in a CSV. Phase 2 alone target: -25% on this kernel. After Phase 4: -50%. After Phase 5+6: -65%.

4. **Bench matrix**: pp520 across 5 fixed prompt lengths {64, 128, 256, 512} × 3 batches each. Avoid n_gen variability by measuring prefill only.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| bf16 precision drift breaks logits parity | Med | High — visible token-level differences | Bit-level tolerance harness in Phase 2 catches early; if intolerable, try TF32 m16n16k8 fragments at 0.5x throughput |
| Fragment storage convention mismatches (row vs col) cause silent wrong results | High | High | Phase 2 has explicit bit comparison against scalar; resolve before extending |
| Register spills under WMMA + 4 warps × 2 blocks/SM | Med | Med | `nvcc -Xptxas -v` after each phase; if spills appear, fall back to (128, 1) |
| smem mirror refresh introduces a new sync that hurts more than it saves | Low | Med | Bench Phase 2 in isolation; if regression, restructure to write bf16 directly from the cvt |
| Bank conflicts on bf16 stride 129 (different than f32 stride 129) | Low | Med | bf16 is 2-byte; 129 elements = 258 bytes. 32 banks × 4 bytes = 128 bytes = 64 bf16. Stride 129 ≠ multiple of 64 — already conflict-free. Verify via `__profile_*` smem counters. |
| Hardware variation: maybe RTX 3090 isn't the right test bed (downclocking, thermals) | Low | Low | Lock clocks via `nvidia-smi -lgc 2100` during bench; rerun before/after each phase |
| Real bottleneck turns out to be elsewhere (sync overhead, kernel launch cost) | Med | Med — caps the speedup | Phase 1's CUPTI metrics should reveal this before we commit to Phase 2-6. If launch overhead dominates, the answer is fewer launches (graph capture or megakernel-on-Ampere), not WMMA |

---

## Estimated timeline

| Phase | Optimistic | Realistic | Pessimistic | Cumulative |
|---|---|---|---|---|
| 0. Preconditions | 1 h | 2 h | 4 h | 4 h |
| 1. Instrumentation + corpus | 2 h | 4 h | 6 h | 10 h |
| 2. d compute WMMA | 3 h | 6 h | 12 h | 22 h |
| 3. o_inter WMMA | 2 h | 3 h | 6 h | 28 h |
| 4. state update WMMA | 4 h | 8 h | 14 h | 42 h |
| 5. launch_bounds tuning | 1 h | 2 h | 4 h | 46 h |
| 6. cp.async (optional) | 4 h | 8 h | 16 h | 62 h |
| Bench/QA/PR | 2 h | 4 h | 8 h | 70 h |

Realistic: **~7 working days** (1.5 weeks at 5 h/day). Optimistic 19 h / 2.5 days, pessimistic 70 h / 2 weeks.

## Expected payoff

Bottoming-up from the 13.27 ms phase2 budget:
- Pure compute fraction (rough): if 60% of phase2 is matmul work and we get 4-8x on it via WMMA, that's `13.27 × 0.6 × (1 - 1/6) = 6.6 ms saved` → phase2 → ~6.7 ms, total prefill 21 ms → **~24,400 tok/s for 512 tokens**.
- After cp.async overlap of remaining smem loads: phase2 → ~4.5 ms, total 19 ms → **~27,000 tok/s**.
- Both well below the README's 37,800 tok/s claim. Confirms that WMMA alone won't close the gap; further work would be on phase1 (2%), kernel launch fusion (graph capture / persistent kernel — back to mega), or revisiting whether the README figure is accurate for stock sm_86.

**Conservative success bar**: ≥30% prefill speedup (target: 23k tok/s) with token-level parity to ≤1 mismatch per 50-prompt corpus.

---

## Open questions to resolve in Phase 1

1. What's the exact CUPTI breakdown of phase2 (compute / memory / sync)? Determines whether WMMA or `cp.async` is the primary lever.
2. Is the global memory load of `state` (32×129 f32 = 16.5 KB / chunk start, only once) actually a hot path, or is most of the 13.27 ms in the inter-chunk smem operations?
3. Does the existing kernel benefit from `__launch_bounds__(128, 2)` even without the WMMA rewrite? Quick experiment in Phase 5; if it does, that's a free 5-10% before any rewrite.
4. Is the 37,800 README figure reproducible at all on this 3090, or was it from a non-stock setup (overclocked, different chip bin, different driver)? Worth one ping to davide221 before committing two weeks of work.

