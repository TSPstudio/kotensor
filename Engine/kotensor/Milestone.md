# Kotensor Roadmap & Industry Competency Milestones

This document specifies the technical requirements, algorithmic designs, and architectural additions necessary to transform Kotensor from an educational autograd and Transformer proof-of-concept into a competitive, production-grade on-device deep learning engine.

---

## The Strategic Positioning

Kotensor does not aim to replace C++ runtimes (`llama.cpp`, `ONNX Runtime`, `TensorRT`) for multi-billion-parameter server inference. Instead, Kotensor dominates an underserved market niche:

1. **Zero Native Dependencies**: 100% pure Kotlin and standard JDK/Android I/O. Zero JNI overhead, zero `.so`/`.dll` binary bloat, and zero multi-architecture ABI splitting (`arm64-v8a`, `armeabi-v7a`, `x86_64`).
2. **First-Class Edge Learning**: Unlike inference-only edge engines, Kotensor retains full reverse-mode autograd, optimizers, and parameter-efficient LoRA training directly in device memory.
3. **Seamless Multiplatform Integration**: Drops into any JVM backend, desktop application, or Android APK with a single Gradle line.

---

## Milestone Breakdown

```
v0.1.0 (Current) ───► v0.2.0 (Inference & Vocab) ───► v0.3.0 (Performance & Concurrency)
                             │                                 │
                             ▼                                 ▼
                      v0.4.0 (Model Importers)        v0.5.0 (Quantization)
                             │                                 │
                             └────────────────► v1.0.0 (Production)
```

---

### Milestone 1 (v0.2.0): Autoregressive Inference & Subword Tokenization

**Goal:** Enable real-world vocabulary coverage and eliminate quadratic inference latency.

#### 1. Key-Value (KV) Caching
* **Files:** [`Attention.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Attention.kt), [`TransformerBlock.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/TransformerBlock.kt), [`Transformer.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Transformer.kt)
* **Problem:** Generating token $t$ currently re-computes attention over all previous $1 \dots t-1$ tokens, scaling total generation time to $O(T^2)$.
* **Technical Specification:**
  - Create `KVCache` holding pre-allocated flat `FloatArray` buffers for keys and values across all layers:
    $$\text{Shape: } [\text{layers}, \text{heads}, \text{maxSeqLen}, \text{headDim}]$$
  - Add decode-mode flag to [`Attention.forward`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Attention.kt):
    - When `step > 0`, input token length is $T = 1$.
    - Compute $Q$ for the current single token.
    - Compute $K_{\text{new}}$ and $V_{\text{new}}$ for the single token and write into `KVCache` at position `step`.
    - Perform causal attention dot product $Q \cdot K_{\text{cached}}^T$ with shape $[1, \text{step} + 1]$.
  - Reduces per-token generation step to $O(1)$ time complexity relative to sequence position.

#### 2. Byte-Pair Encoding (BPE) Tokenizer
* **Files:** [`Tokenizer.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Tokenizer.kt)
* **Problem:** Character-level [`SimpleTokenizer`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Tokenizer.kt) cannot load real pre-trained weights or handle natural language vocabularies (e.g. LLaMA, Qwen, SmolLM).
* **Technical Specification:**
  - Implement `BPETokenizer : Tokenizer` with zero external JSON libraries:
    - Lightweight, zero-dependency parser for Hugging Face `tokenizer.json`.
    - Vocabulary rank table mapping token pairs to merge ranks.
    - Byte-level fallback (mapping raw UTF-8 bytes to unicode code points) ensuring 100% tokenization coverage without out-of-vocabulary crashes.
  - Special token support: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`.

---

### Milestone 2 (v0.3.0): Compute Performance & Memory Efficiency

**Goal:** Accelerate execution from seconds-per-token to interactive UI streaming speeds.

#### 1. Multi-Threaded Tiled Matrix Multiplication
* **Files:** [`Tensor.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Tensor.kt)
* **Problem:** Naive $O(n^3)$ triple loop runs single-threaded, ignoring multi-core CPUs and CPU L1/L2 cache lines.
* **Technical Specification:**
  - **Cache Tiling / Blocking:** Partition matrices $A [M, K]$ and $B [K, N]$ into cache-friendly blocks ($B_M \times B_K \times B_N$) fitting in L1/L2 cache lines ($32 \times 32$ or $64 \times 64$).
  - **Multi-Threading:** Distribute row tiles across worker threads using Java `ForkJoinPool` or `ThreadPoolExecutor` sized to `Runtime.getRuntime().availableProcessors()`.
  - **Loop Unrolling & Register Accumulation:** Inner dot-products unrolled by 4 or 8 accumulators to maximize instruction-level parallelism (ILP).

#### 2. Zero-Allocation Inference Engine
* **Files:** [`Tensor.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Tensor.kt), [`Transformer.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Transformer.kt)
* **Problem:** Frequent allocation of intermediate `FloatArray` instances during autoregressive generation triggers JVM Garbage Collection pauses.
* **Technical Specification:**
  - Implement static activation buffers and in-place ops (`matmulInto`, `addInto`, `siluInPlace`) reserved for inference loops.
  - Zero heap allocations inside the inner `generateStream` loop.

---

### Milestone 3 (v0.4.0): Architecture Generalization & Model Importers

**Goal:** Load and run standard open-source pre-trained checkpoints directly from Hugging Face.

#### 1. Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
* **Files:** [`Attention.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Attention.kt)
* **Problem:** Modern models (Llama 3, SmolLM, Qwen 2.5) use $N_{KV} < N_Q$ to reduce KV cache size.
* **Technical Specification:**
  - Add parameter `numKeyValueHeads: Int` to [`Attention`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/Attention.kt).
  - Broadcast KV heads across query groups during dot-product attention calculation.

#### 2. Automated Hugging Face Config & Weight Importer
* **Files:** New `HuggingFaceModelLoader.kt`, [`SafeTensors.kt`](file:///c:/Users/Dell/Desktop/Kotensor/Engine/kotensor/src/main/kotlin/SafeTensors.kt)
* **Technical Specification:**
  - Minimal JSON parser for `config.json` extracting:
    `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `rms_norm_eps`, `rope_theta`.
  - Declarative weight name mapping converting Hugging Face standard keys to Kotensor structure:
    - `model.layers.{i}.self_attn.q_proj.weight` $\rightarrow$ `layers.{i}.attention.wQ.weight`
    - `model.layers.{i}.mlp.gate_proj.weight` $\rightarrow$ `layers.{i}.feedForward.wGate.weight`
    - `model.layers.{i}.input_layernorm.weight` $\rightarrow$ `layers.{i}.attentionNorm.weight`

---

### Milestone 4 (v0.5.0): Edge Quantization (INT8 / INT4)

**Goal:** Enable running models on consumer mobile devices with strict RAM ceilings.

#### 1. Memory Profile Breakdown
* A 135M parameter model in FP32 requires $\approx 540\text{ MB}$.
* In weight-only INT8: $\approx 145\text{ MB}$.
* In weight-only INT4 (GGUF / AWQ style): $\approx 75\text{ MB}$.

#### 2. Technical Specification
* **Class:** `QuantizedTensor(val qdata: ByteArray, val scales: FloatArray, val zeros: ByteArray?, val shape: IntArray)`
* **Dequantizing MatMul (`matmulDequant`)**:
  - Keep activations in FP32.
  - Dequantize weights on-the-fly inside the inner matrix multiplication dot-product accumulation:
    $$W_{ij} = \text{scale} \cdot (Q_{ij} - \text{zero})$$
  - Never allocate the full FP32 weight matrix in RAM.

---

### Milestone 5 (v1.0.0): Production Hardening & Ecosystem Integration

**Goal:** Official release ready for commercial and enterprise application adoption.

| Component | Deliverable |
| :--- | :--- |
| **Android Sample** | New `:sample-android` module with Jetpack Compose UI demonstrating live token streaming, on-device memory telemetry, and real-time LoRA fine-tuning. |
| **API Documentation** | Automated Dokka build pipeline generating searchable HTML documentation. |
| **Maven Central** | Sonatype Central Portal release under `io.kotensor:kotensor` with automated GPG signing and SHA-256 validation. |
| **Benchmark Suite** | JMH (Java Microbenchmark Harness) benchmarking FLOPs, memory allocations, and tokens/second across sequence lengths. |

---

## Comparison Matrix

| Capability | PyTorch / Python | llama.cpp | ONNX Runtime Mobile | Kotensor (Target v1.0.0) |
| :--- | :--- | :--- | :--- | :--- |
| **Dependencies** | Python, CUDA, C++ | C++, CMake, JNI | C++, JNI, Prebuilt `.so` | **Zero (Pure Kotlin)** |
| **Binary Footprint** | Gigabytes | $5 - 20\text{ MB}$ native `.so` | $15 - 40\text{ MB}$ native `.so` | **$< 100\text{ KB}$ JAR** |
| **Reverse-Mode Autograd** | Yes | No (Inference only) | Complex / Limited | **Yes (Full backprop)** |
| **On-Device LoRA Training**| Yes | No | No | **Yes (Built-in)** |
| **SafeTensors Direct I/O** | Via Python | Via converter | Via converter | **Yes (Pure Kotlin)** |
| **Android Setup Friction** | High (Chaquopy/Termux) | Medium (NDK/CMake) | Medium (ABI packaging) | **None (`implementation`)** |
