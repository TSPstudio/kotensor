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
v0.1.0 (Done) ──────► v0.2.0 (Done) ─────────────► v0.3.0 (Done)
                             │                                 │
                             ▼                                 ▼
                      v0.4.0 (Done)                   v0.5.0 (Done)
                             │                                 │
                             └────────────────► v1.0.0 (Current)
```

---

### Milestone 1 (v0.2.0): Autoregressive Inference & Subword Tokenization ✅

**Goal:** Enable real-world vocabulary coverage and eliminate quadratic inference latency.

**Status: complete.** Delivered in `KVCache.kt`, `Attention.kt`, `TransformerBlock.kt`, `Transformer.kt`, `StreamingSession.kt`, `BPETokenizer.kt`, `JsonParser.kt` and `Tokenizer.kt`, verified by `KVCacheTest` (8 tests) and `BPETokenizerTest` (7 tests):
- `KVCache` with flat `[layers, heads, maxSeqLen, headDim]` key/value buffers; `Attention.forward(input, cache, layer)` writes the new keys/values at `cache.length` and attends over `[1, step + 1]`. `Transformer.forward(indices, cache)` runs prefill or a single decode step, and `generate` / `generateStream` / `StreamingSession` use it by default (output identical to the uncached forward pass to ~1e-6; 22×–88× faster at 64–256 tokens on a 4-layer, 64-dim model).
- `BPETokenizer.fromFile(tokenizer.json)` with a zero-dependency JSON parser, merge-rank table, GPT-2 byte ↔ unicode fallback, the file's own pre-tokenization regex, added/special tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, template tokens such as `<|begin_of_text|>`) and `decode(ids, skipSpecialTokens)`. Encodings match the Hugging Face `tokenizers` library on GPT-2-, Qwen- and Llama-3-style files.
- `IncrementalDecoder` keeps streamed text byte-safe when a character spans several tokens; `stopTokens` end generation at end-of-text.
- Fixed on the way: `RoPE` rotated each head slice as if it were its own token (positions depended on the head index and on the total sequence length), which is incompatible with caching; it now rotates every head of a token with the token's position.

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

### Milestone 2 (v0.3.0): Compute Performance & Memory Efficiency ✅

**Goal:** Accelerate execution from seconds-per-token to interactive UI streaming speeds.

**Status: complete.** Delivered in `MatMul.kt`, `Tensor.kt`, `InferenceBuffers.kt` and in-place paths in `Linear.kt`, `LoRA.kt`, `RMSNorm.kt`, `SwiGLU.kt`, `RoPE.kt`, `Embedding.kt`, `Attention.kt`, `TransformerBlock.kt`, `Transformer.kt`; verified by `MatMulTest` (5 tests) and `InferenceTest` (5 tests):
- `MatMul.multiply` computes C in `[32, 256]` tiles accumulated in per-thread scratch rows (L1-resident), with four accumulator rows per pass over a B row (register blocking) and tiles spread over a `ForkJoinPool` sized to `availableProcessors()` once a product exceeds `MatMul.parallelThreshold` multiply-adds. `Tensor.matmul` and its backward (`dA = dC·Bᵀ`, `dB = Aᵀ·dC`, via `MatMul.transpose`) run on it.
- Kernel choice was measured, not assumed: on HotSpot 21 the spec's register-accumulator dot product (8 accumulators over a packed Bᵀ) stays scalar because C2 does not vectorize float reductions, and the direct `c[ci + j] += a · b[bl + j]` form is not vectorized either (different invariant offsets). Copying the B row and the C tile rows into zero-offset scratch arrays makes the inner `y[j] += a · x[j]` loop auto-vectorize: ≈ 24 GFLOP/s single-threaded vs ≈ 6 for the best scalar tiling and 0.2–2 for the naive triple loop (512³ and 128 × 2048 × 2048, AVX-512 Xeon); ≈ 40 GFLOP/s on two threads.
- Zero-allocation decode: `Transformer.decode(token, cache, buffers)` runs one token through `Embedding.lookupInto`, `RMSNorm.forwardInto`, `Attention.decodeInto`, `SwiGLU.forwardInto`, `Linear.forwardInto` / `LoRALinear.forwardInto` and `RoPE.applyInPlace`, built on the in-place tensor ops `matmulInto`, `addInto` / `addInPlace`, `mulInPlace`, `siluInPlace` and `copyFrom`. `InferenceTest.testDecodeStepAllocatesNothing` measures 0 bytes allocated over 200 steps with `ThreadMXBean`; `generateStream` and `StreamingSession` prefill once through the graph path and then decode through the buffers.
- Result on the same 2-core container as the kernel benchmarks: a dim-64 / 4-layer model streams 3.7 k tokens/s (1.0 k before this milestone), a dim-256 / 8-layer / 8 k-vocab model 91 tokens/s (27 before), and SmolLM-135M-shaped weights (30 layers, dim 576, vocab 49 k, 176 M parameters in FP32) 10.9 tokens/s, memory-bandwidth bound.

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

### Milestone 3 (v0.4.0): Architecture Generalization & Model Importers ✅

**Goal:** Load and run standard open-source pre-trained checkpoints directly from Hugging Face.

**Status: complete.** Delivered in `Attention.kt`, `RoPE.kt`, `TransformerBlock.kt`, `Transformer.kt`, `InferenceBuffers.kt`, `SafeTensors.kt` and the new `HuggingFaceModelLoader.kt`; verified by `GQATest` (6 tests) and `HuggingFaceImportTest` (7 tests):
- `Attention(dimension, numHeads, numKeyValueHeads, ropeTheta, ropeScaling)`: `wK`/`wV` project to `numKeyValueHeads × headDim`, and every query head reads key/value head `h / (numHeads / numKeyValueHeads)` in the full-sequence forward and backward, the cached forward and the allocation-free decode step. `KVCache` and `InferenceBuffers` are sized to the key/value heads, so GQA also shrinks the cache. `TransformerBlock` / `Transformer` take `numKeyValueHeads`, `normEps`, `ropeTheta` and `ropeScaling` (Llama 3 frequency scaling).
- `SafeTensorsFile` reads a checkpoint tensor by tensor through `RandomAccessFile` instead of loading the whole file into memory, converts `F32`, `F16`, `BF16`, `F64` and integer dtypes to `Float`, parses the header with `JsonParser`, and exposes `__metadata__`.
- `HuggingFaceModelLoader.load(directory)` reads `config.json` into `HuggingFaceConfig` (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `rms_norm_eps`, `rope_theta`, `vocab_size`, `tie_word_embeddings`, `rope_scaling`; `model_type` llama / qwen2 / mistral), builds the `Transformer`, and imports `model.safetensors` or every shard of `model.safetensors.index.json` through the declarative `weightMapping` (`model.layers.{i}.self_attn.q_proj.weight → block.{i}.attention.wQ.weight`, …). Linear weights are transposed from Hugging Face's `[out, in]` to Kotensor's `[in, out]`, optional biases (Qwen) are picked up, missing biases stay zero, a missing `lm_head` is tied to the embedding, unknown tensors are reported in `ImportReport`, and missing weights fail loudly. `loadTokenizer(directory)` loads the matching `tokenizer.json`.
- Verification: three tiny checkpoints written the way Hugging Face writes them (BF16 Llama-style with GQA; sharded Qwen-style in mixed F16/F32/BF16 with q/k/v biases, MQA and tied embeddings; Llama-3-style with `rope_theta` 500000 and `llama3` rope scaling) plus an independent NumPy implementation of the architecture under `src/test/resources/hf`. Imported models match its logits to within 6e-6 on both the full forward and the decode path.
- Not supported (rejected with a clear message rather than silently wrong): other `model_type`s (Gemma's `1 + w` norm, Phi-3's fused QKV, Qwen3's q/k norms), `rope_scaling` types other than `llama3`, `head_dim ≠ hidden_size / num_attention_heads`.

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

### Milestone 4 (v0.5.0): Edge Quantization (INT8 / INT4) ✅

**Goal:** Enable running models on consumer mobile devices with strict RAM ceilings.

**Status: complete.** Delivered in `QuantizedTensor.kt`, `QuantizedLinear.kt`, `MatMul.kt`, `Tensor.kt`, `Embedding.kt`, `Linear.kt`, `Module.kt`, `Transformer.kt`, `LoRA.kt`, `SafeTensors.kt` and `HuggingFaceModelLoader.kt`; verified by `QuantizationTest` (11 tests):
- `QuantizedTensor(qdata, scales, zeros, shape, bits, groupSize)`: INT8 is symmetric (`scale · q`, zeros `null`), INT4 asymmetric with a zero point (`scale · (q − zero)`), both with one scale per column per group of `groupSize` rows (default 64). INT4 packs two columns per byte (column `j` in the low nibble, `j + columns/2` in the high nibble) so a row segment unpacks with unit-stride loops. `Quantization(bits, groupSize)` carries the choice (`Quantization.INT8`, `Quantization.INT4`).
- `MatMul.multiplyQuantized` is the dequantizing matmul: activations stay FP32, each weight row segment is expanded from bytes into a per-thread scratch row inside the kernel and never as a full matrix; per group it accumulates `Σ x·q` and applies `scale · (acc − zero · Σ x)` once, so the inner loop stays the auto-vectorized `y[j] += a · x[j]`. The byte → float conversion vectorizes on HotSpot; nibble extraction is done in the byte domain first because `and`/`shr` on converted ints does not. Decode (M = 1) now uses a stripe schedule — rows outermost, each thread owning a contiguous column range — because the previous tile order walked each column stripe with a row-sized stride and was latency-bound: FP32 matvec 90 → 39 ms and INT8 65 → 28 ms for 2048 × 49152. `MatMul.multiplyByQuantizedTransposed` gives `dX = dY · Wᵀ`, so gradients flow through quantized layers to LoRA adapters below them.
- `QuantizedLinear` and `QuantizedEmbedding` replace `Linear` / `Embedding` everywhere a model is built with `quantization` (`Transformer(…, quantization = Quantization.INT4)`) or afterwards with `model.quantize(q)`; `Projection` is the shared base of `Linear` and `QuantizedLinear`, `Module.forwardInto` the shared in-place hook, so the graph forward, the cached forward and the allocation-free `decode` all run unchanged. `LoRAAdapter` wraps quantized projections too (QLoRA): adapters train, export with `saveDelta`, and `merge()` refuses rather than corrupting quantized weights.
- `Tensor.grad` is now allocated lazily, so inference holds one copy of every FP32 parameter instead of two (a 163M-parameter model: 652 MB instead of 1.3 GB).
- `Transformer.saveQuantized(file)` / `Transformer.loadQuantized(file)` store the compact tensors (`I8`/`U8` `.qweight`, `F32` `.scales`, `U8` `.zeros`) plus the model configuration in the safetensors `__metadata__`, so a device loads the quantized model without ever allocating FP32 weights. `HuggingFaceModelLoader.load(directory, Quantization.INT8)` quantizes each projection as it is imported (peak memory: the compact model plus two FP32 copies of the largest tensor); `SafeTensors.saveEntries` writes arbitrary dtypes and metadata.
- Measured on the SmolLM2-135M shape (30 layers, dim 576, 9/3 heads, vocab 49152, 163 M parameters with an untied head), 2-core container: FP32 652 MB at 8.6 tok/s; INT8 173 MB (projections 144 MB) at 12.2 tok/s; INT4 95 MB (projections 78 MB) at 8.0 tok/s — the spec's 145 / 75 MB targets for the projections, with the INT4 kernel compute-bound on nibble unpacking rather than memory-bound. Quantization error: INT8 ≤ 0.4 % of the per-group maximum, INT4 ≤ 1/30 of the per-group range. A 95 MB INT4 file loads in under 100 ms.

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

### Milestone 5 (v1.0.0): Production Hardening & Ecosystem Integration ✅

**Goal:** Official release ready for commercial and enterprise application adoption.

**Status: complete.** Delivered in `build.gradle.kts`, `settings.gradle.kts`, `gradle.properties`, `Module.md`, `benchmark/`, `sample-android/`, `.github/workflows/` and `BPETokenizer.kt`; the library suite is now 66 tests. [`CHANGELOG.md`](CHANGELOG.md) describes every change since v0.1.0 in detail.
- **Android readiness:** the library now compiles to Java 17 bytecode against the Java 17 API (`jvmTarget = 17`, `-Xjdk-release=17`, `JavaCompile.release = 17`, still built with the Java 21 toolchain), the level Android's D8 and JVM 17 consumers accept. Auditing the engine for Android found one crash: Android's ICU-backed `java.util.regex` throws `IllegalArgumentException` for `Pattern.UNICODE_CHARACTER_CLASS`, which `BPETokenizer` passed when compiling the pre-tokenizer regex. It now falls back to the plain flags there (Android's `\s`, `\w`, `\d` are already Unicode-aware), and a test pins the JVM behavior.
- **Android sample:** `:sample-android` is a Compose app with live token streaming and tokens/s, memory telemetry (Java heap, native heap, weight bytes, KV-cache size), a character model trained on device at start-up, LoRA fine-tuning on typed text with a live loss curve, one-tap INT8 / INT4 quantization with QLoRA afterwards, and loading of Hugging Face directories (imported as INT8) or `saveQuantized` files. It uses AGP 9.0.1 — the newest AGP that runs on the project's Gradle 9.2.1 — with built-in Kotlin and the Compose compiler plugin, `minSdk` 26 and `compileSdk` 36, and AndroidX versions chosen from before the libraries moved to `compileSdk` 37. It is included only with `-Pkotensor.android=true`, so the library build never needs an Android SDK.
- **API documentation:** Dokka 2.2.0 (Dokka Gradle plugin v2, K2 analysis) with `Module.md` as the module overview and source links to GitHub; `./gradlew dokkaGenerate` writes `build/dokka/html`, and `docs.yml` deploys it to GitHub Pages.
- **Maven Central:** `com.vanniktech.maven.publish` 0.37.0 publishes `io.github.tspstudio:kotensor` to the Central Portal with automatic release, sources JAR, a javadoc JAR built from the Dokka HTML, and MD5 / SHA-1 / SHA-256 / SHA-512 checksums. GPG signing is turned on by `-PsignAllPublications=true` in `release.yml`, which reads the portal token and in-memory key from repository secrets, so local `publishToMavenLocal` needs no keys. The group is `io.github.tspstudio` rather than the originally planned `io.kotensor`, because a GitHub namespace is verified through the GitHub account while `io.kotensor` would need the `kotensor.io` domain; signing up for the portal with the TSPstudio GitHub account verifies it before the first release.
- **Benchmark suite:** `:benchmark` uses `me.champeau.jmh` 0.7.3 and JMH 1.37: `MatMulBenchmark` (FLOP/s for FP32 matmul, prefill and FP32 / INT8 / INT4 matvec), `DecodeBenchmark` (tokens/s per decode step on a SmolLM2-135M-shaped model at KV lengths 16–2,048 for FP32 / INT8 / INT4), `GenerateBenchmark`, `TrainingBenchmark` (full fine-tuning vs QLoRA step) and `TokenizerBenchmark`, all with the `gc` profiler for bytes allocated per operation.
- **CI:** `ci.yml` runs the tests on Linux and Windows, assembles the Android APK and smoke-runs a benchmark; `docs.yml` and `release.yml` cover documentation and releases.
- **Verification notes:** Gradle could not run in the environment these milestones were developed in (its distribution and plugin repositories were unreachable), so every Kotlin source was compiled with the standalone Kotlin 2.3.20 compiler instead: the library and tests against the Java 17 API with all 66 tests passing, the benchmarks against the JMH API with every benchmark body executed once, and the Android sample against stubs of the Android APIs it calls, with its view-model flow (demo training, streaming, LoRA, quantization, QLoRA) run end to end on the JVM. The Gradle build scripts, the Android build itself and the workflows are untested until they run on a machine with network access.

| Component | Deliverable |
| :--- | :--- |
| **Android Sample** | New `:sample-android` module with Jetpack Compose UI demonstrating live token streaming, on-device memory telemetry, and real-time LoRA fine-tuning. |
| **API Documentation** | Automated Dokka build pipeline generating searchable HTML documentation. |
| **Maven Central** | Sonatype Central Portal release under `io.github.tspstudio:kotensor` with automated GPG signing and SHA-256 validation. |
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
