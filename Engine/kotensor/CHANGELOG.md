# Kotensor changelog — from v0.1.0 to v1.1.0

This document explains everything that changed in Kotensor between the original engine (v0.1.0, "milestone 0") and v1.1.0, following the six milestones of [`Milestone.md`](Milestone.md). For every modification it answers four questions:

- **What changed** — the files and the new or changed behavior.
- **Why it was needed** — the problem it solves, measured where possible.
- **How it works** — the design, in enough detail to read the code with confidence.
- **What we looked for** — the criteria the work had to meet and how it was verified.

## Contents

- [The starting point: v0.1.0](#the-starting-point-v010)
- [At a glance](#at-a-glance)
- [How the work was verified](#how-the-work-was-verified)
- [v0.2.0 — Autoregressive inference and subword tokenization](#v020--autoregressive-inference-and-subword-tokenization)
- [v0.3.0 — Compute performance and memory efficiency](#v030--compute-performance-and-memory-efficiency)
- [v0.4.0 — Architecture generalization and model importers](#v040--architecture-generalization-and-model-importers)
- [v0.5.0 — Edge quantization (INT8 / INT4)](#v050--edge-quantization-int8--int4)
- [v1.0.0 — Production hardening and ecosystem integration](#v100--production-hardening-and-ecosystem-integration)
- [v1.1.0 — Training engine efficiency](#v110--training-engine-efficiency)
- [Cross-cutting changes](#cross-cutting-changes)
- [Known limitations and open items](#known-limitations-and-open-items)

---

## The starting point: v0.1.0

Kotensor v0.1.0 was a complete but educational engine: 19 Kotlin source files and 15 tests. It had a `Tensor` class with reverse-mode autograd, `Linear` / `Embedding` / `RMSNorm` / `RoPE` / `SwiGLU` / causal multi-head `Attention` modules, a pre-norm decoder-only `Transformer`, `SGD` and `AdamW`, a character-level `SimpleTokenizer`, a `DataLoader`, LoRA adapters with a fine-tuning pipeline, F32 `.safetensors` save/load and a `StreamingSession` for token-by-token output.

What it could not do was serve as a real on-device engine:

- every generated token re-ran the model over the whole sequence, so generation slowed down quadratically;
- matrix multiplication was a single-threaded triple loop at roughly 0.2–2 GFLOP/s;
- the character tokenizer could not read the vocabulary of any pre-trained model;
- there was no way to load a pre-trained checkpoint, and the safetensors reader understood only F32 and loaded entire files into memory;
- a 135M-parameter model needed over 1 GB of RAM (FP32 weights plus a gradient array for each);
- nothing was packaged for Android, documented as an API, published, or benchmarked.

## At a glance

| Version | Theme | Main additions | Tests |
| :-- | :-- | :-- | :-- |
| 0.1.0 | Baseline | Autograd, Transformer, LoRA, F32 safetensors | 15 |
| 0.2.0 | Inference and vocabulary | KV cache, prefill/decode generation, stop tokens, BPE tokenizer, JSON parser, incremental decoding, RoPE position fix | 31 |
| 0.3.0 | Performance | Tiled, vectorized, multi-threaded matmul; in-place ops; allocation-free decode | 41 |
| 0.4.0 | Real models | Grouped-query attention, Llama 3 rope scaling, streaming F16/BF16 safetensors reader, Hugging Face importer | 54 |
| 0.5.0 | Edge memory | INT8 / INT4 quantization, dequantizing kernels, QLoRA, compact model files, lazy gradients | 65 |
| 1.0.0 | Production | Java 17 / Android readiness, Compose sample app, Dokka docs, Maven Central pipeline, JMH suite, CI | 66 |
| 1.1.0 | Training efficiency | Loss masking, fused output-layer loss, frozen-weight backpropagation, matmul backward without copies, Vector API kernels, batch telemetry | 78 |

## How the work was verified

Every change was held to the same rule: a new code path must reproduce a trusted reference, not just "look reasonable".

- **Against the existing engine.** Each fast path is compared with the slow, already-trusted path: cached decoding against the full forward pass, the tiled matmul against a naive triple loop, the allocation-free decoder against the graph-building decoder, quantized kernels against FP32 math on the dequantized weights.
- **Against independent references.** The tokenizer is compared token-for-token with the Hugging Face `tokenizers` library, and the model importer with a separate NumPy implementation of the Llama architecture, so mistakes cannot cancel out by being made twice in the same code base.
- **By measurement.** Speed claims come from benchmarks run on the same machine before and after each change, and "allocation-free" is measured with the JVM's per-thread allocation counter rather than assumed.

The development environment could not download Gradle's distribution, the Gradle plugin portal or Maven Central, so the code was compiled with the standalone Kotlin 2.3.20 compiler and the test suites were run with a small reflection-based runner. The same test classes run under `./gradlew test`; you confirmed the Gradle run passed after milestone 2. The Gradle build scripts, the Android build and the GitHub workflows added in v1.0.0 could only be checked against official documentation, not executed — see [Known limitations](#known-limitations-and-open-items).

v1.1.0 was held to the same rules: every fast path is compared with the computation it replaces (the fused loss with `Linear` plus `crossEntropyLoss`, frozen training with unfrozen training, the new kernels with a naive product, the Vector API kernels with the scalar ones). The development environment still could not reach Gradle's repositories, so the library was compiled with the standalone Kotlin 2.3.20 compiler, with the same flags as the build (`-Xjdk-release=17 -Xadd-modules=jdk.incubator.vector`), and the suite was run with the reflection runner twice — on the default scalar kernels and with `--add-modules jdk.incubator.vector` — and repeatedly, to rule out flaky allocation tests. The benchmark sources were compiled against stubs of the JMH annotations and every benchmark body run once, and a client compiled against the 1.0.0 classes was run against 1.1.0 to check binary compatibility. The timings come from hand-timed runs (one warm-up step, then the mean of three) on a 2-core container, so they show relative gains on that machine; the JMH suite is the tool for comparable numbers.

---

## v0.2.0 — Autoregressive inference and subword tokenization

### 1. Key-value cache

**Files:** new `KVCache.kt`; `Attention.kt`, `TransformerBlock.kt`, `Transformer.kt`.

**What changed.** A `KVCache` class holds the keys and values of every position already processed, for every layer and head, in two pre-allocated flat `FloatArray`s laid out as `[layers, heads, maxSequenceLength, headDim]`. `Attention`, `TransformerBlock` and `Transformer` gained a cached forward (`forward(input, cache, layer)` / `forward(indices, cache)`), and `Transformer.newCache(maxSequenceLength)` builds a cache sized for the model.

**Why it was needed.** A language model produces text one token at a time. Without a cache, producing token *t* means running all *t* previous tokens through every layer again, so generating *T* tokens costs on the order of *T²* token-passes. But the keys and values of earlier tokens never change once computed — recomputing them is pure waste. On a 4-layer, 64-dimension model, 256 tokens took 11.7 seconds without a cache.

**How it works.**

- *Prefill:* the prompt is run once; each layer computes its keys and values and writes them into the cache.
- *Decode:* every later step feeds in only the newest token. Its query attends to all cached keys (a `[1, step + 1]` row of scores), and its own key and value are appended at position `cache.length`.
- The cache length is advanced once per model call, after all layers have written, so every layer sees the same starting position.
- Rotary positions continue from `cache.length`, so a token decoded at position 17 is rotated exactly as it would be inside a full 18-token forward pass.
- The cached path records no gradient: it exists for inference, and building an autograd graph for every generated token would defeat its purpose.

**What we looked for.**

- *Exactness:* logits from prefill plus step-by-step decode must equal the full forward pass at every position. They match to within 3.6 × 10⁻⁷ (floating-point rounding), with and without LoRA adapters attached.
- *Same text:* greedy generation with the cache must produce the identical token sequence as the old uncached loop.
- *Safety:* writing past the cache's capacity, or using a cache shaped for a different model, must fail with a clear message instead of corrupting memory; `reset()` must make the cache reusable.
- *Speed:* 22× faster at 64 tokens, 31× at 128 and 88× at 256 tokens (11.7 s → 0.13 s), and the gap keeps growing with length.

### 2. Prefill/decode generation, stop tokens and streaming

**Files:** `Transformer.kt`, `StreamingSession.kt`.

**What changed.** `generate`, `generateStream` and `StreamingSession.stream` now prefill the prompt once and then decode one token per step through the cache. All of them accept `stopTokens`, a set of ids (typically the tokenizer's end-of-text id) at which generation ends without emitting the stop token. Text streaming goes through the new `IncrementalDecoder` (see [6](#6-incremental-decoding)).

**Why it was needed.** The public API should get the cache's speed automatically, without callers managing a cache. Chat and instruction models also signal the end of a reply with a special token, and without stop tokens a caller had no way to end generation there.

**What we looked for.** The old signatures keep working (new parameters have defaults), and the output is identical to the uncached implementation. `generate(prompt, maxTokens = 0)` returns the prompt unchanged, and stopping on the first or second generated token was tested explicitly.

### 3. Rotary-embedding position fix

**Files:** `RoPE.kt`, `Attention.kt`.

**What changed.** `RoPE.apply` now understands a `[tokens, numHeads × headDim]` input and rotates every head of a token by that token's position.

**Why it was needed.** This was a latent bug, found while building the cache. Rotary embeddings encode *where* a token is by rotating its query and key vectors by an angle proportional to its position. The original code handed RoPE the projected queries and keys as if each `headDim`-wide slice were a separate token, so the "position" of head *h* of token *i* was really `(i × numHeads + h) mod sequenceLength`. Positions were scrambled across heads, and the same token got a different rotation depending on the total sequence length. Training still converged, because the model learned whatever signal it was given, but a token's rotation depended on how many tokens followed it — which is impossible to reproduce one token at a time, and wrong for any pre-trained model.

**What we looked for.** A regression test checks that rotating all heads together gives exactly the same numbers as rotating each head on its own with the token's position, and that different positions rotate differently. After the fix, cached decoding matched the full forward pass. Before it, a key rotated on its own differed from the same key rotated inside its sequence by up to 0.58 at positions 1–4.

**Consequence.** Weights trained with v0.1.0 — for example the demo `kotensor_model.safetensors` — produce different outputs after this fix and should be retrained.

### 4. Byte-pair-encoding tokenizer

**File:** new `BPETokenizer.kt`.

**What changed.** `BPETokenizer` implements the `Tokenizer` interface and loads Hugging Face `tokenizer.json` files directly (`BPETokenizer.fromFile` / `fromJson`). It covers byte-level BPE, the scheme used by GPT-2, Llama 3, Qwen 2/2.5, SmolLM and most recent open models.

**Why it was needed.** A model only understands the token ids it was trained with. The character tokenizer could not reproduce any real vocabulary, so pre-trained weights were unusable. The spec also demanded 100% coverage: any input text, including emoji or unusual scripts, must tokenize without "unknown token" failures.

**How it works.**

1. *Added and special tokens first.* Text is split on tokens such as `<|im_start|>`, `<|im_end|>` and `<|endoftext|>`, which always become a single id. Longer tokens are matched first, so one token that is a prefix of another can't steal its match.
2. *Normalization* (NFC/NFD/NFKC/NFKD) if the file asks for it.
3. *Pre-tokenization.* The file's own regular expression (for example Qwen's or Llama 3's) splits text into words; GPT-2's pattern is the default.
4. *Byte-level mapping.* Each word's UTF-8 bytes are mapped to 256 printable stand-in characters (GPT-2's reversible byte table). Every possible byte is in the vocabulary, which is what guarantees full coverage.
5. *Merging.* Adjacent symbols are merged repeatedly, always choosing the pair with the lowest merge rank from the file, until no listed pair remains.
6. *Templates and options.* The `TemplateProcessing` post-processor's prefix and suffix tokens (such as Llama 3's `<|begin_of_text|>`) are added by `encode`, unless `addSpecialTokens = false`. `ignore_merges` (Llama 3) and `unk_token` are honored.

Decoding reverses the byte mapping and can skip special tokens. Words already seen are cached, which matters because natural text repeats the same words constantly.

**What we looked for.**

- *Identical ids to the reference.* Three tokenizers were trained with the Hugging Face `tokenizers` library, one per pre-tokenization family: GPT-2-style, Qwen-style with an explicit split regex, and Llama-3-style with a template and `ignore_merges`. For 16 test strings each — empty text, whitespace runs, tabs and newlines, accents, Japanese, emoji, numbers, chat markup — Kotensor produces exactly the library's ids, with and without special tokens, and the same decoded text.
- *Coverage.* Arbitrary Unicode, including control characters and flags, round-trips exactly through encode and decode.
- *Special tokens.* Each encodes as one id and can be skipped when decoding.

### 5. Zero-dependency JSON reader

**File:** new `JsonParser.kt`.

**What changed.** A small recursive-descent JSON parser returns maps, lists, strings, `Long` / `Double` numbers, booleans and `null`, with full escape handling (including `\uXXXX` surrogate pairs) and strict errors.

**Why it was needed.** `tokenizer.json` (and later `config.json` and safetensors headers) is JSON, and the project's core promise is zero runtime dependencies, which rules out Jackson, Gson or kotlinx.serialization. The previous safetensors loader matched its header with regular expressions, which works for simple headers but breaks on nested metadata.

**What we looked for.** Correct values for every JSON type and escape, map order preserved, and malformed input rejected (missing values, unclosed arrays, trailing garbage) rather than half-parsed.

### 6. Incremental decoding

**File:** `Tokenizer.kt`.

**What changed.** `IncrementalDecoder` turns a stream of token ids into text piece by piece.

**Why it was needed.** With byte-level BPE, one character can be split over several tokens: an emoji is 4 UTF-8 bytes and may arrive as 2–4 tokens. Decoding each token on its own shows broken `�` characters mid-stream. The decoder holds tokens back while the decoded text ends in an incomplete character (at most 4, since a UTF-8 character has at most 4 bytes) and releases the full character once it is complete.

**What we looked for.** Streamed pieces concatenate to exactly the full decode, and no piece ever contains `�` for text that is valid as a whole. Character-level tokenizers pass through unchanged.

---

## v0.3.0 — Compute performance and memory efficiency

### 1. Tiled, vectorized, multi-threaded matrix multiplication

**Files:** new `MatMul.kt`; `Tensor.kt`.

**What changed.** `Tensor.matmul` — forward and backward — now runs on `MatMul.multiply`, a tiled kernel that uses every CPU core.

**Why it was needed.** Matrix multiplication is nearly all of a Transformer's work. The original triple loop reached 0.2–2.3 GFLOP/s, single-threaded, and walked one matrix column-wise, which defeats the CPU cache.

**How it was designed.** Before writing the final kernel, five candidates were benchmarked: the naive loop, a row-wise "saxpy" loop, a tiled saxpy loop, a 4-row register-blocked version, and the spec's suggestion of a dot product with 8 unrolled accumulators over a pre-transposed matrix. Two findings shaped the result:

- HotSpot's C2 compiler does **not** vectorize the dot-product form, because floating-point addition isn't associative and the JIT won't reorder a reduction. The 8-accumulator version stayed scalar, at about 4 GFLOP/s.
- C2 **does** vectorize `y[j] += a * x[j]`, but only when both arrays are indexed by the same plain `j`. A micro-benchmark measured 52 GFLOP/s for that loop and 5 GFLOP/s for the identical loop written with array offsets (`c[ci + j] += a * b[bl + j]`).

So the kernel computes the result in `[32 × 256]` tiles. Each tile is accumulated in per-thread scratch rows (32 KB, about the size of a core's L1 data cache), with the needed row segment of the right-hand matrix copied next to them, so the inner loop is exactly the vectorizable form. Four result rows are updated per pass over each copied row (register blocking: one load feeds four accumulators). Tiles are independent, so a `ForkJoinPool` with one worker per CPU core processes them in parallel once a product exceeds `MatMul.parallelThreshold` (2¹⁸ multiply-adds), below which the cost of handing work to threads would exceed the gain. The backward pass reuses the same kernel on transposed copies (`dA = dC · Bᵀ`, `dB = Aᵀ · dC`).

**What we looked for.**

- *Correctness on awkward shapes:* sizes that aren't multiples of the tile (1×1, 33×65×257, 70×9×513, 4×1000×3 and others) match the naive loop, including the accumulate-and-scale mode.
- *Determinism:* the parallel result is bit-for-bit identical to the single-threaded one.
- *Gradients:* the new backward pass matches naively computed gradients.
- *Speed:* about 24 GFLOP/s per thread and about 40 GFLOP/s on two threads on an AVX-512 machine — 10× the naive loop at 512³ and over 100× at 128 × 2048 × 2048.

### 2. In-place tensor operations

**File:** `Tensor.kt`.

**What changed.** New operations write into tensors the caller already owns and record no autograd graph: `matmulInto` (with optional accumulate and scale), `addInto` / `addInPlace` (with row broadcasting), `mulInPlace`, `siluInPlace` and `copyFrom`.

**Why it was needed.** Every graph-building operation allocates a new output array, a gradient array and a closure for its backward pass. During training that is the price of autograd, but during generation none of it is ever used.

**What we looked for.** Each in-place operation gives the same numbers as its graph-building counterpart and creates no graph.

### 3. Allocation-free decode path

**Files:** new `InferenceBuffers.kt`; `forwardInto` / `decodeInto` / `applyInPlace` / `lookupInto` in `Linear.kt`, `LoRA.kt`, `RMSNorm.kt`, `SwiGLU.kt`, `RoPE.kt`, `Embedding.kt`, `Attention.kt`, `TransformerBlock.kt`; `Transformer.decode` in `Transformer.kt`.

**What changed.** `Transformer.decode(token, cache, buffers)` runs one decode step entirely inside pre-allocated buffers: the residual stream, the normalized input, Q/K/V, the attention context and output, the feed-forward intermediates, the logits and the attention scores. `generateStream` and `StreamingSession` use it for every step after the prefill.

**Why it was needed.** Even with the cache, each step through the graph path allocated dozens of arrays. On phones that churn triggers garbage collections, which show up as visible stutters while text streams. The spec required zero heap allocations inside the generation loop.

**What we looked for.**

- *Identical results:* `decode` matches the graph path and the full forward pass at every position, with and without LoRA.
- *Zero allocations, measured:* after warm-up, 200 consecutive decode steps allocate **0 bytes**, according to the JVM's per-thread allocation counter.
- *Speed:* a 4-layer, 64-dimension model streams 3,700 tokens/s (1,000 before this milestone), and an 8-layer, 256-dimension model with an 8k vocabulary 91 tokens/s (27 before).

### 4. Precomputed rotary frequencies

**File:** `RoPE.kt`. The rotation frequencies `1 / base^(2i/d)` used to be recomputed with `pow` for every element of every token. They are now computed once per layer, which matters on the decode path, where RoPE runs on every token of every layer.

---

## v0.4.0 — Architecture generalization and model importers

### 1. Grouped-query and multi-query attention

**Files:** `Attention.kt`, `TransformerBlock.kt`, `Transformer.kt`, `InferenceBuffers.kt`.

**What changed.** Attention takes a `numKeyValueHeads` parameter. With fewer key/value heads than query heads, each key/value head is shared by a group of query heads: query head *h* uses key/value head `h / (numHeads / numKeyValueHeads)`. Multi-query attention is the special case of a single key/value head.

**Why it was needed.** Modern models — Llama 3, Qwen 2.5, SmolLM2, Mistral — use grouped-query attention to shrink the KV cache and its memory traffic. SmolLM2-135M, for instance, has 9 query heads but only 3 key/value heads. Without GQA, those checkpoints cannot even be loaded, because their key and value matrices have a different shape.

**How it works.** `wK` and `wV` project to `numKeyValueHeads × headDim` instead of the full dimension, and all three attention paths — training forward and backward, cached forward, allocation-free decode — index the shared head. The KV cache and decode buffers store only the key/value heads, so the cache shrinks by the group factor (3× for SmolLM2).

**What we looked for.**

- *Mathematical equivalence:* a GQA model must produce exactly the output of a standard multi-head model whose key/value weights are copies of the shared heads. The test builds that twin and compares logits for single sequences and batches.
- *Correct gradients:* a shared key/value head must receive the sum of its group's gradients, also checked against the twin.
- *All paths agree:* cached and allocation-free decoding match the full forward pass with grouped heads, and a GQA model still trains.
- *Clear errors:* groupings that don't divide evenly are rejected.

### 2. Configurable normalization, rotary base and Llama 3 scaling

**Files:** `RoPE.kt` (`RoPEScaling`), `Transformer.kt`, `TransformerBlock.kt`, `Attention.kt`.

**What changed.** `Transformer` takes `normEps`, `ropeTheta` and `ropeScaling`, mirroring the `rms_norm_eps`, `rope_theta` and `rope_scaling` fields of Hugging Face configs. `RoPEScaling` implements Llama 3's frequency scaling.

**Why it was needed.** These values differ between models — `rope_theta` is 10,000 for Llama 2, 100,000 for SmolLM2 and 500,000 for Llama 3 — and running a model with the wrong value silently produces nonsense. Llama 3.x also rescales the rotary frequencies to extend its context: wavelengths shorter than a threshold are kept, longer ones are divided by a factor, and the band between is blended smoothly. That rescaling has to be reproduced exactly.

**What we looked for.** The low-frequency, high-frequency and blended regions behave as specified (unchanged, divided by the factor, smoothly between), and a Llama 3-style checkpoint matches the NumPy reference (see [5](#5-reference-fixtures-and-the-numpy-reference-implementation)).

### 3. Streaming safetensors reader with half-precision formats

**File:** `SafeTensors.kt` (new `SafeTensorsFile`).

**What changed.** `SafeTensorsFile` opens a checkpoint and reads tensors one at a time, converting F16, BF16, F64 and integer formats to `Float`. It parses the header with `JsonParser` and exposes `names`, `entry(name)` (dtype, shape, byte range), `metadata`, `load(name)` and `readInto(name, array)`. `SafeTensors.load` keeps its old signature and now uses it.

**Why it was needed.** Pre-trained checkpoints are distributed in BF16 or F16, which the old F32-only reader rejected. The old reader also loaded the whole file into one byte array, which caps files at 2 GB and doubles peak memory, and parsed byte offsets as 32-bit integers, which cannot represent positions past 2 GB. Reading tensor by tensor through a `RandomAccessFile`, in 4 MB chunks, keeps peak memory at roughly the model plus one tensor.

**What we looked for.** Exact conversion of edge values: F16's largest value (65,504), its smallest normal and subnormal numbers, negative zero, infinity and NaN, and BF16 values near the float range limits. F32 files round-trip exactly, and both `__metadata__` and header padding are handled.

### 4. Hugging Face configuration and weight importer

**File:** new `HuggingFaceModelLoader.kt` (`HuggingFaceConfig`, `HuggingFaceModelLoader`, `ImportReport`).

**What changed.** `HuggingFaceModelLoader.load(directory)` turns a model directory downloaded from the Hub into a Kotensor `Transformer`, and `loadTokenizer(directory)` loads its matching tokenizer.

**Why it was needed.** This is the milestone's goal: run standard open-source checkpoints directly, without conversion scripts or Python.

**How it works.**

1. `config.json` is read into `HuggingFaceConfig`: hidden size, layers, attention and key/value heads, feed-forward size, norm epsilon, rope theta and scaling, vocabulary size and whether embeddings are tied. Only architectures Kotensor computes exactly are accepted (`llama`, `qwen2`, `mistral`), and anything else is rejected with a clear message rather than loaded and run wrongly.
2. A **declarative mapping table** of regular expressions translates Hugging Face tensor names to Kotensor names, for example `model.layers.{i}.self_attn.q_proj.weight → block.{i}.attention.wQ.weight`. Supporting a new naming scheme means adding rows, not code.
3. **Transposition:** PyTorch stores linear weights as `[out, in]` and Kotensor as `[in, out]`, so projection weights are transposed while loading.
4. **Shards:** with `model.safetensors.index.json`, every shard listed there is read.
5. **Tied embeddings:** when a model has no `lm_head.weight` and its config says embeddings are tied, the output head is built from the transposed embedding table.
6. **Optional biases:** Qwen's query/key/value biases are loaded; models without biases keep zeros.
7. **Strictness:** shape mismatches and missing weights fail with the tensor names, and tensors with no place in Kotensor (such as precomputed `rotary_emb.inv_freq`) are listed in the returned `ImportReport`.

**What we looked for.** Logits that match an independent reference to floating-point precision, on checkpoints covering every feature of the importer (next section).

### 5. Reference fixtures and the NumPy reference implementation

**Files:** `src/test/resources/hf/` (new), `HuggingFaceImportTest.kt`, `GQATest.kt`, `.gitignore`.

**What changed.** The tests include three tiny checkpoints, written in exactly the format Hugging Face uses, together with expected logits from a separate NumPy implementation of the Llama architecture (RMSNorm, rotate-half RoPE, repeated key/value heads, SwiGLU, tied heads):

| Fixture | Covers |
| :-- | :-- |
| `llama_tiny` | BF16 weights, GQA (4 query / 2 key-value heads), explicit `head_dim` in the config, separate `lm_head` |
| `qwen_tiny_sharded` | Two shards with an index file, mixed F16 / F32 / BF16 tensors, query/key/value biases, multi-query attention, tied embeddings, an extra tensor to be ignored |
| `llama3_tiny` | `rope_theta` = 500,000 with Llama 3 rope scaling, tied embeddings |

A separate fixture holds F16 / BF16 / F32 edge values. `.gitignore` excludes `*.safetensors` but now keeps the files under `src/test/resources`, so the fixtures are committed.

**Why it was needed.** A reference written independently of the Kotlin code catches errors that a self-consistency test cannot, such as a transposition done the wrong way in both the writer and the reader.

**What we looked for.** Imported models match the reference logits within 2.4 × 10⁻⁶, 4.3 × 10⁻⁶ and 5.7 × 10⁻⁶ on values up to about 10, and the allocation-free decoder reproduces the same logits token by token (checked on the Llama-style fixture). One mismatch during development turned out to be in the fixture generator, not the engine: it added `1.0` to norm weights after rounding them to BF16, so the expected values used numbers that weren't in the file.

---

## v0.5.0 — Edge quantization (INT8 / INT4)

### 1. Quantized tensor format

**File:** new `QuantizedTensor.kt` (`Quantization`, `QuantizedTensor`).

**What changed.** Weight matrices can be stored in 8 or 4 bits per value instead of 32, with one scale (and for INT4 one zero point) per column for each group of 64 rows.

**Why it was needed.** A 135M-parameter model needs about 540 MB in FP32 — more than many phones allow an app — while its weights carry far less information than 32 bits each.

**How it works.**

- **INT8, symmetric:** `w ≈ scale · q` with `q` in −127…127 and `scale = max |w| / 127` for each group. Rounding error is at most half a step, under 0.4% of the group's largest weight.
- **INT4, asymmetric:** `w ≈ scale · (q − zero)` with `q` in 0…15. The zero point lets the 16 levels cover a lopsided range such as −0.1…+0.4 without wasting levels; error is at most 1/30 of the group's range.
- **Group-wise scales:** a scale per 64 rows of each column adapts to local ranges, so a single outlier costs accuracy in its group only. The overhead is 1/16 byte per weight for INT8 and 5/64 for INT4, giving 1.06 and 0.58 bytes per weight in total.
- **INT4 packing:** column *j* lives in the low nibble of byte *j* and column *j* + *n*/2 in the high nibble, so a row segment unpacks with simple sequential loops that the JIT can vectorize.

**What we looked for.** Error bounds checked against these formulas, all-zero matrices represented exactly, and every packed value readable back through `value(row, column)`.

### 2. Dequantizing matrix multiplication

**Files:** `MatMul.kt` (`multiplyQuantized`, `multiplyByQuantizedTransposed`), `Tensor.kt` (`matmulDequant`, `matmulDequantInto`).

**What changed.** New kernels multiply FP32 activations by quantized weights directly.

**Why it was needed.** The spec required that the full FP32 weight matrix never exist in memory, otherwise quantization would save disk space but not RAM.

**How it works.** Inside the tiled kernel, each weight row segment is expanded from bytes into a small per-thread float buffer just before use. Within a group, scale and zero point are constant for each column, so the kernel accumulates `Σ x · q` and `Σ x` with no scaling at all, then applies `scale · (Σ x·q − zero · Σ x)` once per group. That rearrangement is exact algebra, and it keeps the hot loop identical to the FP32 kernel's vectorizable form. Two further measurements shaped the code: byte-to-float conversion vectorizes on HotSpot, but extracting nibbles on integers after conversion does not, so nibbles are extracted in byte arrays first (0.23–0.27 ns per value versus 0.5–0.8). For gradients, `multiplyByQuantizedTransposed` computes `dX = dY · Wᵀ` one dequantized row at a time, so training signals can flow through frozen quantized layers to adapters beneath them.

**What we looked for.** Results equal to FP32 math on the dequantized matrix (within 1 × 10⁻⁴) on irregular shapes, group sizes and both bit widths, including accumulate mode and the gradient kernel; parallel results identical to single-threaded ones.

### 3. Faster single-row (decode) schedule

**File:** `MatMul.kt`.

**What changed.** Products with a single input row — every decode step — use a "stripe" schedule: each thread owns a contiguous range of output columns and streams through the weight rows in order, instead of the 32-row tiles meant for batches.

**Why it was needed.** Profiling the quantized kernels showed that the tile order made a one-row product jump through memory with a row-sized stride for every column stripe, which made it latency-bound. Streaming each weight row once and in order is what memory hardware is best at.

**What we looked for.** The same numbers as before, and speed: a 2048 × 49152 vector-matrix product went from 90 to 39 ms in FP32 and from 65 to 28 ms in INT8, a 2.2× gain that also benefits non-quantized models.

### 4. Quantized layers and shared abstractions

**Files:** new `QuantizedLinear.kt`; `Linear.kt` (`Projection`), `Embedding.kt` (`TokenEmbedding`, `QuantizedEmbedding`), `Module.kt` (`forwardInto`), `SwiGLU.kt`, `Attention.kt`, `TransformerBlock.kt`.

**What changed.**

- `QuantizedLinear` and `QuantizedEmbedding` mirror `Linear` and `Embedding`.
- `Projection` is the shared base of `Linear` and `QuantizedLinear`, and `TokenEmbedding` the base of both embedding types.
- `Module.forwardInto` gives every module an in-place forward, so the decoder calls `wQ.forwardInto(...)` without knowing whether the layer is FP32, quantized or wrapped in LoRA. A type switch in `Attention` that only knew `Linear` and `LoRALinear` is gone.
- The feed-forward projections and the output head became replaceable, like the attention projections already were.

**Why it was needed.** Quantization had to work in every code path — training graph, cached decoding, allocation-free decoding and LoRA — without duplicating the model classes. Shared base types are what let a layer be swapped without its callers changing.

**What we looked for.** A quantized model gives identical results across the full forward, cached and allocation-free paths and batched input. Against the FP32 model, the tests allow logit differences of up to 2% (INT8) and 25% (INT4) of the largest logit on small random-weight models; INT4 round-to-nearest is lossy, which is why calibrated methods such as GPTQ and AWQ are listed under limitations. The allocation-free guarantee survives quantization: 0 bytes per decode step. Measuring that caught one regression on the way: the tile scheduler's lambda allocated a small object on every call (84 KB over 100 steps), fixed by inlining the scheduler.

### 5. Model-level quantization, accounting and QLoRA

**Files:** `Transformer.kt`, `LoRA.kt`.

**What changed.**

- `model.quantize(Quantization.INT8 or INT4)` converts every projection and the embedding table in place, and `Transformer(…, quantization = …)` builds a compact model from the start.
- `parameterCount()`, `weightBytes()` and `quantization` report the model's state; `namedProjections`, `namedQuantizedLayers` and `namedQuantizedTensors` enumerate its layers.
- `namedParameters()` is now generated from the list of projections instead of written out by hand.
- `LoRALinear` accepts any `Projection`, so adapters attach to quantized layers: **QLoRA**, the standard technique for fine-tuning a compressed model.

**Why it was needed.** First-class on-device learning is Kotensor's reason to exist, and fine-tuning has to keep working after compression. The spec also asked for on-device memory telemetry, which needs accurate accounting.

**What we looked for.** QLoRA training lowers the loss and decodes consistently. Two operations that can't be done correctly are refused with clear messages: merging an adapter into quantized weights (there are no FP32 weights to fold it into — export the adapter instead) and quantizing a model with adapters attached.

### 6. Lazily allocated gradients

**File:** `Tensor.kt`.

**What changed.** A tensor's `grad` array is now created on first access instead of with the tensor.

**Why it was needed.** Every parameter carried a gradient array of its own size, so a model used only for inference held two copies of its weights. For a 163M-parameter FP32 model that was 1.3 GB instead of 652 MB.

**What we looked for.** Training behaves exactly as before (all training tests unchanged), and creating a large tensor no longer allocates its gradient until it is used, measured with the allocation counter.

### 7. Compact model files and quantized import

**Files:** `Transformer.kt` (`saveQuantized`, `loadQuantized`), `SafeTensors.kt` (`SafeTensorsEntry`, `saveEntries`, `readRaw`), `HuggingFaceModelLoader.kt`.

**What changed.**

- `saveQuantized` writes the packed integers (`I8` / `U8`), scales and zero points as ordinary safetensors entries, with the model's configuration in the file's `__metadata__`, so `Transformer.loadQuantized` rebuilds the model with no other information.
- `SafeTensors.saveEntries` writes any dtype plus metadata, with the header padded to 8 bytes as the format recommends.
- `HuggingFaceModelLoader.load(directory, Quantization.INT8)` quantizes each projection while importing it.

**Why it was needed.** The device should never have to hold the FP32 model. Without these, the only way to get a quantized model on a phone would be to load the full FP32 checkpoint there and quantize it, which needs the very memory quantization is meant to save.

**What we looked for.** A save/load round trip that is bit-exact (same packed bytes, scales, biases and logits), and importing with quantization giving exactly the same model as importing in FP32 and then quantizing.

### 8. Results

Measured on a model with SmolLM2-135M's shape (30 layers, dimension 576, 9 query / 3 key-value heads, 49,152-token vocabulary, output head stored separately: 163M parameters), on a 2-core machine:

| Weights | Memory | Decode speed | Load time |
| :-- | :-- | :-- | :-- |
| FP32 | 652 MB | 8.6 tokens/s | — |
| INT8, groups of 64 | 173 MB | 12.2 tokens/s | — |
| INT4, groups of 64 | 95 MB | 8.0 tokens/s | 86 ms from a `saveQuantized` file |

The projection weights alone take 144 MB in INT8 and 78 MB in INT4, close to the spec's 145 MB and 75 MB targets. INT8 is faster than FP32 because decoding is limited by memory bandwidth and INT8 moves a quarter of the bytes. INT4 is limited by nibble unpacking instead, so it trades a little speed for a 7× smaller footprint.

---

## v1.0.0 — Production hardening and ecosystem integration

### 1. Java 17 bytecode against the Java 17 API

**Files:** `build.gradle.kts`.

**What changed.** The library is still built with the Java 21 toolchain, but now emits Java 17 bytecode (`jvmTarget = 17`, `JavaCompile.release = 17`) and is compiled against the Java 17 API (`-Xjdk-release=17`).

**Why it was needed.** Java 21 bytecode can't be used by Java 17 applications, and Android's build tools are most at home with Java 17. Gradle also records the bytecode level in the published metadata and refuses to resolve a dependency that declares a newer level than the consumer. Lowering the target alone isn't enough, because code compiled against JDK 21 could still call methods that don't exist on Java 17; `-Xjdk-release` makes the compiler reject them.

**What we looked for.** The whole engine compiles against the Java 17 API with no changes (class files are version 61, Java 17), and all tests pass on that build.

### 2. Android regular-expression compatibility

**File:** `BPETokenizer.kt`; test in `BPETokenizerTest.kt`.

**What changed.** The pre-tokenizer regex is compiled with `Pattern.UNICODE_CHARACTER_CLASS` where supported, and without it where the platform rejects the flag.

**Why it was needed.** Auditing the engine for Android turned up a crash: Android's `java.util.regex` is implemented on top of ICU and throws `IllegalArgumentException` for this flag (AOSP's source notes it explicitly). Every `BPETokenizer` construction on Android would have failed. The flag isn't needed there, because ICU's `\s`, `\w` and `\d` already match Unicode.

**What we looked for.** On the JVM the flag is still applied (a test checks it), and a genuinely invalid pattern still raises its syntax error instead of being silently retried.

### 3. Android sample app

**Files:** new `sample-android/` (`MainActivity.kt`, `EngineViewModel.kt`, `ModelSource.kt`, `Telemetry.kt`, `ui/KotensorScreen.kt`, manifest, resources, `build.gradle.kts`).

**What changed.** A Jetpack Compose app demonstrates the engine on a phone:

- **Live streaming** of generated text, with tokens per second and a stop button.
- **Memory telemetry**, refreshed every second: Java heap used and maximum, native heap, bytes held by the weights, and the KV-cache size for 2,048 positions.
- **On-device training:** with no model installed, the app trains a small character-level Transformer at start-up while plotting its loss, so it works out of the box with nothing to download.
- **LoRA fine-tuning** on any text typed into the app, with a live loss curve, and **QLoRA** once the model is quantized.
- **One-tap INT8 / INT4 quantization** in place.
- **Real models:** a Hugging Face directory copied into the app's `files/model` folder is imported as INT8 (an FP32 135M model would not fit a typical app heap), and a `saveQuantized` file loads directly.

**How it works.** All engine work runs on a background dispatcher, and the screen observes one immutable state object. Each action returns a status message, so outcomes like "reload the model before quantizing: an adapter is attached" stay visible instead of being overwritten by "Ready".

**Choices and why.**

- *AGP 9.0.1:* the project's Gradle 9.2.1 rules out newer Android Gradle plugins (AGP 9.1 needs Gradle 9.3.1, 9.2 needs 9.4.1, 9.4 needs 9.6), and AGP 9.0 needs at least 9.1.0.
- *Built-in Kotlin:* AGP 9 compiles Kotlin itself, and the old `kotlin-android` plugin is incompatible with its new DSL, so the module applies only the Android and Compose compiler plugins.
- *Library versions:* Compose BOM 2026.01.01, Activity 1.12.4 and Lifecycle 2.10.0 were chosen from before AndroidX moved to `compileSdk` 37, which AGP 9.0 can't compile against. `minSdk` is 26 because the engine uses `ThreadLocal.withInitial`, available from API 26.
- *Opt-in module:* `:sample-android` is included only with `-Pkotensor.android=true`, so building or testing the library never requires an Android SDK, an AGP download or platform 36.

**What we looked for.** Without an Android SDK here, the app's Kotlin code was compiled against stand-ins for the few Android APIs it uses. Its view-model logic was then run end to end on the JVM: demo training (loss 3.56 → 0.02), streaming (about 1,800 tokens/s), LoRA fine-tuning (loss 4.00 → 2.26), the refusal when quantizing with an adapter attached, INT8 quantization, QLoRA (loss 3.08 → 1.79) and telemetry. That run caught the status-message bug described above.

### 4. API documentation

**Files:** `build.gradle.kts`, new `Module.md`, `.github/workflows/docs.yml`.

**What changed.** Dokka 2.2.0 generates searchable HTML documentation (`./gradlew dokkaGenerate` writes `build/dokka/html`), with `Module.md` as the overview page and links from each declaration to its source on GitHub. A workflow publishes it to GitHub Pages on every push to `main`.

**Why this version.** Dokka 2.2.0 is the latest stable release; since 2.1.0 the v2 Gradle plugin and K2 analysis are the defaults, so no opt-in flags are needed.

**Note.** The code base follows a no-comments rule, so the pages document signatures, types and the module overview rather than per-function prose. The README remains the narrative guide.

### 5. Maven Central publishing

**Files:** `build.gradle.kts`, `gradle.properties`, `.github/workflows/release.yml`.

**What changed.** The hand-written `maven-publish` setup under `com.github.TSPstudio` was replaced by `com.vanniktech.maven.publish` 0.37.0, publishing `io.github.tspstudio:kotensor` to the Sonatype Central Portal with automatic release. The publication carries the library JAR, a sources JAR and a javadoc JAR containing the Dokka HTML, plus MD5, SHA-1, SHA-256 and SHA-512 checksums for every file (`mavenCentralChecksums`). Pushing a `v*` tag runs the tests, signs everything with GPG, uploads it and attaches the JARs and a `SHA256SUMS` file to a GitHub release.

**Why these choices.**

- *Coordinates:* the Central Portal only accepts groups under a namespace whose ownership has been verified. The spec originally named `io.kotensor`, and that namespace is proven with a DNS record on the `kotensor.io` domain. `io.github.tspstudio` is proven by the GitHub account instead, and signing up for the portal through GitHub verifies it automatically. Maven Central does not require the group to match the Kotlin package, so the code keeps `io.kotensor.core` and nothing that imports it changes.
- *Plugin and version:* Sonatype shut down its old OSSRH service in 2025, and the Central Portal is the supported route. Version 0.37.0 is the current release; it requires Gradle 9.0+ and Kotlin 2.2+, both satisfied.
- *Checksums:* the plugin uploads only MD5 and SHA-1 by default, and the spec asked for SHA-256 validation, hence the explicit list.
- *Signing only in CI:* signing is enabled by `-PsignAllPublications=true` in the release workflow, not in `gradle.properties`, so `publishToMavenLocal` keeps working on any machine without keys. Credentials and keys come from repository secrets.

**Before the first release.** The `io.github.tspstudio` namespace must be verified on the Central Portal. Signing up for the portal with the TSPstudio GitHub account does this automatically; an account created another way verifies it with a temporary public repository on that GitHub account, named after the verification key the portal shows.

### 6. JMH benchmark suite

**Files:** new `benchmark/` (`build.gradle.kts`, five benchmark classes, `Models.kt`, a tokenizer resource).

**What changed.** A JMH suite measures what matters for this engine:

| Benchmark | Measures |
| :-- | :-- |
| `MatMulBenchmark` | FLOP/s for a 512³ matmul, a 32-row prefill product, and the 2048 × 8192 decode product in FP32, INT8 and INT4 |
| `DecodeBenchmark` | Tokens/s of one decode step on a SmolLM2-135M-shaped model at KV-cache lengths 16, 128, 512 and 2,048, for FP32, INT8 and INT4 |
| `GenerateBenchmark` | End-to-end tokens/s of `generate` and of `StreamingSession`, and prompt prefill time |
| `TrainingBenchmark` | Time per full fine-tuning step (AdamW on all weights) and per QLoRA step |
| `TokenizerBenchmark` | BPE encode and decode throughput |

Every run includes JMH's `gc` profiler, which reports bytes allocated per operation next to the throughput, so allocation regressions show up in the same report as speed.

**Why it was needed.** Hand-timed loops, like the ones used during development, are distorted by JIT warm-up, dead-code elimination and noise. JMH controls for these, which makes before/after comparisons trustworthy.

**How it was kept fast.** Decode benchmarks fill the KV cache with random keys and values instead of decoding thousands of tokens during setup, since attention's cost depends on how many positions are cached, not on their values.

**What we looked for.** Every benchmark compiles against the JMH API, and every setup and benchmark body was executed once to catch runtime errors. The plugin, `me.champeau.jmh` 0.7.3, is the latest release; it predates Gradle 9, and its recent issues concern Gradle 9 deprecation warnings, so it is expected to work with warnings — one of the things to confirm on the first Gradle run.

### 7. Continuous integration

**Files:** `.github/workflows/ci.yml`, `docs.yml`, `release.yml`.

**What changed.** `ci.yml` runs the tests on Linux and Windows, assembles the Android sample's APK and smoke-runs one benchmark. `docs.yml` deploys the API documentation. `release.yml` publishes to Maven Central.

**What we looked for.** Workflows run in bash on both operating systems and mark `gradlew` executable, since a file committed from Windows often loses that permission. All three files parse as valid workflow YAML.

**Location.** GitHub only reads workflows from the repository root, so the three files belong in `Kotensor/.github/workflows/`, and their paths assume that location. They are delivered as separate files rather than written into the project, because `.github` folders are protected from automated writes; adding them is a deliberate step for the repository owner.

### 8. Build configuration

**Files:** `settings.gradle.kts`, `gradle.properties`, `.gitignore`.

**What changed.** Repositories are declared centrally in `settings.gradle.kts` (including Google's repository for Android), `:benchmark` is always included and `:sample-android` only on request, the Gradle daemon gets 2 GB for documentation generation, the checksum list for Maven Central is configured, and Android build outputs and `local.properties` are ignored. Every plugin version was checked against its official release notes for compatibility with the existing Gradle 9.2.1 and Kotlin 2.3.20, so the core build doesn't need a Gradle upgrade.

---

## v1.1.0 — Training engine efficiency

LoRA fine-tuning of a Qwen2-0.5B-sized model (896-dimensional, 24 layers, a 151,936-token vocabulary) showed where the training engine lost its time. Inference was already fast; backpropagation was not. The output projection produced a `[tokens × 151,936]` logits tensor and back-propagated through it in full, the loss was computed over prompt tokens that should not be learned, the backward pass computed weight gradients for every frozen projection, the backward matmul copied its weight transposed on every call, dot products ran without SIMD, and long training runs printed nothing between epochs. Version 1.1.0 addresses each of these; Milestone 6 in [`Milestone.md`](Milestone.md) records the plan and its estimates.

### 1. Loss masking

**Files:** `Loss.kt`, `DataLoader.kt`, `FineTuningPipeline.kt`.

**What changed.** `crossEntropyLoss(logits, targets, ignoreIndex = IGNORE_INDEX)` skips every row whose target is `IGNORE_INDEX` (−100, the convention of PyTorch and Hugging Face). `DataLoader` takes an optional `labels` array the same length as `tokens`: the target of position *i* is `labels[i + 1]` instead of `tokens[i + 1]`. `DataLoader.promptMasked(examples, batchSize, sequenceLength)` builds tokens and labels from (prompt, response) pairs, with every prompt position labelled `IGNORE_INDEX`.

**Why it was needed.** In instruction tuning, the model should learn to produce the response, not to predict the system prompt and the user's question. Computing the loss on prompt tokens trains the wrong objective and costs compute: in the plan's example batch of 64 tokens only about 18 belong to the response, so about 70% of the output-layer work was spent on tokens that should have been ignored.

**How it works.** The rows whose target is not ignored are collected once. Only those rows get a softmax, a loss term and a gradient, and the probabilities are stored for those rows only. The loss is the mean over the active rows. If every target is masked, the loss is 0 and has no graph, so `backward()` does nothing. `promptMasked` labels the prompt, so the first response token is still learned (it is predicted from the last prompt token) and predictions across example boundaries are masked automatically.

**What we looked for.** The masked loss equals the loss of the active rows computed on their own, masked rows receive exactly zero gradient, targets outside the vocabulary other than the ignore index are still rejected, and float targets of −100 (the `Tensor` overload) work the same way. `DataLoader` returns targets from `labels`, and `promptMasked` produces the expected arrays.

### 2. Fused output projection and cross-entropy

**Files:** `Loss.kt` (`fusedCrossEntropyLoss`), `Transformer.kt` (`hiddenStates`, `loss`).

**What changed.** `fusedCrossEntropyLoss(hidden, head, targets, ignoreIndex)` computes the output projection and the cross-entropy together, for a `Linear` or `QuantizedLinear` head. `Transformer.hiddenStates(input)` returns the final normalized hidden states, and `Transformer.loss(input, targets)` combines the two; it falls back to `crossEntropyLoss(lmHead(hidden), …)` for any other kind of head. `FineTuningPipeline.trainStep` trains through `model.loss`.

**Why it was needed.** For 64 tokens and Qwen's vocabulary, the unfused path allocated the 9.7-million-float logits tensor, a probability array of the same size, a gradient array for the logits, a gradient array for the matmul output and, in the matmul backward, a transposed copy of the 136-million-float head weight — about 0.7 GB of temporary memory per step — and it computed the head's weight gradient even when the head was frozen.

**How it works.** The active rows (see 1) are processed a few at a time, so that a chunk of logits is at most 4 Mi floats (16 MB); with Qwen's vocabulary that is 27 rows, so a masked batch fits in one chunk. For each chunk the rows are gathered, projected with the ordinary matmul kernel and the bias added; each row's maximum and sum of exponentials give its loss term. The chunk is then turned in place into the gradient of the logits, `(softmax − one-hot) / active rows`, and multiplied by the transposed head weight to give the gradient of the hidden states (`multiplyTransposedB`, or `multiplyByQuantizedTransposed` for a quantized head). These hidden-state gradients, `active rows × dimension` floats, are kept, together with each row's maximum and sum; the logits chunk is reused. `backward()` scales them by the upstream gradient and adds them to the hidden states. If the head's weight needs a gradient, the backward pass recomputes each chunk's logits from the stored row statistics and accumulates `hiddenᵀ · gradient` (`multiplyTransposedA`); the bias gradient is collected during the forward pass. Because the gradient is computed as `exp(logit − max) / sum`, exactly like `crossEntropyLoss`, the fused and unfused results agree to within rounding.

**What we looked for.** Loss and gradients (hidden states, weight, bias) equal to `Linear` followed by `crossEntropyLoss`, within 1 × 10⁻⁶, with masked rows; the same with an INT4 head; a 524,288-token vocabulary that forces several chunks; and `Transformer.loss` equal to `crossEntropyLoss(model.forward(…))`, including the adapter gradients, with and without the Vector API. A frozen head never allocates a weight gradient.

**Results.** The output layer alone — projection and loss for 64 tokens, forward and backward, 896 × 151,936 weights, on the 2-core container used for all measurements in this section:

| | Seconds |
| :-- | --: |
| 1.0.0 (unfused; the head's weight gradient always computed) | 2.7 |
| 1.1.0, frozen head, all 64 tokens | 2.0 |
| 1.1.0, frozen head, 18 response tokens | 0.70 |
| 1.1.0 with the Vector API, all 64 tokens | 0.93 |
| 1.1.0 with the Vector API, 18 response tokens | 0.33 |

### 3. Frozen parameters and early backward termination

**Files:** `Tensor.kt`, `Module.kt`, `LoRA.kt`, `AdamW.kt`, `SGD.kt`, `FineTuningPipeline.kt`, `Attention.kt`, `RMSNorm.kt`, `RoPE.kt`, `Embedding.kt`, `Loss.kt`.

**What changed.**

- `Tensor.requiresGrad`: leaves (parameters and inputs) start with `true`, and the result of an operation requires a gradient only if one of its inputs does. `Tensor.hasGrad` tells whether the gradient array has been allocated.
- `Module.requiresGrad` reads or sets the flag of all of a module's parameters, so `module.requiresGrad = false` freezes it.
- Every backward operation skips the gradient of an operand that does not require one: a frozen matmul weight gets no `Aᵀ · dC`, a frozen bias or norm weight no accumulation, attention no query or key gradient when neither is needed.
- `backward()` only visits tensors that require a gradient, so a subgraph made only of frozen tensors is never entered.
- `LoRAAdapter` freezes every base weight when it is attached and restores the flags in `merge()`. The new `blockIndices` parameter attaches adapters to selected blocks only, for example `LoRAAdapter(model, blockIndices = 20 until 24)`.
- `AdamW` and `SGD` skip frozen tensors, and `zeroGrad()` no longer allocates gradient arrays that were never used.
- `FineTuningPipeline` clears stray gradients only where they exist, instead of touching (and so allocating) every base weight's gradient.
- `QuantizedEmbedding` outputs do not require gradients (their weights cannot be trained).

**Why it was needed.** Only the adapters are trained, yet 1.0.0 computed the full weight gradient of every frozen projection — as much work as the input gradient itself — and then threw it away. Worse, every gradient computed allocated a gradient array the size of its weight, so LoRA training held two copies of the model: the 24-layer Qwen2-0.5B-shaped model ran out of a 6 GB heap in its first step.

**How it works.** The flag is computed when a tensor is created, from its parents, so it costs one pass over two or three parents. The topological sort recurses only into parents that require a gradient. Inside each backward closure, every operand's gradient is guarded by its own flag, and the gradient arrays are only touched — and so allocated — when they are needed. With adapters on the query and value projections, the residual stream above the lowest adapted block still needs gradients, because it carries the loss back to the adapters below, but no frozen weight does; below the lowest adapted block, nothing does, and the backward pass never runs there.

**What we looked for.** Adapter gradients identical whether the base is frozen or not (two identically seeded models), no gradient array ever allocated for a frozen weight, a probe closure in a frozen subgraph never called, blocks below the lowest adapter left out of the graph, frozen weights unchanged by `AdamW` with weight decay and by `SGD`, `merge()` restoring the flags, and every pre-existing test unchanged.

### 4. Matrix-multiplication backward without transposed copies

**Files:** `MatMul.kt` (`multiplyTransposedA`, `multiplyTransposedB`), `Tensor.kt`.

**What changed.** `Tensor.matmul` back-propagates with `dA = dC · Bᵀ` through `MatMul.multiplyTransposedB` and `dB = Aᵀ · dC` through `MatMul.multiplyTransposedA`, both public, with the same `accumulate` / `alpha` options as `multiply`.

**Why it was needed.** 1.0.0 transposed both operands into new arrays on every backward call: for Qwen's output head that was a 544 MB copy per step, just to put the data in the order the forward kernel expects.

**How it works.** `Aᵀ · B` needs no copy at all: the tile kernel now reads its left operand through a row stride and a column stride, so a transposed `A` is just another pair of strides. `A · Bᵀ` is a matrix of dot products, which HotSpot does not vectorize; on the scalar path, `B` is therefore transposed one chunk at a time — at most 4 MB, in 32 × 32 cache blocks, spread over the worker threads — and each chunk goes through the vectorizable tile kernel. With the Vector API (see 5), `A · Bᵀ` is computed directly as SIMD dot products: 32-row by 64-column tiles, k in chunks of 1,024, four dot products per load of the `A` row.

**What we looked for.** Equality with a naive product on odd shapes (1 × 1 × 1 up to 40 × 2,100 × 33, several k chunks), with accumulation and a negative `alpha`, in both backends; parallel results identical to serial ones; and the existing `matmul` gradient test unchanged. Throughput on 2 threads, in GFLOP/s, for the input gradient `dC · Bᵀ` of training-shaped products:

| Product | 1.0.0 | 1.1.0 | 1.1.0 + Vector API |
| :-- | --: | --: | --: |
| [64 × 896] · [896 × 896]ᵀ | 19.9 | 20.7 | 57.1 |
| [64 × 4,864] · [896 × 4,864]ᵀ | 19.3 | 23.5 | 73.7 |
| [64 × 896] · [4,864 × 896]ᵀ | 20.4 | 25.8 | 75.1 |
| [18 × 37,984] · [896 × 37,984]ᵀ (a quarter of Qwen's head) | 4.8 | 14.8 | 60.7 |

### 5. Optional JVM Vector API backend

**Files:** new `SimdKernels.kt`; `MatMul.kt`, `build.gradle.kts`, new `src/main/resources/META-INF/proguard/kotensor.pro`.

**What changed.** `SimdKernels` implements the matmul inner loops — `y += a · x` for one or four rows, and one or four dot products — with `jdk.incubator.vector.FloatVector`, `SPECIES_PREFERRED` and fused multiply-add. `MatMul` loads it when the JVM has the `jdk.incubator.vector` module (JDK 17 or newer, started with `--add-modules jdk.incubator.vector`) and uses it in the tiled FP32 and dequantizing kernels, the quantized backward kernel and `A · Bᵀ`. `MatMul.vectorApiAvailable` reports whether it loaded, `MatMul.useVectorApi` turns it off and on at run time, and `-Dkotensor.vectorApi=false` keeps it from loading.

**Why it was needed.** The C2 compiler auto-vectorizes `y[j] += a · x[j]` but never a floating-point reduction, because vectorizing it would reorder additions. Every dot product — the input gradient of every projection and the quantized backward kernel — therefore ran one float at a time. The Vector API states the vector operations explicitly, so the JIT emits AVX2 or AVX-512 (or NEON) instructions regardless.

**How it works.** `MatMul` references the backend only through an internal interface and creates it by reflection after checking that `jdk.incubator.vector.FloatVector` resolves, then compares one `axpy` and one dot product with the scalar code before using it. Without the module, or on Android — whose runtime has no Vector API — nothing is loaded and the scalar kernels run exactly as before; the JAR carries a `-dontwarn jdk.incubator.vector.**` rule, so R8 builds of Android apps don't fail on the missing classes. The library is compiled with `-Xadd-modules=jdk.incubator.vector`, still against the Java 17 API. Single-row forward products — every decode step — always use the scalar kernels: they are bound by memory bandwidth, not arithmetic, and the Vector API's objects are only eliminated in fully optimized code, so running decode on it made a few bytes of allocation appear in 3 of 15 runs of the zero-allocation test.

**What we looked for.** Every kernel with the Vector API agrees with the scalar kernel within 1 × 10⁻³ (tiled, single-row, dequantizing, quantized backward and `A · Bᵀ`); the whole suite passes in both modes; when the module is missing, `useVectorApi` stays `false` and cannot be turned on. `./gradlew test` now also runs the new `vectorApiTest` task, the same suite with the module added. On 2 threads the tiled matmul becomes 1.2–1.6× faster with the Vector API, and `A · Bᵀ` 1.8–4.1× faster (table in 4).

### 6. Per-batch training telemetry

**Files:** `FineTuningPipeline.kt`.

**What changed.** `train(dataLoader, epochs, onBatchEnd, onEpochEnd)` calls `onBatchEnd(epoch, batch, totalBatches, loss)` after every batch, with 1-based batch numbers. A trailing lambda still binds to `onEpochEnd`, so existing calls compile unchanged. A batch whose targets are all masked is skipped without an optimizer step (`trainStep` returns 0) and left out of the epoch average.

**Why it was needed.** A CPU epoch can take minutes, and with only an end-of-epoch callback nothing was printed in between, so training looked frozen — in Gradle runs especially. The callback is what a progress line such as `[Epoch 1/3 | Batch 45/150 | Loss: 2.14 | 30%]` needs.

**What we looked for.** One report per batch with the right epoch, index and total; the epoch average; the trailing-lambda form; and a batch without targets leaving the adapter unchanged.

### 7. Compatibility

Version 1.1.0 is a drop-in replacement for 1.0.0, in source and in bytecode. The changed signatures keep their old forms: `crossEntropyLoss(logits, targets)` and `DataLoader(tokens, batchSize, sequenceLength)` through `@JvmOverloads`, `LoRAAdapter(model, rank, alpha)` through a secondary constructor, and `train(dataLoader, epochs, onEpochEnd)` through a hidden overload. A client compiled against 1.0.0 was run unchanged against 1.1.0 to confirm it.

Behavior changes to be aware of: attaching a `LoRAAdapter` freezes the base weights until `merge()`, so their gradient arrays stay empty; optimizers ignore tensors whose `requiresGrad` is `false`; and the cross-entropy losses treat −100 as "ignore", which earlier versions rejected as out of range.

### 8. Results

LoRA fine-tuning steps measured with `FineTuningPipeline.trainStep` on random weights with Qwen2-0.5B's shape — dimension 896, 14 query / 2 key-value heads, feed-forward size 4,864, a 151,936-token vocabulary with a separate output head, rank-8 adapters on the query and value projections — and a batch of 64 tokens, of which the last 18 are response tokens when masked. The machine is a 2-core Intel Xeon container with AVX-512 running JDK 21 with a 6 GB heap; each figure is the mean of three steps after a warm-up step.

| Seconds per step | 1.0.0 | 1.1.0 | 1.1.0, masked | 1.1.0 + Vector API | 1.1.0 + Vector API, masked |
| :-- | --: | --: | --: | --: | --: |
| 4 layers | 3.87 | 2.97 | 1.46 | 1.47 | 0.81 |
| 8 layers | 4.84 | 3.95 | 2.07 | 2.05 | 1.19 |
| 24 layers (full depth) | out of memory | 6.44 | 4.62 | 3.66 | 3.04 |

| Heap after a step | 1.0.0 | 1.1.0 |
| :-- | --: | --: |
| 4 layers | 2.77 GB | 1.39 GB |
| 8 layers | 3.35 GB | 1.68 GB |
| 24 layers | more than 6 GB | 2.84 GB |

The plan estimated 2.5–3.5 seconds per step for the full model once masking and the fused loss were in place (Milestone 6); on this 2-core machine the full model takes 3.0 seconds with the Vector API and 4.6 seconds without it. With the output layer fixed, the 24 frozen blocks are now the larger cost — about 0.11 (Vector API) to 0.17 seconds per block per step here, for their forward pass and the input gradient that has to flow through them to reach the adapters — which is what adapting only the top blocks with `blockIndices` avoids.

---

## Cross-cutting changes

### Documentation

- `README.md` was updated with each milestone and fully revised for v1.0.0: a new introduction and feature list, a "What's new" table, a quick start for running a pre-trained model, Maven Central coordinates in place of the old placeholders, Android requirements, and sections on the Android sample, benchmarks, API documentation and releasing. The public "Contributing" section was removed.
- `Milestone.md` records the status of each milestone and how it was delivered.
- `Module.md` (new) is the Dokka overview page, and this file is new.
- For v1.1.0 the README gained sections on freezing, loss masking, training efficiency (with the measurements) and the Vector API backend, its API, limitation, test and benchmark tables were updated, and `Milestone.md` and `Module.md` record the new milestone and APIs.

### Code style

From milestone 3 on, the code is written without comments or KDoc, following the project's no-comments rule; the v1.1.0 code follows it too. Comments from earlier milestones were removed in your own clean-up pass, and later write-backs restored the comment-free versions wherever a file had been regenerated. The files on disk are comment-free throughout.

### Versions and coordinates

The version went 0.1.0 → 0.2.0 → 0.3.0 → 0.4.0 → 0.5.0 → 1.0.0 → 1.1.0. The Maven coordinates changed from `com.github.TSPstudio:kotensor` to `io.github.tspstudio:kotensor`. The spec originally named `io.kotensor:kotensor`; the GitHub namespace was chosen because it is verified through the GitHub account rather than a domain (see *Maven Central publishing* under v1.0.0).

### Test suite growth

| Suite | Tests | Added in | Purpose |
| :-- | :-- | :-- | :-- |
| `CoreTensorTest` | 4 | 0.1.0 | Tensor operations, activations, broadcasting, SGD convergence |
| `TransformerTest` | 3 | 0.1.0 | Block forward/backward, cross-entropy, end-to-end training |
| `Phase5Test` | 5 | 0.1.0 | Safetensors, persistence, LoRA, streaming, fine-tuning pipeline |
| `RegressionTest` | 4 | 0.1.0, +1 in 0.2.0 | Batch independence, LoRA merge, rotary positions, weight names |
| `KVCacheTest` | 8 | 0.2.0 | Cache layout, cached = full forward, greedy equality, stop tokens, limits |
| `BPETokenizerTest` | 8 | 0.2.0, +1 in 1.0.0 | Hugging Face equality, coverage, special tokens, templates, JSON, regex flags |
| `MatMulTest` | 5 | 0.3.0 | Kernel vs naive, parallel = serial, gradients, in-place ops |
| `InferenceTest` | 5 | 0.3.0 | Allocation-free decode equals graph path, zero bytes allocated |
| `GQATest` | 6 | 0.4.0 | GQA equals replicated multi-head, gradients, all paths, rope scaling |
| `HuggingFaceImportTest` | 7 | 0.4.0 | NumPy-reference logits, shards, biases, tied heads, dtypes, errors |
| `QuantizationTest` | 11 | 0.5.0 | Error bounds, kernels, model paths, QLoRA, round trips, memory |
| `TrainingTest` | 12 | 1.1.0 | Loss masking, fused loss, freezing and early termination, transposed kernels, Vector API vs scalar, data loading, telemetry |
| **Total** | **78** | | |

---

## Known limitations and open items

**Build and release.** v1.0.0 was built, tested and published to Maven Central by the GitHub workflows. v1.1.0 adds `-Xadd-modules=jdk.incubator.vector` to the compiler options and a `vectorApiTest` task that `./gradlew test` runs after the regular tests, so CI and the release workflow check both backends.

**Engine limitations that remain:**

- Greedy decoding only; there is no temperature or top-k / top-p sampling.
- Quantization is round-to-nearest per group, without calibration (GPTQ / AWQ), and GGUF files are not read.
- LoRA covers the query and value projections only, in all blocks or a chosen subset.
- Loss masking applies to the loss: attention still sees masked tokens, and examples packed into one sequence can attend to earlier examples.
- The Vector API backend needs `--add-modules jdk.incubator.vector` on JDK 17+, because the module is still incubating; Android always runs the scalar kernels.
- The tokenizer handles byte-level BPE; SentencePiece-style tokenizers (Llama 2, Mistral) and pre-tokenizers other than `Split` and `ByteLevel` (such as `Digits`) are not supported.
- The importer accepts the `llama`, `qwen2` and `mistral` architectures, and rejects others rather than guessing.
- Performance was tuned on HotSpot; on Android, ART compiles the same code differently, so phone throughput should be measured with the sample app.
