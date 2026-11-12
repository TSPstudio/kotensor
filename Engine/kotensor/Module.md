# Module Kotensor

Pure Kotlin automatic differentiation and decoder-only Transformer engine for the JVM and Android.

The whole stack is plain Kotlin on top of the standard library and `java.io` / `java.nio`: tensors with
reverse-mode autograd, RMSNorm / rotary / SwiGLU Transformer blocks with grouped-query attention,
KV-cached decoding with a buffered prompt prefill and an allocation-free decode step, greedy or sampled generation, a tiled multi-threaded
matmul with an optional JVM Vector API backend, byte-level BPE tokenization, LoRA fine-tuning with
frozen-weight backpropagation, loss masking and a fused output-layer loss, INT8 / INT4 weight quantization
with memory-mapped model files and tied embeddings, `.safetensors` I/O and a Hugging Face checkpoint importer.

# Package io.kotensor.core

Every public type of the engine lives in this package.

- Core: `Tensor`, `noGrad`, `MatMul`, `Quantization`, `QuantizedTensor`, `Module`
- Layers: `Linear`, `QuantizedLinear`, `Embedding`, `QuantizedEmbedding`, `TiedQuantizedEmbedding`, `RMSNorm`, `RoPE`, `SwiGLU`, `Attention`, `TransformerBlock`, `Transformer`, `KVCache`, `PromptPrefix`, `InferenceBuffers`
- Training: `mseLoss`, `crossEntropyLoss`, `fusedCrossEntropyLoss`, `IGNORE_INDEX`, `SGD`, `AdamW`, `DataLoader`, `LoRALinear`, `LoRAAdapter`, `FineTuningPipeline`
- Text: `Tokenizer`, `SimpleTokenizer`, `BPETokenizer`, `IncrementalDecoder`, `StreamingSession`, `Sampling`, `Sampler`
- I/O: `SafeTensors`, `SafeTensorsFile`, `HuggingFaceModelLoader`, `HuggingFaceConfig`, `JsonParser`
