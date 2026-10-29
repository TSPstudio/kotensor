# Module Kotensor

Pure Kotlin automatic differentiation and decoder-only Transformer engine for the JVM and Android.

The whole stack is plain Kotlin on top of the standard library and `java.io` / `java.nio`: tensors with
reverse-mode autograd, RMSNorm / rotary / SwiGLU Transformer blocks with grouped-query attention,
KV-cached and allocation-free decoding, a tiled multi-threaded matmul, byte-level BPE tokenization,
LoRA fine-tuning, INT8 / INT4 weight quantization, `.safetensors` I/O and a Hugging Face checkpoint importer.

# Package io.kotensor.core

Every public type of the engine lives in this package.

- Core: `Tensor`, `MatMul`, `Quantization`, `QuantizedTensor`, `Module`
- Layers: `Linear`, `QuantizedLinear`, `Embedding`, `QuantizedEmbedding`, `RMSNorm`, `RoPE`, `SwiGLU`, `Attention`, `TransformerBlock`, `Transformer`, `KVCache`, `InferenceBuffers`
- Training: `mseLoss`, `crossEntropyLoss`, `SGD`, `AdamW`, `DataLoader`, `LoRALinear`, `LoRAAdapter`, `FineTuningPipeline`
- Text: `Tokenizer`, `SimpleTokenizer`, `BPETokenizer`, `IncrementalDecoder`, `StreamingSession`
- I/O: `SafeTensors`, `SafeTensorsFile`, `HuggingFaceModelLoader`, `HuggingFaceConfig`, `JsonParser`
