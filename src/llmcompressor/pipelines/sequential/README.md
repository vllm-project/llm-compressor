# Sequential Pipeline #
The sequential pipeline is a data pipeline, primarily used compressing models with the
[GPTQModifier](/src/llmcompressor/modifiers/quantization/gptq/base.py) or the
[SparseGPTModifier](/src/llmcompressor/modifiers/obcq/base.py).

If, when using this pipeline, you encounter a `torch.fx.proxy.TraceError`, see the
[Model Tracing Guide](/src/llmcompressor/transformers/tracing/GUIDE.md).