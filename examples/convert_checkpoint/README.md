# Converting an AutoAWQ Checkpoint

`examples/convert_checkpoint/autoawq_example.py` converts an existing asymmetric AutoAWQ GEMM W4A16 checkpoint to compressed-tensors, verifies every quantized layer against the source tensors, and optionally runs a Transformers loading/inference smoke test on CPU.

Follow the [checkpoint conversion guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/convert/) for installation, a pinned source checkpoint, tensor layouts, verification details, and inference commands.
