# Select a Model and Dataset

Before you start compressing, select the model you'd like to compress and a calibration dataset that is representative of your use case. LLM Compressor supports a variety of models and integrates natively with Hugging Face Transformers and Model Hub, so a great starting point is to use a model from the Hugging Face Model Hub. LLM Compressor also supports many datasets from the Hugging Face Datasets library, making it easy to find a suitable dataset for calibration.

For this guide, we'll use the `TinyLlama` model and the `open_platypus` dataset for calibration. You can replace these with your own model and dataset as needed.

## Next Steps

- [Choosing the right compression scheme](choosing-scheme.md)
- [Choosing the right quantization, sparsity, and transform-based algorithms](choosing-algo.md)
- [Compress your first model](compress.md)
- [Deploy with vLLM](deploy.md)
