# Choosing your dataset

Depending on your selected algorithm or scheme, you may also require a dataset. Many quantization algorithms, such as GPTQ, AWQ, SmoothQuant, and AutoRound, require a calibration dataset to analyze activation patterns and optimize weight transformations. This dataset helps the algorithm identify which weights and activations are most critical to preserve during compression. LLM Compressor also supports many datasets from the Hugging Face Datasets library, making it easy to find a suitable dataset for calibration.

## Algorithms requiring datasets

- **AWQ**
- **GPTQ**
- **AutoRound**
- **SmoothQuant**

!!! info
    RTN (Round-to-Nearest) quantization is data-free and can compress models without any calibration dataset. However, calibration-based methods typically achieve better accuracy recovery, especially at lower bit-widths.

## Schemes requiring datasets

Quantization schemes where activations are quantized non-dynamically (i.e the scales to quantize the activations are not determined during inference time) will also require a dataset. 

These Include:
- **NVFP4**: Data is required to calibrate the activation scales, allowing quantization of the activatins to FP4 during inference
- **Static-Per Tensor Activation Quantization**: Commonly used with FP8 and INT8 weight quantization, if you are targeting a static-per tensor scheme for activation quantization, data is required to calibrate a single scale which enables quantization of the activations to 8 bits during inference

## Key considerations

When selecting a calibration dataset, consider the following factors:

### Domain alignment
The calibration dataset should be representative of your target use case. For general-purpose language models, common choices include:

- **General text datasets**: WikiText or C4 for broad language understanding
- **Instruction-tuning data**:  UltraChat for instruction-following models
- **Domain-specific data**: E.g. code datasets for coding models

Some popular datasets include: 

| Dataset | Best for | Description |
|---------|----------|-------------|
| `ultrachat-200k` | Instruction-following models | High-quality conversational data for chat and assistant models |
| `open-platypus` | General instruction models | Diverse instruction-following examples |
| `wikitext-2-raw-v1` | General language models | Clean Wikipedia text for broad language understanding |
| `c4` | General pre-training | Large-scale web text for general-purpose models |

### Dataset size
Most calibration algorithms work well with relatively small datasets:

- **Typical range**: 128-512 samples is sufficient for most models
- **Trade-off**: More samples improve representation but increase compression time
- **Recommendation**: Start with 128-256 samples; increase if accuracy recovery is insufficient

LLM Compressor makes it easy to use popular calibration datasets from Hugging Face by providing access to processed datasets. Users only need to pass in a string with the dataset name or a supported dataset. Begin with a standard dataset like `ultrachat-200k`, then iterate if needed. Alternatively, you can use your own custom dataset and pass in the dataset object into LLM Compressor for calibration as well.

LLM Compressor provides easy access to popular calibration datasets from Hugging Face. For [supported datasets](https://github.com/vllm-project/llm-compressor/tree/main/src/llmcompressor/transformers/data), simply pass the dataset name as a string (e.g., ultrachat-200k). Start with a standard dataset and iterate as needed. You can also use custom datasets by passing a dataset object directly to LLM Compressor for calibration.

- [Compress your first model](compress.md)
- [Deploy with vLLM](deploy.md)
