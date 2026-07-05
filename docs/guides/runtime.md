# Runtime requirements for LLM Compressor

The following are typical runtimes for each LLM Compressor algorithm based on runs using Meta-Llama-3-8B-Instruct on a NVIDIA A100 Tensor Core GPU.   

| Algorithm| Estimated Time 
|--------|-------------|
| **RTN (QuantizationModifier)** <br> Weights only (no activation quant) | ~ 1 minutes |
| **RTN (QuantizationModifier)** <br> Weights and activations | ~ 20 minutes  |
| **GPTQ** (weights only) | ~ 30 minutes | 
| **AWQ** (weights only) | ~ 30 minutes | 