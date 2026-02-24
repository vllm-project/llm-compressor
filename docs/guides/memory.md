# Memory requirements for LLM Compressor

When compressing a model you should be aware that the memory requirements are dependent on model size and the algorithm used, such as GPTQ/SparseGPT.  

This section will go through how to calculate the CPU and GPU memory requirements for each algorithm using several popular models, an 8B, a 684B, and a model with vision capabilities, as examples. 

The GPTQ/SparseGPT requires a large amount of auxiliary memory. GPTQ/SparseGPT allocates an auxiliary hessian matrix for any layers that are onloaded to the GPU. This is because the hessian matrices have to be almost as large as the weights they are trying to represent. 

Also, larger models, like DeepSeek R1 use a large amount of CPU memory, and models with large vision towers, such as command A, may use large amounts of GPU memory. 

## Things to note when calculating memory requirements for LLM Compressor:

1. A 1B model uses 2Gb of memory to load:
    ```
	mem(1B parameters) ~= (1B parameters) * (2 bytes / parameter) = 2B bytes ~= 2Gb
    ```

2. How text decoder layers and vision tower layers are loaded on to GPU differs significantly. 
    
    In the case of text decoder layers, LLM Compressor dynamically loads one layer at a time into the GPU for computation. The rest of the model remains in CPU memory. 

    However, vision tower layers are loaded onto GPU all at once. Unlike the text model, vision towers are not split up into individual layers before onloading to the GPU. This can create a GPU memory bottleneck for models whose vision towers are larger than their text layers.		

    At this time LLM Compressor does not quantise the vision tower as quantization is generally not worth the tradeoff between latency/throughput and accuracy loss.   

3. LLM Compressor does not currently support tensor parallelism for compression. Supporting this feature will allow layers to be sharded across GPUs, leading to reduced memory usage per GPU and faster compression.

## QuantizationModifier or Round-To-Nearest (RTN)

The quantization modifier, RTN, does not require any additional memory beyond the storage needed for its quantization parameters (scales/zeros). 

If we ignore these scales and zero points from our calculation, we can estimate the following memory requirements:


| Model| CPU requirements | GPU requirements |
|--------|-------------|----------------------------|
| **Meta-Llama-3-8B-Instruct** | mem(8B params) ~= 16Gb | mem(1 Layer) ~= 0.5Gb |
| **DeepSeek-R1-0528-BF16** | mem(684B params) ~= 1368Gb | mem(1 Layer) ~= 22.4Gb|
| **Qwen2.5-VL-7B-Instruct** | mem(7B params) ~= 14Gb | max(mem(1 Text Layer)~= 0.4B, mem(Vision tower)~=1.3B) ~= 1.3Gb |

## GPT Quantization(GPTQ)/ Sparse GPT 

The GPTQ/ SparseGPT algorithms differ from the RTN in that they must also allocate an auxiliary hessian matrices for any layers that are onloaded to the GPU. 

This hessian matrix is used to increase the accuracy recovery of the algorithm, and is approximately the same size as the original weights.

| Model| CPU requirements | GPU requirements |
|--------|-------------|----------------------------|
| **Meta-Llama-3-8B-Instruct** |mem(8B params) ~= 16Gb | mem(1 Layer) * 2 ~= 1Gb |
| **DeepSeek-R1-0528-BF16** | mem(684B params) ~= 1368Gb | mem(1 Layer) * 2 ~= 44.8Gb |
| **Qwen2.5-VL-7B-Instruct** | mem(7B params) ~= 14Gb | max(mem(1 Text Layer)~= 0.4B, mem(Vision tower)~=1.3B)*2 ~= 2.6Gb |