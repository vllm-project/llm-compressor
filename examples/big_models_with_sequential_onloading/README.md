## Big Modeling with Sequential Onloading ##
### What is Sequential Onloading? ###
Sequential onloading is a memory-efficient approach for compressing large language models (LLMs) using only a single GPU. Instead of loading the entire model into memory—which can easily require hundreds of gigabytes—this method loads and compresses one layer at a time. The outputs are offloaded before the next layer is processed, dramatically reducing peak memory usage while maintaining high compression fidelity.

<p align="center">
    <img src="assets/sequential_onloading.png"/>
</p>

For more information, see the [RedHat AI blog post](https://developers.redhat.com/articles/2025/05/09/llm-compressor-optimize-llms-low-latency-deployments#generalizing_to_multimodal_and_moe_architectures) or the [LLM Compressor Office Hours Recording](https://www.youtube.com/watch?v=GrhuqQDmBk8).

### Using Sequential Onloading ###
Sequential onloading is enabled by default within LLM Compressor. To disable sequential onloading, add the `pipeline="basic"` argument to the LLM Compressor `oneshot` function call.