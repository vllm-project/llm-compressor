---
weight: -6
---

# Deploy with vLLM

Once you've compressed your model using LLM Compressor, you can deploy it for efficient inference using vLLM. This guide walks you through the deployment process, using the output from the [Compress Your Model](compress.md) guide. If you haven't completed that step, change the model arguments in the code snippets below to point to your desired model.

vLLM is a high-performance inference engine designed for large language models, providing support for various quantization formats and optimized for both single and multi-GPU setups. It also offers an OpenAI-compatible API for easy integration with existing applications.

## Prerequisites

Before deploying your model, ensure you have the following prerequisites:
- **Operating System:** Linux (recommended for GPU support)
- **Python Version:** 3.9 or newer
- **Available GPU:** For optimal performance, it's recommended to use a GPU. vLLM supports a range of accelerators, including NVIDIA GPUs, AMD GPUs, TPUs, and other accelerators.
- **vLLM Installed:** Ensure you have vLLM installed. You can install it using pip:
  ```bash
  pip install vllm
  ```

## Python API

vLLM provides a Python API for easy integration with your applications, enabling you to load and use your compressed model directly in your Python code. To test the compressed model, use the following code:

```python
from vllm import LLM, SamplingParams

model = LLM("./TinyLlama-1.1B-Chat-v1.0-INT8")
sampling_params = SamplingParams(max_tokens=256)
outputs = model.generate("What is machine learning?", sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

After running the above code, you should see the generated output from your compressed model. This confirms that the model is loaded and ready for inference.

## HTTP Server

vLLM also provides an HTTP server for serving your model via a RESTful API that is compatible with OpenAI's API definitions. This allows you to easily integrate your model into existing applications or services.
To start the HTTP server, use the following command:

```bash
vllm serve "TinyLlama-1.1B-Chat-v1.0-INT8"
```

By default, the server will run on `localhost:8000`. You can change the host and port by using the `--host` and `--port` flags. Now that the server is running, you can send requests to it using any HTTP client. For example, you can use `curl` to send a request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama-1.1B-Chat-v1.0-INT8",
        "messages": [{"role": "user", "content": "What is machine learning?"}],
        "max_tokens": 256
    }'
```

This will return a JSON response with the generated text from your model. You can also use any HTTP client library in your programming language of choice to send requests to the server.
