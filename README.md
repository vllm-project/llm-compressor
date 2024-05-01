# compressed_tensors

This repository extends a [safetensors](https://github.com/huggingface/safetensors) format to efficiently store sparse and/or quantized tensors on disk. `compressed-tensors` format supports multiple compression types to minimize the disk space and facilitate the tensor manipulation.

## Motivation

### Reduce disk space by saving sparse tensors in a compressed format

The compressed format stores the data much more efficiently by taking advantage of two properties of tensors:

- Sparse tensors -> due to a large number of entries that are equal to zero.
- Quantized -> due to their low precision representation.

### Introduce an elegant interface to save/load compressed tensors

The library provides the user with the ability to compress/decompress tensors. The properties of tensors are defined by human-readable configs, allowing the users to understand the compression format at a quick glance.

## Installation

### Pip

```bash
pip install compressed-tensors
```

### From source

```bash
git clone https://github.com/neuralmagic/compressed-tensors
cd compressed-tensors
pip install -e .
```

## Getting started

### Saving/Loading Compressed Tensors (Bitmask Compression)

The function `save_compressed` uses the `compression_format` argument to apply compression to tensors.
The function `load_compressed` reverses the process: converts the compressed weights on disk to decompressed weights in device memory.

```python
from compressed_tensors import save_compressed, load_compressed, BitmaskConfig
from torch import Tensor
from typing import Dict

# the example BitmaskConfig method efficiently compresses 
# tensors with large number of zero entries 
compression_config = BitmaskConfig()

tensors: Dict[str, Tensor] = {"tensor_1": Tensor(
    [[0.0, 0.0, 0.0], 
     [1.0, 1.0, 1.0]]
)}
# compress tensors using BitmaskConfig compression format (save them efficiently on disk)
save_compressed(tensors, "model.safetensors", compression_format=compression_config.format)

# decompress tensors (load_compressed returns a generator for memory efficiency)
decompressed_tensors = {}
for tensor_name, tensor in load_compressed("model.safetensors", compression_config = compression_config):
    decompressed_tensors[tensor_name] = tensor
```

## Saving/Loading Compressed Models (Bitmask Compression)

We can apply bitmask compression to a whole model. For more detailed example see `example` directory.
```python
from compressed_tensors import save_compressed_model, load_compressed, BitmaskConfig
from transformers import AutoModelForCausalLM

model_name = "neuralmagic/llama2.c-stories110M-pruned50"
model = AutoModelForCausalLM.from_pretrained(model_name)

original_state_dict = model.state_dict()

compression_config = BitmaskConfig()

# save compressed model weights
save_compressed_model(model, "compressed_model.safetensors", compression_format=compression_config.format)

# load compressed model weights (`dict` turns generator into a dictionary)
state_dict = dict(load_compressed("compressed_model.safetensors", compression_config))
```

For more in-depth tutorial on bitmask compression, refer to the [notebook](https://github.com/neuralmagic/compressed-tensors/blob/d707c5b84bc3fef164aebdcd97cb6eaa571982f8/examples/bitmask_compression.ipynb).
