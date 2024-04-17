# Save/Load Compressed SafeTensors

## Motivation

* Reduce disk space by saving in a compressed format for sparse models. Models in this compressed format will be loaded by vLLM for more efficient inference
* Set up the save/load architecture such that we can easily expand to additional compression formats in the future. The config should be human readable so users can understand the compression format at a quick glance

## SafeTensors File Format

For each parameter in the uncompressed state_dict, we store the following attributes 
needed for decompression in the compressed state_dict:

* compressed tensor
* bitmask
* uncompressed shape
* row offsets

```python
# dense
{
    PARAM_NAME: uncompressed_tensor
}

# compressed
{
    PARAM_NAME.compressed: compressed_tensor # 1d tensor
    PARAM_NAME.bitmask: value # 2d bitmask tensor (nrows x (ncols / 8))
    PARAM_NAME.shape: value # uncompressed shape tensor
    PARAM_NAME.row_offsets: value # 1d offsets tensor
}
```

Config information gets stored in the HF config file
```json
// config.json
{
    "sparsity_config": {
        "format": "sparse_bitmask", // "dense_sparsity" for original tensor format

        // informational
        "sparsity_structure": "unstructured", // or 2:4, 8:16 etc...
        "global_sparsity": "0.5"
    }
}
```

## Saving/Loading Interface 

Loading in a compressed model requires no interface changes

```python
from sparseml.transformers.utils import SparseAutoModelForCausalLM

# should contain model.safetensors or model.safetensors.index.json
model_path = "/PATH/TO/COMPRESSED_MODEL"

model = SparseAutoModelForCausalLM.from_pretrained(
    model_name_or_path=model_path,
    **model_kwargs,
)
```

Saving a compressed model with an explicitly provided compression config. The config
is saved to the model's `config.json` file. **Note:** the model must have been 
initialized with SparseAutoModelForCausalLM.from_pretrained()

```python
from compressed_tensors import BitmaskConfig

output_dir = "/PATH/TO/SAVE/COMPRESSED_MODEL"
sparsity_config = BitmaskConfig()

model.save_pretrained(
    save_directory=output_dir,
    sparsity_config=sparsity_config,
)
```

Saving a compressed model, inferring the config from the model attributes

```python
model.save_pretrained(
    save_directory=output_dir,
    save_compressed=True
)
```

Saving a model in the dense format. If the model has at least 5% global sparsity a 
sparsity config will still be included in `config.json` with format `dense_sparsity`

```python
model.save_pretrained(
    save_directory=output_dir
)
```

Saving a model in the dense format, bypassing the sparsity config calculation. When the
`skip_compression_stats` flag is set, no sparsity config will be written to 
`config.json`

```python
model.save_pretrained(
    save_directory=output_dir
    skip_compression_stats=True
)
```

## Enable Compression During One-Shot and Sparse Finetunining
Models that are saved in a supported compressed format on disk will automatically be
decompressed when loaded as input to `sparseml.transformers.oneshot` or 
`sparseml.transformers.train`

To enable compression on save after oneshot or finetuning simply add the 
`save_compressed=True` argument to `sparseml.transformers.oneshot` or 
`sparseml.transformers.train`

```python
from sparseml.transformers import train

train(
    save_compressed=True,
    model="neuralmagic/TinyLlama-1.1B-Chat-v1.0-pruned2.4",
    recipe=RECIPE,
    dataset=DATASET
)
```


## Example Code

Loads a 60% sparse model, compresses it using the inferred bitmask compression, then 
reloads the compressed model.

```python
from sparseml.transformers import SparseAutoModelForCausalLM
from sparseml.utils.pytorch.utils import measure_cuda_memory
import torch

MODEL_PATH = "zoo:llama2-7b-open_platypus_orca_llama2_pretrain-pruned60"
OUTPUT_PATH = "./test_compress_output"
RECIPE = "zoo:llama2-7b-open_platypus_orca_llama2_pretrain-pruned60"

torch.cuda.set_device(0)
with measure_cuda_memory() as m:
    model = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda:0")
print(f"Load dense model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")

sparsity_config = getattr(model,"sparsity_config", None)
print(f"Sparsity config before compression: {sparsity_config}")
with measure_cuda_memory() as m:
    model.save_pretrained(OUTPUT_PATH, save_compressed=True)
print(f"Save compressed model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")

torch.cuda.set_device(1)
with measure_cuda_memory() as m:
    model_again = SparseAutoModelForCausalLM.from_pretrained(
        OUTPUT_PATH, device_map="cuda:1"
    )
print(f"Load compressed model peak GPU {m.overall_peak_memory / float(2**30):.4f} GB")
sparsity_config = getattr(model_again,"sparsity_config", None)
print(f"Sparsity config after compression: {sparsity_config}")
```
