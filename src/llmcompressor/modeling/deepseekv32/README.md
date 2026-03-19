This module was created with the following actions:
- copy over model.py and kernel.py from checkpoint (https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/inference)
- ModelArgs -> ModelConfig subclasses PretrainedConfig
- Add DeepseekV32PreTrainedModel and DeepseekV32ForCausalLM classes that wrap Transformer as model field, instead of DeepseekV3Model
- replace names corresponding to convert.py (https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/convert.py#L12-L33) but NOT for Indexer class
- Gate module rename bias to e_score_correction_bias, add @property bias to get around attribute not found issues
- merge into config.json any fields not already existing in https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/config_671B_v3.2.json
- change `tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True` to `tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False`
- drop Linear/RowParallel/ColumnParallelLinear classes, use torch.nn.Linear instead
- Indexer refactor:
    - replace `fp8_index` tilelang kernel with `bf16_index` built in torch primitives (by claude). TODO validate this somehow
    - remove all hard-coded logic for fp8 block encoding
- Rename Transformer input var name from `tokens` to `input_ids`
- add bias=False to all Linear constructors in model def (defaulted to False on original model def's Linear class)
- wrap all dist calls in `if world_size > 1` conditional
- add `.float()` to LayerNorm, weight/bias dtypes change to bf16 at layer 25(?)
- change weights_proj to bfloat16, pass in x to weights_proj forward pass instead of x.float()

Model was created with the following actions:
- run compressed-tensors/examples/convert_checkpoint/deepseek32_fpblock_example.py
- copy over missing values (excluding "dtype") into config.json, from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/config_671B_v3.2.json
- run llm-compressor/examples/disk_offloading/deepseek_v32_example.py
