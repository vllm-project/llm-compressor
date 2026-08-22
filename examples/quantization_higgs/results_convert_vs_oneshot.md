# HIGGS: convert_checkpoint vs oneshot Comparison

Investigation into a config serialization bug that caused degraded
inference quality when using `convert_checkpoint` with HIGGS.

## The Bug

`generate_config_groups()` in `entrypoints/higgs/utils.py` named config
groups with scheme suffixes (e.g., `group_0_FP8_DYNAMIC`,
`group_1_NVFP4A16`). The `oneshot()` path strips these suffixes during
save (via `QuantizationConfig.from_pretrained()` which re-indexes to
`group_0`, `group_1`), but `convert_checkpoint` preserves the original
names from the `QuantizationConfig` object.

vLLM failed to correctly match layers to quantization schemes when
config group names contained suffixes, resulting in degraded inference
quality despite the quantized weights being byte-identical.

## Fix

Changed `generate_config_groups()` to use plain `group_{idx}` names
instead of `group_{idx}_{scheme_name}`, matching the convention used
by the rest of the compressed-tensors ecosystem.

## Verification

### Step 1: Confirm weights are identical

Compared the HIGGS 4.5-bit convert and oneshot checkpoints tensor by
tensor: 639 shared tensors, 0 differing. All `weight_packed`,
`weight_scale`, and `weight_global_scale` values are byte-identical.
Fused group global scales are correctly shared in both.

### Step 2: Isolate the config as the cause

Copied the convert checkpoint and manually renamed config groups from
`group_0_FP8_DYNAMIC` / `group_1_NVFP4A16` to `group_0` / `group_1`.
PPL changed from 9.4118 to 9.1921 — matching the oneshot result exactly.

### Step 3: Verify the code fix

Re-ran the full sweep with the fixed `generate_config_groups()`. All
results now match the oneshot path and show monotonic improvement.

## Results at 6.0 avg bits

| Method | Config Group Names | word_perplexity |
|--------|--------------------|-----------------|
| convert_checkpoint (before fix) | group_0_FP8_DYNAMIC, group_1_NVFP4A16 | 9.2471 |
| convert_checkpoint (after fix) | group_0, group_1 | 9.0381 |
| oneshot + QuantizationModifier | group_0, group_1 | 9.0381 |

## Impact on sweep results

| Target Bits | Before Fix | After Fix | Delta |
|-------------|------------|-----------|-------|
| 4.5 | 9.4118 | 9.1921 | -0.22 |
| 5.0 | 9.3373 | 9.1225 | -0.21 |
| 5.5 | 9.2847 | 9.0672 | -0.22 |
| 6.0 | 9.2471 | 9.0381 | -0.21 |
| 6.5 | 9.2123 | 9.0003 | -0.21 |
| 7.0 | 9.1760 | 8.9685 | -0.21 |
| 7.5 | 9.1400 | 8.9293 | -0.21 |

The bug inflated PPL by ~0.21 across all bitwidths. With the fix,
convert_checkpoint and oneshot produce identical results.
