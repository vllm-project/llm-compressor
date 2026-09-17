# HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization

HIGGS is an automated mixed-precision quantization system that uses Integer Linear Programming (ILP) to select optimal quantization schemes for each layer in a neural network. This technique is based on the paper: [Pushing the Limits of Large Language Model Quantization via the
Linearity Theorem](https://arxiv.org/pdf/2411.17525)


## Quick start

```python
from llmcompressor import model_free_ptq
from llmcompressor.entrypoints.higgs import get_higgs_config

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

config = get_higgs_config(
    model_stub=model_id,
    candidate_schemes=["NVFP4A16", "FP8_DYNAMIC"],
    targets="Linear",
    ignore=["lm_head", "re:.*embed_tokens"],
    target_avg_bitwidth=6.0,
    allow_unquantized=True,
)

model_free_ptq(
    model_stub=model_id,
    save_directory="./Llama-3.1-8B-HIGGS",
    config=config,
)
```

`get_higgs_config` only chooses schemes. Apply the returned config with
`model_free_ptq` when every selected scheme is data-free, or with `oneshot` and
a `QuantizationModifier` or `GPTQModifier` when calibration is required.

## Optimization

For every matched weight tensor, HIGGS:

1. Fake-quantizes the weight with each candidate scheme and records its MSE.
2. Assigns a heuristic importance value based on parameter count, layer depth,
   and module type.
3. Detects fused groups, including attention projections, MLP gate/up
   projections, and MoE experts.
4. Solves an ILP that minimizes weighted MSE while assigning exactly one choice
   to each tensor.
5. Converts the solution into compressed-tensors config groups. Individual MoE
   expert targets are collapsed into regular expressions in the resulting
   config.

The optional weight constraint is parameter-weighted over tensors matched by
`targets` after `ignore` is applied:

```text
sum(params[layer] * weight_bits[choice]) / sum(params[layer])
    <= target_avg_bitwidth
```

`target_avg_act_bitwidth` adds an analogous constraint using parameter count as
a proxy for activation cost. Ignored tensors are outside both averages.

Fused-layer constraints are enabled by default. Set
`enforce_fused_layer_constraints=False` only when the intended runtime supports
different schemes within those fused groups.

## Unquantized layers

`allow_unquantized` defaults to `True`. This adds a synthetic 16-bit choice with
zero quantization MSE to the ILP. Tensors assigned to that choice remain in the
original dtype and are omitted from `config.config_groups`; it is not emitted as
a quantization scheme. Set `allow_unquantized=False` to require every matched
tensor to use one of `candidate_schemes`.

Because the bitwidth constraints are upper bounds, HIGGS need not consume the
entire budget. With no `target_avg_bitwidth`, it warns and simply chooses the
lowest-MSE option for each tensor, which will generally be the unquantized choice
when it is enabled.

## Candidate schemes

Candidate entries can be compressed-tensors preset names or
`QuantizationScheme` objects:

```python
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor.entrypoints.higgs import get_higgs_config

custom_w4 = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type="int",
        strategy="group",
        group_size=128,
        symmetric=True,
    ),
)

config = get_higgs_config(
    model_stub="meta-llama/Meta-Llama-3.1-8B-Instruct",
    candidate_schemes=[custom_w4, "W8A16"],
    target_avg_bitwidth=6.0,
)
```

Preset availability is defined by compressed-tensors. Common choices include
`W4A16`, `W8A16`, `NVFP4A16`, `NVFP4`, `FP8`, and `FP8_DYNAMIC`.
Static activation schemes such as `NVFP4` and `FP8` require calibration when the
config is applied. Weight-only or dynamic-activation schemes such as
`NVFP4A16` and `FP8_DYNAMIC` can be applied with `model_free_ptq`.

## API

```python
get_higgs_config(
    model_stub,
    candidate_schemes,
    targets="Linear",
    ignore=None,
    enforce_fused_layer_constraints=True,
    target_avg_bitwidth=None,
    target_avg_act_bitwidth=None,
    device=None,
    allow_unquantized=True,
)
```

When `device` is omitted, HIGGS uses CUDA if available and otherwise uses the
CPU. The returned `QuantizationConfig` preserves `ignore` and has compressed
status so it can be passed directly to the model-free converter architecture or
used to construct a modifier.

## Source layout

```text
src/llmcompressor/entrypoints/higgs/
├── __init__.py   # public exports
├── base.py       # MSE collector and get_higgs_config
├── ilp_solver.py # PuLP/CBC formulation
└── utils.py      # MSE, alpha, fusion, and config helpers
```

## Tests

```bash
pytest tests/llmcompressor/entrypoints/higgs -v
```

The ILP solver uses the CBC backend included with PuLP.
