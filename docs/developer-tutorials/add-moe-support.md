# Background

## How are Mixture of Experts (MoE) models calibrated?

Mixture of Experts (MoE) models route each token to only a subset of expert layers. This creates a calibration problem: Given that calibration datasets are relatively small, some experts will not be activated, or activated very infrequently. This can result in poorly calibrated quantization parameters, numerical instability, or NaNs.

LLM Compressor and other quantization frameworks resolve this by replacing MoE modules with calibration-friendly versions that route all tokens through all experts during calibration, while keeping only the routed expert outputs for the final result.

LLM Compressor also performs an operation called **linearization**, by which fused experts (with 3D weights) are loaded as a sequence of unfused, 2D linear weights. Linearization is useful because most algorithms (GPTQ, AWQ, Quantization) are built around calibration of 2D linear weights. Unfusing the weights can also lead to better GPU utilization during compression in DDP cases. In most cases, the checkpoint remains unfused and can be loaded by vLLM as unfused, although re-fusion into 3D weights may be necessary for some models.

# Adding MoE Support

Since upgrading to [transformers v5](https://github.com/vllm-project/llm-compressor/pull/2647), LLM Compressor automatically handles MoE support for nearly all model architectures supported by transformers.

Mixture of Experts (MoE) models should be loaded with the `load_context` provided by LLM Compressor in order to ensure that they are loaded correctly and optimally for calibration.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils import load_context

model_id = "zai-org/GLM-4.7"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Transformers v5 Background ##

One of the primary goals of [transformers v5](https://github.com/huggingface/transformers/releases/tag/v5.0.0) was to accelerate model inference when loading with the transformers framework. For MoE models, this means loading the model experts as fused experts. Fused experts allow the model to use fast and efficient [`group_mm`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/moe.py) and [`batch_mm`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/moe.py) operations during inference.

This is achieved through 2 systems:
1. Each model architecture has a dedicated [conversion_mapping](https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py) which defines how the model should be loaded and saved. For MoEs, this typically means fusing 2D weights into 3D weights.
2. Each model architecture uses the [`use_experts_implementation`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/moe.py#L526) decorator to designate experts modules. These experts modules are highly standardized, which allows LLM Compressor to standardize its conversions. For adding support for architectures which do not use the `use_experts_implementation` decorator, see [Adding Custom MoE Definitions](#adding-custom-moe-definitions).

## MoE Conversion Before Loading

Many checkpoints already store weights as unfused, 2D linear weights. Transformers typically uses the [conversion_mapping](https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py) to fuse them into 3D weights on load. However, LLM Compressor allows you to skip this step by adding explicit 2D mappings. Examples of 2D conversion mappings can be found [here](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/moe/conversion_mappings.py).

When contributing a new mapping, be sure to add your model architecture to `test_linearize.py`.

Adding a conversion mapping is the most efficient way to load your model. For models which do not have conversion mappings, they will fallback to performing conversion after loading.

## MoE Conversion After Loading

The vast majority of models can be converted after loading. Conversion after loading is often slow, since it may require converting from 2D -> 3D and then back to 2D. However, it is guaranteed to work for nearly all MoE model definitions.

For implementation details, see [LinearExperts2D](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/moe/linear_experts.py).

## Adding Custom MoE Definitions

For models which do not use the standard `use_experts_implementation` decorator, you may need to add a custom model definition. This is not required for the vast majority of models. **Do not add a new model definition if your model definition already uses the `use_experts_implementation` decorator**.

1. Define a linearized model definition. This definition should specify an `__init__` method for initializing parameters, a `from_experts_module` method for specifying how to convert from the original module to the new definition, and a `forward` method which uses `get_calibrate_all_experts_flag` to calibrate all experts.

```python
class GraniteMoeLinearExperts(LinearExperts2D):
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = False

    @classmethod
    @torch.no_grad()
    def from_experts_module(
        cls, experts: "GraniteMoeParallelExperts", config: GraniteMoeConfig
    ):
        assert experts.num_experts == config.num_local_experts

        with skip_weights_initialize():
            self = cls(
                experts.num_experts, experts.input_size, experts.output_size, config
            )
            self.num_experts = experts.num_experts

        for i in range(experts.num_experts):
            self[i].weight.copy_(experts.weight[i])

        # copy offloading from original
        offload_kwargs = get_cache_init_kwargs(experts)
        for module in self.modules():
            offload_module(module, **offload_kwargs)

        return self

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        config: GraniteMoeConfig,
    ) -> None:
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

        torch.nn.ModuleList.__init__(
            self,
            [
                torch.nn.Linear(input_size, output_size, bias=False, dtype=config.dtype)
                for _ in range(num_experts)
            ],
        )

    def forward(self, inputs: torch.Tensor, expert_size: list[int]):
        output_list = []

        for i in range(self.num_experts):
            if get_calibrate_all_experts_flag():
                expert_out = self[i](inputs).split(expert_size, dim=0)[i]
            else:
                expert_out = self[i](inputs.split(expert_size, dim=0)[i])
            output_list.append(expert_out)

        return torch.cat(output_list, dim=0)
```

2. Make sure the definition is registered

```python
# register in registry
LinearExperts2D._registry[GraniteMoeParallelExperts] = GraniteMoeLinearExperts
```

```python
class LinearExperts2D(torch.nn.ModuleList):
    # custom model definitions
    _registry: ClassVar[dict[type[torch.nn.Module], type["LinearExperts2D"]]] = dict()

    @classmethod
    def get_registration(
        cls, key: type[torch.nn.Module], default: Any = None
    ) -> type["LinearExperts2D"]:
        from .granitemoe import GraniteMoeLinearExperts  # noqa: F401
        from .llama4 import Llama4LinearExperts  # noqa: F401
        # Add your import here

        return cls._registry.get(key, default)
```

3. Add a test to `test_linearize.py` to ensure that outputs are similar before and after linearization.

```python
def test_linearize_moe_granite():
    config = GraniteMoeConfig(hidden_size=512, intermediate_size=1024)
    experts = GraniteMoeParallelExperts(
        config.num_local_experts, config.hidden_size, config.intermediate_size
    )
    init.normal_(experts.weight, mean=0.0, std=config.initializer_range)

    mock_model = DummyModel(experts, config)
    linearize_moe(mock_model)
    assert mock_model.module is not experts

    hidden_states = torch.randn(NUM_TEST_TOKENS, config.hidden_size, dtype=config.dtype)
    expert_size = [
        (NUM_TEST_TOKENS // config.num_local_experts)
        for _ in range(config.num_local_experts)
    ]
    expert_size[-1] += NUM_TEST_TOKENS % config.num_local_experts
    true_outputs = experts(hidden_states, expert_size)
    outputs = mock_model(hidden_states, expert_size)
    with moe_calibration_context():
        calib_outputs = mock_model(hidden_states, expert_size)

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
    assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE
```