from enum import Enum
from typing import Iterable, List, Literal, Optional

from compressed_tensors import match_modules_set, match_named_modules, align_module_device
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from pydantic import Field, ValidationInfo, field_validator
from transformers import PreTrainedModel

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling import fuse_norm_linears, normalize_embedding
from llmcompressor.modifiers import Modifier

from .mappings import SpinQuantMapping, infer_mapping_from_model
from .norm_mappings import NormMapping, infer_norm_mapping_from_model


class SpinquantRotation(str, Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


class SpinQuantModifier(Modifier, use_enum_values=True):
    """
    Implements the transforms according to
    [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)  # noqa: E501

    Transforms (rotations) are extra layers added to a model which reduce the accuracy
    loss induced by quantization. This is achived through "rotating" weights and
    activations into a space with a smaller dynamic range of values, thus decreasing
    the range of scales required for quantization.

    The SpinQuant authors describe four different rotations which can be applied to a
    model. R1 and R2 are "offline" rotations, meaning that they can be fused into
    existing weights and therefore do not induce runtime cost. R3 and R4 are "online"
    rotations, meaning that they require additional computation at runtime.

    :param rotations: A list containing the names of rotations to apply to the model.
        Possible rotations include R1, R2, R3, and R4
    :param transform_type: The type of transform to apply to the model.
        `"hadamard"` has the least performance cost but only supports sizes which are
        powers of power of two.
        `"random-matrix"` has more performance cost, but supports a much larger set of
            sizes.
        `"random-matrix"` has the greatest performance cost, but supports any size
    :param randomize: if True, create distinct transforms for each application
    :param learnable: if True, attach gradients to transform weights for training
    :param mappings: Specifies layers within a model to target for transforms.
        A mapping will be inferred if None is provided
    :param norm_mappings: Specifies layers within a model to target for norm fusing.
        A mapping will be inferred if None is provided
    :param transform_config: Optional transform config for overriding provided arguments
    """

    rotations: List[SpinquantRotation] = Field(
        default_factory=lambda: ["R1", "R2"], exclude=True
    )
    transform_type: Literal["hadamard", "random-hadamard", "random-matrix"] = Field(
        default="hadamard", exclude=True
    )
    randomize: bool = Field(default=False, exclude=True)
    learnable: bool = Field(default=False, exclude=True)

    # norm mappings separate from spinquant mappings to allow users to
    # override spinquant mappings with transform_config without overriding norms
    mappings: Optional[SpinQuantMapping] = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    norm_mappings: Optional[List[NormMapping]] = Field(
        default=None,
        repr=False,
        exclude=True,
    )

    # optional override for more fine-grained control
    # also included in recipe serialization
    transform_config: Optional[TransformConfig] = Field(default=None, repr=False)

    @field_validator("randomize", "learnable", mode="before")
    def validate_not_implemented(cls, value, info: ValidationInfo):
        raise NotImplementedError(f"{info.field_name} is not supported right now")

    @field_validator("rotations", mode="before")
    def validate_rotations(cls, value):
        if isinstance(value, Iterable):
            return tuple(v.upper() for v in value)
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.transform_config is not None:
            return True

        self.mappings = infer_mapping_from_model(state.model)
        self.norm_mappings = infer_norm_mapping_from_model(state.model)

        config_groups = {}
        if SpinquantRotation.R1 in self.rotations:
            config_groups["R1"] = self._create_r1_scheme()

        if SpinquantRotation.R2 in self.rotations:
            config_groups["R2"] = self._create_r2_scheme(state.model)

        if SpinquantRotation.R3 in self.rotations:
            config_groups["R3"] = self._create_r3_scheme()

        if SpinquantRotation.R4 in self.rotations:
            config_groups["R4"] = self._create_r4_scheme()

        self.transform_config = TransformConfig(config_groups=config_groups)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # output_full = state.model(**state.model.dummy_inputs)

        # needs to happen after the model has been hooked to execute on the GPU
        # otherwise we're applying weight transforms on CPU
        self._prenormalize_embeddings(state.model)
        self._fuse_norms(state.model)

        # output_fuse = state.model(**state.model.dummy_inputs)

        apply_transform_config(state.model, self.transform_config)

        # output_transform = state.model(**state.model.dummy_inputs)

        from compressed_tensors.quantization import apply_quantization_config, QuantizationConfig, QuantizationScheme
        from compressed_tensors.quantization.quant_scheme import W4A16
        q_config = QuantizationConfig(config_groups={"": QuantizationScheme(targets=["Linear"], weights=W4A16["weights"])}, ignore=["lm_head"])
        apply_quantization_config(state.model, q_config)
        mock_calibrate_forward(state.model)

        # output_quant = state.model(**state.model.dummy_inputs)

        # loss = torch.nn.MSELoss()
        # fuse_loss = loss(output_fuse.logits, output_full.logits)
        # transform_loss = loss(output_transform.logits, output_full.logits)
        # quant_loss = loss(output_quant.logits, output_full.logits)

        # assert fuse_loss < transform_loss < quant_loss

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        elif event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            pass

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True

    def on_finalize(self, state: State, **kwargs) -> bool:
        if not self.ended_:
            self.on_end(state, None)

        return True

    def _prenormalize_embeddings(self, model: PreTrainedModel):
        for _, embedding in match_named_modules(
            model, [self.mappings.embedding], warn_on_fail=True
        ):
            normalize_embedding(embedding)

    def _fuse_norms(self, model: PreTrainedModel):
        for mapping in self.norm_mappings:
            for norm, *linears in match_modules_set(
                model, (mapping.norm, *mapping.linears)
            ):
                fuse_norm_linears(norm, linears)

    def _create_r1_scheme(self) -> TransformScheme:
        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            apply=[
                TransformArgs(
                    targets=[
                        self.mappings.embedding,
                        self.mappings.attn_o,
                        *self.mappings.mlp_out,
                    ],
                    location="weight_output",
                ),
                TransformArgs(
                    targets=[
                        self.mappings.attn_q,
                        self.mappings.attn_k,
                        self.mappings.attn_v,
                        *self.mappings.mlp_in,
                        self.mappings.lm_head,
                    ],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r2_scheme(self, model: PreTrainedModel) -> TransformScheme:
        config = model.config

        if hasattr(config, "head_dim"):
            head_dim = config.head_dim
        elif hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            head_dim = config.hidden_size // config.num_attention_heads
        else:
            raise NotImplementedError()

        return TransformScheme(
            type=self.transform_type,
            randomize=self.randomize,
            requires_grad=self.learnable,
            head_dim=head_dim,
            apply=[
                TransformArgs(targets=[self.mappings.attn_v], location="weight_output"),
                TransformArgs(
                    targets=[self.mappings.attn_o],
                    location="weight_input",
                    inverse=True,
                ),
            ],
        )

    def _create_r3_scheme(self) -> TransformScheme:
        raise NotImplementedError()

    def _create_r4_scheme(self) -> TransformScheme:
        raise NotImplementedError()


import torch
from compressed_tensors import update_offload_parameter
from compressed_tensors.quantization.utils import calculate_qparams


def mock_calibrate_forward(model: torch.nn.Module):
    for name, module in model.named_modules():
        if (scheme := getattr(module, "quantization_scheme", None)):
            if scheme.weights.strategy == "group":
                group_size = scheme.weights.group_size
                num_groups = module.weight.size(-1) // group_size
                values = module.weight.unflatten(-1, (num_groups, group_size))

            elif scheme.weights.strategy == "channel":
                values = module.weight.unflatten(-1, (1, module.weight.size(-1)))

            max_values = values.max(dim=-1).values
            min_values = values.min(dim=-1).values
            scale, zero_point = calculate_qparams(min_values, max_values, scheme.weights)

            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)

            # mock_group_fake_quantize(module)


def mock_group_fake_quantize(module: torch.nn.Module):
    with align_module_device(module):
        scale = module.weight_scale
        zero_point = module.weight_zero_point
        original_dtype = module.weight.dtype

        # quantize
        x = module.weight
        x_q = (x.unflatten(-1, (scale.size(-1), -1)) / scale[:, :, None]) + zero_point[:, :, None]
        x_q = torch.round(x_q)
        x_q = torch.clamp(x_q, -8, 7)  # unlike current impl, round then clamp

        # dequantize
        x_qdq = (x_q - zero_point[:, :, None]) * scale[:, :, None]
        x_qdq = x_qdq.flatten(-2, -1)

        #print(f"quant_loss: {torch.nn.MSELoss()(x_qdq, module.weight.data)}")
        update_offload_parameter(module, "weight", x_qdq.to(original_dtype))