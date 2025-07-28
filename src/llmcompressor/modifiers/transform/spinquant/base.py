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


import torch

from llmcompressor.modeling import normalize_embedding, fuse_norm_linears

from compressed_tensors.utils import is_match
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from compressed_tensors.utils import align_module_device, update_offload_parameter


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

        # # needs to happen after the model has been hooked to execute on the GPU
        # # otherwise we're applying weight transforms on CPU
        # self._prenormalize_embeddings(state.model)
        # self._fuse_norms(state.model)

        # apply_transform_config(state.model, self.transform_config)

        model = state.model

        normalize_embedding(model.model.embed_tokens)
        for layer in model.model.layers:
            fuse_norm_linears(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            )
            fuse_norm_linears(
                layer.post_attention_layernorm,
                [layer.mlp.up_proj, layer.mlp.gate_proj],
            )
        fuse_norm_linears(
            model.model.norm,
            [model.lm_head],
        )
        print("normalized embeddings and fused norms")

        transform_and_quant(model)
        print("transformed and quanted")

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


def transform(weight: torch.Tensor, loc: str):
    if loc == "embed_output":
        hadamard = deterministic_hadamard_matrix(weight.size(1), weight.dtype, weight.device)
        return (weight @ hadamard) / torch.tensor(hadamard.size(0)).sqrt()

    if loc == "weight_output":
        hadamard = deterministic_hadamard_matrix(weight.size(0), weight.dtype, weight.device)
        return (hadamard.T @ weight) / torch.tensor(hadamard.size(0)).sqrt()
    
    if loc == "weight_input":
        hadamard = deterministic_hadamard_matrix(weight.size(1), weight.dtype, weight.device)
        inv = hadamard.T
        return (weight @ inv.T) / torch.tensor(hadamard.size(0)).sqrt()

    assert False


def calibrate_fake_quantize(weight: torch.Tensor) -> torch.Tensor:
    # calibrate
    group_size = 128
    num_groups = weight.size(-1) // group_size
    values = weight.unflatten(-1, (num_groups, group_size))

    max_values = values.max(dim=-1).values
    min_values = values.min(dim=-1).values

    value_range = torch.maximum(max_values.abs(), min_values.abs()) * 2
    scale = value_range / (7 + 8)
    scale = scale.clamp(min=torch.finfo(torch.float32).eps)
    zero_point = torch.zeros_like(scale)

    # quantize
    x = weight
    x_q = (x.unflatten(-1, (scale.size(-1), -1)) / scale[:, :, None]) + zero_point[:, :, None]
    x_q = torch.round(x_q)
    x_q = torch.clamp(x_q, -8, 7)  # unlike current impl, round then clamp

    # dequantize
    x_qdq = (x_q - zero_point[:, :, None]) * scale[:, :, None]
    x_qdq = x_qdq.flatten(-2, -1)
    return x_qdq


def transform_and_quant(model: torch.nn.Module, do_transform=True):
    for name, module in model.named_modules():
        if is_match(name, module, "re:.*embed_tokens$"):
            with align_module_device(module):
                transformed = transform(module.weight, "embed_output")

        elif any(is_match(name, module, t) for t in ["re:.*o_proj$", "re:.*down_proj$"]):
            with align_module_device(module):
                transformed = transform(module.weight, "weight_output")
            
        elif any(is_match(name, module, t) for t in ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$", "re:.*up_proj$", "re:.*gate_proj$", "lm_head"]):
            with align_module_device(module):
                transformed = transform(module.weight, "weight_input")

        else:
            continue

        with align_module_device(module):
            quant = calibrate_fake_quantize(module.weight)
            transformed_quant = calibrate_fake_quantize(transformed)

            loss = torch.nn.MSELoss()
            with torch.no_grad():
                quant_loss = loss(quant, module.weight)
                transform_quant_loss = loss(transformed_quant, transformed)

            if not transform_quant_loss < quant_loss < 1e-05:
                print((name.rjust(32), transform_quant_loss, quant_loss))

            if "embed_tokens" or "lm_head" in name:
                if do_transform:
                    update_offload_parameter(module, "weight", transformed)
            else:
                if do_transform:
                    update_offload_parameter(module, "weight", transformed_quant)

                else:
                    update_offload_parameter(module, "weight", quant)