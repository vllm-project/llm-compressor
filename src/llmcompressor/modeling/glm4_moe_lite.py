from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmcompressor.utils.dev import skip_weights_initialize

from .glm_moe_dsa import CalibrationGlmMoeDsaMoE
from .moe_context import MoECalibrationModule

if TYPE_CHECKING:
    from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
        Glm4MoeLiteConfig,
    )
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteNaiveMoe,
    )


@MoECalibrationModule.register("Glm4MoeLiteMoE")
class CalibrationGlm4MoeLiteMoE(CalibrationGlmMoeDsaMoE):
    """
    Calibration version of Glm4MoeLiteMoE that unfuses 3D expert parameters into
    individual MLP modules (nn.Linear) so they can be quantized.

    GLM-4.7-Flash Lite stores routed experts in a ``Glm4MoeLiteNaiveMoe`` module
    using 3D parameters (``gate_up_proj``, ``down_proj``) instead of ``nn.Linear``
    submodules.  Since llm-compressor targets ``Linear`` modules, the original
    routed experts are invisible to quantization and remain BF16 unless they are
    unpacked.

    Inherits routing logic (:meth:`route_tokens_to_experts`) and forward pass
    from :class:`CalibrationGlmMoeDsaMoE`, overriding only expert creation to
    use ``Glm4MoeLiteMLP`` modules.
    """

    def _get_num_experts(self, config) -> int:
        return config.n_routed_experts

    def _make_experts(self, config, original_experts) -> torch.nn.ModuleList:
        return SequentialGlm4MoeLiteExperts(config, original_experts)


class SequentialGlm4MoeLiteExperts(torch.nn.ModuleList):
    """
    Unpacks 3D expert parameter tensors into individual Glm4MoeLiteMLP modules so
    each routed expert has standard ``nn.Linear`` projections visible to
    ``targets="Linear"``.
    """

    def __init__(self, config: Glm4MoeLiteConfig, original: Glm4MoeLiteNaiveMoe):
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteMLP,
        )

        self.num_experts = config.n_routed_experts
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    Glm4MoeLiteMLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]
            gate_proj, up_proj = gate_up.chunk(2, dim=0)

            self[i].gate_proj.weight.data = gate_proj.clone().contiguous()
            self[i].up_proj.weight.data = up_proj.clone().contiguous()
            self[i].down_proj.weight.data = down.clone().contiguous()
