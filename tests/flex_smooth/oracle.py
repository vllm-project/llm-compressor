# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Execute external original definitions; no reference algorithm is vendored."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tests.quarot.source_loader import source_definitions


class FlexOracle:
    def __init__(self, root):
        namespace = {
            "torch": torch,
            "np": np,
            "ABC": ABC,
            "abstractmethod": abstractmethod,
            "Tuple": Tuple,
            "List": List,
            "Optional": Optional,
            "Union": Union,
            "Any": Any,
            "Dict": Dict,
            "nn": torch.nn,
            "dataclass": dataclass,
            "get_logger": lambda: logging.getLogger("oracle"),
        }
        self.hashes = {}

        def load(relative, names):
            module, digest = source_definitions(root / relative, names, namespace)
            self.hashes[relative] = digest
            return module

        self.search = load(
            "msmodelslim/processor/anti_outlier/flex_smooth/alpha_beta_search.py",
            [
                "quant_int8sym",
                "quant_int8asym",
                "BaseAlphaBetaSearcher",
                "FlexSmoothAlphaBetaSearcher",
            ],
        )
        self.scales = load(
            "msmodelslim/processor/anti_outlier/common/scale_computation.py",
            [
                "MQGAScaleParams",
                "compute_weight_scale",
                "compute_multi_weight_scale",
                "prepare_mqga_parameters",
                "reduce_scales_for_mqga_max",
                "reduce_scales_for_mqga_mean",
                "BaseScaleCalculator",
                "FlexSmoothScaleCalculator",
            ],
        )
        self.subgraphs = load(
            "msmodelslim/processor/anti_outlier/common/subgraph_type.py",
            ["Subgraph", "NormLinearSubgraph", "OVSubgraph"],
        )
        for name in ("Subgraph", "NormLinearSubgraph", "OVSubgraph"):
            namespace[name] = getattr(self.subgraphs, name)
        # Context recording is metadata-only; numerical fusers are unchanged.
        namespace["get_current_context"] = lambda: None
        self.fusion = load(
            "msmodelslim/processor/anti_outlier/common/subgraph_fusion.py",
            [
                "apply_smooth_scale_shift",
                "SubgraphFusionStrategy",
                "NormLinearSubgraphFusion",
                "OVSubgraphFusion",
            ],
        )
        adapter_types = load(
            "msmodelslim/core/graph/adapter_types.py",
            [
                "SUPPORTED_SUBGRAPH_TYPES",
                "MappingConfig",
                "FusionConfig",
                "AdapterConfig",
            ],
        )
        namespace.update(
            {
                name: getattr(adapter_types, name)
                for name in ("MappingConfig", "FusionConfig", "AdapterConfig")
            }
        )
        # Explicit single-rank expert range; distributed scheduling is not emulated.
        namespace["_get_expert_range"] = lambda config: (0, config.n_routed_experts)
        relative = "msmodelslim/model/glm_5_2/model_adapter.py"
        adapter, digest = source_definitions(
            root / relative,
            ["get_adapter_config_for_subgraph"],
            namespace,
            class_name="GLM52ModelAdapter",
        )
        self.hashes[relative] = digest
        self.mapping = adapter.get_adapter_config_for_subgraph
