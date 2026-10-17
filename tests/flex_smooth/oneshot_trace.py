# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Observe real oneshot callbacks without replacing their implementation."""

from importlib import import_module

import torch
from torch.utils._pytree import tree_map

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import FlexSmoothModifier, QuaRotModifier
from llmcompressor.modifiers.transform.flex_smooth.mappings import glm_mappings
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context


def inspect_fine_partitions(monkeypatch):
    """Check the actual graph, including experts hidden by starred-call wrapping."""
    pipeline = import_module("llmcompressor.pipelines.sequential.pipeline")
    original = pipeline.trace_subgraphs
    report = {}

    def trace(model, sample_input, *args, **kwargs):
        subgraphs = original(model, sample_input, *args, **kwargs)
        names = {module: name for name, module in model.named_modules()}
        members = [set(subgraph.submodules(model)) for subgraph in subgraphs]
        for item in glm_mappings(model, ("norm-linear", "ov")):
            participants = {model.get_submodule(name) for name in item.targets}
            assert sum(participants <= group for group in members) == 1
            assert sum(bool(participants & group) for group in members) == 1
        # Two attention groups, dense/shared MLPs, two routed experts and a head.
        assert len(subgraphs) == 7
        expert_partitions = []
        for index in range(2):
            expert = model.get_submodule(f"model.layers.1.mlp.experts.{index}")
            positions = [i for i, group in enumerate(members) if expert in group]
            assert len(positions) == 1
            expert_partitions.extend(positions)
        assert len(set(expert_partitions)) == 2
        for i in expert_partitions:
            assert not any(".self_attn" in names[module] for module in members[i])

        # Execute real traced subgraphs, not just the transformed full model.
        with calibration_forward_context(model), DisableQuantization(model):
            # The calibration context disables lm_head. Compare its real inputs.
            hidden = []
            handle = model.model.norm.register_forward_hook(
                lambda module, args, output: hidden.append(output.detach().clone())
            )
            try:
                model(**sample_input)
                namespace = dict(sample_input)
                for subgraph in subgraphs:
                    # Model an accelerator roundtrip: inputs must not alias the
                    # CPU cache. A CPU-only replay otherwise hides in-place bugs.
                    inputs = tree_map(
                        lambda value: value.clone()
                        if isinstance(value, torch.Tensor)
                        else value,
                        {key: namespace[key] for key in subgraph.input_names},
                    )
                    output = subgraph.forward(model, **inputs)
                    namespace.update(output)
                    for key in subgraph.consumed_names:
                        namespace.pop(key, None)
            finally:
                handle.remove()
            assert len(hidden) == 2
            torch.testing.assert_close(hidden[1], hidden[0], atol=0, rtol=0)
        report.update(
            subgraph_count=len(subgraphs),
            modules=[sorted(names[module] for module in group) for group in members],
            expert_partitions=expert_partitions,
            replay_matches_full_forward=True,
        )
        return subgraphs

    monkeypatch.setattr(pipeline, "trace_subgraphs", trace)
    return report


class OneshotTrace:
    def __init__(self, monkeypatch):
        self.events = []
        self.transformed = {}
        self.observed = set()
        self.model = None
        self.flex = None
        self.captures = 0

        initialize = QuantizationModifier.on_initialize
        rotate = QuaRotModifier.on_calibration_start
        start_flex = FlexSmoothModifier.on_calibration_start
        start_quant = QuantizationModifier.on_calibration_start
        capture_factory = FlexSmoothModifier._capture
        smooth = FlexSmoothModifier._smooth
        quant = import_module("llmcompressor.modifiers.quantization.quantization.base")
        observe = quant.observe

        def initialize_quant(modifier, state, **kwargs):
            result = initialize(modifier, state, **kwargs)
            self.model = state.model
            self.events.append("quant.initialize")
            assert any(
                getattr(module, "quantization_scheme", None) is not None
                for module in state.model.modules()
            )
            return result

        def rotate_weights(modifier, state, event, **kwargs):
            result = rotate(modifier, state, event, **kwargs)
            assert state.model.config.quarot_config["status"] == "applied"
            self.events.append("quarot.applied")
            return result

        def register_flex(modifier, state, event, **kwargs):
            assert state.model.config.quarot_config["status"] == "applied"
            self.flex = modifier
            result = start_flex(modifier, state, event, **kwargs)
            self.events.append("flex.start")
            return result

        def register_quant(modifier, state, event, **kwargs):
            result = start_quant(modifier, state, event, **kwargs)
            self.events.append("quant.start")
            return result

        def capture(modifier, key, model):
            original = capture_factory(modifier, key, model)

            def hook(module, args, kwargs):
                if modifier._collecting and key not in modifier._processed:
                    assert model.config.quarot_config["status"] == "applied"
                    assert not getattr(module, "quantization_enabled", False)
                    self.captures += 1
                return original(module, args, kwargs)

            return hook

        def smooth_weights(modifier, model, item):
            result = smooth(modifier, model, item)
            for name in item.targets:
                self.transformed[name] = (
                    model.get_submodule(name).weight.detach().clone()
                )
            self.events.append("flex.applied:" + item.source)
            return result

        def observe_weights(modules, base_name):
            modules = list(modules)
            if base_name == "weight":
                names = {module: name for name, module in self.model.named_modules()}
                assert self.model.config.quarot_config["status"] == "applied"
                for module in modules:
                    name = names[module]
                    for item in self.flex._plan:
                        if name in item.targets:
                            # This assertion fails if Quantization runs before Flex
                            # at a sequential boundary, even if final weights match.
                            assert item.source in self.flex._processed, name
                            assert name in self.transformed, name
                            torch.testing.assert_close(
                                module.weight,
                                self.transformed[name],
                                atol=0,
                                rtol=0,
                            )
                            self.observed.add(name)
                    self.events.append("weight.observe:" + name)
            return observe(modules, base_name)

        monkeypatch.setattr(QuantizationModifier, "on_initialize", initialize_quant)
        monkeypatch.setattr(QuaRotModifier, "on_calibration_start", rotate_weights)
        monkeypatch.setattr(FlexSmoothModifier, "on_calibration_start", register_flex)
        monkeypatch.setattr(
            QuantizationModifier, "on_calibration_start", register_quant
        )
        monkeypatch.setattr(FlexSmoothModifier, "_capture", capture)
        monkeypatch.setattr(FlexSmoothModifier, "_smooth", smooth_weights)
        monkeypatch.setattr(quant, "observe", observe_weights)

    def verify(self, pipeline):
        assert self.events.index("quant.initialize") < self.events.index(
            "quarot.applied"
        )
        assert self.events.index("quarot.applied") < self.events.index("flex.start")
        assert self.captures > 0
        assert len(self.observed) >= 6
        smooth = [
            i
            for i, event in enumerate(self.events)
            if event.startswith("flex.applied:")
        ]
        observe = [
            i
            for i, event in enumerate(self.events)
            if event.startswith("weight.observe:")
        ]
        assert len(smooth) == 6
        if pipeline == "default":
            # Independent runs QuaRot datafree, Flex sequential, MXFP datafree.
            assert max(smooth) < self.events.index("quant.start") < min(observe)
        else:
            assert self.events.index("quant.start") < min(smooth)
        return {
            "events": self.events,
            "floating_calibration_hook_calls": self.captures,
            "post_flex_observed_modules": sorted(self.observed),
            "observer_saw_post_flex_weights": True,
        }
