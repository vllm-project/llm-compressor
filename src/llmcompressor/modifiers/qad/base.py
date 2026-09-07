import math
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import torch
from compressed_tensors.quantization import QuantizationStatus, enable_quantization
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import getattr_chain, patch_attr, update_offload_parameter
from loguru import logger
from pydantic import Field, PrivateAttr
from torch.fx import Graph, GraphModule

from llmcompressor.core import State, active_session
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.sequential.helpers import Subgraph

__all__ = ["QADModifier"]


@dataclass
class _SubgraphBatch:
    inputs: dict[str, Any]
    target: Any
    loss_mask: torch.Tensor | None


def _map_tensors(value: Any, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    return value


def _masked_mse(prediction, target, loss_mask):
    if prediction.shape != target.shape:
        raise ValueError("Student and teacher output shapes differ")
    error = (prediction.float() - target.float()).square()
    per_token_loss = error.mean(dim=-1) if error.ndim else error
    if loss_mask is None:
        return per_token_loss.mean()
    mask = loss_mask.to(device=prediction.device, dtype=per_token_loss.dtype)
    while mask.ndim < per_token_loss.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = torch.broadcast_to(mask, per_token_loss.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Loss mask shape {tuple(loss_mask.shape)} cannot broadcast to "
            f"output loss shape {tuple(per_token_loss.shape)}"
        ) from error
    if mask.sum().item() == 0:
        raise ValueError("Loss mask does not contain any valid tokens")
    return (per_token_loss * mask).sum() / mask.sum()


def _output_loss(prediction, target, loss_mask):
    """Average MSE over floating output leaves; ignore structural metadata."""
    losses = []

    def visit(pred, ref):
        if isinstance(ref, torch.Tensor):
            if ref.is_floating_point() and ref.numel():
                if not isinstance(pred, torch.Tensor):
                    raise ValueError("Student and teacher output structures differ")
                losses.append(_masked_mse(pred, ref.to(pred.device), loss_mask))
        elif isinstance(ref, dict):
            if not isinstance(pred, dict) or pred.keys() != ref.keys():
                raise ValueError("Student and teacher output structures differ")
            for key in ref:
                visit(pred[key], ref[key])
        elif isinstance(ref, (tuple, list)):
            if not isinstance(pred, (tuple, list)) or len(pred) != len(ref):
                raise ValueError("Student and teacher output structures differ")
            for p, r in zip(pred, ref):
                visit(p, r)

    visit(prediction, target)
    if not losses:
        raise ValueError("QAD requires a floating output affected by quantized weights")
    return torch.stack(losses).mean()


def _quantized_modules(modules):
    return [
        module
        for module in dict.fromkeys(modules)
        if getattr_chain(module, "quantization_scheme.weights", None) is not None
    ]


def _distillation_graph(model, subgraph, quantized_modules, include_propagation=False):
    """Keep the traced computation and select outputs dependent on quantized weights.

    Calibration replaces the LM head with a meta forward. For an ignored head,
    reconstruct its hidden-state input instead of allocating vocabulary logits.
    All other operations, branches and residual paths retain their traced order.
    """
    head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    quantized = set(quantized_modules)
    if head in quantized:
        raise ValueError(
            "QAD requires the output LM head to be ignored by quantization"
        )
    graph = Graph()
    mapping = {}
    affected = set()
    weights = {id(module.weight) for module in quantized}
    for node in subgraph.graph.nodes:
        if node.op == "call_module" and model.get_submodule(node.target) is head:
            mapping[node] = mapping[node.args[0]]
            continue
        if node.op == "output":

            def select(n):
                mapped = mapping[n]
                return mapped if mapped in affected else None

            output = graph.output(torch.fx.map_arg(node.args[0], select))
            if not output.all_input_nodes:
                # The terminal pipeline subgraph exports {} because it has no
                # consumers. Retain its terminal weight-dependent values for loss.
                graph.erase_node(output)
                output = graph.output(
                    {n.name: n for n in graph.nodes if n in affected and not n.users}
                )
            if include_propagation:
                # Return both the loss targets and the complete boundary values
                # in a single teacher forward. Independent branches may be needed
                # by a later subgraph even though they do not enter this loss.
                output.args = (
                    {
                        "target": output.args[0],
                        "propagation": torch.fx.map_arg(
                            node.args[0], lambda n: mapping[n]
                        ),
                    },
                )
            continue
        copied = graph.node_copy(node, lambda n: mapping[n])
        mapping[node] = copied
        direct = (
            node.op == "call_module"
            and any(m in quantized for m in model.get_submodule(node.target).modules())
        ) or (
            node.op == "get_attr" and id(getattr_chain(model, node.target)) in weights
        )
        if direct or any(n in affected for n in copied.all_input_nodes):
            affected.add(copied)
    result = GraphModule(model, graph, "QADSubgraph")
    result.graph.eliminate_dead_code()
    result.recompile()
    return result


class QADModifier(Modifier):
    """Jointly distill quantized weights in each traced sequential subgraph.

    Place after a weight quantization modifier, e.g. QuantizationModifier (RTN)
    or GPTQModifier. Teacher outputs are captured before the current subgraph is
    quantized. ``teacher_mode="local"`` uses the student's inputs, including
    upstream quantization error. ``teacher_mode="full"`` maintains an independent
    original-model activation stream from the model input. Neither mode loads a
    second teacher model; full mode runs each stage before its weights change.
    Qparams remain fixed throughout QAD and final weight materialization.

    ``num_epochs`` and early stopping apply separately to each subgraph. Cached
    batches are split into training/validation sets. By default these are the PTQ
    calibration batches. Pass ``qad_dataset`` and ``qad_dataset_args`` to oneshot
    to configure an independent stream, including its microbatch size. Gradient
    accumulation combines microbatches.
    """

    requires_calibration_data: bool = True
    teacher_mode: Literal["local", "full"] = "local"
    num_epochs: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2.0e-6, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float | None = Field(default=1.0, gt=0)
    seed: int = 42
    target_offload_device: str = "cpu"
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    early_stopping_patience: int = Field(default=3, ge=1)
    validation_relative_min_delta: float = Field(default=1.0e-3, ge=0)

    _graph: GraphModule | None = PrivateAttr(default=None)
    _teacher_activations: IntermediatesCache | None = PrivateAttr(default=None)
    _next_teacher_subgraph: int = PrivateAttr(default=0)
    _num_subgraphs: int = PrivateAttr(default=0)
    _batches: list[_SubgraphBatch] = PrivateAttr(default_factory=list)
    _name: str = PrivateAttr(default="")
    _device: torch.device = PrivateAttr(default_factory=lambda: torch.device("cpu"))
    _optimizer_steps: dict[str, int] = PrivateAttr(default_factory=dict)
    _best_validation_losses: dict[str, float] = PrivateAttr(default_factory=dict)
    _epochs_completed: dict[str, int] = PrivateAttr(default_factory=dict)
    _validation_histories: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    @property
    def optimizer_steps(self):
        return dict(self._optimizer_steps)

    @property
    def best_validation_losses(self):
        return dict(self._best_validation_losses)

    @property
    def epochs_completed(self):
        return dict(self._epochs_completed)

    @property
    def validation_histories(self):
        return {
            name: list(values) for name, values in self._validation_histories.items()
        }

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.teacher_mode == "full":
            # These methods may change future weights or hidden representations
            # before the teacher reaches them. Reusing those weights would no
            # longer produce the original model's intermediate outputs.
            from llmcompressor.modifiers.transform.awq import AWQModifier
            from llmcompressor.modifiers.transform.quip import QuIPModifier
            from llmcompressor.modifiers.transform.smoothquant import (
                SmoothQuantModifier,
            )
            from llmcompressor.modifiers.transform.spinquant import SpinQuantModifier

            modifiers = getattr(active_session().lifecycle.recipe, "modifiers", ())
            if any(
                isinstance(
                    m,
                    (AWQModifier, SmoothQuantModifier, QuIPModifier, SpinQuantModifier),
                )
                for m in modifiers
            ):
                raise ValueError(
                    "Full QAD teacher does not support smoothing or rotation "
                    "modifiers; "
                    "use RTN/GPTQ without these transforms, or teacher_mode='local'"
                )
        modules = _quantized_modules(state.model.modules())
        if not modules:
            raise ValueError(
                "Place QADModifier after a weight quantization modifier, such as "
                "QuantizationModifier (RTN) or GPTQModifier, in the same recipe"
            )
        if any(
            getattr(m, "quantization_status", None) == QuantizationStatus.COMPRESSED
            for m in modules
        ):
            raise ValueError("QAD needs the original model before weight compression")
        return True

    def on_calibration_start(
        self, state, event, subgraphs=None, dataset_args=None, **kwargs
    ):
        if subgraphs is None or dataset_args is None:
            raise ValueError('QADModifier requires pipeline="sequential"')
        if not dataset_args.propagate_error:
            raise ValueError("QADModifier requires propagate_error=True")
        self._teacher_activations = None
        self._next_teacher_subgraph = 0
        self._num_subgraphs = len(subgraphs)
        # A shared weight cannot be updated in an earlier stage without changing
        # the unquantized teacher for a later stage. Sharing within a stage is fine.
        owners = {}
        for index, subgraph in enumerate(subgraphs):
            for module in _quantized_modules(subgraph.submodules(state.model)):
                # Inspect parameter identity without onloading all model weights.
                previous = owners.setdefault(id(module._parameters["weight"]), index)
                if previous != index:
                    raise ValueError(
                        "QAD found quantized weights shared across subgraphs; "
                        "group their uses into the same sequential subgraph"
                    )

    @torch.no_grad()
    def on_sequential_epoch_start(
        self,
        state,
        event,
        modules,
        subgraph: Subgraph,
        activations: IntermediatesCache,
        subgraph_index: int,
        additional_activations: dict[str, IntermediatesCache] | None = None,
        **kwargs,
    ):
        self._clear_subgraph()
        quantized = _quantized_modules(modules)
        if not quantized and self.teacher_mode == "local":
            return
        separate_data = "qad" in (additional_activations or {})
        if separate_data:
            activations = additional_activations["qad"]
        self._split_batch_indices(
            len(activations)
        )  # validate before expensive forwards
        self._name = f"subgraph_{subgraph_index}"
        if quantized:
            self._graph = _distillation_graph(state.model, subgraph, quantized)
        teacher_graph = (
            _distillation_graph(
                state.model, subgraph, quantized, include_propagation=True
            )
            if self.teacher_mode == "full"
            else self._graph
        )
        input_names = (
            {n.target for n in self._graph.graph.nodes if n.op == "placeholder"}
            if self._graph is not None
            else set()
        )

        def cache(tensor):
            if tensor.is_meta:
                raise ValueError("QAD cannot cache a meta teacher output")
            return tensor.detach().to(self.target_offload_device, copy=True)

        # Preserve the pipeline's quantization flags. DisableQuantization restores
        # everything to enabled, which would corrupt the following GPTQ pass.
        with ExitStack() as stack:
            stack.enter_context(HooksMixin.disable_hooks())
            for module in teacher_graph.modules():
                if hasattr(module, "quantization_enabled"):
                    stack.enter_context(
                        patch_attr(module, "quantization_enabled", False)
                    )
            try:
                if self.teacher_mode == "full":
                    self._prepare_teacher_stream(activations, subgraph_index)
                for batch_index in range(len(activations)):
                    inputs = activations.fetch(batch_index, input_names)
                    cached_inputs = _map_tensors(inputs, cache)
                    if self.teacher_mode == "full":
                        teacher_inputs = self._teacher_activations.fetch(
                            batch_index, subgraph.input_names
                        )
                        result = teacher_graph(**teacher_inputs)
                        target = _map_tensors(result["target"], cache)
                        self._teacher_activations.update(
                            batch_index,
                            _map_tensors(
                                result["propagation"], lambda t: t.detach().clone()
                            ),
                        )
                        self._teacher_activations.delete(
                            batch_index, subgraph.consumed_names
                        )
                    else:
                        target = _map_tensors(teacher_graph(**inputs), cache)
                    if not quantized:
                        continue
                    if separate_data:
                        mask = activations.fetch(batch_index, ["loss_mask"]).get(
                            "loss_mask"
                        )
                    else:
                        mask = (
                            state.loss_masks[batch_index]
                            if state.loss_masks is not None
                            else None
                        )
                    self._batches.append(
                        _SubgraphBatch(
                            cached_inputs,
                            target,
                            cache(mask) if mask is not None else None,
                        )
                    )
                if self.teacher_mode == "full":
                    self._next_teacher_subgraph += 1
                    if self._next_teacher_subgraph == self._num_subgraphs:
                        self._teacher_activations = None
            except Exception:
                self._clear_subgraph()
                self._teacher_activations = None
                raise
        if quantized:
            logger.info(
                "QAD cached {} {} teacher batches for {}",
                len(self._batches),
                self.teacher_mode,
                self._name,
            )

    def _prepare_teacher_stream(self, activations, subgraph_index):
        if subgraph_index != self._next_teacher_subgraph:
            raise ValueError("Full QAD teacher requires all subgraphs in order from 0")
        if self._teacher_activations is None:
            if subgraph_index != 0:
                raise ValueError("Full QAD teacher activation stream is missing")
            self._teacher_activations = IntermediatesCache.empty(
                len(activations), torch.device(self.target_offload_device)
            )
            for index in range(len(activations)):
                self._teacher_activations.update(
                    index,
                    _map_tensors(
                        activations.fetch(index), lambda t: t.detach().clone()
                    ),
                )
        if len(self._teacher_activations) != len(activations):
            raise ValueError("QAD teacher/student batch counts changed between stages")

    def on_sequential_epoch_end(self, state, event, modules, **kwargs):
        if self._graph is None:
            return
        try:
            with HooksMixin.disable_hooks():
                self._optimize_subgraph(_quantized_modules(modules))
        except Exception:
            self._teacher_activations = None
            raise
        finally:
            self._clear_subgraph()

    def _optimize_subgraph(self, modules):
        trainable = []
        for module in modules:
            if (
                not isinstance(module.weight, torch.nn.Parameter)
                or not module.weight.is_floating_point()
            ):
                raise ValueError(
                    "QAD requires floating-point weights for fake quantization"
                )
            for key in ("weight_scale", "weight_global_scale"):
                value = getattr(module, key, None)
                if key == "weight_global_scale" and value is None:
                    continue
                if (
                    value is None
                    or value.is_meta
                    or not torch.isfinite(value).all()
                    or not (value > 0).all()
                ):
                    raise ValueError(
                        f"QAD requires initialized {key}; run the preceding "
                        "quantization modifier before QAD at the subgraph boundary"
                    )
            trainable.append(module.weight)
        trainable = list(dict.fromkeys(trainable))
        self._device = trainable[0].device
        # FP32 optimizer state/master weights avoid Adam's epsilon underflow and
        # sub-ULP updates when the model's execution weights are FP16/BF16.
        masters = [
            torch.nn.Parameter(p.detach().float().clone())
            if p.dtype in (torch.float16, torch.bfloat16)
            else p
            for p in trainable
        ]
        optimizer = torch.optim.AdamW(
            masters, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        flags = {p: p.requires_grad for p in self._graph.parameters()}
        self._graph.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
        for module in modules:
            enable_quantization(module)
        try:
            train_indices, validation_indices = self._split_batch_indices(
                len(self._batches)
            )
            initial_train = self._evaluate(train_indices)
            steps, epochs, best_loss = self._train_with_validation(
                optimizer, trainable, masters, train_indices, validation_indices
            )
            self._materialize_quantized_weights(modules)
            final_train = self._evaluate(train_indices)
            final_validation = self._evaluate(validation_indices)
            self._optimizer_steps[self._name] = steps
            self._epochs_completed[self._name] = epochs
            self._best_validation_losses[self._name] = best_loss
            logger.info(
                "QAD {}: {} updates over {} epochs; train MSE {:.6e} -> {:.6e}, "
                "final validation MSE {:.6e}",
                self._name,
                steps,
                epochs,
                initial_train,
                final_train,
                final_validation,
            )
        finally:
            for parameter, requires_grad in flags.items():
                parameter.grad = None
                parameter.requires_grad_(requires_grad)

    def _train_with_validation(
        self, optimizer, trainable, masters, train_indices, validation_indices
    ):
        generator = torch.Generator().manual_seed(self.seed)
        best_loss = self._evaluate(validation_indices)
        reference_loss = best_loss
        best_weights = self._snapshot_weights(trainable)
        history = [best_loss]
        patience = steps = epochs = 0
        for epoch in range(1, self.num_epochs + 1):
            order = torch.randperm(len(train_indices), generator=generator).tolist()
            steps += self._train_epoch(
                optimizer, trainable, masters, [train_indices[i] for i in order]
            )
            epochs = epoch
            loss = self._evaluate(validation_indices)
            history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_weights = self._snapshot_weights(trainable)
            improvement = (reference_loss - loss) / max(
                abs(reference_loss), torch.finfo(torch.float32).tiny
            )
            if improvement >= self.validation_relative_min_delta:
                reference_loss, patience = loss, 0
            else:
                patience += 1
            logger.info(
                "QAD {} epoch {} validation MSE {:.6e}, best {:.6e}, patience {}/{}",
                self._name,
                epoch,
                loss,
                best_loss,
                patience,
                self.early_stopping_patience,
            )
            if patience >= self.early_stopping_patience:
                break
        self._restore_weights(trainable, best_weights)
        self._validation_histories[self._name] = history
        return steps, epochs, best_loss

    def _train_epoch(self, optimizer, trainable, masters, indices):
        steps = 0
        with torch.enable_grad():
            for start in range(0, len(indices), self.gradient_accumulation_steps):
                group = indices[start : start + self.gradient_accumulation_steps]
                optimizer.zero_grad(set_to_none=True)
                for parameter in trainable:
                    parameter.grad = None
                for index in group:
                    (self._batch_loss(self._batches[index]) / len(group)).backward()
                for parameter, master in zip(trainable, masters):
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise ValueError(f"Nonfinite QAD gradient in {self._name}")
                        if master is not parameter:
                            master.grad = parameter.grad.float()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(masters, self.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    for parameter, master in zip(trainable, masters):
                        if parameter is not master:
                            parameter.copy_(master)
                steps += 1
        return steps

    def _split_batch_indices(self, batch_count):
        if batch_count < 2:
            raise ValueError("QAD validation requires at least two calibration batches")
        validation_count = min(
            batch_count - 1, max(1, math.ceil(batch_count * self.validation_fraction))
        )
        order = torch.randperm(
            batch_count, generator=torch.Generator().manual_seed(self.seed)
        ).tolist()
        return order[validation_count:], order[:validation_count]

    def _snapshot_weights(self, parameters):
        return [
            p.detach().to(self.target_offload_device, copy=True) for p in parameters
        ]

    @staticmethod
    @torch.no_grad()
    def _restore_weights(parameters, weights):
        for parameter, weight in zip(parameters, weights):
            parameter.copy_(weight)

    @torch.no_grad()
    def _evaluate(self, indices):
        return sum(self._batch_loss(self._batches[i]).item() for i in indices) / len(
            indices
        )

    def _batch_loss(self, batch):
        inputs = _map_tensors(batch.inputs, lambda t: t.to(self._device))
        prediction = self._graph(**inputs)
        loss = _output_loss(prediction, batch.target, batch.loss_mask)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite QAD loss in {self._name}")
        return loss

    @staticmethod
    @torch.no_grad()
    def _materialize_quantized_weights(modules):
        for module in modules:
            weight = forward_quantize(
                module, module.weight, "weight", module.quantization_scheme.weights
            )
            update_offload_parameter(module, "weight", weight)

    def _clear_subgraph(self):
        self._batches.clear()
        self._graph = None

    def on_calibration_end(self, state, event, **kwargs):
        self._clear_subgraph()
        self._teacher_activations = None

    def on_finalize(self, state, **kwargs):
        self._clear_subgraph()
        self._teacher_activations = None
        return True
