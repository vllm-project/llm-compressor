from itertools import product
from typing import Callable, Iterator

import torch
from compressed_tensors.quantization import QuantizationStrategy, forward_quantize
from compressed_tensors.utils import (
    get_execution_device,
    get_lowest_common_ancestor_name,
    getattr_chain,
    patch_attrs,
    update_offload_parameter,
)
from loguru import logger
from torch.nn import Module
from torch.utils._pytree import tree_leaves
from tqdm import tqdm

from llmcompressor.core import active_session
from llmcompressor.modifiers.awq.mappings import AWQMapping, ResolvedMapping
from llmcompressor.modifiers.quantization.calibration import call_observer
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.observers.base import Observer


def validate_and_get_smooth_layer(
    smooth_layers: list[Module],
    module_to_name: dict[Module, str],
    mapping: AWQMapping,
) -> tuple[Module, str]:
    if len(smooth_layers) > 1:
        raise ValueError(
            "AWQ needs to match a single smoothlayer for each mapping but "
            f"got {[module_to_name.get(s) for s in smooth_layers]}"
            f" for mapping: {mapping}"
        )

    smooth_layer = smooth_layers[0]
    smooth_name = module_to_name.get(smooth_layer)
    return smooth_layer, smooth_name


def flatten_balance_layers(
    nested_balance_layers: list[list[Module]],
    module_to_name: dict[Module, str],
) -> tuple[list[Module], list[str]]:
    # [[b00, b01, ...], [b10, b11, ...], ...] -> [b00, b01, ..., b10, b11, ...]
    balance_layers = tree_leaves(nested_balance_layers)
    balance_names = [module_to_name.get(balance_layer) for balance_layer in balance_layers]
    return balance_layers, balance_names


def get_mapping_skip_reason(
    smooth_layer: Module,
    smooth_name: str,
    balance_layers: list[Module],
    balance_names: list[str],
    targeted_names: set[str],
) -> str | None:
    if not _check_layers_are_compatible(
        smooth_layer, smooth_name, balance_layers, balance_names
    ):
        return " because found incompatible balance layers"

    any_targeted = smooth_name in targeted_names or any(
        balance_name in targeted_names for balance_name in balance_names
    )
    if not any_targeted:
        return " because no layers are targeted for quantization"
    if len(balance_layers) == 0:
        return " because no balance layers were found"
    return None


def resolve_activation_hook_target(
    mapping: AWQMapping,
    ancestor: Module,
    ancestor_name: str,
) -> Module | None:
    if not mapping.activation_hook_target:
        return None

    activation_hook_target = getattr_chain(ancestor, mapping.activation_hook_target)
    if activation_hook_target is None:
        raise ValueError(
            f"activation_hook_target '{mapping.activation_hook_target}'"
            f" not found on parent module '{ancestor_name}'"
        )
    return activation_hook_target


def extract_masked_activations(activations: torch.Tensor) -> torch.Tensor:
    # Get loss mask for current batch from state.
    session = active_session()
    state = session.state
    loss_masks = state.loss_masks if state else None
    batch_idx = state.current_batch_idx if state else -1
    loss_mask = loss_masks[batch_idx] if loss_masks and batch_idx >= 0 else None

    if loss_mask is None:
        return activations.flatten(0, -2)

    # Mask: [batch, seq] -> [batch, seq, 1]
    mask = loss_mask.to(activations.device).unsqueeze(-1)
    flat_activations = activations.flatten(0, -2)  # [batch*seq, hidden]
    flat_mask = mask.flatten(0, -2).squeeze(-1)
    return flat_activations[flat_mask.bool()]


def should_skip_smoothing_for_outputs(
    mapping: ResolvedMapping, fp16_outputs: list[torch.Tensor]
) -> bool:
    if len(fp16_outputs) == 0 or all(output.numel() == 0 for output in fp16_outputs):
        logger.info(
            f"Skipping smooth_layer {mapping.smooth_name}, no activations "
            "found to scale. This can occasionally occur in MoE models when "
            "certain experts are not activated by calibration samples."
        )
        return True
    if not all(output.isfinite().all() for output in fp16_outputs):
        logger.warning(
            f"Skipping smooth_layer {mapping.smooth_name}, NaN or inf outputs "
            "found during forward pass of the parent module "
            f"{mapping.parent_name}. The model is either generating NaN output "
            "with provided calibration data set, or the mappings are incorrectly "
            "set and modifying the model in undesired ways. If you encounter "
            "this consistently, raise an issue at "
            "https://github.com/vllm-project/llm-compressor/issues"
        )
        return True
    return False


@torch.no_grad()
def apply_scale_to_module(
    module: Module,
    scales: torch.Tensor,
    balance_layers: list[Module],
    smooth_layer: Module,
    orig_layer_weights: dict[Module, torch.Tensor],
) -> None:
    scales = scales.to(module.weight.device)
    if module in balance_layers:
        update_offload_parameter(
            module,
            "weight",
            orig_layer_weights[module].to(module.weight.device) * scales.view(1, -1),
        )
    elif module == smooth_layer:
        if module.weight.ndim == 1:
            update_offload_parameter(module, "weight", module.weight.div_(scales))
        else:
            # Edge case when smooth layer number of out_features is not equal to
            # balance layer number of in_features (e.g. fused qkv_proj -> o_proj).
            # Default to scaling the last output features because the desired smooth
            # layer is v_proj.
            weight = module.weight
            weight[-scales.size(0) :].div_(scales.view(-1, 1))
            update_offload_parameter(module, "weight", weight)

        if hasattr(module, "bias") and module.bias is not None:
            update_offload_parameter(module, "bias", module.bias.div_(scales))


def get_balance_layers_with_weight_quantization(
    balance_layers: list[Module],
) -> list[Module]:
    return [
        layer
        for layer in balance_layers
        if hasattr(layer, "quantization_scheme")
        and hasattr(layer.quantization_scheme, "weights")
    ]


def create_memoryless_weight_observers(balance_layers: list[Module]) -> list[Observer]:
    return [
        Observer.load_from_registry(
            "memoryless_minmax",
            base_name="weight",
            args=balance_layer.quantization_scheme.weights,
            module=balance_layer,
        )
        for balance_layer in balance_layers
    ]


def compute_scales_for_ratio(
    ratio: float,
    use_duo_scaling: bool,
    x_mean: torch.Tensor,
    w_mean: torch.Tensor | None,
) -> torch.Tensor:
    # NOTE: s^-1 * x is fused here, according to paper.
    if use_duo_scaling:
        scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
    else:
        scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
    scales = scales / (scales.max() * scales.min()).sqrt()
    scales[torch.isinf(scales)] = 1
    scales[torch.isnan(scales)] = 1
    return scales


def apply_quantized_balance_weights(
    balance_layers: list[Module],
    scale_view: torch.Tensor,
    orig_layer_weights: dict[Module, torch.Tensor],
) -> None:
    for balance_layer in balance_layers:
        w_qscheme = balance_layer.quantization_scheme.weights
        balance_layer.weight.data.copy_(
            orig_layer_weights[balance_layer].to(scale_view.device) * scale_view
        )

        should_calculate_gparam = w_qscheme.strategy == QuantizationStrategy.TENSOR_GROUP
        call_observer(
            balance_layer,
            "weight",
            balance_layer.weight,
            should_calculate_gparam=should_calculate_gparam,
        )
        balance_layer.weight.data = (
            forward_quantize(
                balance_layer,
                balance_layer.weight,
                "weight",
                w_qscheme,
            )
            / scale_view
        ).to(balance_layer.weight.dtype)


def apply_tensor_group_fusion_if_needed(
    parent: Module, balance_layers: list[Module]
) -> None:
    # Apply fused global scales for TENSOR_GROUP during grid search to match
    # inference behavior.
    if balance_layers and all(
        getattr(layer.quantization_scheme.weights, "strategy", None)
        == QuantizationStrategy.TENSOR_GROUP
        for layer in balance_layers
    ):
        update_fused_layer_weight_global_scales(parent)


def compute_scale_losses(
    mapping: ResolvedMapping,
    fp16_outputs: list[torch.Tensor],
    orig_layer_weights: dict[torch.nn.Module, torch.Tensor],
    get_grid_search_means: Callable[
        [ResolvedMapping, torch.device], tuple[torch.Tensor, torch.Tensor | None]
    ],
    get_grid_configuration: Callable[[], tuple[int, list[bool]]],
    run_samples: Callable[[Module], list[torch.Tensor]],
    compute_loss: Callable[[list[torch.Tensor], list[torch.Tensor]], float],
) -> list[dict[str, float | bool | torch.Tensor]]:
    device = get_execution_device(mapping.parent)
    x_mean, w_mean = get_grid_search_means(mapping, device)
    n_grid, duo_scalings = get_grid_configuration()
    balance_layers_to_patch = get_balance_layers_with_weight_quantization(
        mapping.balance_layers
    )

    history: list[dict[str, float | bool | torch.Tensor]] = []
    best_error_so_far = float("inf")
    with patch_attrs(
        balance_layers_to_patch,
        "weight_observer",
        create_memoryless_weight_observers(balance_layers_to_patch),
    ):
        total_iterations = n_grid * len(duo_scalings)
        pbar = tqdm(
            product(range(n_grid), duo_scalings),
            total=total_iterations,
            desc=f"Grid search for {mapping.smooth_name}",
            leave=False,
        )
        for grid_idx, use_duo_scaling in pbar:
            ratio = grid_idx / n_grid
            scales = compute_scales_for_ratio(
                ratio=ratio,
                use_duo_scaling=use_duo_scaling,
                x_mean=x_mean,
                w_mean=w_mean,
            )
            scale_view = scales.view(1, -1).to(device)

            apply_quantized_balance_weights(
                balance_layers=balance_layers_to_patch,
                scale_view=scale_view,
                orig_layer_weights=orig_layer_weights,
            )
            apply_tensor_group_fusion_if_needed(
                mapping.parent, balance_layers_to_patch
            )

            int_w_outputs = run_samples(mapping.parent)
            loss = compute_loss(fp16_outputs, int_w_outputs)
            del int_w_outputs

            history.append(
                {
                    "ratio": ratio,
                    "duo_scaling": use_duo_scaling,
                    "error": loss,
                    "scales": scales.clone(),
                }
            )
            if loss < best_error_so_far:
                best_error_so_far = loss
            pbar.set_postfix({"best_error": f"{best_error_so_far:.3e}"})

    return history


def select_best_scales_from_losses(
    loss_history: list[dict[str, float | bool | torch.Tensor]],
) -> tuple[torch.Tensor, float, float, float]:
    if len(loss_history) == 0:
        raise Exception("No scales were evaluated during AWQ grid search")

    initial_error = loss_history[0]["error"]
    best_entry = min(loss_history, key=lambda entry: entry["error"])
    best_error = best_entry["error"]
    best_ratio = best_entry["ratio"]
    best_scales = best_entry["scales"]

    if not torch.isfinite(torch.tensor(best_error)):
        logger.debug(loss_history)
        raise Exception(
            "No finite loss was found in best scalesgrid search. This typically "
            "means NaN values are appearing in the forward pass of the parent "
            "module. If you encounter this error, raise an issue at "
            "https://github.com/vllm-project/llm-compressor/issues"
        )

    return best_scales, best_ratio, best_error, initial_error


def _check_layers_are_compatible(
    smooth_layer, smooth_name, balance_layers, balance_names
):
    """
    returns True if they are all compatible
    returns False if any smooth & balance layers are incompatible
    """
    for balance_layer, balance_name in zip(balance_layers, balance_names):
        # exclude v_proj->o_proj mappings whose shapes are incompatible
        # https://github.com/mit-han-lab/llm-awq/pull/67#issuecomment-1681632777
        if (
            isinstance(smooth_layer, torch.nn.Linear)
            and isinstance(balance_layer, torch.nn.Linear)
            and balance_name.endswith(".o_proj")
            and (
                (
                    smooth_name.endswith(".v_proj")
                    and smooth_layer.out_features != balance_layer.in_features
                )
                or (
                    smooth_name.endswith(".qkv_proj")
                    and smooth_layer.out_features != 3 * balance_layer.in_features
                )
            )
        ):
            return False
    return True


def get_lowest_common_ancestor_with_avoid(
    balance_names: Iterator[str], model: Module, avoid=torch.nn.ModuleList
):
    """
    Get the lowest ancestor that is not the avoided class/type.
    see compressed_tensors.utils.get_lowest_common_ancestor_name
    for detail on case handling.

    NOTE: primarily used to exclude parents of type ModuleList, which don't play
    nicely with hooks because their forward method is never directly
    called for MoE models. See Qwen3MoeSparseMoeBlock for example, experts
    are selected based on router output and their forward method is called.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L233
    """
    ancestor_name = get_lowest_common_ancestor_name(balance_names)

    while True:
        if ancestor_name == "":
            return "", model
        ancestor = model.get_submodule(ancestor_name)
        if not isinstance(ancestor, avoid):
            return ancestor_name, ancestor
        ancestor_name = ".".join(ancestor_name.split(".")[:-1])


def accumulate_mean(
    inp: torch.Tensor,
    prev_mean_and_count: tuple[torch.FloatTensor, int] | None,
) -> tuple[torch.FloatTensor, int]:
    sum_added = inp.sum(dim=0)
    num_added = inp.size(0)
    if prev_mean_and_count is None:
        return sum_added / num_added, num_added

    prev_mean, prev_count = prev_mean_and_count
    prev_mean = prev_mean.to(inp.device)

    prev_sum = prev_mean * prev_count
    new_count = prev_count + num_added

    return (prev_sum + sum_added) / new_count, new_count
