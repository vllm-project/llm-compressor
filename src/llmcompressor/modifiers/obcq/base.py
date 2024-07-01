from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.core.state import State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.obcq.utils.sgpt_wrapper import SparseGptWrapper
from llmcompressor.modifiers.utils.layer_compressor import LayerCompressor
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_no_split_params,
    get_prunable_layers,
)

__all__ = ["SparseGPTModifier"]


class SparseGPTModifier(Modifier):
    """
    Modifier for applying the one-shot SparseGPT algorithm to a model

    Lifecycle:
        - on_initialize
            - initialize_compression()
                - compressible_layers()
                - LayerCompressor.pre_compress()
            - apply_compression()
                - run_calibration_forward()
                - LayerCompressor.compress()
                - LayerCompressor.post_compress()
                - LayerCompressor.revert_layer_wrappers()

    | Sample yaml:
    |   test_stage:
    |       obcq_modifiers:
    |           SparseGPTModifier:
    |               sparsity: 0.5
    |               mask_structure: "2:4"
    |               sequential_update: True
    |               dampening_frac: 0.001
    |               block_size: 128

    :param sparsity: Sparsity to compress model to
    :param sparsity_profile: Can be set to 'owl' to use Outlier Weighed
        Layerwise Sparsity (OWL), more information can be found
        in the paper https://arxiv.org/pdf/2310.05175
    :param owl_m: Number of outliers to use for OWL
    :param owl_lmbda: Lambda value to use for OWL
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param preserve_sparsity_mask: Whether or not to preserve the sparsity mask
        during when applying sparsegpt, this becomes useful when starting from a
        previously pruned model, defaults to False.
    """

    sparsity: Union[float, List[float]] = 0.0
    sparsity_profile: Optional[str] = None
    owl_m: Optional[int] = None
    owl_lmbda: Optional[float] = None
    mask_structure: str = "0:0"
    sequential_update: Optional[bool] = False
    targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    preserve_sparsity_mask: bool = False

    model: Optional[Any] = None
    layer_compressors_: Optional[List[Any]] = None
    prunen_: Optional[int] = None
    prunem_: Optional[int] = None
    compressible_layers_: Optional[List] = None

    def on_initialize_structure(self, state: State, **kwargs):
        """
        Initialize the structure of the model for compression.
        This modifier does not modifiy the model structure, so this method
        is a no-op.

        :param state: session state storing input model and calibration data
        """
        return True

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the OBCQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        if not self.initialized_structure_:
            self.on_initialize_structure(state, **kwargs)

        if self.sparsity == 0.0:
            raise ValueError(
                "To use the SparseGPTModifier, target sparsity must be > 0.0"
            )

        modifiable_model = state.model
        calibration_dataloader = state.data.calib

        if self.targets is None:
            # if no targets are provided, default to the modules that shouldn't be
            # split by FSDP. For Transformers models this is equivalent to the
            # decoder layers (ie LlamaDecoderLayer)
            self.targets = get_no_split_params(modifiable_model)

        self.initialize_compression(modifiable_model, calibration_dataloader)
        self.apply_compression(calibration_dataloader)

        return True

    def on_finalize(self, state: State, **kwargs):
        """
        Nothing to do on finalize, on this level.
        Quantization Modifier if any will be finalized in the subclass

        :param state: session state storing input model and calibration data
        :param kwargs: additional arguments
        :return: True
        """
        return True

    def initialize_compression(
        self,
        model: Module,
        dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None,
    ):
        """
        Setup for SparseGPT, initializes the model, device,
        and other parameters, also initilializes the
        compressible layers of model, and sets the device

        :param model: model to initialize for compression
        """
        self.model = model
        self.compressible_layers_ = self.compressible_layers()
        self.layer_compressors_ = []
        self._infer_mask_block_size()

        if self.sparsity_profile is not None and self.sparsity_profile.lower() == "owl":
            logger.info(
                "Inferring layer-wise sparsities from "
                f"{len(dataloader)} calibration samples..."
            )
            self.sparsity = self._infer_layer_sparsity(dataloader)
        self._validate_layerwise_sparsity()

        for idx, (name, layer) in enumerate(self.compressible_layers_.items()):
            logger.info(f"Preparing {name} for compression")
            if isinstance(self.sparsity, Dict):
                layer_sparsity = self.sparsity[name]
            elif isinstance(self.sparsity, List):
                layer_sparsity = self.sparsity[idx]
            else:  # float
                layer_sparsity = self.sparsity
            args = self._pruning_arguments(layer_sparsity)
            comp_cls = self._compression_class()
            compressor = LayerCompressor(comp_cls, self.model, layer, idx, name, args)
            if not self.sequential_update:
                # add all batch processing hooks before the forward pass
                compressor.pre_compress()
            self.layer_compressors_.append(compressor)

    def compressible_layers(self) -> Dict:
        """
        Retrieves the modules corresponding to a list of
        compressible layer names

        :precondition: self.model is set and is a torch.nn.Module
        :return: dictionary of modules to compress
        """
        if not isinstance(self.model, Module):
            raise ValueError(
                "`self.model` must be a PyTorch Module to use "
                f"the {self.__class__.__qualname__} modifier but got "
                f"{type(self.model)} instead"
            )

        return get_layers(self.targets, self.model)

    @torch.no_grad()
    def apply_compression(
        self, dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None
    ) -> Dict:
        """
        Run Wanda on the loaded model, using dataloader as calibration data

        :param dataloader: calibration data for WANDA
        """
        class_name = self.__class__.__name__.replace("PyTorch", "")
        logger.info(
            f"Running {class_name} calibration with "
            f"{len(dataloader) if dataloader else 0} samples..."
        )
        if not self.sequential_update:
            # in non-sequential mode we run one forward batch for all modules
            run_calibration_forward(self.model, dataloader, mask_padding=True)

        num_layers = len(self.compressible_layers_)
        for idx, layer_compressor in enumerate(self.layer_compressors_):
            layer_sparsity = layer_compressor.args["sparsity"]
            logger.info(
                f"\n===== Compressing layer {idx+1}/{num_layers} "
                f"to sparsity {layer_sparsity} ====="
            )

            # Prune/quantize using SparseGPT
            if self.sequential_update:
                # in sequential mode we run one forward pass for each module we
                # want to compress, this will be really slow but allows compression in
                # earlier layers to affect later layers
                layer_compressor.pre_compress()
                logger.info(f"Calibrating {layer_compressor.name}...")
                run_calibration_forward(self.model, dataloader, mask_padding=True)
            layer_compressor.compress()
            layer_compressor.post_compress()
            layer_compressor.revert_layer_wrappers()
            torch.cuda.empty_cache()

    def _validate_layerwise_sparsity(self):
        if isinstance(self.sparsity, float):
            # single sparsity will be applied to all layers
            return

        target_layers = list(self.compressible_layers_.keys())

        if len(target_layers) != len(self.sparsity):
            raise ValueError(
                "Number of layer targets must match the number of sparsities. "
                "Received {len(target_layers)} layers and "
                f"{len(self.sparsity)} sparsities"
            )

    def _pruning_arguments(self, sparsity):
        """
        Gather the parameters needed for root module compression in a dict

        :param sparsity: target sparsity
        :return: dict of params for pruning
        """
        return {
            "sparsity": sparsity,
            "prunen": self.prunen_,
            "prunem": self.prunem_,
            "blocksize": self.block_size,
            "percdamp": self.dampening_frac,
            "preserve_sparsity_mask": self.preserve_sparsity_mask,
        }

    def _compression_class(self):
        """
        :return: wrapper class used for root modules of this compression class
        """
        return SparseGptWrapper

    def _infer_mask_block_size(self):
        """
        Infer the mask block size from the mask structure.
        Parses mask_structure of the form N:M where N, M are integers that
        define a custom block shape; and sets prunen_ and prunem_ accordingly.

        :post-condition: prunen_ and prunem_ are set
        """
        if self.mask_structure is None:
            raise ValueError("mask_structure must be defined")

        self.prunen_, self.prunem_ = list(map(int, self.mask_structure.split(":")))

    def _infer_layer_sparsity(self, calibration_dataloader):
        acts = _get_activations(self.model, calibration_dataloader)
        sparsegpt_groups = {}
        for name, layer in self.compressible_layers_.items():
            prunable_layers = get_prunable_layers(layer)
            z = [
                m.weight.abs() * acts[f"{name}.{n}"].unsqueeze(0)
                for n, m in prunable_layers.items()
            ]
            sparsegpt_groups[name] = torch.cat([item.flatten().cpu() for item in z])

        acts = None
        del acts
        torch.cuda.empty_cache()

        outlier_ratios = {}
        for group in sparsegpt_groups:
            threshold = torch.mean(sparsegpt_groups[group]) * self.owl_m
            outlier_ratios[group] = (
                100
                * (sparsegpt_groups[group] > threshold).sum().item()
                / sparsegpt_groups[group].numel()
            )
        outlier_ratios_arr = np.array([outlier_ratios[k] for k in outlier_ratios])
        for k in outlier_ratios:
            outlier_ratios[k] = (outlier_ratios[k] - outlier_ratios_arr.min()) * (
                1
                / (outlier_ratios_arr.max() - outlier_ratios_arr.min())
                * self.owl_lmbda
                * 2
            )
        outlier_ratios_arr = np.array([outlier_ratios[k] for k in outlier_ratios])
        sparsities = {
            k: 1
            - (
                outlier_ratios[k]
                - np.mean(outlier_ratios_arr)
                + (1 - float(self.sparsity))
            )
            for k in outlier_ratios
        }
        logger.info(f"OWL sparsities for sp={self.sparsity} are:")
        for k in sparsities:
            logger.info(f"Sparsity for {k}: {sparsities[k]}")
        return sparsities


@torch.no_grad()
def _get_activations(model, data_loader, nsamples=128):
    import functools

    model.eval()
    acts = {}

    def save_acts(module, input, name):
        if isinstance(input, tuple):
            input = input[0]
        if name not in acts:
            acts[name] = 1.0 / nsamples * input.detach().pow(2).sum(dim=(0, 1)).sqrt()
        else:
            acts[name] += 1.0 / nsamples * input.detach().pow(2).sum(dim=(0, 1)).sqrt()

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            hooks.append(
                mod.register_forward_pre_hook(functools.partial(save_acts, name=name))
            )
    device = next(model.parameters()).device
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
        batch = None
    torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    return acts
