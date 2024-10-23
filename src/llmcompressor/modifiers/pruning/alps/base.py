from typing import Any, List, Optional, Union

from llmcompressor.modifiers.obcq import SparseGPTModifier

from llmcompressor.modifiers.pruning.alps.utils.alps_wrapper import ALPSWrapper

__all__ = ["ALPSModifier"]


class ALPSModifier(SparseGPTModifier):
    """
    Modifier for applying the one-shot ALPS algorithm to a model

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
    |           ALPSModifier:
    |               sparsity: 0.5
    |               mask_structure: "2:4"
    |               sequential_update: True
    |               dampening_frac: 0.001

    :param sparsity: Sparsity to compress model to
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    """

    sparsity: Union[float, List[float]] = 0.0
    sparsity_profile: Optional[str] = None
    owl_m: Optional[int] = None
    owl_lmbda: Optional[float] = None
    mask_structure: str = "0:0"
    sequential_update: Optional[bool] = False
    targets: Union[str, List[str], None] = None
    dampening_frac: Optional[float] = 0.01
    preserve_sparsity_mask: bool = False

    model: Optional[Any] = None
    layer_compressors_: Optional[List[Any]] = None
    prunen_: Optional[int] = None
    prunem_: Optional[int] = None
    compressible_layers_: Optional[List] = None

    save_hidden_states: Optional[bool] = None
    hidden_states_dir: Optional[str] = None

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
            "percdamp": self.dampening_frac,
            "preserve_sparsity_mask": self.preserve_sparsity_mask,
        }

    def _compression_class(self):
        """
        :return: wrapper class used for root modules of this compression class
        """
        return ALPSWrapper

