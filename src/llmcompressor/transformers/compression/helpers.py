# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
from tqdm import tqdm

from llmcompressor.pytorch.utils import get_linear_layers

__ALL__ = [
    "tensor_follows_mask_structure",
    "infer_sparsity_structure_from_stage_modifiers",
    "infer_sparsity_structure_from_model",
]


def tensor_follows_mask_structure(tensor, mask: str = "2:4") -> bool:
    """
    :param tensor: tensor to check
    :param mask: mask structure to check for, in the format "n:m"
    :return: True if the tensor follows the mask structure, False otherwise.
        Note, some weights can incidentally be zero, so we check for
        atleast n zeros in each chunk of size m
    """

    n, m = tuple(map(int, mask.split(":")))
    # Reshape the tensor into chunks of size m
    tensor = tensor.view(-1, m)

    # Count the number of zeros in each chunk
    zero_counts = (tensor == 0).sum(dim=1)

    # Check if the number of zeros in each chunk atleast n
    # Greater than sign is needed as some weights can incidentally
    # be zero
    return torch.all(zero_counts >= n).item()


def infer_sparsity_structure_from_stage_modifiers(
    stage_modifiers: List["StageModifier"],  # noqa E501
) -> Optional[str]:
    """
    Determines the sparsity structure, if any exists, given the
    list of stage modifiers

    :param stage_modifiers: non-empty list of stage modifiers
    :return: sparsity structure as a string or None
    """
    for stage in stage_modifiers:
        if stage.applied:
            for modifier in stage.modifiers:
                if hasattr(modifier, "mask_structure"):
                    sparsity_structure = modifier.mask_structure
                    return sparsity_structure
    return None


def infer_sparsity_structure_from_model(model: torch.nn.Module) -> Optional[str]:
    """
    Determines the sparsity structure, if any exists, given the model

    :param model: model to check for sparsity structure
    :return: sparsity structure as a string or None
    """

    # check for the common sparsity structures
    structures = {"2:4"}
    for sparsity_structure in structures:
        linear_modules = get_linear_layers(model)
        linear_modules_with_sparsity_structure = [
            tensor_follows_mask_structure(layer.weight)
            for layer in tqdm(
                linear_modules.values(),
                desc="Checking whether model follows "
                f"{sparsity_structure} sparsity structure",
            )
        ]
        # if the majority of the linear modules follow the sparsity structure
        # we can assume that the model follows the sparsity structure
        # (taking into consideration the fact that some Linear layers like the
        # embedding layer might not be sparse)
        if (
            sum(linear_modules_with_sparsity_structure)
            > len(linear_modules_with_sparsity_structure) * 0.8
        ):
            return sparsity_structure

    return None
