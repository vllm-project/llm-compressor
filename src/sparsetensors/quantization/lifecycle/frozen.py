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


from sparsetensors.quantization.quant_config import QuantizationStatus
from torch.nn import Module


__all__ = [
    "freeze_module_quantization",
]


def freeze_module_quantization(module: Module):
    if not getattr(module, "quantization_scheme", None):
        # no quantization scheme nothing to do
        return

    # delete observers from module
    observer_names = []
    for submodule_name, _ in module.named_modules():
        if "." not in submodule_name and submodule_name.endswith("_observer"):
            # delete any observers that belong directly to this module
            observer_names.append(submodule_name)
    for observer_name in observer_names:
        delattr(module, observer_name)

    module.quantization_status = QuantizationStatus.FROZEN
