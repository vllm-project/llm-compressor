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

from typing import Dict

from sparsezoo.utils.registry import RegistryMixin


class PreprocessingFunctionRegistry(RegistryMixin):
    pass


@PreprocessingFunctionRegistry.register()
def custom_evolved_codealpaca_dataset(data: Dict):
    PROMPT_DICT = """[Instruction]:\n{instruction}\n\n[Response]:"""
    data["prompt"] = PROMPT_DICT.format_map(data)
    data["text"] = data["prompt"] + data["output"]
    return data
