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

from trl import SFTTrainer as TRLSFTTrainer

from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["SFTTrainer"]


class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    def _prepare_dataset(self, dataset, *args, **kwargs):
        if "input_ids" in dataset.column_names:
            # dataset is already tokenized, skip preprocessing
            return dataset

        return super()._prepare_dataset(dataset, *args, **kwargs)
