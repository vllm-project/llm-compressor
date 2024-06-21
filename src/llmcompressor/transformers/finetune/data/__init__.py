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

# flake8: noqa

from .base import TextGenerationDataset
from .c4 import C4Dataset
from .cnn_dailymail import CNNDailyMailDataset
from .custom import CustomDataset
from .data_args import DataTrainingArguments
from .evolcodealpaca import EvolCodeAlpacaDataset
from .gsm8k import GSM8KDataset
from .open_platypus import OpenPlatypusDataset
from .ptb import PtbDataset
from .ultrachat_200k import UltraChatDataset
from .wikitext import WikiTextDataset
