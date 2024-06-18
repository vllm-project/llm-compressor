<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Sparse Finetuning with TRL's SFTTrainer

The `SessionManagerMixin` can be added to other Trainer classes that inherit from 
[Hugging Face's Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).

For example, we can add SparseML support to TRL's SFTTrainer like so: 

```python
from trl import SFTTrainer as TRLSFTTrainer

class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    ...
```

The new `SFTTrainer` class can now apply SparseML recipes and modifiers during 
supervised finetuning, will full support for all of the original TRL features. The full
class is defined in the script `sft_trainer.py` and requires very minimal 
additional code: just a dataset load override to support passing in tokenized datasets 
to the Trainer. 

### Examples

* Script `ex_trl_sft_data.py`: finetunes a 50% sparse Llama-7b model,
using TRL's dataset preprocessing. Sparsity is maintained throughout training by 
applying a `ConstantPruningModifier` recipe to the `SFTTrainer` 

* Script `ex_trl_distillation.py`: finetunes a 50% sparse Llama-7b 
model using knowledge distillation from a dense Llama-7b model. Sparsity is maintained 
throughout training with a `ConstantPruningModifier` and layer-wise knowledge 
distillation is handled by the `OutputDistillationModifier`