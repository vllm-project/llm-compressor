# Sparse Finetuning with TRL's SFTTrainer

The `SessionManagerMixin` can be added to other Trainer classes that inherit from 
[Hugging Face's Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer).

For example, we can add LLM Compressor support to TRL's SFTTrainer like so: 

Note: install `trl` using `pip install trl`

```python
from trl import SFTTrainer as TRLSFTTrainer

class SFTTrainer(SessionManagerMixIn, TRLSFTTrainer):
    ...
```

The new `SFTTrainer` class can now apply LLM Compressor recipes and modifiers during 
supervised finetuning, will full support for all of the original TRL features. The full
class is defined in the script `sft_trainer.py` and requires very minimal 
additional code: just a dataset load override to support passing in tokenized datasets 
to the Trainer. 

### Examples

* Script `ex_trl_constant.py`: finetunes a 50% sparse Llama-7b model,
using TRL's dataset preprocessing. Sparsity is maintained throughout training by 
applying a `ConstantPruningModifier` recipe to the `SFTTrainer` 

* Script `ex_trl_distillation.py`: finetunes a 50% sparse Llama-7b 
model using knowledge distillation from a dense Llama-7b model. Sparsity is maintained 
throughout training with a `ConstantPruningModifier` and layer-wise knowledge 
distillation is handled by the `OutputDistillationModifier`