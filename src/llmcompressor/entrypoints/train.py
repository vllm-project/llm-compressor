from transformers import HfArgumentParser

from llmcompressor.args import DatasetArguments, ModelArguments, TrainingArguments
from llmcompressor.core import LLMCompressor


def train(**kwargs):
    """
    Fine-tuning entrypoint that supports vanilla fine-tuning and
    knowledge distillation for compressed model using `oneshot`.


    This entrypoint is responsible the entire fine-tuning lifecycle, including
    preprocessing (model and tokenizer/processor initialization), fine-tuning,
    and postprocessing (saving outputs). The intructions for fine-tuning compressed
    model can be specified by using a recipe.

    - **Input Keyword Arguments:**
        `kwargs` are parsed into:
        - `model_args`: Arguments for loading and configuring a pretrained model
          (e.g., `AutoModelForCausalLM`).
        - `dataset_args`: Arguments for dataset-related configurations, such as
          calibration dataloaders.
        - `recipe_args`: Arguments for defining and configuring recipes that specify
          optimization actions.
        - `training_args`: rguments for defining and configuring training parameters

        Parsers are defined in `src/llmcompressor/args/`.

    - **Lifecycle Overview:**
        The fine-tuning lifecycle consists of three steps:
        1. **Preprocessing**:
            - Instantiates a pretrained model and tokenizer/processor.
            - Ensures input and output embedding layers are untied if they share
              tensors.
            - Patches the model to include additional functionality for saving with
              quantization configurations.
        2. **Training**:
            - Finetunes the model using a global `CompressionSession` and applies
              recipe-defined modifiers (e.g., `ConstantPruningModifier`,
                `OutputDistillationModifier`)
        3. **Postprocessing**:
            - Saves the model, tokenizer/processor, and configuration to the specified
              `output_dir`.

    - **Usage:**
        ```python
        train(model=model, recipe=recipe, dataset=dataset)

        ```

    """
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    model_args, dataset_args, training_args = parser.parse_dict(kwargs)

    compressor = LLMCompressor(**model_args)
    compressor.set_train_dataset(**dataset_args)
    compressor.train(**training_args)
