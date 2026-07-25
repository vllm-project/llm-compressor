---
name: create-tiny-model
description: Create and manage tiny models for testing and development. Includes utilities for saving tiny models, inspecting tensors, and finetuning workflows.
args:
  model_id:
    type: string
    description: The HuggingFace model ID to use (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
    required: false
---

# Create Tiny Model Skill

This skill creates a tiny version of a known model for testing and experimentation purposes

## Arguments

- `model_id` (optional): The HuggingFace model ID to use for creating or working with tiny models. Examples: "Qwen/Qwen2.5-0.5B-Instruct", "facebook/opt-125m"

## Available Scripts

Scripts are located in `.claude/skills/create-tiny-model/scripts/`:

- `inspect_config.py` - Inspect the config fields of a model without downloading all files
- `save_tiny_model.py` - Template for saving a tiny version of the model
- `inspect_tensors.py` - Inspect and analyze tensors in models
- `finetune.py` - Finetune tiny models on a toy dataset
- `validate_tiny_model.py` - Validate that the tiny model was created correctly

## Templates

Templates are located in `.claude/skills/create-tiny-model/templates/`:

- `README_TEMPLATE.md` - Template for generating model README with placeholders for model details


## Steps to creating a tiny model

When this skill is invoked, the following steps will be completed:

1. **Inspect config**: Use `inspect_config.py` to understand the model configuration fields. Specifically find which fields control the number of layers, layer types, and the number of parameters. Check if the model is multimodal. If the model is multimodal, remember to always load with `...ForConditionalGeneration` rather than `AutoModelForCausalLM`

2. **Create tiny model**: Make a copy of `save_tiny_model.py`. Modify the copy to create a tiny version of the given model which maintains the same architecture as the original model (at least one of each attention type in the original model, etc.) but with ~1B parameters. It's okay to create a slightly bigger model so long as the architecture is still represented.

IMPORTANT: Start by only modifying the number of layers in the model. If the model is significantly larger than 1B parameters, then consider reducing the hidden sizes, number of experts, and other configurations.

3. **Fine-tune**: Fine tune the model on a toy dataset using `finetune.py`. This validates that the model can actually learn. Note: vision-language models may require script modifications to load correctly. Make sure the target perplexity is ~3.0, a model with a high perplexity with respect to the toy dataset is not considered valid.

If the model is a vision-language model, do not try to fine tune on a vision dataset, only fine tune on the provided text dataset. Continue to load with `...ForConditionalGeneration`.

4. **Validate checkpoint structure**: Make sure that the saved model checkpoint structure is analogous to the checkpoint structure of the original large model checkpoint. The `transformers` library can sometimes contain bugs where models are saved in invalid checkpoint structures. First, inspect the original checkpoint structure using the HuggingFace Hub API or by checking `https://huggingface.co/{model_id}/resolve/main/model.safetensors.index.json`. If this file does not exist, download the original checkpoint directly. Use `inspect_tensors.py` to inspect the checkpoint format of the saved model and/or the downloaded model. If the two structures do not match, create a converter script to convert our tiny saved checkpoint structure into a checkpoint structure which matches the original. Do not try to match mtp layers.

5. **Validate model**: Confirm that the model loads and inferences correctly using `validate_tiny_model.py`.

6. **Generate README**: Create a comprehensive README.md for the model using the template at `templates/README_TEMPLATE.md`. Fill in all placeholders with actual values:
   - `{model_name}` - Name of the tiny model directory
   - `{base_model_id}` - Original HuggingFace model ID
   - `{architecture}` - Model architecture type (from config.model_type)
   - `{total_params}` - Total parameter count in billions
   - `{activated_params}` - Activated parameter count for MoE models
   - `{config_table}` - Markdown table comparing original vs tiny config
   - `{checkpoint_description}` - Description of checkpoint structure (single file vs sharded)
   - `{validation_output}` - Output from running validate_tiny_model.py
   - `{additional_notes}` - Any additional notes about the model

7. **Upload** Upload the model to HuggingFace Hub using:
   ```bash
   hf upload inference-optimization/{tiny-model-id} {path-to-model} --repo-type=model
   hf collections add-item inference-optimization/tiny-models {tiny-model-id} model
   ```

   If you do not have permissions within the `inference-optimization` org, skip this step.

   IMPORTANT: Make sure the tiny model id reflects the number of parameters in the tiny model, not the base model. For example, if the base model is `Qwen/Qwen3-30B-A3B`, then the tiny model should be something like `Qwen3-1B-A0.6B`

   IMPORTANT: Make sure to upload the **fine tuned model**, not the base model.

8. **Copy to working directory**: Copy the final model to the project's working directory for easy access.

   IMPORTANT: Make sure to copy the **fine tuned model**, not the base model.

Make sure that, if you create any extra files, that they are created in a temporary directory, not in the skills folder.