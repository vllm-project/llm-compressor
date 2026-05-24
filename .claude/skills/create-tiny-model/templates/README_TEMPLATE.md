---
license: mit
base_model:
- {base_model_id}
library_name: transformers
---

# {model_name}

This is a tiny version of [{base_model_id}](https://huggingface.co/{base_model_id}) created for testing and development.

## Model Details

- **Base Model**: {base_model_id}
- **Architecture**: {architecture}
- **Total Parameters**: {total_params}B
- **Activated Parameters**: {activated_params}

## Configuration Changes

The following parameters were reduced from the original model:

{config_table}

## Checkpoint Structure

{checkpoint_description}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

input_ids = tokenizer("According to all known laws", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

## Creation Process

This model was created using the llm-compressor `create-tiny-model` claude skill.

{Creation process}

## Notes

{additional_notes}
