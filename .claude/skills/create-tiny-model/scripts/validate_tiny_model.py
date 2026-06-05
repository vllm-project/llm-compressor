"""
Validate the perplexity score of a tiny model
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEXT = """According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible."""
TARGET_PERPLEXITY = 10.0


def main():
    parser = argparse.ArgumentParser(description="Inspect model configs")
    parser.add_argument("model_id", type=str)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # perplexity test
    input_ids = tokenizer(TEXT, return_tensors="pt").input_ids.to(model.device)
    output = model(input_ids=input_ids, labels=input_ids)
    perplexity = torch.exp(output.loss).item()
    if perplexity <= TARGET_PERPLEXITY:
        print(f"Success: {perplexity} <= {TARGET_PERPLEXITY}")
    else:
        print(f"Failure: {perplexity} > {TARGET_PERPLEXITY}")

    # generation test
    print("\n" + "="*50)
    print("Generating sample text:")
    outputs = model.generate(
        input_ids=input_ids[:, :10],
        max_length=20,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    print("="*50)


if __name__ == "__main__":
    main()
