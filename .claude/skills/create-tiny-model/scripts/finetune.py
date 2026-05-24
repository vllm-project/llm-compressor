"""
Fine-tune a language model on famous internet copypastas until target perplexity is reached.
"""

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import math
import argparse


COPYPASTAS = [
    """According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible.""",
    """I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or as I've recently taken to calling it, GNU plus Linux. Linux is not an operating system unto itself, but rather another free component of a fully functioning GNU system made useful by the GNU corelibs, shell utilities and vital system components comprising a full OS as defined by POSIX.""",
    """The FitnessGram Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues. The 20 meter pacer test will begin in 30 seconds. Line up at the start. The running speed starts slowly, but gets faster each minute after you hear this signal.""",
    """Did you ever hear the tragedy of Darth Plagueis The Wise? I thought not. It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life.""",
]


class CopypastaDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        for text in texts:
            encoded = tokenizer(
                text + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.encodings.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": encoded["input_ids"].squeeze().clone(),
            })
            self.encodings[-1]["labels"][self.encodings[-1]["attention_mask"] == 0] = -100

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


class PerplexityStoppingCallback(TrainerCallback):
    def __init__(self, target_perplexity, consecutive_steps=15):
        self.target_perplexity = target_perplexity
        self.consecutive_steps = consecutive_steps
        self.steps_below_threshold = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            perplexity = math.exp(logs["loss"])

            if perplexity <= self.target_perplexity:
                self.steps_below_threshold += 1
                if self.steps_below_threshold >= self.consecutive_steps:
                    control.should_training_stop = True
            else:
                self.steps_below_threshold = 0

        return control


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on copypastas")
    parser.add_argument(
        "model",
        type=str,
        help="Model name or path"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for saved model"
    )
    parser.add_argument(
        "--target_perplexity",
        type=float,
        default=3.0,
        help="Target perplexity for early stopping (default: 3.0)"
    )
    parser.add_argument(
        "--consecutive_steps",
        type=int,
        default=15,
        help="Number of consecutive steps perplexity must be below target before stopping (default: 15)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum training steps (default: 1000)"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Creating dataset with {len(COPYPASTAS)} copypastas")
    dataset = CopypastaDataset(COPYPASTAS, tokenizer)

    # Repeat the dataset multiple times for more training data
    train_dataset = dataset

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=100,  # High number, will stop early based on perplexity
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        logging_first_step=True,
        max_steps=args.max_steps,
        weight_decay=0.01,
        dataloader_drop_last=False,
    )

    perplexity_callback = PerplexityStoppingCallback(
        target_perplexity=args.target_perplexity,
        consecutive_steps=args.consecutive_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[perplexity_callback],
    )

    print(f"Starting training (target perplexity: {args.target_perplexity})")
    trainer.train()

    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")

    # Generate a sample to verify the model works
    print("\n" + "="*50)
    print("Generating sample text:")
    print("="*50)
    model.eval()
    prompt = "According to all known laws"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=20,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(generated_text)
        print(f"\nGenerated {len(outputs[0])} tokens")
        print("="*50)


if __name__ == "__main__":
    main()
