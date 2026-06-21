"""Extract and save activations from LLM layer norms."""

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096
LAYER_IDX = 15


def main():
    print("="*70)
    print("EXTRACTING LLM ACTIVATIONS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="validation", streaming=True)
    dataset = dataset.take(NUM_SAMPLES)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )

    # Prepare dataloader
    dataloader = DataLoader(
        dataset.map(tokenize, batched=True, remove_columns=["text"]),
        batch_size=1
    )

    # Collect activations
    print(f"\nCollecting activations from layer {LAYER_IDX}...")
    input_norm_acts = []
    post_attn_norm_acts = []

    layer = model.model.layers[LAYER_IDX]

    def input_norm_hook(module, input, output):
        act = output.detach().cpu().to(torch.bfloat16)
        if len(act.shape) == 3:
            input_norm_acts.append(act)

    def post_attn_norm_hook(module, input, output):
        act = output.detach().cpu().to(torch.bfloat16)
        if len(act.shape) == 3:
            post_attn_norm_acts.append(act)

    h1 = layer.input_layernorm.register_forward_hook(input_norm_hook)
    h2 = layer.post_attention_layernorm.register_forward_hook(post_attn_norm_hook)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= NUM_SAMPLES:
                break
            model(batch["input_ids"].to(device))
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{NUM_SAMPLES} samples")

    h1.remove()
    h2.remove()

    # Stack into tensors
    input_norm_tensor = torch.cat(input_norm_acts, dim=0)  # [64, seq_len, hidden_dim]
    post_attn_tensor = torch.cat(post_attn_norm_acts, dim=0)

    print(f"\nActivation shapes:")
    print(f"  Input layernorm: {list(input_norm_tensor.shape)}")
    print(f"  Post-attention layernorm: {list(post_attn_tensor.shape)}")

    # Save to cache
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_path = cache_dir / 'activations_post_input_ln.pt'
    post_attn_path = cache_dir / 'activations_post_attention_ln.pt'

    print(f"\nSaving activations...")
    torch.save(input_norm_tensor, input_path)
    torch.save(post_attn_tensor, post_attn_path)

    print(f"  Saved to {input_path}")
    print(f"  Saved to {post_attn_path}")

    # Report sizes
    input_size_mb = input_path.stat().st_size / (1024**2)
    post_attn_size_mb = post_attn_path.stat().st_size / (1024**2)

    print(f"\nFile sizes:")
    print(f"  Input layernorm: {input_size_mb:.1f} MB")
    print(f"  Post-attention layernorm: {post_attn_size_mb:.1f} MB")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
