"""Oneshot compression with Greedy Multi-Scale Decomposition.

This script applies greedy multi-scale decomposition (cascaded MPO + Low-Rank)
to compress model weights while targeting activation SNR.

Run on GPU for best performance.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from typing import Optional
import re

from llmcompressor.modifiers.experimental import GreedyMultiScaleLinear


# Configuration
# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

SAVE_DIR = MODEL_ID.split("/")[-1] + "-greedy-multiscale"

# Compression settings
TARGET_SNR_DB = 25.0  # Target activation SNR
MAX_STAGES = 10  # Maximum cascade iterations
TUCKER_NUM_MODES = 8  # Number of modes for Tucker decomposition
TUCKER_RANK = 2  # Rank ratio for Tucker core (low for small components)
KRONECKER_FACTOR_SIZE = None  # Kronecker factor size (None = auto)
BLOCKTT_BLOCK_SIZE = 512  # Block size for Block Tensor Train
BLOCKTT_NUM_CORES = 3  # Number of TT cores per block
BLOCKTT_RANK = 0.2  # Rank ratio for Block TT
BLOCKDIAG_NUM_BLOCKS = 16  # Number of diagonal blocks (local clusters)
BLOCKDIAG_RANK = 64  # Low-rank component rank (global communication)
SPARSE_SPARSITY = 0.1  # Column sparsity (0.1 = keep 90% of columns)
USE_KRONECKER = True  # Include Kronecker decomposition
USE_BLOCKTT = True  # Include Block Tensor Train
USE_SPARSE = True  # Include column-sparse stages in cascade

# Calibration settings
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"
NUM_CALIBRATION_SAMPLES = 32  # Increased for better activation statistics
MAX_SEQUENCE_LENGTH = 2048

# Layer targeting
COMPRESS_TARGETS = [
    # "model.layers.0.self_attn.q_proj",  # All attention projections
    "re:.*self_attn.(q|k|v|o)_proj$",  # All attention projections
    "re:.*mlp.(gate|up|down)_proj$",  # Uncomment to include MLP layers
]

IGNORE_LAYERS = [
    "lm_head",
    "embed_tokens",
]


def get_calib_dataset(tokenizer):
    """Load calibration dataset."""
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*10}]",
    )

    def preprocess(example):
        return {
            "input_ids": tokenizer.encode(example["text"].strip()[:MAX_SEQUENCE_LENGTH])
        }

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )

    return ds


def collect_layer_activations(model, layer_name, dataloader, device="cuda"):
    """Collect input activations for a specific layer."""
    activations = []

    # Navigate to layer
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)

    def hook(module, input, output):
        # input is a tuple, take first element
        act = input[0].detach().cpu()
        # Flatten if needed: (batch, seq, hidden) -> (batch*seq, hidden)
        if len(act.shape) == 3:
            batch, seq, hidden = act.shape
            act = act.reshape(batch * seq, hidden)
        activations.append(act)

    handle = layer.register_forward_hook(hook)

    # Forward passes
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            model(input_ids)

    handle.remove()

    # Concatenate all activations
    all_acts = torch.cat(activations, dim=0)
    return all_acts


def compress_model_greedy_multiscale(
    model,
    dataloader,
    target_layers,
    ignore_layers,
    device="cuda",
):
    """Apply greedy multi-scale compression to model."""
    import re

    print("\n" + "=" * 100)
    print("Greedy Multi-Scale Compression")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Target SNR: {TARGET_SNR_DB} dB")
    print(f"  Max stages: {MAX_STAGES}")
    print(f"  Tucker: num_modes={TUCKER_NUM_MODES}, rank={TUCKER_RANK}")
    print(f"  Kronecker: factor_size={KRONECKER_FACTOR_SIZE}, enabled={USE_KRONECKER}")
    print(
        f"  BlockTT: block_size={BLOCKTT_BLOCK_SIZE}, num_cores={BLOCKTT_NUM_CORES}, rank={BLOCKTT_RANK}, enabled={USE_BLOCKTT}"
    )
    print(f"  Sparse: sparsity={SPARSE_SPARSITY}, enabled={USE_SPARSE}")
    print(f"  BlockDiag+LR: num_blocks={BLOCKDIAG_NUM_BLOCKS}, rank={BLOCKDIAG_RANK}")
    print()

    # Collect all linear layers
    layers_to_compress = []

    def find_linear_layers(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if should be ignored
            should_ignore = False
            for ignore_pattern in ignore_layers:
                if ignore_pattern.startswith("re:"):
                    if re.match(ignore_pattern[3:], full_name):
                        should_ignore = True
                        break
                elif ignore_pattern in full_name:
                    should_ignore = True
                    break

            if should_ignore:
                continue

            # Check if should be compressed
            should_compress = False
            for target_pattern in target_layers:
                if target_pattern.startswith("re:"):
                    if re.match(target_pattern[3:], full_name):
                        should_compress = True
                        break
                elif (
                    target_pattern in full_name
                    or target_pattern == type(child).__name__
                ):
                    should_compress = True
                    break

            if should_compress and isinstance(child, nn.Linear):
                layers_to_compress.append((full_name, child, module, name))
            else:
                find_linear_layers(child, full_name)

    find_linear_layers(model)

    print(f"Found {len(layers_to_compress)} layers to compress:")
    for layer_name, _, _, _ in layers_to_compress:
        print(f"  - {layer_name}")
    print()

    # Compress each layer
    total_original_params = 0
    total_compressed_params = 0

    for layer_name, layer, parent_module, attr_name in layers_to_compress:
        print(f"\n{'='*100}")
        print(f"Compressing: {layer_name}")
        print(f"  Shape: {layer.weight.shape}")
        print(f"  Params: {layer.weight.numel():,}")
        print(f"{'='*100}")

        # Collect activations for this layer
        print("Collecting activations...")
        input_activations = collect_layer_activations(
            model, layer_name, dataloader, device
        )
        print(f"Collected {input_activations.shape[0]} activation samples")

        # Create greedy multiscale decomposition
        gms = GreedyMultiScaleLinear.from_linear(
            layer,
            input_activations=input_activations,
            target_snr_db=TARGET_SNR_DB,
            max_stages=MAX_STAGES,
            tucker_num_modes=TUCKER_NUM_MODES,
            tucker_rank=TUCKER_RANK,
            kronecker_factor_size=KRONECKER_FACTOR_SIZE,
            blocktt_block_size=BLOCKTT_BLOCK_SIZE,
            blocktt_num_cores=BLOCKTT_NUM_CORES,
            blocktt_rank=BLOCKTT_RANK,
            blockdiag_num_blocks=BLOCKDIAG_NUM_BLOCKS,
            blockdiag_rank=BLOCKDIAG_RANK,
            sparse_sparsity=SPARSE_SPARSITY,
            use_kronecker=USE_KRONECKER,
            use_blocktt=USE_BLOCKTT,
            use_sparse=USE_SPARSE,
            verbose=True,
        )

        # Replace layer
        setattr(parent_module, attr_name, gms)

        total_original_params += layer.weight.numel()
        total_compressed_params += gms.num_params

        print(f"\n✓ Compressed {layer_name}")

    # Summary
    print(f"\n{'='*100}")
    print("COMPRESSION SUMMARY")
    print(f"{'='*100}")
    print(f"Layers compressed: {len(layers_to_compress)}")
    print(f"Original params: {total_original_params:,}")
    print(f"Compressed params: {total_compressed_params:,}")
    print(f"Param ratio: {total_compressed_params/total_original_params:.2f}x")
    print(f"Compression: {100*(1-total_compressed_params/total_original_params):.1f}%")
    print(f"{'='*100}\n")


def main():
    # Load model and tokenizer
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare calibration data
    print("Loading calibration dataset...")
    calib_dataset = get_calib_dataset(tokenizer)

    # Create dataloader
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0
        )
        return {"input_ids": input_ids}

    dataloader = DataLoader(calib_dataset, batch_size=4, collate_fn=collate_fn)

    # Apply compression
    compress_model_greedy_multiscale(
        model=model,
        dataloader=dataloader,
        target_layers=COMPRESS_TARGETS,
        ignore_layers=IGNORE_LAYERS,
        device="cuda",
    )

    # Test generation
    print("\n" + "=" * 100)
    print("SAMPLE GENERATION")
    print("=" * 100)

    model.eval()
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("=" * 100 + "\n")

    # Save compressed model
    print(f"Saving compressed model to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"\n✓ Done! Compressed model saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
