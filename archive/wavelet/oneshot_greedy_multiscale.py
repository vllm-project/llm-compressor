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
TARGET_SNR_DB = 35.0  # Target activation SNR (30-35 dB range)
MAX_REFINEMENT_ITERS = 3  # Phase 2 refinement iterations
TOTAL_PARAM_BUDGET = 0.50  # Target 50% of original params

# ASVD (primary backbone — "fast climber" at low SNR)
# Rank is determined dynamically by efficiency knee, not by budget fraction.
# svd_budget_fraction is the MAX fraction of total budget ASVD can use.
USE_ASVD = True
SVD_BUDGET_FRACTION = 0.8  # Up to 80% of budget for ASVD (knee usually stops earlier)

# Tucker (residual structure, Step 3)
TUCKER_NUM_MODES = 2
TUCKER_RANK = 0.5

# SpectralMPO — DISABLED (slow climber: DCT needs ~98% of modes for 99% energy)
SPECTRAL_MPO_PARAM_BUDGET = 0.10
SPECTRAL_MPO_BLOCK_SIZE = 512
SPECTRAL_MPO_NUM_CORES = 3
SPECTRAL_MPO_RANK = 0.5

# Kronecker — DISABLED (consistently 0 dB contribution)
KRONECKER_FACTOR_SIZE = None
USE_KRONECKER = False

# Block Tensor Train (Step 3, if budget remains)
BLOCKTT_BLOCK_SIZE = 512
BLOCKTT_NUM_CORES = 3
BLOCKTT_RANK = 0.15
USE_BLOCKTT = True

# Block-Diagonal + Low-Rank (Step 3, if budget remains)
BLOCKDIAG_NUM_BLOCKS = 16
BLOCKDIAG_RANK = 64

# Column Sparse — "fast climber" at high SNR (Step 2)
# Budget allocated automatically: whatever ASVD doesn't use goes to sparse.
SPARSE_SPARSITY = 0.1  # Not used directly — budget determines column count

# Feature flags
USE_SPECTRAL_MPO = False  # Disabled: DCT is a slow climber
USE_SPARSE = True  # Adaptive sparse after ASVD knee

# Calibration settings
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"
NUM_CALIBRATION_SAMPLES = 128  # Increased for better activation statistics
MAX_SEQUENCE_LENGTH = 2048

# Layer targeting
COMPRESS_TARGETS = [
    # "model.layers.0.self_attn.q_proj",  # All attention projections
    "re:.*model.layers.15.self_attn.(q|k|v|o)_proj$",  # All attention projections
    "re:.*model.layers.15.mlp.(gate|up|down)_proj$",  # MLP layers
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


def collect_decoder_outputs(model, decoder_layer_name, dataloader, device="cuda"):
    """Collect output activations for a decoder layer."""
    outputs = []

    parts = decoder_layer_name.split(".")
    layer = model
    for part in parts:
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

    def hook(module, input, output):
        # Decoder layers return a tuple (hidden_states, ...)
        out = output[0] if isinstance(output, tuple) else output
        out = out.detach().cpu()
        if len(out.shape) == 3:
            out = out.reshape(-1, out.shape[-1])
        outputs.append(out)

    handle = layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch["input_ids"].to(device))
    handle.remove()
    return torch.cat(outputs, dim=0)


def _decoder_layer_name(layer_name):
    """Extract decoder layer name from a linear layer name.

    'model.layers.15.self_attn.q_proj' -> 'model.layers.15'
    """
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return ".".join(parts[: i + 2])
    return None


def _compute_snr_simple(original, approx):
    """Compute SNR in dB."""
    signal_power = torch.var(original)
    mse = torch.mean((original - approx) ** 2)
    return 10 * torch.log10(signal_power / (mse + 1e-10)).item()


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
    print(f"  Max refinement iters: {MAX_REFINEMENT_ITERS}")
    print(f"  Param budget: {TOTAL_PARAM_BUDGET:.0%} of original")
    print(f"  ASVD: budget_fraction={SVD_BUDGET_FRACTION}, enabled={USE_ASVD}")
    print(f"  Tucker: num_modes={TUCKER_NUM_MODES}, rank={TUCKER_RANK}")
    print(
        f"  SpectralMPO: budget={SPECTRAL_MPO_PARAM_BUDGET}, block_size={SPECTRAL_MPO_BLOCK_SIZE}, cores={SPECTRAL_MPO_NUM_CORES}, rank={SPECTRAL_MPO_RANK}, enabled={USE_SPECTRAL_MPO}"
    )
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

    # Collect baseline decoder layer outputs before any compression
    decoder_baselines = {}
    for layer_name, _, _, _ in layers_to_compress:
        dec_name = _decoder_layer_name(layer_name)
        if dec_name and dec_name not in decoder_baselines:
            print(f"Collecting baseline decoder outputs for {dec_name}...")
            decoder_baselines[dec_name] = collect_decoder_outputs(
                model, dec_name, dataloader, device
            )
            print(f"  Collected {decoder_baselines[dec_name].shape[0]} samples, "
                  f"dim={decoder_baselines[dec_name].shape[1]}")

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
            max_refinement_iters=MAX_REFINEMENT_ITERS,
            total_param_budget=TOTAL_PARAM_BUDGET,
            use_asvd=USE_ASVD,
            svd_budget_fraction=SVD_BUDGET_FRACTION,
            tucker_num_modes=TUCKER_NUM_MODES,
            tucker_rank=TUCKER_RANK,
            spectral_mpo_param_budget=SPECTRAL_MPO_PARAM_BUDGET,
            spectral_mpo_block_size=SPECTRAL_MPO_BLOCK_SIZE,
            spectral_mpo_num_cores=SPECTRAL_MPO_NUM_CORES,
            spectral_mpo_rank=SPECTRAL_MPO_RANK,
            kronecker_factor_size=KRONECKER_FACTOR_SIZE,
            blocktt_block_size=BLOCKTT_BLOCK_SIZE,
            blocktt_num_cores=BLOCKTT_NUM_CORES,
            blocktt_rank=BLOCKTT_RANK,
            blockdiag_num_blocks=BLOCKDIAG_NUM_BLOCKS,
            blockdiag_rank=BLOCKDIAG_RANK,
            sparse_sparsity=SPARSE_SPARSITY,
            use_spectral_mpo=USE_SPECTRAL_MPO,
            use_kronecker=USE_KRONECKER,
            use_blocktt=USE_BLOCKTT,
            use_sparse=USE_SPARSE,
            layer_name=layer_name,
            verbose=True,
        )

        # Replace layer
        setattr(parent_module, attr_name, gms)

        total_original_params += layer.weight.numel()
        total_compressed_params += gms.num_params

        # Measure decoder layer SNR (cumulative impact of all compressions so far)
        dec_name = _decoder_layer_name(layer_name)
        if dec_name and dec_name in decoder_baselines:
            decoder_output = collect_decoder_outputs(
                model, dec_name, dataloader, device
            )
            decoder_snr = _compute_snr_simple(
                decoder_baselines[dec_name], decoder_output
            )
            print(f"\n  Decoder layer SNR ({dec_name}): {decoder_snr:.2f} dB")

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
