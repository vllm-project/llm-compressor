# HIGGS: ILP-Based Mixed-Precision Quantization

HIGGS (Heuristic ILP-Guided Grouped Scheme) automatically selects optimal quantization schemes for each layer in your model.

## Quick Start

```bash
python llama3_higgs_example.py
```

This will:
1. Download Meta-Llama-3-8B-Instruct
2. Evaluate W4A16 and W8A8 schemes on each layer
3. Solve ILP to select optimal scheme per layer
4. Save quantized model to `./Meta-Llama-3-8B-Instruct-HIGGS`
5. Run a test generation

## How It Works

HIGGS follows a two-phase approach:

### Phase 1: MSE Collection & ILP Solving
1. For each layer, quantize with each candidate scheme
2. Compute MSE (Mean Squared Error) between quantized and original weights
3. Calculate alpha (importance) values using a depth-based heuristic
4. Solve ILP to minimize: `sum(MSE * alpha)` across all layers
   - Subject to: one scheme per layer
   - Constraint: fused layers (gate_proj+up_proj, q+k+v) use same scheme
   - Optional: average bitwidth constraint

### Phase 2: Apply Quantization
1. Apply ILP-selected schemes to each layer
2. Save quantized model with mixed-precision config

## Customization

Edit `llama3_higgs_example.py` to customize:

```python
# Use different model
MODEL_ID = "meta-llama/Llama-3.1-70B"

# Try more schemes
CANDIDATE_SCHEMES = ["W4A16", "W8A8", "FP8_DYNAMIC"]

# Constrain average bitwidth
optimal_config = ilp_quantize(
    model_stub=MODEL_ID,
    save_directory="./quantized",
    candidate_schemes=CANDIDATE_SCHEMES,
    target_avg_bitwidth=4.5,  # Force 4.5-bit average
    max_workers=16,
    device="cuda:0",
)
```

### Available Schemes
- `W4A16`: 4-bit weights, 16-bit activations
- `W8A8`: 8-bit weights, 8-bit activations
- `FP8_DYNAMIC`: Dynamic FP8 quantization

## WNaM Activation Quantization Sweep

```bash
python wnam_activation_sweep.py
```

Generates config-only mixed-precision allocations using WNaM candidate schemes
(W2A4 through W8A16) with dual weight/activation bitwidth constraints.

## Perplexity Measurement

```bash
python measure_ppl.py --model ./my-quantized-model --max-model-len 4096
```

Measures wikitext-2 perplexity using vLLM with `prompt_logprobs`. No lm_eval
dependency — uses vLLM directly for model loading and inference.

## Why HIGGS?

**Better Quality:** Different layers have different sensitivity to quantization. HIGGS allocates more bits to sensitive layers and fewer bits to robust layers, achieving better quality than uniform quantization at the same compression ratio.

**Automatic:** No manual tuning required. HIGGS automatically discovers the optimal scheme assignment.

**Hardware-Aware:** Respects hardware fusion constraints (e.g., gate_proj and up_proj must use same scheme for efficient kernels).

**Flexible:** Supports any combination of candidate schemes and optional bitwidth constraints.

## Expected Runtime

For Llama-3-8B with 2 candidate schemes:
- Phase 1 (MSE + ILP): ~10-15 minutes (GPU recommended)
- Phase 2 (Quantization): ~5-10 minutes
- Total: ~15-25 minutes

## Output

The quantized model is saved in HuggingFace format with:
- `model.safetensors` (or sharded `.safetensors` files)
- `config.json` with `quantization_config` containing mixed-precision groups
- Compatible with `transformers.AutoModelForCausalLM.from_pretrained()`

## Loading Quantized Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./Meta-Llama-3-8B-Instruct-HIGGS")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Run inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```
