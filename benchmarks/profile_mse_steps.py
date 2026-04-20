"""
Profile how many MSE grid search steps run before early stopping on real
Llama-3-8B weights.  Uses QuantizationModifier with memoryless_mse observer
(W8A16 channel + W4A16 group-128, weights only) and logs the step count for
every _grid_search_mse call.

Usage:
    python benchmarks/profile_mse_steps.py
"""

from collections import Counter

from compressed_tensors.quantization import preset_name_to_scheme
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers.compile_config import set_torch_compile
from llmcompressor.observers.mse_quant import enable_step_logging, get_step_log

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- W8A16 channel (weight-only) with MSE observer ---
scheme_w8 = preset_name_to_scheme("W8A16", ["Linear"])
scheme_w8.weights.observer = "memoryless_mse"

# --- W4A16 group-128 (weight-only) with MSE observer ---
scheme_w4 = preset_name_to_scheme("W4A16", ["Linear"])
scheme_w4.weights.observer = "memoryless_mse"

set_torch_compile(False)

for label, scheme in [("W8A16-channel", scheme_w8), ("W4A16-group128", scheme_w4)]:
    print(f"\n{'='*60}")
    print(f"Scheme: {label}")
    print(f"{'='*60}")

    # Reload model fresh for each scheme
    mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

    recipe = QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    )

    enable_step_logging(True)

    oneshot(
        model=mdl,
        recipe=recipe,
        num_calibration_samples=0,
    )

    steps = get_step_log()
    enable_step_logging(False)

    if not steps:
        print("  No grid search calls recorded!")
        continue

    counts = Counter(steps)
    print(f"\n  Total _grid_search_mse calls: {len(steps)}")
    print(f"  Min steps:  {min(steps)}")
    print(f"  Max steps:  {max(steps)}")
    print(f"  Mean steps: {sum(steps) / len(steps):.1f}")
    median = sorted(steps)[len(steps) // 2]
    print(f"  Median:     {median}")
    print(f"\n  Distribution:")
    for n in sorted(counts.keys()):
        bar = "#" * counts[n]
        print(f"    {n:4d} steps: {counts[n]:4d} calls  {bar}")

    print(f"\n  Percentiles:")
    ss = sorted(steps)
    for p in [25, 50, 75, 90, 95, 99, 100]:
        idx = min(len(ss) - 1, int(len(ss) * p / 100))
        print(f"    p{p:<3d}: {ss[idx]} steps")

    del mdl
