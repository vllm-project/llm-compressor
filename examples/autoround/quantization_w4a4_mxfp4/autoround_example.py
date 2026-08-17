import argparse
from pathlib import Path

from auto_round.calib_dataset import get_dataset
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

parser = argparse.ArgumentParser(description="AutoRound W4A4-MXFP4 Quantization")
parser.add_argument(
    "model_id",
    nargs="?",
    default="Qwen/Qwen3-8B",
    help="Model name or path (positional)",
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model name or path (overrides positional argument)",
)
parser.add_argument(
    "--iters",
    type=int,
    default=200,
    help="Number of iterations for AutoRound",
)
parser.add_argument(
    "--disable-torch-compile",
    action="store_true",
    help="Disable torch.compile in AutoRound tuning",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="AutoRound tuning batch size",
)
parser.add_argument(
    "--num-calibration-samples",
    type=int,
    default=128,
    help="Number of calibration samples",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum calibration sequence length",
)
parser.add_argument(
    "--sequential-targets",
    nargs="+",
    default=None,
    help="Sequential targets passed to llm-compressor oneshot",
)
args = parser.parse_args()
args.model = args.model or args.model_id

# Select model and load it.
model_id = args.model
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = args.num_calibration_samples
MAX_SEQUENCE_LENGTH = args.max_seq_length
# Get aligned calibration dataset.

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# Configure the quantization algorithm to run.
#   * quantize the model to W4A4-MXFP4 with AutoRound
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="MXFP4",
    ignore=["lm_head"],
    iters=args.iters,
    enable_torch_compile=not args.disable_torch_compile,
    batch_size=args.batch_size,
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better mmlu score
    shuffle_calibration_samples=False,
    moe_calibrate_all_experts=False,
    sequential_targets=args.sequential_targets,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = Path(model_id).name + "-W4A4-MXFP4-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
