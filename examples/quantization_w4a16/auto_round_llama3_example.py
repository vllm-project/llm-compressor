import os

_DEBUG = os.environ.get("DEBUG", "0") == "1"
IS_LLAMA = os.environ.get("MODEL", "LLAMA") == "LLAMA"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import AutoRoundModifier

# from llmcompressor.modifiers.quantization import QuantizationModifier as AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct"
model_id = "Qwen/Qwen2.5-0.5B"
model_id = "/data5/yliu7/HF_HOME/Qwen/Qwen2.5-0.5B"
model_id = "/data5/yliu7/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct"

model_dir = "/storage/yiliu7"
# model_id=f"{model_dir}/meta-llama/Meta-Llama-3.1-8B-Instruct"

model_dir = "/storage/yiliu7"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
if IS_LLAMA:
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
else:
    model_name = "Qwen/Qwen2.5-0.5B/"

model_id = f"{model_dir}/{model_name}"


# model_id = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

if _DEBUG:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2  # Use a smaller model for testing
    # Fix configuration validation issues
    # config.layer_types = config.layer_types[: config.num_hidden_layers]

    # Load the tokenizer and model
    if "Qwen" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2ForCausalLM(config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = LlamaForCausalLM(config)
    model.to(torch.bfloat16)
    NUM_CALIBRATION_SAMPLES = 3
    MAX_SEQUENCE_LENGTH = 16
    iters = 4

else:
    # Select number of samples. 512 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    light = {"batch_size": 8, "iters": 50, "seqlen": 2048, "nsamples": 128, "lr": 5e-3}
    light = {"batch_size": 8, "iters": 200, "seqlen": 2048, "nsamples": 128, "lr": 5e-3}

    light = {"batch_size": 8, "iters": 200, "seqlen": 2048, "nsamples": 128, "lr": None}
    # light = {"batch_size": 8, "iters": 200, "seqlen": 2048, "nsamples": 32, "lr": None}
    NUM_CALIBRATION_SAMPLES = light["nsamples"]
    MAX_SEQUENCE_LENGTH = light["seqlen"]
    iters = light["iters"]

# Select calibration dataset.
from auto_round.calib_dataset import get_dataset

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with AutoRound with a group size 128
recipe = AutoRoundModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head"], iters=iters
)


# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # !!! shuffle_calibration_samples: True -> mmlu 0.6574
    # !!! shuffle_calibration_samples: False -> mmlu 0.66
    shuffle_calibration_samples=False,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample = tokenizer("Explain AI in ", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
SAVE_DIR = (
    f"{model_dir}/"
    + model_id.rstrip("/").split("/")[-1]
    + "-W4A16-G128-disbale-shuffule-ar"
)
print(f"Saving quantized model to {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
