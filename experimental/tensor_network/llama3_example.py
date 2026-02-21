from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from llmcompressor import oneshot, active_session
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.experimental import TensorNetworkModifier


# Select model and load it.
# MODEL_ID = "Qwen/Qwen3-30B-A3B"
# MODEL_ID = "CohereLabs/c4ai-command-r-plus"
# MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_ID = "tencent/Hunyuan-A13B-Instruct"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_ID = "Qwen/Qwen3-0.6B"
# MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
# MODEL_ID = "meta-llama/Llama-2-7b-hf"
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "microsoft/Phi-3-medium-4k-instruct"

SAVE_DIR = MODEL_ID.split("/")[-1] + "-tn"


# Configure the quantization algorithm to run.
recipe = [
    TensorNetworkModifier(
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*mlp.shared_expert_gate$",
            "re:visual.*",
        ],
        targets=["Linear"],
        # offload_device=torch.device("cpu"),
        # best if block size is equal to some value**num_cores
        # 8**3=512, 16**3=4096
        num_cores=3,
        block_size=512,
    ),
]

# Select calibration dataset.
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 2  # 256
MAX_SEQUENCE_LENGTH = 16


def get_calib_dataset(tokenizer):
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*10}]",
    )

    def preprocess(example):
        return {"input_ids": tokenizer.encode(example["text"].strip())}

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )

    return ds


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    ###
    ### Apply algorithms.
    ###
    oneshot(
        model=model,
        dataset=get_calib_dataset(tokenizer),
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        log_dir=None,
    )

    # Confirm generations of the quantized model look sane.
    dispatch_for_generation(model)
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save to disk compressed.
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
