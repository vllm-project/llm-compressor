import torch
from datasets import load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation

# NOTE: transformers 4.49.0 has an attribute error with DeepSeek.
# Please consider either downgrading your transformers version to a
# previous version or upgrading to a version where this bug is fixed

# select a Mixture of Experts model for quantization
MODEL_ID = "deepseek-ai/DeepSeek-V2.5"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# define a llmcompressor recipe for W416 quantization
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = "deepseek_recipe_w4a16.yaml"

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
    trust_remote_code_model=True,
)

# Confirm generations of the quantized model look sane.
# Generation is broken for deepseek models when using the latest transformers package
if Version(__version__) < Version("4.48"):
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
    print("==========================================")
else:
    print(
        "WARNING: cannot perform sample generation of "
        "deepseek models with transformers >= 4.48"
    )

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)


# Run the model on vLLM
try:
    from vllm import LLM, SamplingParams

    vllm_installed = True
except ImportError:
    vllm_installed = False

if vllm_installed:
    print("vLLM installed, running using vLLM")
    sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
    llm = LLM(
        model=SAVE_DIR,
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=1042,
        dtype=torch.half,
    )
    prompts = [
        "The capital of France is",
        "The president of the US is",
        "My name is",
    ]

    outputs = llm.generate(prompts, sampling_params)
    print("================= vLLM GENERATION ======================")
    for output in outputs:
        assert output
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("PROMPT", prompt)
        print("GENERATED TEXT", generated_text)
