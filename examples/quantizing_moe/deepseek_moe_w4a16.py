import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

# select a Mixture of Experts model for quantization
MODEL_ID = "deepseek-ai/DeepSeek-V2.5"

# adjust based off number of desired GPUs
# if not enough memory is available, some layers will automatically be offlaoded to cpu
device_map = calculate_offload_device_map(
    MODEL_ID,
    reserve_for_hessians=True,
    num_gpus=2,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


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

SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16"


oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
)
model.save_pretrained(SAVE_DIR)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")


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
