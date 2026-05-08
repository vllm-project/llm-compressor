import torch
from datasets import load_dataset
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from compressed_tensors.offload import init_dist, load_offloaded_model

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "google/gemma-4-31B-it"

# Load model.
init_dist()
with load_offloaded_model():
    model = Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

# MoE calibration is handled automatically by the pipeline.
# The `SequentialGemma4TextExperts` modules (from `llmcompressor.modeling.gemma4`)
# will be applied to enable proper expert handling and vLLM compatibility.

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*embed.*",
        "re:.*router",
        "re:.*vision_tower.*",
    ],
)
DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messgages = []
    for message in example["messages"]:
        messgages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messgages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)


"""
## Serve gemma-4 full-precision
chg run -g 2 -- vllm serve google/gemma-4-31B-it --port 9000 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --tensor-parallel-size=2 \
  --max-num-batched-tokens 8192 \
  --served-model-name=gemma-4-31B-it \
  --chat-template ~/projects/vllm/examples/tool_chat_template_gemma4.jinja \
  --reasoning-parser gemma4 \
  --max-model-len 200000 \
  --gpu-memory-utilization 0.90

## Serve gemma-4 W4A16
chg run -g 2 -- vllm serve ~/projects/gemma-4-31B-it-W4A16 --port 9001 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --tensor-parallel-size=2 \
  --max-num-batched-tokens 8192 \
  --served-model-name=gemma-4-31B-it-W4A16 \
  --chat-template ~/projects/vllm/examples/tool_chat_template_gemma4.jinja \
  --reasoning-parser gemma4 \
  --max-model-len 200000 \
  --gpu-memory-utilization 0.90

## SSH port forward (on local machine)
ssh -L 9000:localhost:9000 -L 9001:localhost:9001 brian-dellabetta@a100-08

## Sample curl command
curl http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-31B-it",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
curl http://localhost:9001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-31B-it-W4A16",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'

## Launch Claude Code
CLAUDE_CODE_USE_VERTEX=0 CLAUDE_CODE_AUTO_COMPACT_WINDOW=400000 \
  ANTHROPIC_BASE_URL=http://localhost:8000 \
  ANTHROPIC_DEFAULT_OPUS_MODEL=gemma-4-31B-it \
  ANTHROPIC_DEFAULT_SONNET_MODEL=gemma-4-31B-it \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=gemma-4-31B-it \
  ANTHROPIC_AUTH_TOKEN=dummy \
  claude

## Run oneshot 
chg run -g 3 -- torchrun --nproc-per-node=3 /home/brian-dellabetta/projects/llm-compressor/examples/quantization_w4a16/gemma4_example.py
"""
