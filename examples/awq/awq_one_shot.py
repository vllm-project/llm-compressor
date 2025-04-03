import os

import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_ID= "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_ID = "meta-llama/Llama-2-7b-hf"
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

#
# 1) LLMCOMPRESSOR quantize
#


def run_llmc_awq() -> AutoModelForCausalLM:
    OUTPUT_DIR = MODEL_ID.split("/")[-1] + f"-llmc-awq-{NUM_CALIBRATION_SAMPLES}"
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    )

    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = [
        AWQModifier(bits=4, apply_clip=False, symmetric=False),
        QuantizationModifier(
            ignore=["lm_head"],
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        type=QuantizationType.INT,
                        dynamic=False,
                        symmetric=False,
                        strategy=QuantizationStrategy.GROUP,
                        group_size=128,
                    ),
                )
            },
        ),
    ]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype="auto"
    )

    def get_calib_dataset(tokenizer):
        from datasets import load_dataset

        ds = load_dataset(
            DATASET_ID,
            split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*100}]",
        )

        def preprocess(example):
            return {
                "input_ids": tokenizer.encode(example["text"].strip())[
                    :MAX_SEQUENCE_LENGTH
                ]
            }

        ds = (
            ds.shuffle(seed=42)
            .map(preprocess, remove_columns=ds.column_names)
            .filter(lambda example: len(example["input_ids"]) >= MAX_SEQUENCE_LENGTH)
            .select(range(NUM_CALIBRATION_SAMPLES))
        )

        return ds

    oneshot(
        model=model,
        dataset=get_calib_dataset(tokenizer=tokenizer),
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print("Done! model saved to", OUTPUT_DIR)

    return model, OUTPUT_DIR


#
# 2) AUTOAWQ quantize
#


def run_auto_awq() -> AutoModelForCausalLM:
    OUTPUT_DIR = (
        MODEL_ID.split("/")[-1] + f"-auto-awq-{NUM_CALIBRATION_SAMPLES}-quant-only"
    )
    from awq import AutoAWQForCausalLM

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(MODEL_ID, device_map="cuda:0")

    # Quantize
    model.quantize(
        tokenizer,
        apply_clip=False,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        },
    )
    model = model.model.to("cuda:0")

    # Save quantized model
    model.save_quantized(OUTPUT_DIR)

    return model, OUTPUT_DIR


#
# RUN
#

# print("RUNNING AUTOAWQ")
# model, OUTPUT_DIR = run_auto_awq()
print("RUNNING LLMCAWQ")
model, OUTPUT_DIR = run_llmc_awq()

#
# EVAL
#

# NOTE: lm_eval --model vllm is failing with vllm==0.8.1 if using V1
os.environ["VLLM_USE_V1"] = "0"

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args={
        "pretrained": OUTPUT_DIR,
        "add_bos_token": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
    },
    tasks=["wikitext", "gsm8k"],
    num_fewshot=5,
    batch_size=8,
)
print("DONE", results["results"])
