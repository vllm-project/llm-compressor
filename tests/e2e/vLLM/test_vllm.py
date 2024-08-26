from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from vllm import LLM, SamplingParams

import unittest

from tests.testing_utils import requires_gpu

@requires_gpu
class TestvLLM(unittest.TestCase):
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    prompts = ["The capital of France is", "The president of the US is", "My name is"]

    # Load model.
    model = SparseAutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per channel via ptq
    #   * quantize the activations to fp8 with dynamic per token
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe)

    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
    print("==========================================")

    # Save to disk in compressed-tensors format.
    SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
    llm  = LLM(model=model_path)
    outputs = llm.generate(prompts, sampling_params)
    print(outputs)
    assert output

