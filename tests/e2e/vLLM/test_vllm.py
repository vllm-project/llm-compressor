import unittest

from datasets import load_dataset
from parameterized import parameterized_class
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from tests.testing_utils import parse_params, requires_gpu, requires_torch

CONFIGS_DIRECTORY = "tests/e2e/vLLM/configs"


@requires_gpu
@requires_torch
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestvLLM(unittest.TestCase):
    model = None
    scheme = None
    dataset_id = None
    dataset_split = None

    def test_vllm(self):
        print("========== RUNNING ==============")
        print(self.scheme)

        MODEL_ID = self.model
        prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]
        oneshot_kwargs = {}
        # Load model.
        model = SparseAutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype="auto"
        )
        oneshot_kwargs["model"] = model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                )
            }

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                add_special_tokens=False,
            )

        if self.dataset_id:
            NUM_CALIBRATION_SAMPLES = 512
            MAX_SEQUENCE_LENGTH = 2048

            ds = load_dataset(self.dataset_id, split=self.dataset_split)
            ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
            ds = ds.map(preprocess)
            ds = ds.map(tokenize, remove_columns=ds.column_names)
            oneshot_kwargs["dataset"] = ds
            oneshot_kwargs["max_seq_length"] = MAX_SEQUENCE_LENGTH
            oneshot_kwargs["num_calibration_samples"] = NUM_CALIBRATION_SAMPLES

        oneshot_kwargs["recipe"] = QuantizationModifier(
            targets="Linear", scheme=self.scheme, ignore=["lm_head"]
        )

        # Apply quantization.
        oneshot(**oneshot_kwargs)

        # Confirm generations of the quantized model look sane.
        print("========== SAMPLE GENERATION ==============")
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
            "cuda"
        )
        output = model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0]))
        print("==========================================")

        # Save to disk in compressed-tensors format.
        SAVE_DIR = MODEL_ID.split("/")[1] + f"-{self.scheme}"
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

        # Run vLLM with saved model
        sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
        llm = LLM(model=SAVE_DIR)
        outputs = llm.generate(prompts, sampling_params)
        print(outputs)
        assert outputs
