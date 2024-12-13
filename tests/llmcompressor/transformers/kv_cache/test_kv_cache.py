import os
from pathlib import Path

import pytest
from accelerate import init_empty_weights
from compressed_tensors.quantization.lifecycle import KVCacheScaleType
from compressed_tensors.quantization.utils.helpers import iter_named_quantizable_modules
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor.core import reset_session
from llmcompressor.transformers import oneshot

NUM_CALIBRATION_SAMPLES = 16
MAX_SEQUENCE_LENGTH = 512
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

KV_CACHE_GPTQ_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/kv_cache/configs/gptq_kv_cache"
)

MODEL_IDS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
]


@pytest.fixture(scope="session")
def oneshot_fixture():
    def _oneshot_fixture(tmp_path: Path):
        num_bits = 8
        _type = "float"
        strategy = "tensor"
        dynamic = False
        symmetric = True
        recipe = f"""
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    kv_cache_scheme:
                        num_bits: {num_bits}
                        type: {_type}
                        strategy: {strategy}
                        dynamic: {dynamic}
                        symmetric: {symmetric}
        """
        used_args = dict(
            num_bits=num_bits,
            _type=_type,
            strategy=strategy,
            dynamic=dynamic,
            symmetric=symmetric,
        )
        oneshot_args = dict(
            dataset="open_platypus",
            recipe=recipe,
            num_calibration_samples=16,
        )
        for model_id in MODEL_IDS:
            oneshot_args["output_dir"] = os.path.join(tmp_path, model_id)
            used_args["output_dir"] = oneshot_args["output_dir"]
            yield oneshot(model=model_id, **oneshot_args), used_args

    return _oneshot_fixture


def test_kv_cache_config_format(oneshot_fixture, tmp_path):
    _, used_args = next(oneshot_fixture(tmp_path))
    output_dir = used_args["output_dir"]
    config = AutoConfig.from_pretrained(str(output_dir))
    quant_config = config.quantization_config
    assert quant_config is not None
    assert quant_config["kv_cache_scheme"] is not None

    kv_cache_scheme = quant_config["kv_cache_scheme"]
    assert kv_cache_scheme["num_bits"] == used_args["num_bits"]
    assert kv_cache_scheme["type"] == used_args["_type"]
    assert kv_cache_scheme["strategy"] == used_args["strategy"]
    assert kv_cache_scheme["dynamic"] == used_args["dynamic"]
    assert kv_cache_scheme["symmetric"] == used_args["symmetric"]


def test_kv_cache_model_state_dict_attr(oneshot_fixture, tmp_path):
    model, used_args = next(oneshot_fixture(tmp_path))
    output_dir = used_args["output_dir"]
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(str(output_dir))

    counts = 0
    for name, submodule in iter_named_quantizable_modules(
        model, include_children=False, include_attn=True
    ):
        counts += 1
        assert "self_attn" in name
        assert hasattr(submodule, KVCacheScaleType.VALUE.value)
        assert hasattr(submodule, KVCacheScaleType.KEY.value)
    assert counts > 0


@pytest.fixture(scope="session")
def kv_cache_fixture():
    def _kv_cache_fixture(recipe: str, tmp_path: Path):
        num_bits = 8
        _type = "float"
        strategy = "tensor"
        dynamic = False
        symmetric = True

        recipe = recipe.format(
            num_bits=num_bits,
            _type=_type,
            strategy=strategy,
            dynamic=dynamic,
            symmetric=symmetric,
        )

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
        ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                )
            }

        ds = ds.map(preprocess)

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names)

        output_dir = os.path.join(tmp_path, model_id[-1].replace("-", "_"))

        oneshot_args = dict(
            model=model_id,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            output_dir=output_dir,
        )

        oneshot(**oneshot_args)
        reset_session()

        yield (
            output_dir,
            {
                "num_bits": num_bits,
                "_type": _type,
                "strategy": strategy,
                "dynamic": dynamic,
                "symmetric": symmetric,
                "output_dir": output_dir,
            },
        )

    return _kv_cache_fixture


def test_kv_cache_gptq_config_format(kv_cache_fixture, tmp_path):
    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                kv_cache_scheme:
                    num_bits: {num_bits}
                    type: {_type}
                    strategy: {strategy}
                    dynamic: {dynamic}
                    symmetric: {symmetric}
    """

    output_dir, used_args = next(kv_cache_fixture(recipe, tmp_path))

    config = AutoConfig.from_pretrained(output_dir)
    quant_config = config.quantization_config
    assert quant_config is not None
    assert quant_config.get("kv_cache_scheme") is not None

    kv_cache_scheme = quant_config["kv_cache_scheme"]
    assert kv_cache_scheme["num_bits"] == used_args["num_bits"]
    assert kv_cache_scheme["type"] == used_args["_type"]
    assert kv_cache_scheme["strategy"] == used_args["strategy"]
    assert kv_cache_scheme["dynamic"] == used_args["dynamic"]
    assert kv_cache_scheme["symmetric"] == used_args["symmetric"]

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(output_dir)

    counts = 0
    for name, submodule in iter_named_quantizable_modules(
        model, include_children=False, include_attn=True
    ):
        counts += 1
        assert "self_attn" in name
        assert hasattr(submodule, KVCacheScaleType.VALUE.value)
        assert hasattr(submodule, KVCacheScaleType.KEY.value)

    assert counts > 0


def test_kv_cache_gptq_model_state_dict_attr(kv_cache_fixture, tmp_path):
    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                kv_cache_scheme:
                    num_bits: {num_bits}
                    type: {_type}
                    strategy: {strategy}
                    dynamic: {dynamic}
                    symmetric: {symmetric}
            GPTQModifier:
                sequential_update: false
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 4
                            type: "int"
                            symmetric: true
                            strategy: "channel"
                            actorder: False
                        targets: ["Linear"]
    """

    output_dir, _ = next(kv_cache_fixture(recipe, tmp_path))

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(output_dir)

    counts = 0
    for name, submodule in iter_named_quantizable_modules(
        model, include_children=False, include_attn=True
    ):
        counts += 1
        assert "self_attn" in name
        assert hasattr(submodule, KVCacheScaleType.VALUE.value)
        assert hasattr(submodule, KVCacheScaleType.KEY.value)

    assert counts > 0
