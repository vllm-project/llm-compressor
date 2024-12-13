import os
import shutil
import tempfile
import unittest

import torch
from compressed_tensors.quantization.lifecycle import KVCacheScaleType
from compressed_tensors.quantization.utils.helpers import iter_named_quantizable_modules
from datasets import load_dataset
from parameterized import parameterized_class
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor.core import reset_session
from llmcompressor.transformers import oneshot
from tests.testing_utils import parse_params

NUM_CALIBRATION_SAMPLES = 16
MAX_SEQUENCE_LENGTH = 512
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

KV_CACHE_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/kv_cache/configs/kv_cache"
)
KV_CACHE_GPTQ_CONFIGS_DIRECTORY = (
    "tests/llmcompressor/transformers/kv_cache/configs/gptq_kv_cache"
)


@parameterized_class(parse_params(KV_CACHE_CONFIGS_DIRECTORY))
class TestKVCache(unittest.TestCase):
    model = None

    @classmethod
    def setUpClass(cls):
        cls.num_bits = 8
        cls._type = "float"
        cls.strategy = "tensor"
        cls.dynamic = False
        cls.symmetric = True

        cls.recipe = f"""
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    kv_cache_scheme:
                        num_bits: {cls.num_bits}
                        type: {cls._type}
                        strategy: {cls.strategy}
                        dynamic: {cls.dynamic}
                        symmetric: {cls.symmetric}
        """

        cls.used_args = dict(
            num_bits=cls.num_bits,
            _type=cls._type,
            strategy=cls.strategy,
            dynamic=cls.dynamic,
            symmetric=cls.symmetric,
        )

        cls.oneshot_args = dict(
            dataset="open_platypus",
            recipe=cls.recipe,
            num_calibration_samples=16,
        )

        cls.model_args = {}
        output_dir = os.path.join(
            tempfile.gettempdir(),
            "output",
            cls.model.split(os.path.sep)[-1].replace("-", "_"),
        )
        cls.oneshot_args["output_dir"] = output_dir
        cls.model_args[output_dir] = cls.used_args
        oneshot(model=cls.model, **cls.oneshot_args)
        reset_session()

    def test_kv_cache_config_format(self):
        for output_dir, used_args in self.model_args.items():
            config = AutoConfig.from_pretrained(output_dir)
            quant_config = config.quantization_config
            self.assertIsNotNone(quant_config)
            self.assertIsNotNone(quant_config["kv_cache_scheme"])

            # check the values match
            kv_cache_scheme = quant_config["kv_cache_scheme"]
            self.assertEqual(kv_cache_scheme["num_bits"], used_args["num_bits"])
            self.assertEqual(kv_cache_scheme["type"], used_args["_type"])
            self.assertEqual(kv_cache_scheme["strategy"], used_args["strategy"])
            self.assertEqual(kv_cache_scheme["dynamic"], used_args["dynamic"])
            self.assertEqual(kv_cache_scheme["symmetric"], used_args["symmetric"])

    def test_kv_cache_model_state_dict_attr(self):
        for output_dir in self.model_args.keys():
            model = AutoModelForCausalLM.from_pretrained(output_dir)

            counts = 0
            for name, submodule in iter_named_quantizable_modules(
                model, include_children=False, include_attn=True
            ):
                counts += 1
                self.assertIn("self_attn", name)
                self.assertTrue(hasattr(submodule, KVCacheScaleType.VALUE.value))
                self.assertTrue(hasattr(submodule, KVCacheScaleType.KEY.value))

            self.assertGreater(counts, 0)

    @classmethod
    def tearDownClass(cls):
        for output_dir in cls.model_args.keys():
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)


@parameterized_class(parse_params(KV_CACHE_GPTQ_CONFIGS_DIRECTORY))
class TestGPTQKVCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp_path = tempfile.TemporaryDirectory()
        cls.tmp_paths = []
        cls.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_kv_cache_model_populate_kv_scales_only(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

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

        kv_cache_num_bits = 8
        kv_cache_type = "float"
        kv_cache_strategy = "tensor"
        kv_cache_dynamic = False
        kv_cache_symmetric = True

        recipe = f"""
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    kv_cache_scheme:
                        num_bits: {kv_cache_num_bits}
                        type: {kv_cache_type}
                        strategy: {kv_cache_strategy}
                        dynamic: {kv_cache_dynamic}
                        symmetric: {kv_cache_symmetric}
        """
        output_dir = self.model_id.split("/")[-1].replace("-", "_") + "_gptq_1"

        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            output_dir=output_dir,
        )

        config = AutoConfig.from_pretrained(output_dir)
        quant_config = config.quantization_config

        self.assertIsNotNone(quant_config)
        self.assertIsNotNone(quant_config.get("kv_cache_scheme"))

        # Check the values match
        scheme = quant_config["kv_cache_scheme"]
        self.assertEqual(scheme["num_bits"], kv_cache_num_bits)
        self.assertEqual(scheme["type"], kv_cache_type)
        self.assertEqual(scheme["strategy"], kv_cache_strategy)
        self.assertEqual(scheme["dynamic"], kv_cache_dynamic)
        self.assertEqual(scheme["symmetric"], kv_cache_symmetric)

        # Check that config_group is empty - no weight, [input|output]_activations
        self.assertEqual(len(quant_config["config_groups"]), 0)

        # Check for vllm loading
        self.assertEqual(quant_config["quant_method"], "compressed-tensors")

        model = AutoModelForCausalLM.from_pretrained(output_dir)

        counts = 0
        for name, submodule in iter_named_quantizable_modules(
            model, include_children=False, include_attn=True
        ):
            self.assertIn("self_attn", name)
            self.assertTrue(hasattr(submodule, KVCacheScaleType.VALUE.value))
            self.assertTrue(hasattr(submodule, KVCacheScaleType.KEY.value))
            counts += 1

        self.assertGreater(counts, 0)

    def test_kv_cache_with_gptq(
        self,
    ):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

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

        kv_cache_num_bits = 8
        kv_cache_type = "float"
        kv_cache_strategy = "tensor"
        kv_cache_dynamic = False
        kv_cache_symmetric = True

        recipe = f"""
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    kv_cache_scheme:
                        num_bits: {kv_cache_num_bits}
                        type: {kv_cache_type}
                        strategy: {kv_cache_strategy}
                        dynamic: {kv_cache_dynamic}
                        symmetric: {kv_cache_symmetric}
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
        output_dir = self.model_id.split("/")[-1].replace("-", "_") + "_gptq_2"

        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            output_dir=output_dir,
        )

        config = AutoConfig.from_pretrained(output_dir)
        quant_config = config.quantization_config

        self.assertIsNotNone(quant_config)
        self.assertIsNotNone(quant_config.get("kv_cache_scheme"))

        # Check the values match
        scheme = quant_config["kv_cache_scheme"]
        self.assertEqual(scheme["num_bits"], kv_cache_num_bits)
        self.assertEqual(scheme["type"], kv_cache_type)
        self.assertEqual(scheme["strategy"], kv_cache_strategy)
        self.assertEqual(scheme["dynamic"], kv_cache_dynamic)
        self.assertEqual(scheme["symmetric"], kv_cache_symmetric)

        model = AutoModelForCausalLM.from_pretrained(output_dir)

        counts = 0
        for name, submodule in iter_named_quantizable_modules(
            model, include_children=False, include_attn=True
        ):
            self.assertIn("self_attn", name)
            self.assertTrue(hasattr(submodule, KVCacheScaleType.VALUE.value))
            self.assertTrue(hasattr(submodule, KVCacheScaleType.KEY.value))
            counts += 1

        self.assertGreater(counts, 0)

    @classmethod
    def tearDownClass(cls):
        for output_dir in cls.tmp_paths:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
