import unittest

import pytest
from datasets import IterableDataset, load_dataset
from parameterized import parameterized

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.transformers.finetune.runner import StageRunner
from llmcompressor.transformers.finetune.training_args import TrainingArguments
from tests.testing_utils import requires_torch


@pytest.mark.unit
class TestConcentrationTokenization(unittest.TestCase):
    def setUp(self):
        self.data_args = DataTrainingArguments(
            dataset="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            concatenate_data=True,
        )

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    def test_concatenation_tokenization(self):
        wiki_manager = TextGenerationDataset.load_from_registry(
            self.data_args.dataset,
            data_args=self.data_args,
            split="train[:5%]",
            processor=self.tiny_llama_tokenizer,
        )
        raw_dataset = wiki_manager.load_dataset()
        self.assertGreater(len(raw_dataset), 0)
        self.assertEqual(raw_dataset.split, "train[:5%]")
        self.assertEqual(raw_dataset.info.config_name, "wikitext-2-raw-v1")
        tokenized_dataset = wiki_manager()
        self.assertIn("input_ids", tokenized_dataset.features)
        self.assertIn("labels", tokenized_dataset.features)
        for i in range(len(tokenized_dataset)):
            self.assertEqual(
                len(tokenized_dataset[i]["input_ids"]), wiki_manager.max_seq_length
            )


@pytest.mark.unit
class TestNoPaddingTokenization(unittest.TestCase):
    def setUp(self):
        self.data_args = DataTrainingArguments(
            dataset="open_platypus", pad_to_max_length=False
        )

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    @pytest.mark.usefixtures("tiny_llama_tokenizer")
    def test_no_padding_tokenization(self):
        op_manager = TextGenerationDataset.load_from_registry(
            self.data_args.dataset,
            data_args=self.data_args,
            split="train[5%:10%]",
            processor=self.tiny_llama_tokenizer,
        )
        dataset = op_manager.load_dataset()  # load
        dataset = op_manager.map(  # preprocess
            dataset,
            op_manager.preprocess,
            batched=False,
            num_proc=op_manager.data_args.preprocessing_num_workers,
        )
        dataset = op_manager.rename_columns(dataset)  # rename
        self.assertGreater(len(dataset), 0)
        ex_item = dataset[0]["text"]
        self.assertIn("Below is an instruction that describes a task", ex_item)

        self.assertEqual(dataset.split, "train[5%:10%]")
        tokenized_dataset = op_manager()
        self.assertIn("input_ids", tokenized_dataset.features)
        self.assertIn("labels", tokenized_dataset.features)
        print(tokenized_dataset[0]["input_ids"])

        for i in range(len(tokenized_dataset)):
            self.assertLessEqual(
                len(tokenized_dataset[i]["input_ids"]), op_manager.max_seq_length
            )


@pytest.mark.unit
class TestMaxSeqLenClipped(unittest.TestCase):
    def setUp(self):
        self.data_args = DataTrainingArguments(
            dataset="open_platypus", max_seq_length=4096
        )

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    def test_max_seq_len_clipped(self):
        op_manager = TextGenerationDataset.load_from_registry(
            self.data_args.dataset,
            data_args=self.data_args,
            split="train[80%:]",
            processor=self.tiny_llama_tokenizer,
        )

        self.assertEqual(
            op_manager.max_seq_length, self.tiny_llama_tokenizer.model_max_length
        )


@pytest.mark.unit
class TestDatasetKwargsAndPercent(unittest.TestCase):
    def setUp(self):
        self.data_args = DataTrainingArguments(
            dataset="wikitext",
            raw_kwargs={
                "data_files": {
                    "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet"
                }
            },
        )

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    def test_dataset_kwargs_and_percentages(self):
        c4_manager_a = TextGenerationDataset.load_from_registry(
            self.data_args.dataset,
            data_args=self.data_args,
            split="train[5%:10%]",
            processor=self.tiny_llama_tokenizer,
        )
        raw_dataset_a = c4_manager_a.load_dataset()

        c4_manager_b = TextGenerationDataset.load_from_registry(
            self.data_args.dataset,
            data_args=self.data_args,
            split="train[5%:15%]",
            processor=self.tiny_llama_tokenizer,
        )
        raw_dataset_b = c4_manager_b.load_dataset()

        self.assertEqual(len(raw_dataset_b), 2 * len(raw_dataset_a))


@pytest.mark.unit
class TestDatasets(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    @parameterized.expand(
        [
            ["ptb", "penn_treebank", "train[:5%]", False],
            ["gsm8k", "main", "train[:5%]", True],
            ["ultrachat_200k", "default", "train_sft[:2%]", False],
        ]
    )
    def test_datasets(self, dataset_key, dataset_config, split, do_concat):
        data_args = DataTrainingArguments(
            dataset=dataset_key,
            dataset_config_name=dataset_config,
            concatenate_data=do_concat,
            trust_remote_code_data=True,
        )
        manager = TextGenerationDataset.load_from_registry(
            data_args.dataset,
            data_args=data_args,
            split=split,
            processor=self.tiny_llama_tokenizer,
        )
        raw_dataset = manager.load_dataset()
        self.assertGreater(len(raw_dataset), 0)
        self.assertEqual(raw_dataset.split, split)
        self.assertEqual(raw_dataset.info.config_name, dataset_config)

        tokenized_dataset = manager()
        self.assertIn("input_ids", tokenized_dataset.features)
        self.assertIn("labels", tokenized_dataset.features)
        for i in range(len(tokenized_dataset)):
            if do_concat:
                self.assertEqual(
                    len(tokenized_dataset[i]["input_ids"]), manager.max_seq_length
                )
            else:
                self.assertLessEqual(
                    len(tokenized_dataset[i]["input_ids"]), manager.max_seq_length
                )


@pytest.mark.skip("Dataset load broken on Hugging Face")
@pytest.mark.unit
class TestEvol(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    def setUp(self):
        self.data_args = DataTrainingArguments(
            dataset="evolcodealpaca",
            dataset_config_name=None,
            concatenate_data=False,
        )

    def test_evol(self):
        evol_manager = TextGenerationDataset.load_from_registry(
            self.data_args.dataset,
            data_args=self.data_args,
            split="train[:2%]",
            processor=self.tiny_llama_tokenizer,
        )
        raw_dataset = evol_manager.load_dataset()
        self.assertGreater(len(raw_dataset), 0)
        self.assertEqual(raw_dataset.split, "train[:2%]")

        tokenized_dataset = evol_manager()
        self.assertIn("input_ids", tokenized_dataset.features)
        self.assertIn("labels", tokenized_dataset.features)
        for i in range(len(tokenized_dataset)):
            self.assertLessEqual(
                len(tokenized_dataset[i]["input_ids"]), evol_manager.max_seq_length
            )


@pytest.mark.unit
class TestStreamLoading(unittest.TestCase):
    def setUp(self):
        self.data_args = DataTrainingArguments(
            dataset="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            concatenate_data=True,
            streaming=True,
        )

    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    def test_stream_loading(self):
        manager = TextGenerationDataset.load_from_registry(
            self.data_args.dataset,
            data_args=self.data_args,
            split="train",
            processor=self.tiny_llama_tokenizer,
        )

        processed = manager()
        self.assertIsInstance(processed, IterableDataset)
        with pytest.raises(TypeError):
            # in streaming mode we don't know the length of the dataset
            _ = len(processed)

        # confirm tokenization of streamed item works correctly
        item = next(iter(processed))
        self.assertIn("labels", item)
        self.assertEqual(len(item["input_ids"]), manager.max_seq_length)


@pytest.mark.unit
class TestSplitLoading(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer

    @parameterized.expand(
        [["train"], ["train[60%:]"], [{"train": "train[:20%]"}], [None]]
    )
    def test_split_loading(self, split_def):
        from llmcompressor.transformers.finetune.model_args import ModelArguments

        data_args = DataTrainingArguments(
            dataset="open_platypus",
            splits=split_def,
            trust_remote_code_data=True,
        )
        training_args = TrainingArguments(do_train=True, output_dir="dummy")
        model_args = ModelArguments(model=None)
        stage_runner = StageRunner(
            model_args=model_args, data_args=data_args, training_args=training_args
        )
        stage_runner.populate_datasets(processor=self.tiny_llama_tokenizer)

        train_dataset = stage_runner.get_dataset_split("train")
        assert train_dataset is not None
        self.assertIsInstance(train_dataset[0], dict)


@requires_torch
@pytest.mark.unit
class TestTokenizationDataset(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def prepare_fixture(self, tiny_llama_tokenizer):
        self.tiny_llama_tokenizer = tiny_llama_tokenizer
        dataset = load_dataset("garage-bAInd/Open-Platypus")["train"]
        self.num_calib_samples = 256
        self.max_seq_len = 512
        self.dataset = dataset.shuffle(seed=42).select(range(self.num_calib_samples))

    def test_load_tokenized_data(self):
        import torch

        from llmcompressor.transformers.finetune.data.data_helpers import (
            format_calibration_data,
        )

        def preprocess(sample):
            concat_text = "INPUT: " + sample.get("input", "")
            concat_text += "INSTRUCTIONS: " + sample.get("instruction", "")
            concat_text += "OUTPUT: " + sample.get("output", "")

            return self.tiny_llama_tokenizer(
                concat_text, padding=False, max_length=self.max_seq_len, truncation=True
            )

        tokenized_dataset = self.dataset.map(
            preprocess, remove_columns=["input", "output", "instruction", "data_source"]
        )
        stage_runner = StageRunner(
            model_args=None,
            data_args=DataTrainingArguments(
                dataset=tokenized_dataset, shuffle_calibration_samples=False
            ),
            training_args=TrainingArguments(do_oneshot=True),
        )
        stage_runner.populate_datasets(processor=None)
        calib_dataset = stage_runner.get_dataset_split("calibration")
        self.assertEqual(len(calib_dataset), self.num_calib_samples)
        data_cols = calib_dataset.column_names
        self.assertEqual(len(data_cols), 2)
        self.assertIn("input_ids", data_cols)
        self.assertIn("attention_mask", data_cols)

        # confirm turning shuffle off works
        calib_dataloader = format_calibration_data(
            tokenized_dataset=calib_dataset,
            num_calibration_samples=self.num_calib_samples,
            do_shuffle=stage_runner._data_args.shuffle_calibration_samples,
        )
        self.assertEqual(len(calib_dataloader), self.num_calib_samples)
        dataloader_sample = next(iter(calib_dataloader))["input_ids"]
        diff = dataloader_sample - torch.Tensor(calib_dataset[0]["input_ids"])
        self.assertEqual(torch.sum(diff), 0)
