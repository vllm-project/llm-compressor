import pytest
import torch
from datasets import IterableDataset, load_dataset

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets import format_calibration_data, get_processed_dataset
from llmcompressor.transformers import TextGenerationDataset


@pytest.mark.unit
def test_concatenation_tokenization(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        concatenate_data=True,
    )

    wiki_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train[:5%]",
        processor=tiny_llama_tokenizer,
    )
    raw_dataset = wiki_manager.load_dataset()
    assert len(raw_dataset) > 0
    assert raw_dataset.split == "train[:5%]"
    assert raw_dataset.info.config_name == "wikitext-2-raw-v1"

    tokenized_dataset = wiki_manager()
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    for i in range(len(tokenized_dataset)):
        assert len(tokenized_dataset[i]["input_ids"]) == wiki_manager.max_seq_length


@pytest.mark.unit
def test_no_padding_tokenization(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(dataset="open_platypus", pad_to_max_length=False)

    op_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train[5%:7%]",
        processor=tiny_llama_tokenizer,
    )
    dataset = op_manager.load_dataset()  # load
    dataset = op_manager.map(  # preprocess
        dataset,
        op_manager.preprocess,
        batched=False,
        num_proc=op_manager.dataset_args.preprocessing_num_workers,
    )
    dataset = op_manager.rename_columns(dataset)  # rename
    assert len(dataset) > 0
    ex_item = dataset[0]["text"]
    assert "Below is an instruction that describes a task" in ex_item

    assert dataset.split == "train[5%:7%]"
    tokenized_dataset = op_manager()
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    print(tokenized_dataset[0]["input_ids"])

    for i in range(len(tokenized_dataset)):
        assert len(tokenized_dataset[i]["input_ids"]) <= op_manager.max_seq_length


@pytest.mark.unit
def test_max_seq_len_clipped(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(dataset="open_platypus", max_seq_length=4096)

    op_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train[95%:]",
        processor=tiny_llama_tokenizer,
    )

    assert op_manager.max_seq_length == tiny_llama_tokenizer.model_max_length


@pytest.mark.unit
def test_dataset_kwargs_and_percent(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(
        dataset="wikitext",
        raw_kwargs={
            "data_files": {"train": "wikitext-2-raw-v1/train-00000-of-00001.parquet"}
        },
    )

    c4_manager_a = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train[5%:6%]",
        processor=tiny_llama_tokenizer,
    )
    raw_dataset_a = c4_manager_a.load_dataset()

    c4_manager_b = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train[6%:8%]",
        processor=tiny_llama_tokenizer,
    )
    raw_dataset_b = c4_manager_b.load_dataset()

    assert len(raw_dataset_b) == 2 * len(raw_dataset_a)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset_key, dataset_config, split, do_concat",
    [
        ["gsm8k", "main", "train[:5%]", True],
        ["ultrachat_200k", "default", "train_sft[:1%]", False],
    ],
)
def test_datasets(tiny_llama_tokenizer, dataset_key, dataset_config, split, do_concat):
    dataset_args = DatasetArguments(
        dataset=dataset_key,
        dataset_config_name=dataset_config,
        concatenate_data=do_concat,
    )
    manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split=split,
        processor=tiny_llama_tokenizer,
    )
    raw_dataset = manager.load_dataset()
    assert len(raw_dataset) > 0
    assert raw_dataset.split == split
    assert raw_dataset.info.config_name == dataset_config

    tokenized_dataset = manager()
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    for i in range(len(tokenized_dataset)):
        if do_concat:
            assert len(tokenized_dataset[i]["input_ids"]) == manager.max_seq_length
        else:
            assert len(tokenized_dataset[i]["input_ids"]) <= manager.max_seq_length


@pytest.mark.skip("Dataset load broken on Hugging Face")
@pytest.mark.unit
def test_evol(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(
        dataset="evolcodealpaca",
        dataset_config_name=None,
        concatenate_data=False,
    )

    evol_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train[:2%]",
        processor=tiny_llama_tokenizer,
    )
    raw_dataset = evol_manager.load_dataset()
    assert len(raw_dataset) > 0
    assert raw_dataset.split == "train[:2%]"

    tokenized_dataset = evol_manager()
    assert "input_ids" in tokenized_dataset.features
    assert "labels" in tokenized_dataset.features
    for i in range(len(tokenized_dataset)):
        assert len(tokenized_dataset[i]["input_ids"]) <= evol_manager.max_seq_length


@pytest.mark.unit
def test_stream_loading(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        concatenate_data=True,
        streaming=True,
    )

    manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split="train",
        processor=tiny_llama_tokenizer,
    )

    processed = manager()
    assert isinstance(processed, IterableDataset)
    with pytest.raises(TypeError):
        # in streaming mode we don't know the length of the dataset
        _ = len(processed)

    # confirm tokenization of streamed item works correctly
    item = next(iter(processed))
    assert "labels" in item
    assert len(item["input_ids"]) <= manager.max_seq_length


@pytest.mark.unit
@pytest.mark.parametrize("split_def", ["train[95%:]", {"train": "train[:5%]"}])
def test_split_loading(split_def, tiny_llama_tokenizer):
    dataset_args = DatasetArguments(
        dataset="open_platypus",
        splits=split_def,
    )

    dataset = get_processed_dataset(
        dataset_args=dataset_args, processor=tiny_llama_tokenizer
    )

    assert dataset is not None
    assert isinstance(dataset, dict)


@pytest.fixture
def open_platypus_dataset():
    num_calibration_samples = 64
    dataset = load_dataset("garage-bAInd/Open-Platypus")["train"]
    dataset = dataset.shuffle(seed=42).select(range(num_calibration_samples))
    return dataset


def test_load_tokenized_data(open_platypus_dataset, tiny_llama_tokenizer):
    num_calibration_samples = len(open_platypus_dataset)
    max_seq_length = 512

    def preprocess(sample):
        concat_text = "INPUT: " + sample.get("input", "")
        concat_text += "INSTRUCTIONS: " + sample.get("instruction", "")
        concat_text += "OUTPUT: " + sample.get("output", "")

        return tiny_llama_tokenizer(
            concat_text, padding=False, max_length=max_seq_length, truncation=True
        )

    tokenized_dataset = open_platypus_dataset.map(
        preprocess, remove_columns=["input", "output", "instruction", "data_source"]
    )

    dataset_args = DatasetArguments(
        dataset=tokenized_dataset, shuffle_calibration_samples=False
    )

    dataset = get_processed_dataset(
        dataset_args=dataset_args,
        processor=tiny_llama_tokenizer,
        do_oneshot=True,
        do_train=False,
    )
    calib_dataset = dataset["calibration"]

    assert len(calib_dataset) == num_calibration_samples
    data_cols = calib_dataset.column_names
    assert len(data_cols) == 2
    assert "input_ids" in data_cols
    assert "attention_mask" in data_cols

    # confirm turning shuffle off works

    calib_dataloader = format_calibration_data(
        tokenized_dataset=calib_dataset,
        num_calibration_samples=num_calibration_samples,
        do_shuffle=dataset_args.shuffle_calibration_samples,
    )
    assert len(calib_dataloader) == num_calibration_samples
    dataloader_sample = next(iter(calib_dataloader))["input_ids"]
    diff = dataloader_sample - torch.Tensor(calib_dataset[0]["input_ids"])
    assert torch.sum(diff) == 0
