import pytest

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.finetune.data import (
    C4Dataset,
    OpenPlatypusDataset,
    TextGenerationDataset,
    WikiTextDataset,
)


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_c4_initializes(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(dataset="c4", concatenate_data=True)
    c4_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split=None,
        processor=tiny_llama_tokenizer,
    )
    assert isinstance(c4_manager, TextGenerationDataset)
    assert isinstance(c4_manager, C4Dataset)
    assert c4_manager.dataset_args.text_column == "text"
    assert not c4_manager.padding
    assert c4_manager.max_seq_length == dataset_args.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_wikitext_initializes(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(
        dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
    )
    wiki_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split=None,
        processor=tiny_llama_tokenizer,
    )
    assert isinstance(wiki_manager, TextGenerationDataset)
    assert isinstance(wiki_manager, WikiTextDataset)
    assert wiki_manager.dataset_args.text_column == "text"
    assert wiki_manager.padding == "max_length"
    assert wiki_manager.max_seq_length == dataset_args.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_open_platypus_initializes(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(dataset="open_platypus", pad_to_max_length=False)
    op_manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split=None,
        processor=tiny_llama_tokenizer,
    )
    assert isinstance(op_manager, TextGenerationDataset)
    assert isinstance(op_manager, OpenPlatypusDataset)
    assert op_manager.dataset_args.text_column == "text"
    assert not op_manager.padding
    assert op_manager.max_seq_length == dataset_args.max_seq_length
