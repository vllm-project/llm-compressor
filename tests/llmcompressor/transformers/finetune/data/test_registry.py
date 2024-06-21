import pytest

from llmcompressor.transformers.finetune.data import (
    C4Dataset,
    OpenPlatypusDataset,
    TextGenerationDataset,
    WikiTextDataset,
)
from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_c4_initializes(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(dataset="c4", concatenate_data=True)
    c4_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split=None,
        tokenizer=tiny_llama_tokenizer,
    )
    assert isinstance(c4_manager, TextGenerationDataset)
    assert isinstance(c4_manager, C4Dataset)
    assert c4_manager.text_column == "text"
    assert not c4_manager.padding
    assert c4_manager.max_seq_length == data_args.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_wikitext_initializes(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(
        dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
    )
    wiki_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split=None,
        tokenizer=tiny_llama_tokenizer,
    )
    assert isinstance(wiki_manager, TextGenerationDataset)
    assert isinstance(wiki_manager, WikiTextDataset)
    assert wiki_manager.text_column == "text"
    assert wiki_manager.padding == "max_length"
    assert wiki_manager.max_seq_length == data_args.max_seq_length


@pytest.mark.usefixtures("tiny_llama_tokenizer")
def test_open_platypus_initializes(tiny_llama_tokenizer):
    data_args = DataTrainingArguments(dataset="open_platypus", pad_to_max_length=False)
    op_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split=None,
        tokenizer=tiny_llama_tokenizer,
    )
    assert isinstance(op_manager, TextGenerationDataset)
    assert isinstance(op_manager, OpenPlatypusDataset)
    assert op_manager.text_column == "text"
    assert not op_manager.padding
    assert op_manager.max_seq_length == data_args.max_seq_length
