import re
from typing import List

from loguru import logger

from llmcompressor.args import DatasetArguments, TrainingArguments
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.transformers.finetune.data.data_helpers import make_dataset_splits
from llmcompressor.typing import Processor


def get_processed_dataset(
    data_args: "DatasetArguments",
    training_args: "TrainingArguments",
    processor: Processor,
    add_labels: bool = True,
):
    """
    Loads datasets for each flow based on data_args, stores a Dataset for each
    enabled flow in datasets

    :param processor: processor or tokenizer to use for dataset tokenization
    :param add_labels: if True, add labels column to dataset splits
    """
    if data_args.dataset is None:
        logger.warning(
            "Running oneshot without calibration data. This is expected for "
            "weight-only and dynamic quantization"
        )
        return

    splits = data_args.splits
    tokenized_datasets = {}

    def _get_split_name(inp_str):
        # strip out split name, for ex train[60%:] -> train
        match = re.match(r"(\w*)\[.*\]", inp_str)
        if match is not None:
            return match.group(1)
        return inp_str

    if splits is None:
        splits = {"all": None}
    elif isinstance(splits, str):
        splits = {_get_split_name(splits): splits}
    elif isinstance(splits, List):
        splits = {_get_split_name(s): s for s in splits}

    # default to custom dataset if dataset provided isn't a string
    registry_id = data_args.dataset if isinstance(data_args.dataset, str) else "custom"
    for split_name, split_str in splits.items():
        dataset = data_args.dataset
        if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
            # dataset is already tokenized
            tokenized_datasets[split_name] = dataset
        else:
            # dataset needs to be tokenized
            dataset_manager = TextGenerationDataset.load_from_registry(
                registry_id,
                data_args=data_args,
                split=split_str,
                processor=processor,
            )
            tokenized_datasets[split_name] = dataset_manager(add_labels=add_labels)

    return make_dataset_splits(
        tokenized_datasets,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        do_predict=training_args.do_predict,
    )
