import math
from typing import TYPE_CHECKING, Optional, Union

from llmcompressor.args import DatasetArguments, TrainingArguments
from llmcompressor.core import State
from llmcompressor.core.llmcompressor.utils import add_dataclass_annotations
from llmcompressor.datasets.utils import get_processed_dataset
from llmcompressor.transformers.finetune.trainer import Trainer
from llmcompressor.typing import DatasetType

if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator


class HFSFTMixin:
    state: State
    train_dataset: Optional[DatasetType] = None
    eval_dataset: Optional[DatasetType] = None
    train_data_collator: Optional["DataCollator"] = None

    @add_dataclass_annotations(DatasetArguments)
    def set_train_dataset(self, dataset: Union[str, DatasetType], **kwargs):
        dataset_args = DatasetArguments(dataset=dataset, **kwargs)

        self.train_dataset = get_processed_dataset(
            dataset_args, self.state.processor, add_labels=True
        )

    @add_dataclass_annotations(DatasetArguments)
    def set_eval_dataset(self, dataset: Union[str, DatasetType], **kwargs):
        dataset_args = DatasetArguments(dataset=dataset, **kwargs)

        self.eval_dataset = get_processed_dataset(
            dataset_args, self.state.processor, add_labels=True
        )

    @add_dataclass_annotations(TrainingArguments)
    def train(self, **kwargs):
        args = TrainingArguments(**kwargs)

        # TODO: warn if max_seq_len conflicts between dataset and training args

        # train model
        trainer = Trainer(
            model=self.state.model,
            args=args,
            data_collator=self.train_data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.state.processor,
        )
        checkpoint = args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # save metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.train_dataset)
        metrics["perplexity"] = math.exp(metrics["train_loss"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # save model
        trainer.save_model(output_dir=args.output_dir)
