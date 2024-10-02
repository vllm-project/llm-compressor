from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from
    """

    model: str = field(
        metadata={
            "help": (
                "A pretrained model or a string as a path to pretrained model, "
                "HF stub, or model identifier from huggingface.co/models."
            )
        },
    )
    distill_teacher: Optional[str] = field(
        default=None,
        metadata={
            "help": "Teacher model (a trained text generation model)",
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained data from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizers. Default True"},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use "
            "(can be a branch name, tag name or commit id)"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)"
        },
    )
    precision: str = field(
        default="auto",
        metadata={"help": "Precision to cast model weights to, default to auto"},
    )

    tie_word_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Whether the model's input and output word embeddings "
            "should be tied. Note that this is only relevant if the "
            "model has a output word embedding layer."
        },
    )
    trust_remote_code_model: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models to execute their "
            "own modeling files. This option should only be set to True for "
            "repositories you trust and in which you have read the code"
        },
    )
