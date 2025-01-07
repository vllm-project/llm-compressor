from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecipeArguments:
    """Recipe and session variables"""

    recipe: Optional[str] = field(  # runner py, test_gen.py
        default=None,
        metadata={
            "help": "Path to a LLM Compressor sparsification recipe",
        },
    )
    recipe_args: Optional[List[str]] = field(  # text_gen.py
        default=None,
        metadata={
            "help": (
                "List of recipe arguments to evaluate, of the format key1=value1 "
                "key2=value2"
            )
        },
    )
    clear_sparse_session: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to clear CompressionSession data between runs."},
    )
