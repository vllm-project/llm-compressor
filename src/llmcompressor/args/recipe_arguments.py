"""
Recipe argument classes for LLM compression workflows.

Defines dataclass-based argument containers for configuring sparsification
recipes, compression sessions, and stage-based execution parameters used in
model compression and optimization workflows.
"""

from dataclasses import dataclass, field


@dataclass
class RecipeArguments:
    """Recipe and session variables"""

    recipe: str | None = field(
        default=None,
        metadata={
            "help": "Path to a LLM Compressor sparsification recipe",
        },
    )
    recipe_args: list[str] | None = field(
        default=None,
        metadata={
            "help": (
                "List of recipe arguments to evaluate, of the format key1=value1 "
                "key2=value2"
            )
        },
    )
    clear_sparse_session: bool | None = field(
        default=False,
        metadata={
            "help": (
                "Whether to clear CompressionSession/CompressionLifecycle ",
                "data between runs.",
            )
        },
    )
    stage: str | None = field(
        default=None,
        metadata={"help": ("The stage of the recipe to use for oneshot / train.",)},
    )
