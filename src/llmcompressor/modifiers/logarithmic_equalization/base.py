import warnings

from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

__all__ = ["LogarithmicEqualizationModifier"]


class LogarithmicEqualizationModifier(SmoothQuantModifier):
    """
    .. deprecated::
        ``LogarithmicEqualizationModifier`` is deprecated and will be removed in a
        future release. Use ``SmoothQuantModifier`` with
        ``algorithm="log_equalization"``
        instead::

            SmoothQuantModifier:
              algorithm: log_equalization
              mappings: [...]

    Implements the Logarithmic Equalization Algorithm from
    https://arxiv.org/abs/2308.15987. This modifier is now an alias for
    ``SmoothQuantModifier(algorithm="log_equalization")``.
    """

    def __init__(self, **kwargs):
        warnings.warn(
            "LogarithmicEqualizationModifier is deprecated and will be removed in a "
            "future release. Use SmoothQuantModifier with "
            "algorithm='log_equalization' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("algorithm", "log_equalization")
        super().__init__(**kwargs)
