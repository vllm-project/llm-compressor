from torch.nn import Module


def _get_config_metadata(model: Module, field: str):
    """Read a config metadata field, preferring text_config when available."""
    config = getattr(model, "config", None)
    if config is None:
        return None

    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, field, None) if text_config is not None else None
    if value is not None:
        return value

    return getattr(config, field, None)


def get_hybrid_attention_config(model: Module) -> tuple[list[str], int] | None:
    """
    Extract layer_types and num_hidden_layers from a model with hybrid attention.

    Checks both top-level config and text_config (for VL models like Qwen3.5).
    Returns ``(layer_types, num_hidden_layers)`` or ``None`` if the model does not
    expose the metadata needed to identify hybrid attention layers.
    """
    layer_types = get_config_layer_types(model)
    num_layers = _get_config_metadata(model, "num_hidden_layers")

    if layer_types is None or num_layers is None:
        return None

    has_full = "full_attention" in layer_types
    has_linear = "linear_attention" in layer_types
    if not (has_full and has_linear):
        return None

    return layer_types, num_layers


def get_config_layer_types(model: Module) -> list[str] | None:
    """
    Extract layer_types from a model config, checking text_config first when present.
    """
    return _get_config_metadata(model, "layer_types")
