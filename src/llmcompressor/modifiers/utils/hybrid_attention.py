from torch.nn import Module


def get_hybrid_attention_config(model: Module) -> tuple[list[str], int] | None:
    """
    Extract layer_types and num_hidden_layers from a model with hybrid attention.

    Checks both top-level config and text_config (for VL models like Qwen3.5).
    Returns ``(layer_types, num_hidden_layers)`` or ``None`` if the model does not
    expose the metadata needed to identify hybrid attention layers.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)

    if layer_types is None or num_layers is None:
        return None

    has_full = "full_attention" in layer_types
    has_linear = "linear_attention" in layer_types
    if not (has_full and has_linear):
        return None

    return layer_types, num_layers


def get_hybrid_attention_layer_types(model: Module) -> list[str] | None:
    """
    Extract layer_types from a model config, checking text_config first when present.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    text_config = getattr(config, "text_config", config)
    return getattr(text_config, "layer_types", None)
