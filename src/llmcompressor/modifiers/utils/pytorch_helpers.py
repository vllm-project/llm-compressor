from torch.nn import Module

__all__ = [
    "is_moe_model",
]


def is_moe_model(model: Module) -> bool:
    """
    Check if the model is a mixture of experts model

    :param model: the model to check
    :return: True if the model is a mixture of experts model
    """

    # Check for MoE components
    for _, module in model.named_modules():
        module_name = module.__class__.__name__
        if "MoE" in module_name or "Expert" in module_name:
            return True

    # Check config for MoE attributes
    if hasattr(model, "config"):
        if any(
            "moe" in attr.lower() or "expert" in attr.lower()
            for attr in dir(model.config)
        ):
            return True

    return False
