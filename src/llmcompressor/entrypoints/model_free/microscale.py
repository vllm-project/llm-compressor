import torch
from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy

__all__ = ["get_fused_names", "is_microscale_scheme"]


def is_microscale_scheme(scheme: QuantizationScheme) -> bool:
    assert scheme.weights is not None
    return scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP


def get_fused_names(tensors: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    fused_names = {}

    for name in tensors:
        parts = name.rsplit(".")
        if len(parts) < 3:
            continue

        parent, module, param = parts[-3:]

        if (
            ("attn" in parent or "attention" in parent)
            and module == "q_proj"
            and param == "weight"
        ):
            parent_name = ".".join((*parts[:-3], parent))
            q_name = ".".join((parent_name, "q_proj", param))
            k_name = ".".join((parent_name, "k_proj", param))
            v_name = ".".join((parent_name, "v_proj", param))

            submodule_names = [q_name, k_name, v_name]

            if all(name in tensors for name in submodule_names):
                assert parent_name not in fused_names
                fused_names[parent_name] = submodule_names

        if "mlp" in parent and module == "gate_proj" and param == "weight":
            parent_name = ".".join((*parts[:-3], parent))
            gate_name = ".".join((parent_name, "gate_proj", param))
            up_name = ".".join((parent_name, "up_proj", param))

            submodule_names = [gate_name, up_name]

            if all(name in tensors for name in submodule_names):
                assert parent_name not in fused_names
                fused_names[parent_name] = submodule_names

    return fused_names
