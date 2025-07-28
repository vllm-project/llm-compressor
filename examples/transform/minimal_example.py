import torch

from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from llmcompressor.modeling import normalize_embedding, fuse_norm_linears

from compressed_tensors.utils import is_match
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix


def transform(weight: torch.Tensor, loc: str):
    if loc == "embed_output":
        hadamard = deterministic_hadamard_matrix(weight.size(1), weight.dtype, "cuda")
        return (weight @ hadamard) / torch.tensor(hadamard.size(0)).sqrt()

    if loc == "weight_output":
        hadamard = deterministic_hadamard_matrix(weight.size(0), weight.dtype, "cuda")
        return (hadamard.T @ weight) / torch.tensor(hadamard.size(0)).sqrt()
    
    if loc == "weight_input":
        hadamard = deterministic_hadamard_matrix(weight.size(1), weight.dtype, "cuda")
        inv = hadamard.T
        return (weight @ inv.T) / torch.tensor(hadamard.size(0)).sqrt()

    assert False


def calibrate_fake_quantize(weight: torch.Tensor) -> torch.Tensor:
    # calibrate
    group_size = 128
    num_groups = weight.size(-1) // group_size
    values = weight.unflatten(-1, (num_groups, group_size))

    max_values = values.max(dim=-1).values
    min_values = values.min(dim=-1).values

    value_range = torch.maximum(max_values.abs(), min_values.abs()) * 2
    scale = value_range / (7 + 8)
    scale = scale.clamp(min=torch.finfo(torch.float32).eps)
    zero_point = torch.zeros_like(scale)

    # quantize
    x = weight
    x_q = (x.unflatten(-1, (scale.size(-1), -1)) / scale[:, :, None]) + zero_point[:, :, None]
    x_q = torch.round(x_q)
    x_q = torch.clamp(x_q, -8, 7)  # unlike current impl, round then clamp

    # dequantize
    x_qdq = (x_q - zero_point[:, :, None]) * scale[:, :, None]
    x_qdq = x_qdq.flatten(-2, -1)
    return x_qdq


def transform_and_quant(model: torch.nn.Module, do_transform=True):
    for name, module in model.named_modules():
        if is_match(name, module, "re:.*embed_tokens$"):
            transformed = transform(module.weight, "embed_output")

        elif any(is_match(name, module, t) for t in ["re:.*o_proj$", "re:.*down_proj$"]):
            transformed = transform(module.weight, "weight_output")
            
        elif any(is_match(name, module, t) for t in ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$", "re:.*up_proj$", "re:.*gate_proj$", "lm_head"]):
            transformed = transform(module.weight, "weight_input")

        else:
            continue

        quant = calibrate_fake_quantize(module.weight)
        transformed_quant = calibrate_fake_quantize(transformed)

        loss = torch.nn.MSELoss()
        with torch.no_grad():
            quant_loss = loss(quant, module.weight)
            transform_quant_loss = loss(transformed_quant, transformed)

        if not transform_quant_loss < quant_loss < 1e-05:
            print((name.rjust(32), transform_quant_loss, quant_loss))

        if "embed_tokens" in name or "lm_head" in name:
            if do_transform:
                module.weight.data = transformed
        else:
            if do_transform:
                module.weight.data = transformed_quant

            else:
                module.weight.data = quant


if __name__ == "__main__":
    # Select model and load it.
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # print("loaded model")

    #normalize_embedding(model.model.embed_tokens)
    for layer in model.model.layers:
        fuse_norm_linears(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        fuse_norm_linears(
            layer.post_attention_layernorm,
            [layer.mlp.up_proj, layer.mlp.gate_proj],
        )
    fuse_norm_linears(
        model.model.norm,
        [model.lm_head],
    )
    print("normalized embeddings and fused norms")

    transform_and_quant(model, do_transform=False)
    print("transformed and quanted")

    SAVE_DIR = MODEL_ID.split("/")[1] + "-minimal-zero-no-embed-norm"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {key: value.to("cuda") for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")