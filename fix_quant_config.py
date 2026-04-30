import json
import re
import sys


def rename_key(key: str) -> str:
    if not key.startswith("model."):
        return key

    key = key[len("model."):]

    key = re.sub(
        r"\.compressor\.indexer\.(ape|wgate|wkv)", r".indexer.compressor.\1", key
    )
    key = key.replace(".compressor.indexer.kv_norm.", ".indexer.compressor.norm.")
    key = re.sub(
        r"\.compressor\.indexer\.(weights_proj|wq_b)", r".indexer.\1", key
    )
    key = key.replace(".compressor.kv_norm.", ".compressor.norm.")
    key = key.replace(".self_attn.", ".attn.")
    key = key.replace(".mlp.", ".ffn.")
    key = key.replace(".input_layernorm.", ".attn_norm.")
    key = key.replace(".post_attention_layernorm.", ".ffn_norm.")
    key = key.replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
    key = key.replace(".shared_experts.up_proj.", ".shared_experts.w3.")
    key = key.replace(".shared_experts.down_proj.", ".shared_experts.w2.")

    return key


def fix_target_regex(target: str) -> str:
    if not target.startswith("re:"):
        return rename_key(target)

    regex = target[3:]
    # group_0: "re:model.*attn.*(wkv|wo_a|wo_b|wq_a|wq_b)$"
    #        → "re:.*attn.*(wkv|wo_a|wo_b|wq_a|wq_b)$"
    # group_0: "re:model.*attn\\.compressor.*(wgate|wkv)$"
    #        → "re:.*attn\\.compressor.*(wgate|wkv)$"
    # group_1: "re:model.*mlp.*(gate|up|down)_proj$"
    #        → "re:.*ffn.*(gate|up|down)_proj$"
    regex = regex.replace("model.*", ".*", 1)
    regex = regex.replace("mlp", "ffn")
    return "re:" + regex


def main():
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = json.load(f)

    qc = config["quantization_config"]

    for group in qc["config_groups"].values():
        group["targets"] = [fix_target_regex(t) for t in group["targets"]]

    qc["ignore"] = [rename_key(k) for k in qc["ignore"]]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print("Updated quantization_config in", config_path)
    print()
    for name, group in qc["config_groups"].items():
        print(f"  {name} targets: {group['targets']}")
    print(f"  ignore: {qc['ignore'][:3]}... ({len(qc['ignore'])} total)")


if __name__ == "__main__":
    main()
