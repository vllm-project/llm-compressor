from __future__ import annotations

import gc
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-8B"
SCHEME = "W2A16"
PROMPT_MESSAGES = [{"role": "user", "content": "Introduce yourself in one sentence."}]
MAX_NEW_TOKENS = 100
OUTPUT_NOTE = Path(__file__).with_name("qwen3_w2a16_rtn_sweep.md")

TARGET_GROUPS = {
    "qkv": ["re:.*self_attn\\.(q|k|v)_proj$"],
    "o": ["re:.*self_attn\\.o_proj$"],
    "mlp": ["re:.*mlp\\.(gate|up|down)_proj$"],
}

VARIANTS = [
    ("dense", []),
    ("qkv", ["qkv"]),
    ("o", ["o"]),
    ("mlp", ["mlp"]),
    ("qkv_o", ["qkv", "o"]),
    ("qkv_mlp", ["qkv", "mlp"]),
    ("o_mlp", ["o", "mlp"]),
    ("qkv_o_mlp", ["qkv", "o", "mlp"]),
]


def flatten_targets(enabled_groups: list[str]) -> list[str]:
    targets: list[str] = []
    for group in enabled_groups:
        targets.extend(TARGET_GROUPS[group])
    return targets


def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def generate_text(model, tokenizer) -> str:
    dispatch_model(model)
    prompt = tokenizer.apply_chat_template(
        PROMPT_MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )
    sample = tokenizer(prompt, return_tensors="pt")
    sample = {key: value.to(model.device) for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.decode(output[0])


def run_variant(name: str, enabled_groups: list[str]) -> str:
    print(f"\n===== {name} =====")
    print(f"enabled_groups={enabled_groups or ['none']}")
    model, tokenizer = load_model_and_tokenizer()
    try:
        if enabled_groups:
            recipe = QuantizationModifier(
                targets=flatten_targets(enabled_groups),
                scheme=SCHEME,
                ignore=["lm_head"],
            )
            oneshot(model=model, precision="auto", recipe=recipe)
        generated = generate_text(model, tokenizer)
        print(generated)
        return generated
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_note(results: list[tuple[str, list[str], str]]):
    prompt = PROMPT_MESSAGES[0]["content"]
    lines = [
        "# Qwen3-8B W2A16 RTN 2-Bit Sweep",
        "",
        "- Model: `Qwen/Qwen3-8B`",
        "- Scheme: `W2A16`",
        "- Method: `QuantizationModifier` RTN",
        "- Preset group size: `128`",
        f"- Prompt: `{prompt}`",
        "",
        "## Variants",
        "",
        "| Variant | QKV | O | MLP |",
        "| --- | --- | --- | --- |",
    ]

    for name, enabled_groups, _text in results:
        enabled = set(enabled_groups)
        lines.append(
            f"| `{name}` | {'on' if 'qkv' in enabled else 'off'} | "
            f"{'on' if 'o' in enabled else 'off'} | "
            f"{'on' if 'mlp' in enabled else 'off'} |"
        )

    lines.extend(["", "## Generations", ""])
    for name, enabled_groups, text in results:
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- Enabled groups: `{', '.join(enabled_groups) if enabled_groups else 'none'}`",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- This document records qualitative generations only; it is not an accuracy benchmark.",
            "- Each variant is run from a fresh model load; no checkpoints are saved.",
            "",
        ]
    )

    OUTPUT_NOTE.write_text("\n".join(lines))
    print(f"\nWrote {OUTPUT_NOTE}")


def main():
    results: list[tuple[str, list[str], str]] = []
    for name, enabled_groups in VARIANTS:
        text = run_variant(name, enabled_groups)
        results.append((name, enabled_groups, text))
    write_note(results)


if __name__ == "__main__":
    main()
