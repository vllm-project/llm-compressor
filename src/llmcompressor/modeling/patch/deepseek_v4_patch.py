import importlib.util
import os

from huggingface_hub import hf_hub_download

__all__ = ["install_dsv4_chat_template"]


# DeepSeek-V4 does not ship a Jinja `chat_template`; instead it renders conversations
# with a custom `encode_messages()` helper (see the model's `encoding/encoding_dsv4.py`
# and https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#chat-template). The
# helpers below bridge that encoder to chat datasets, which drive calibration through
# `processor.apply_chat_template(...)`.
def _load_dsv4_encoder(model_path: str):
    """Import the `encode_messages` helper bundled with the DeepSeek-V4 checkpoint."""
    encoder_path = os.path.join(model_path, "encoding", "encoding_dsv4.py")
    if not os.path.exists(encoder_path):
        # `model_path` is a hub repo id. Re-uploads of DeepSeek-V4 may not carry the
        # `encoding/` folder over, so fetch the helper from the checkpoint it ships with
        encoder_path = hf_hub_download(
            repo_id=model_path, filename="encoding/encoding_dsv4.py"
        )
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_messages(messages):
    """Reshape raw chat messages for DeepSeek-V4's `encode_messages`.

    * assistant `thought` -> `reasoning_content` (rendered inside thinking blocks
      in thinking mode). SWE-smith trajectories store identical text in `thought`
      and `content`, so the duplicated `content` is dropped to avoid emitting the
      deliberation twice.
    * assistant `tool_calls` are already OpenAI-style and pass through unchanged.
    * `tool` observations keep the OpenAI `tool` role (with a scalar `tool_call_id`);
      `encode_messages` merges them into the preceding user turn as `<tool_result>`.
    * list-of-parts content (e.g. tool observations) is flattened to plain text.
    """
    normalized = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n\n".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )

        if role == "assistant":
            thought = msg.get("thought")
            new_msg = {
                "role": "assistant",
                "reasoning_content": thought or "",
                "content": (
                    ""
                    if thought is not None and content == thought
                    else (content or "")
                ),
            }
            if msg.get("tool_calls"):
                new_msg["tool_calls"] = msg["tool_calls"]
            normalized.append(new_msg)
        elif role == "tool":
            tool_call_ids = msg.get("tool_call_ids") or []
            normalized.append(
                {
                    "role": "tool",
                    "content": content or "",
                    "tool_call_id": (
                        tool_call_ids[0]
                        if tool_call_ids
                        else msg.get("tool_call_id", "")
                    ),
                }
            )
        else:  # system / user
            normalized.append({"role": role, "content": content or ""})

    return normalized


def install_dsv4_chat_template(
    tokenizer, model_path, thinking_mode="thinking", reasoning_effort="low"
):
    """Route `tokenizer.apply_chat_template` through DeepSeek-V4's `encode_messages`.

    DeepSeek-V4 has no Jinja chat template, so this installs one backed by the
    checkpoint's `encode_messages` helper. This lets chat datasets render
    conversations (including agent trajectories with reasoning + tool calls + tool
    observations) in the format the model was trained on.

    :param thinking_mode: ``"thinking"`` renders assistant turns with reasoning
        (``thought`` / ``reasoning_content``) inside thinking blocks;
        ``"chat"`` renders assistant content directly
    :param reasoning_effort: thinking-mode effort prompt, one of ``"low"``
        (default, adds nothing), ``"high"``, or ``"max"``

    Calibration renders complete conversations, so `drop_thinking=False` keeps every
    reasoning turn. `encode_messages` inserts the trailing ``<｜Assistant｜>``
    generation prompt itself based on message transitions, so `add_generation_prompt`
    is a no-op.
    """
    encoder = _load_dsv4_encoder(model_path)

    def apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        text = encoder.encode_messages(
            _normalize_messages(conversation),
            thinking_mode=thinking_mode,
            drop_thinking=False,
            reasoning_effort=reasoning_effort,
        )
        if tokenize:
            return tokenizer(text, **kwargs)["input_ids"]
        return text

    tokenizer.apply_chat_template = apply_chat_template
    return tokenizer
