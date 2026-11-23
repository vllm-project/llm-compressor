import json
import math
import os

import torch
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.moe import load_model_config
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from loguru import logger
from transformers import AutoTokenizer

__all__ = ["OverthinkPenalizer"]

# dtypes of weights which are already quantized and can no longer be penalized
_QUANTIZED_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.uint8,
    torch.int8,
    torch.int32,
}


class OverthinkPenalizer(Converter):
    """
    Bake the overthinking penalty of "Quantized Reasoning Models Think They
    Need to Think Longer, but They Do Not" (https://arxiv.org/pdf/2606.00206)
    into a checkpoint's lm_head.

    The paper shows that reasoning models, and aggressively quantized ones in
    particular, often reach a correct intermediate answer but fail to commit
    to it, instead re-opening reasoning branches with hesitation markers such
    as "wait", "but" and "alternatively". Penalizing those overthinking tokens
    with a single fixed strength is training-free, adds no inference overhead,
    and shortens chain-of-thought by 12-23% while preserving accuracy.

    The paper subtracts a fixed penalty from the logits of the overthinking
    tokens at each decoding step. A constant logit offset cannot be expressed
    in a bias-free lm_head, so this converter instead scales the lm_head rows
    of the overthinking tokens by exp(-lam), and any runtime which loads the
    checkpoint is penalized with no extra machinery. This is NOT equivalent
    to the paper's penalty and `lam` is not the paper's lambda: a logit z
    becomes z * exp(-lam), i.e. it is reduced by (1 - exp(-lam)) * z. The
    penalty therefore grows with the logit, and is largest exactly where a
    token is about to be sampled, where its logit is close to the top logit
    of the distribution (typically 20-35 for current LLMs). For example, on
    Qwen3.6-35B-A3B lam=0.5 lowers a competitive marker's logit by ~11 nats
    (the paper sweeps 0.5-4) and removes ~96% of the markers from generated
    text, and lam>=1.5 removes them entirely. On that model the penalty still
    shortened reasoning (by 15-19% on GPQA-Diamond at lam=0.5 and 2.5) without
    hurting accuracy, but because the markers are effectively banned, the
    token list must contain only tokens which are safe to ban. For
    checkpoints with tied embeddings the lm_head is untied first, so that only
    the output logits and not the input embeddings of the penalized tokens are
    modified.

    Construct with :meth:`from_pretrained` and use alone or chain it before
    quantization, e.g.

    ```python
    model_free_ptq(
        MODEL_ID,
        SAVE_DIR,
        scheme="FP8_DYNAMIC",
        converter=OverthinkPenalizer.from_pretrained(
            MODEL_ID, "overthink_tokens.json", lam=0.5
        ),
    )
    ```

    The paper's 50-token list (Table 8, every token with a leading space) is
    shipped at the repository root as `overthink_tokens.json`.

    :param token_ids: vocabulary ids of the overthinking tokens to penalize
    :param lam: penalty strength. The logits of the overthinking tokens are
        scaled by exp(-lam), so larger values penalize more strongly. Negative
        values scale the logits up and boost the tokens instead, which is much
        stronger than the paper's negative-lambda ablation (lam=-0.5 multiplies
        the logits by 1.65) and makes generations degenerate
    :param tie_word_embeddings: whether the source checkpoint ties `lm_head` to
        the embedding. If so, the saved config is untied, since the penalized
        `lm_head` and the embedding differ
    :param emit_lm_head: whether to create `lm_head.weight` from the embedding,
        for tied checkpoints which do not save `lm_head.weight`
    """

    def __init__(
        self,
        token_ids: list[int],
        lam: float,
        tie_word_embeddings: bool = False,
        emit_lm_head: bool = False,
    ):
        if lam < 0:
            logger.warning(
                f"lam={lam} scales the logits of the overthinking tokens up by "
                f"{math.exp(-lam):.2f}x, which boosts them far more strongly than "
                "the paper's negative-lambda ablation and degenerates generations"
            )
        self.token_ids = token_ids
        self.lam = lam
        self.tie_word_embeddings = tie_word_embeddings
        self.emit_lm_head = emit_lm_head

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        tokens_path: str | os.PathLike,
        lam: float,
    ) -> "OverthinkPenalizer":
        """
        Build the converter from a model stub or checkpoint path, resolving the
        overthinking tokens to vocabulary ids with the model's tokenizer.

        :param model_name_or_path: HuggingFace stub or local checkpoint path,
            used to load the tokenizer and the model config
        :param tokens_path: path to a json file containing the overthinking
            tokens, as a list of token strings such as `[" wait", " but"]`.
            Each entry must resolve to a single vocabulary entry; tokens which
            do not are skipped with a warning. A `_` prefix is read as a
            leading space, following the convention of the paper's token lists
        :param lam: penalty strength. The logits of the overthinking tokens
            are scaled by exp(-lam)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        with open(tokens_path) as file:
            tokens = json.load(file)

        token_ids = []
        for token in tokens:
            token_id = _token_to_id(tokenizer, token)
            if token_id is None:
                logger.warning(
                    f"Skipping overthinking token {token!r}: not a single "
                    f"token in the vocabulary of {model_name_or_path}"
                )
            else:
                token_ids.append(token_id)
        # a token list may contain duplicates, but each token is penalized once
        token_ids = list(dict.fromkeys(token_ids))
        if not token_ids:
            raise ValueError(
                f"No tokens from {tokens_path} were found in the vocabulary "
                f"of {model_name_or_path}"
            )

        model_files = get_checkpoint_files(model_name_or_path)
        weight_map = get_weight_map(model_files)
        config = load_model_config(model_files)
        text_config = config.get("text_config", config)
        tie_word_embeddings = config.get(
            "tie_word_embeddings", text_config.get("tie_word_embeddings", False)
        )

        has_lm_head = "lm_head.weight" in weight_map
        if not has_lm_head and "model.embed_tokens.weight" not in weight_map:
            raise ValueError(
                f"Could not find lm_head.weight or model.embed_tokens.weight in "
                f"{model_name_or_path}. OverthinkPenalizer expects a language "
                "model checkpoint with standard tensor names"
            )

        return cls(
            token_ids=token_ids,
            lam=lam,
            tie_word_embeddings=tie_word_embeddings,
            emit_lm_head=tie_word_embeddings and not has_lm_head,
        )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Scale down the lm_head weights of the overthinking tokens."""
        if "lm_head.weight" in tensors:
            _penalize(tensors["lm_head.weight"], self.token_ids, self.lam)
        elif self.emit_lm_head and "model.embed_tokens.weight" in tensors:
            # tied embeddings: untie so the penalty is applied to the output
            # head without changing the input embeddings of the tokens
            lm_head = tensors["model.embed_tokens.weight"].clone()
            _penalize(lm_head, self.token_ids, self.lam)
            tensors["lm_head.weight"] = lm_head
        return tensors

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.process(tensors)

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        return config

    def update_model_config(self, model_config: dict) -> dict:
        if self.tie_word_embeddings:
            model_config["tie_word_embeddings"] = False
            if isinstance(model_config.get("text_config"), dict):
                model_config["text_config"]["tie_word_embeddings"] = False
            logger.info("Untied word embeddings to apply the lm_head penalty")
        return model_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()


def _token_to_id(tokenizer, token: str) -> int | None:
    """
    Resolve a token string to the id of a single vocabulary entry. A `_`
    prefix is read as a leading space, the convention used by the paper's
    token lists. Leading spaces fold into the token the way the tokenizer
    expects, e.g. as "Ġ" in byte-level BPE or "▁" in sentencepiece. Returns
    None when the token is not a single vocabulary entry
    """
    if token.startswith("_"):
        token = " " + token[1:]

    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return token_id

    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) == 1 and ids[0] != tokenizer.unk_token_id:
        return ids[0]
    return None


def _penalize(weight: torch.Tensor, token_ids: list[int], lam: float):
    if weight.dtype in _QUANTIZED_DTYPES:
        raise ValueError(
            "Cannot penalize a quantized lm_head. Apply OverthinkPenalizer "
            "before quantization, or dequantize the checkpoint first"
        )
    # scale the rows toward zero: the logit of each token becomes exp(-lam)
    # times its original value. The reduction (1 - exp(-lam)) * logit is
    # largest where the token is about to be sampled, so even small lam is a
    # strong penalty there (see the class docstring)
    weight[token_ids] *= math.exp(-lam)
