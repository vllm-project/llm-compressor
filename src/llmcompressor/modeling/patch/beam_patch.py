from contextlib import contextmanager

import torch
from compressed_tensors.utils import patch_attr
from transformers.models.beam import modeling_beam


def forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """Traceable equivalent of ``BeamModel.forward``.

    Transformers' implementation builds the per-layer-type position embeddings and
    masks inside dictionaries. LLM Compressor's FX tracer wraps those blocks into
    opaque functions, so mutations to the dictionaries are lost during tracing.
    This version keeps the same computation but avoids that in-place mutation.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        past_key_values = modeling_beam.DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if position_ids is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        position_ids = (
            torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            + past_seen_tokens
        )
        position_ids = position_ids.unsqueeze(0)

    if isinstance(attention_mask, dict):
        full_mask = attention_mask["full_attention"]
        sliding_mask = attention_mask["sliding_attention"]
    else:
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        full_mask = modeling_beam.create_causal_mask(**mask_kwargs)
        sliding_mask = modeling_beam.create_sliding_window_causal_mask(**mask_kwargs)

    sliding_position_embeddings = self.rotary_emb(
        inputs_embeds, position_ids, "sliding_attention"
    )
    sliding_attention_scaling = None
    full_position_embeddings = None
    full_attention_scaling = modeling_beam.compute_attention_scaling(position_ids)

    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(
        self.layers[: self.config.num_hidden_layers]
    ):
        layer_type = self.config.layer_types[layer_idx]
        if layer_type == "sliding_attention":
            position_embeddings = sliding_position_embeddings
            attention_scaling = sliding_attention_scaling
            layer_attention_mask = sliding_mask
        else:
            position_embeddings = full_position_embeddings
            attention_scaling = full_attention_scaling
            layer_attention_mask = full_mask

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            attention_scaling=attention_scaling,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return modeling_beam.MoeModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=past_key_values
    )


@contextmanager
def patch_beam_for_tracing():
    """Make Beam's model forward traceable by LLM Compressor's FX tracer."""
    with patch_attr(modeling_beam.BeamModel, "forward", forward):
        yield
