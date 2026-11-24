from contextlib import contextmanager

from compressed_tensors.utils import patch_attr


@contextmanager
def patch_llama4_image_outputs():
    """
    Work around a transformers>=5.24.0 bug in ``Llama4ForConditionalGeneration``.

    The multimodal forward builds its output with
    ``image_outputs=outputs.image_outputs``, where ``outputs`` is the text
    ``Llama4ForCausalLM`` output (a ``MoeCausalLMOutputWithPast``). That output
    class never defines an ``image_outputs`` field and no output recorder
    populates one, so *every* forward -- text-only or multimodal -- raises
    ``AttributeError: 'MoeCausalLMOutputWithPast' object has no attribute
    'image_outputs'``. This breaks both calibration and sequential tracing.

    This context sets a class-level ``image_outputs`` default of ``None`` on the
    text output class so the attribute read resolves instead of raising. A
    class-level default is used (rather than wrapping the forward) because it
    also holds during FX symbolic tracing, where the wrapped forward is not
    re-invoked.

    This is a workaround for an upstream transformers bug; it can be removed once
    ``Llama4ForConditionalGeneration.forward`` no longer reads a missing
    ``image_outputs`` attribute from the text model output.
    """
    # imported lazily so the patch module can be imported without llama4 present
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

    with patch_attr(MoeCausalLMOutputWithPast, "image_outputs", None):
        yield
