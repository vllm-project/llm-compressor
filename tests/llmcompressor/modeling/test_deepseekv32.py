import inspect

from llmcompressor.modeling.deepseekv32.model import Transformer


def test_transformer_forward_does_not_use_inference_mode():
    """
    Regression test for https://github.com/vllm-project/llm-compressor/issues/2745

    Transformer.forward was originally decorated with @torch.inference_mode(),
    copied from the upstream inference-only reference implementation. Once
    llm-compressor calibrates this model, GPTQ and the offload cache need to
    write compressed weights back in place, which crashes on inference
    tensors ("Inplace update to inference tensor outside InferenceMode is not
    allowed"). @torch.no_grad() gives the same gradient-disabling behavior
    without that restriction.
    """
    source = inspect.getsource(Transformer.forward)
    assert "inference_mode" not in source
