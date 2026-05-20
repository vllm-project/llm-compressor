def test_fp8_compressed_inference_regex_matching():
    """Test that regex patterns in fp8_compressed_inference example match expected modules.

    This example doesn't contain explicit quantization patterns (it loads an already
    quantized model), but we test the model can be loaded with skip_weights_download.
    """
    from transformers import AutoModelForCausalLM

    from llmcompressor.utils.dev import skip_weights_download

    model_id = "nm-testing/tinyllama-fp8-dynamic-compressed"

    # Verify the model can be loaded with skip_weights_download
    with skip_weights_download():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="meta")

    assert model is not None, "Model should be loaded successfully"
