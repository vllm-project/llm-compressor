import os
import torch
from transformers import AutoModelForCausalLM
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.modeling.moe.linearize import load_linearized_moe

def test_linearize_moe_model(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    input_ids = torch.randint(1024, size=(1, 64), device="cuda")

    with skip_weights_download(skip_init=False):  # TODO: test with cpu offloading
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V4-Flash",
            num_hidden_layers=3,
            default_num_hash_layers=1,
            num_nextn_predict_layers=1,
            device_map="cuda",
        )

    true_outputs = model(input_ids=input_ids).logits
    model.save_pretrained(offload_dir)
    del model

    # with load_linearized_moe():
    #     linearized_model = AutoModelForCausalLM.from_pretrained(tmp_dir)

    # # check calibrate_all_experts=True
    # with ...:
    #     linearized_model = model(*input_ids)
    #     assert linearized_model == true_outputs

    # # check calibrate_all_experts=False
    # with ...:
    #     linearized_model = model(*input_ids)
    #     assert linearized_model == true_outputs
    
    
    # linearized_model.save_pretrained(model)
    # # check checkpoint keys for experts.N.up_proj structure
