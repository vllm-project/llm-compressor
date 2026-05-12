from transformers import AutoModelForCausalLM
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.modeling.moe.linearize import load_linearized_moe

# def test_linearize_moe_model(tmp_dir):
#     input_ids = # random 2048 tokens

#     with skip_weights_download(skip_init=False):
#         model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", num_hidden_layers=3, num_hash_layers=1, num_nextn_predict_layers=1)

#     true_outputs = model(*input_ids)
#     model.save_pretrained(tmp_dir)
#     del model

#     with load_linearized_moe():
#         linearized_model = AutoModelForCausalLM.from_pretrained(tmp_dir)

#     # check calibrate_all_experts=True
#     with ...:
#         linearized_model = model(*input_ids)
#         assert linearized_model == true_outputs

#     # check calibrate_all_experts=False
#     with ...:
#         linearized_model = model(*input_ids)
#         assert linearized_model == true_outputs
    
    
#     linearized_model.save_pretrained(model)
#     # check checkpoint keys for experts.N.up_proj structure


with skip_weights_download(skip_init=False):
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", num_hidden_layers=3, num_hash_layers=1, num_nextn_predict_layers=1)
    