import transformers
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import numpy as np
import safetensors
from safetensors import safe_open
import os
import json
from tqdm import tqdm 



def get_stats_of_layer(tensors):

    stats_layer = {}
    for linear_ in tqdm(tensors):
        stats_layer[linear_] = {}
        stats_layer[linear_]["min"] = torch.min(tensors[linear_]).item()
        stats_layer[linear_]["max"] = torch.max(tensors[linear_]).item()
        stats_layer[linear_]["mean"] = torch.mean(tensors[linear_]).item()
        stats_layer[linear_]["median"] = torch.median(tensors[linear_]).item()
        stats_layer[linear_]["std"] = torch.std(tensors[linear_]).item()
        # float16_tensor = tensors[linear_].to(torch.float16).cpu().numpy().flatten()
        # stats_layer[linear_]["kurtosis"] = kurtosis(float16_tensor)

    return stats_layer


def store_histograms(tensors, layer, model_path, log=True):

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Histogram of Linear Operators of Layer {layer}")
    tensor_keys = sorted(list(tensors.keys()))
    for i, linear_ in enumerate(tensor_keys):
        tensor = tensors[linear_].to(torch.float16).cpu().numpy().flatten()
        axs[i//4, i%4].hist(tensor, bins=100, log=log)
        axs[i//4, i%4].set_title(linear_)

    plt.savefig(f"{model_path}/histograms/histogram_layer_{layer}.png", dpi=300)
    plt.close()

if __name__ == "__main__":

    model_id = "meta-llama/Meta-Llama-3-70B"
    weight_path = "/nm/drive0/shashata/weight-analysis/dense_llama_3_70B"
    cache_dir = "/nm/drive0/shashata/weight-analysis"
    presaved_path = f"{cache_dir}/models--{model_id.replace('/', '--')}"

    if not os.path.exists(presaved_path):
        # os.makedirs(presaved_path)
        model = SparseAutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='auto',
            torch_dtype='auto',
            cache_dir=cache_dir
        )
        model.save_pretrained(weight_path)

    linear_operators = ['mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj']
    layer_index_file = f"{weight_path}/model.safetensors.index.json"

    # open json file as dictionary
    with open(layer_index_file, "r") as f:
        layer_index = json.load(f)['weight_map']
        layer_keys = list(layer_index.keys())

        # find the max layer number
        max_layer = max([int(x.split('.')[2]) for x in layer_keys if 'layers' in x])
        print("Total Layers ->", max_layer+1)

    min_layer = 0

    stats = {}
    print("Starting to work with layers")
    for layer in range(min_layer, max_layer+1):
        print(f"Layer {layer}")
        layer_files = []
        layer_opearators = []
        layer_tensors = {}

        for op in linear_operators:
            layer_opearators.extend(x for x in layer_keys if f"layers.{layer}.{op}" in x)

        for lo in layer_opearators:
            if layer_index[lo] not in layer_files:
                layer_files.append(layer_index[lo])

        print(layer_files)
        print(layer_opearators)
        if len(layer_files) == 1:
            with safe_open(f"{weight_path}/{layer_files[0]}", 'pt', device='cpu') as f:
                for k in layer_opearators:
                    layer_tensors[k] = f.get_tensor(k)
        elif len(layer_files) > 1:
            for lf in layer_files:
                with safe_open(f"{weight_path}/{lf}", 'pt', device='cpu') as f:
                    for k in layer_opearators:
                        if k in f.keys():
                            layer_tensors[k] = f.get_tensor(k)

        
        for k in layer_tensors.keys():
            print(k, layer_tensors[k].shape)
        
        layer_stats = get_stats_of_layer(layer_tensors)
        stats.update(layer_stats)
        print(layer_stats)
        # print(stats)
        # print(stats[layer])

        store_histograms(layer_tensors, layer, weight_path, log=True)

    # save the stats using json
    with open(f"{weight_path}/model_stats.json", "w") as f:
        json.dump(stats, f)
