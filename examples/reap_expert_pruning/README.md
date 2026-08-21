# Mixture of Experts (MoE) Compression with REAP Expert Pruning

## Algorithm

REAP stands for Router-weighted Expert Activation Pruning, introduced in [REAP the Experts: Why Pruning Prevails for
One-Shot MoE Compression](https://arxiv.org/pdf/2510.13999). The algorithm reduces the VRAM footprint of models by structurally removing individual experts in their MoE layers. In order to determine which experts to remove, the algorithm uses a calibration dataset, tracking router and expert outputs that construct a saliency score. The equation for the REAP saliency score is as follows:

```
S_j = (1/N_j) ∑ (g_j(t) * ||f_j(t)||_2)
```

where the sum is over the N_j tokens routed to expert j, and:

- `g_j(t)` is the router gate weight assigned to expert `j` for token `t` (the coefficient that multiplies the expert's output when combining experts)
- `f_j(t)` is expert `j`'s output activation for token `t`
- `||f_j(t)||_2` is the L2 norm of that activation
- `N_j` is the total number of tokens routed to expert `j` during calibration

As the paper describes, saliency is a heuristic for the relevance of an expert in its respective layer. Therefore, REAP proceeds after calibration by removing the experts with the lowest saliency in each layer, up to a certain sparsity.

## Usage

REAP is useful for reducing the VRAM footprint of MoE models without sacrificing model quality. When applied, it can allow models to fit on less total GPUs, drastically reducing inference costs.

However, excess pruning can lead to a degradation in model quality, as shown in the results below. Additionally, the dataset choice for calibration is important. If calibration samples leave out certain features or topics, it is possible that experts normally relevant for these samples will be marked as irrelevant overall and removed. Exercise care and know your application before pruning.

## Results

The end-to-end examples in `examples/reap_expert_pruning` were used to compress two different models. 

The first, `Qwen/Qwen3-30B-A3B-Instruct-2507`, is a 30.5B parameter model, with only 3.3B parameters active during inference. It has 48 layers with 128 experts per layer. Only 8 experts are activated per token.

The second, `moonshotai/Moonlight-16B-A3B-Instruct`, is a relatively smaller 16B parameter model, with 3B parameters active during inference. It has 27 layers with 64 experts per layer. 6 experts are activated per token. The model uses a DeepSeek-V3 architecture with a group-limited router. This is supported by the `llm-compressor` REAP implementation.

To test, both models were pruned to 25% and 50% sparsity. Pruning for each was calibrated with 512 samples (max token length 2048) from the `train_sft` split of HuggingFace's `ultrachat_200k` dataset.

Then, the compressed models were evaluated on the `gsm8k_platinum_cot_llama` task from [LM Eval](https://github.com/EleutherAI/lm-evaluation-harness). Evaluation was done with the `local_chat_completions` backend of LM Eval, sending requests to a vLLM server. `MAX_LENGTH` was set to 8192 tokens and `MAX_GEN_TOKS` to 4096. For the Qwen models, the generation config parameters of `temperature=0.7, top_p=0.8, top_k=20` were used. For the Moonlight models, reasonable defaults of `temperature=0.7, top_p=0.9, top_k=0` were used. The task was executed with an `N_SHOT` of 5, and the results were averaged across seeds 1234-1236.

| Model | Config | Flexible | Strict | Recovery (Strict) | Size (GiB) |
|-------|--------|----------|--------|-------------------|------------|
| Qwen3-30B-A3B (128e) | baseline | 0.9672 | 0.9672 | - | 56.93 |
|  | REAP 25% -> 96 | 0.9559 | 0.9562 | 98.86% | 43.43 |
|  | REAP 50% -> 64 | 0.9653 | 0.9653 | 99.80% | 29.92 |
|       |        |          |        |                   |            |
| Moonlight-16B-A3B (DeepSeek-V3 arch, 64e) | baseline | 0.8348 | 0.8326 | - | 29.88 |
|  | REAP 25% -> 48 | 0.6647 | 0.6642 | 79.77% | 23.17 |
|  | REAP 50% -> 32 | 0.1368 | 0.1414 | 16.99% | 16.47 |

As can be seen, the Qwen3 model has very high recovery, both at 25% and 50% sparsity. The smaller Moonlight model has less redundancy across its experts, so it has a modest recovery at 25% sparsity and an extremely poor recovery at 50% sparsity. This demonstrates that REAP is only effective to the extent to which the original model's experts are redundant.