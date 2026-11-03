---
title: "NVFP4 sweep on Llama-3.1-8B-Instruct, including long-context MRCR"
tags:
  - NVFP4
  - GPTQ
  - AWQ
  - AutoRound
  - SmoothQuant
  - Llama-3.1-8B
  - calibration
  - long-context
---

<div class="entry-byline" markdown>
[![dsikka](https://github.com/dsikka.png?size=64){ .entry-avatar }](https://github.com/dsikka)
**[@dsikka](https://github.com/dsikka)** posted on **2026-07-13**, updated **2026-08-06**
</div>

# Llama-3.1-8B-Instruct NVFP4 Evaluation Results

Baseline: `RedHatAI/Llama-3.1-8B-Instruct` (FP16)  
Seeds: 1234, 2345, 3456  
Generation params: temperature=0.6, top_p=0.9  
Serving: vLLM v0.25.1, TP=1, max-model-len=20480  

## Summary

- GTPQ models were calibrated with Ultrachat and Perfect Blend
- Autoround models were calibrated using the autoround default dataset NeelNanda/pile-10k and a second set of models were calibrated using Ultrachat for an easier comparison
- An attempt at MRCR results are below; note: the task was missing from MLR's fork so claude added it based on other longcontext evals available 

Scripts: https://github.com/vllm-project/llm-compressor/compare/nvfp4_scripts?expand=1

| Model | Top-2 finishes | Tasks |
|-------|---------------|-------|
| NVFP4-GPTQ-damp05-perf-blend | 4 / 5 | gsm8k_platinum_cot_llama, mmlu_pro_chat, ifeval, math_500 |
| NVFP4-GPTQ | 2 / 5 | mmlu_cot_llama, ifeval |
| NVFP4-GPTQ-perf-blend-more-data | 2 / 5 | gsm8k_platinum_cot_llama, math_500 |
| NVFP4-GPTQ-AWQ-imatrix-perf-blend | 2 / 5 | mmlu_pro_chat, math_500 |
| NVFP4-GPTQ-imatrix-perf-blend | 2 / 5 | gsm8k_platinum_cot_llama, mmlu_cot_llama |
| NVFP4-GPTQ-perf-blend | 1 / 5 | mmlu_pro_chat |
| NVFP4-AutoRound-AWQ | 1 / 5 | ifeval |

**GPTQ calibration data:** Perf-blend variants all match or beat base GPTQ (94.2%), while fineweb (93.2%) and red-pajama (93.0%) hurt recovery due to math_500 drops. damp05-perf-blend is the new best overall GPTQ variant (95.6%), +1.0pp over the original perf-blend. perf-blend-more-data is second (95.0%). MSE loss and lower damping are neutral.

**GPTQ vs AutoRound:** Base GPTQ leads base AutoRound by 2.3pp (94.2% vs 91.9%), ahead on all 5 tasks. Best AutoRound (AWQ-Ultrachat-More-Iter, 93.8%) still trails best GPTQ (damp05-perf-blend, 95.6%).

**GPTQ with/without AWQ:** AWQ hurts GPTQ overall (94.2% → 92.6%), mainly due to a 7.8pp math_500 drop. Perf-blend recovers most of this (92.6% → 94.0%). Imatrix + perf-blend combined yields 94.5%.

## Top GPTQ Variants (Recovery)

| Model | gsm8k | mmlu_cot | mmlu_pro | ifeval | math_500 | Average |
|-------|-------|----------|----------|--------|----------|---------|
| NVFP4-GPTQ | 0.9498 | 0.9592 | 0.8969 | 0.9888 | 0.9153 | 0.9420 |
| NVFP4-GPTQ-perf-blend | 0.9638 | 0.9587 | 0.9147 | 0.9800 | 0.9125 | 0.9459 |
| NVFP4-GPTQ-damp001-perf-blend | 0.9612 | 0.9543 | 0.9096 | 0.9628 | 0.9266 | 0.9429 |
| NVFP4-GPTQ-damp05-perf-blend | 0.9736 | 0.9580 | **0.9163** | **0.9927** | **0.9393** | **0.9560** |
| NVFP4-GPTQ-mse-perf-blend | 0.9688 | 0.9569 | 0.9012 | 0.9824 | 0.9012 | 0.9421 |
| NVFP4-GPTQ-perf-blend-more-data | 0.9704 | 0.9577 | 0.9109 | 0.9800 | 0.9294 | 0.9497 |
| NVFP4-GPTQ-imatrix-perf-blend | **0.9762** | **0.9594** | 0.9089 | 0.9771 | 0.9040 | 0.9451 |
| NVFP4-GPTQ-AWQ-imatrix-perf-blend | 0.9647 | 0.9497 | 0.9118 | 0.9859 | 0.9139 | 0.9452 |

## gsm8k_platinum_cot_llama

Metric: `exact_match,strict-match`

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|-------|------|-----------|-----------|-----------|----------|
| RedHatAI/Llama-3.1-8B-Instruct (FP16) | 0.8671 | 0.8677 | 0.8635 | 0.8701 | -  |
| NVFP4 | 0.8103 | 0.7990 | 0.8139 | 0.8180 | 0.9345 |
| NVFP4-AWQ | 0.8246 | 0.8296 | 0.8213 | 0.8230 | 0.9510 |
| NVFP4-SmoothQuant | 0.8170 | 0.8040 | 0.8280 | 0.8189 | 0.9422 |
| NVFP4-GPTQ | 0.8235 | 0.8147 | 0.8321 | 0.8238 | 0.9498 |
| NVFP4-GPTQ-perf-blend | 0.8357 | 0.8371 | 0.8404 | 0.8296 | 0.9638 |
| NVFP4-GPTQ-damp001-perf-blend | 0.8335 | 0.8280 | 0.8313 | 0.8412 | 0.9612 |
| NVFP4-GPTQ-damp05-perf-blend | 0.8442 | 0.8462 | 0.8371 | 0.8495 | **0.9736** |
| NVFP4-GPTQ-mse-perf-blend | 0.8401 | 0.8329 | 0.8478 | 0.8395 | 0.9688 |
| NVFP4-GPTQ-perf-blend-more-data | 0.8415 | 0.8379 | 0.8304 | 0.8561 | **0.9704** |
| NVFP4-GPTQ-AWQ | 0.8332 | 0.8172 | 0.8462 | 0.8362 | 0.9609 |
| NVFP4-GPTQ-AWQ-imatrix | 0.8282 | 0.8280 | 0.8329 | 0.8238 | 0.9552 |
| NVFP4-GPTQ-AWQ-perf-blend | 0.8349 | 0.8387 | 0.8346 | 0.8313 | 0.9628 |
| NVFP4-GPTQ-AWQ-imatrix-perf-blend | 0.8365 | 0.8296 | 0.8453 | 0.8346 | 0.9647 |
| NVFP4-GPTQ-imatrix | 0.8208 | 0.8230 | 0.8213 | 0.8180 | 0.9466 |
| NVFP4-GPTQ-imatrix-no-static | 0.8211 | 0.8139 | 0.8371 | 0.8122 | 0.9469 |
| NVFP4-GPTQ-imatrix-perf-blend | 0.8464 | 0.8478 | 0.8470 | 0.8445 | **0.9762** |
| NVFP4-GPTQ-BlockSize16 | 0.8340 | 0.8313 | 0.8371 | 0.8337 | 0.9619 |
| NVFP4-GPTQ-SmoothQuant | 0.8175 | 0.8156 | 0.8089 | 0.8280 | 0.9428 |
| NVFP4-GPTQ-fineweb | 0.8224 | 0.8114 | 0.8230 | 0.8329 | 0.9485 |
| NVFP4-GPTQ-red-pajama | 0.8340 | 0.8321 | 0.8379 | 0.8321 | 0.9619 |
| NVFP4-AutoRound | 0.8134 | 0.8073 | 0.8139 | 0.8189 | 0.9380 |
| NVFP4-AutoRound-Ultrachat | 0.8188 | 0.8114 | 0.8304 | 0.8147 | 0.9443 |
| NVFP4-AutoRound-AWQ | 0.8252 | 0.8238 | 0.8321 | 0.8197 | 0.9517 |
| NVFP4-AutoRound-AWQ-Ultrachat | 0.8238 | 0.8172 | 0.8346 | 0.8197 | 0.9501 |
| NVFP4-AutoRound-AWQ-Ultrachat-More-Iter | 0.8128 | 0.8056 | 0.8205 | 0.8122 | 0.9373 |
| NVFP4-AutoRound-SmoothQuant | 0.8219 | 0.8172 | 0.8288 | 0.8197 | 0.9479 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat | 0.8202 | 0.8131 | 0.8271 | 0.8205 | 0.9460 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat-More-Iter | 0.8169 | 0.8147 | 0.8255 | 0.8106 | 0.9421 |

## mmlu_cot_llama

Metric: `exact_match,strict_match`

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|-------|------|-----------|-----------|-----------|----------|
| RedHatAI/Llama-3.1-8B-Instruct (FP16) | 0.7274 | 0.7272 | 0.7255 | 0.7295 | -  |
| NVFP4 | 0.6840 | 0.6784 | 0.6833 | 0.6902 | 0.9403 |
| NVFP4-AWQ | 0.6866 | 0.6834 | 0.6880 | 0.6884 | 0.9439 |
| NVFP4-SmoothQuant | 0.6859 | 0.6847 | 0.6868 | 0.6863 | 0.9430 |
| NVFP4-GPTQ | 0.6977 | 0.6972 | 0.6983 | 0.6976 | **0.9592** |
| NVFP4-GPTQ-perf-blend | 0.6973 | 0.6975 | 0.6969 | 0.6976 | 0.9587 |
| NVFP4-GPTQ-damp001-perf-blend | 0.6942 | 0.6952 | 0.6926 | 0.6946 | 0.9543 |
| NVFP4-GPTQ-damp05-perf-blend | 0.6968 | 0.6962 | 0.6968 | 0.6975 | 0.9580 |
| NVFP4-GPTQ-mse-perf-blend | 0.6961 | 0.6944 | 0.6975 | 0.6963 | 0.9569 |
| NVFP4-GPTQ-perf-blend-more-data | 0.6966 | 0.6945 | 0.6957 | 0.6996 | 0.9577 |
| NVFP4-GPTQ-AWQ | 0.6929 | 0.6967 | 0.6899 | 0.6921 | 0.9526 |
| NVFP4-GPTQ-AWQ-imatrix | 0.6883 | 0.6883 | 0.6886 | 0.6879 | 0.9462 |
| NVFP4-GPTQ-AWQ-perf-blend | 0.6927 | 0.6966 | 0.6907 | 0.6908 | 0.9523 |
| NVFP4-GPTQ-AWQ-imatrix-perf-blend | 0.6908 | 0.6898 | 0.6901 | 0.6926 | 0.9497 |
| NVFP4-GPTQ-imatrix | 0.6948 | 0.6924 | 0.6958 | 0.6963 | 0.9552 |
| NVFP4-GPTQ-imatrix-no-static | 0.6969 | 0.6960 | 0.6951 | 0.6995 | 0.9580 |
| NVFP4-GPTQ-imatrix-perf-blend | 0.6979 | 0.6993 | 0.6958 | 0.6985 | **0.9594** |
| NVFP4-GPTQ-BlockSize16 | 0.6955 | 0.6947 | 0.6954 | 0.6963 | 0.9561 |
| NVFP4-GPTQ-SmoothQuant | 0.6925 | 0.6940 | 0.6902 | 0.6934 | 0.9521 |
| NVFP4-GPTQ-fineweb | 0.6966 | 0.6943 | 0.6999 | 0.6956 | 0.9577 |
| NVFP4-GPTQ-red-pajama | 0.6974 | 0.6976 | 0.6968 | 0.6977 | **0.9587** |
| NVFP4-AutoRound | 0.6891 | 0.6907 | 0.6905 | 0.6862 | 0.9474 |
| NVFP4-AutoRound-Ultrachat | 0.6903 | 0.6866 | 0.6925 | 0.6919 | 0.9490 |
| NVFP4-AutoRound-AWQ | 0.6952 | 0.6978 | 0.6947 | 0.6932 | 0.9558 |
| NVFP4-AutoRound-AWQ-Ultrachat | 0.6924 | 0.6926 | 0.6924 | 0.6923 | 0.9519 |
| NVFP4-AutoRound-AWQ-Ultrachat-More-Iter | 0.6912 | 0.6906 | 0.6913 | 0.6917 | 0.9502 |
| NVFP4-AutoRound-SmoothQuant | 0.6955 | 0.6964 | 0.6973 | 0.6929 | 0.9562 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat | 0.6925 | 0.6911 | 0.6938 | 0.6925 | 0.9520 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat-More-Iter | 0.6943 | 0.6952 | 0.6913 | 0.6963 | 0.9544 |

## mmlu_pro_chat

Metric: `exact_match,custom-extract`

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|-------|------|-----------|-----------|-----------|----------|
| RedHatAI/Llama-3.1-8B-Instruct (FP16) | 0.4707 | 0.4670 | 0.4723 | 0.4727 | -  |
| NVFP4 | 0.4164 | 0.4117 | 0.4200 | 0.4176 | 0.8847 |
| NVFP4-AWQ | 0.4157 | 0.4131 | 0.4128 | 0.4212 | 0.8832 |
| NVFP4-SmoothQuant | 0.4202 | 0.4203 | 0.4178 | 0.4225 | 0.8927 |
| NVFP4-GPTQ | 0.4222 | 0.4229 | 0.4221 | 0.4215 | 0.8969 |
| NVFP4-GPTQ-perf-blend | 0.4306 | 0.4287 | 0.4341 | 0.4289 | **0.9147** |
| NVFP4-GPTQ-damp001-perf-blend | 0.4282 | 0.4270 | 0.4279 | 0.4295 | 0.9096 |
| NVFP4-GPTQ-damp05-perf-blend | 0.4313 | 0.4291 | 0.4311 | 0.4338 | **0.9163** |
| NVFP4-GPTQ-mse-perf-blend | 0.4242 | 0.4289 | 0.4225 | 0.4212 | 0.9012 |
| NVFP4-GPTQ-perf-blend-more-data | 0.4288 | 0.4311 | 0.4265 | 0.4287 | 0.9109 |
| NVFP4-GPTQ-AWQ | 0.4251 | 0.4271 | 0.4243 | 0.4240 | 0.9032 |
| NVFP4-GPTQ-AWQ-imatrix | 0.4228 | 0.4239 | 0.4206 | 0.4240 | 0.8983 |
| NVFP4-GPTQ-AWQ-perf-blend | 0.4258 | 0.4293 | 0.4261 | 0.4220 | 0.9046 |
| NVFP4-GPTQ-AWQ-imatrix-perf-blend | 0.4292 | 0.4279 | 0.4337 | 0.4260 | **0.9118** |
| NVFP4-GPTQ-imatrix | 0.4207 | 0.4193 | 0.4236 | 0.4192 | 0.8938 |
| NVFP4-GPTQ-imatrix-no-static | 0.4192 | 0.4190 | 0.4197 | 0.4190 | 0.8907 |
| NVFP4-GPTQ-imatrix-perf-blend | 0.4278 | 0.4249 | 0.4343 | 0.4242 | 0.9089 |
| NVFP4-GPTQ-BlockSize16 | 0.4219 | 0.4243 | 0.4210 | 0.4204 | 0.8963 |
| NVFP4-GPTQ-SmoothQuant | 0.4201 | 0.4184 | 0.4214 | 0.4205 | 0.8925 |
| NVFP4-GPTQ-fineweb | 0.4234 | 0.4181 | 0.4244 | 0.4279 | 0.8996 |
| NVFP4-GPTQ-red-pajama | 0.4227 | 0.4209 | 0.4190 | 0.4281 | 0.8979 |
| NVFP4-AutoRound | 0.4184 | 0.4122 | 0.4222 | 0.4207 | 0.8888 |
| NVFP4-AutoRound-Ultrachat | 0.4248 | 0.4269 | 0.4194 | 0.4282 | 0.9026 |
| NVFP4-AutoRound-AWQ | 0.4212 | 0.4231 | 0.4190 | 0.4215 | 0.8948 |
| NVFP4-AutoRound-AWQ-Ultrachat | 0.4223 | 0.4255 | 0.4185 | 0.4230 | 0.8972 |
| NVFP4-AutoRound-AWQ-Ultrachat-More-Iter | 0.4266 | 0.4236 | 0.4309 | 0.4252 | 0.9062 |
| NVFP4-AutoRound-SmoothQuant | 0.4224 | 0.4226 | 0.4215 | 0.4231 | 0.8974 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat | 0.4273 | 0.4258 | 0.4292 | 0.4268 | 0.9077 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat-More-Iter | 0.4224 | 0.4205 | 0.4234 | 0.4234 | 0.8975 |

## ifeval

Metric: `inst_level_strict_acc,none`

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|-------|------|-----------|-----------|-----------|----------|
| RedHatAI/Llama-3.1-8B-Instruct (FP16) | 0.8161 | 0.8153 | 0.8225 | 0.8106 | -  |
| NVFP4 | 0.7990 | 0.7986 | 0.7974 | 0.8010 | 0.9790 |
| NVFP4-AWQ | 0.7906 | 0.7962 | 0.7878 | 0.7878 | 0.9688 |
| NVFP4-SmoothQuant | 0.7998 | 0.7986 | 0.8010 | 0.7998 | 0.9800 |
| NVFP4-GPTQ | 0.8070 | 0.8141 | 0.8094 | 0.7974 | **0.9888** |
| NVFP4-GPTQ-perf-blend | 0.7998 | 0.8082 | 0.7866 | 0.8046 | 0.9800 |
| NVFP4-GPTQ-damp001-perf-blend | 0.7858 | 0.7854 | 0.7866 | 0.7854 | 0.9628 |
| NVFP4-GPTQ-damp05-perf-blend | 0.8102 | 0.8034 | 0.8082 | 0.8189 | **0.9927** |
| NVFP4-GPTQ-mse-perf-blend | 0.8018 | 0.8129 | 0.7986 | 0.7938 | 0.9824 |
| NVFP4-GPTQ-perf-blend-more-data | 0.7998 | 0.7842 | 0.8118 | 0.8034 | 0.9800 |
| NVFP4-GPTQ-AWQ | 0.7970 | 0.8022 | 0.8022 | 0.7866 | 0.9766 |
| NVFP4-GPTQ-AWQ-imatrix | 0.7946 | 0.8034 | 0.7794 | 0.8010 | 0.9737 |
| NVFP4-GPTQ-AWQ-perf-blend | 0.8054 | 0.7866 | 0.8237 | 0.8058 | 0.9868 |
| NVFP4-GPTQ-AWQ-imatrix-perf-blend | 0.8046 | 0.8034 | 0.8034 | 0.8070 | 0.9859 |
| NVFP4-GPTQ-imatrix | 0.7942 | 0.7974 | 0.7878 | 0.7974 | 0.9732 |
| NVFP4-GPTQ-imatrix-no-static | 0.8038 | 0.8010 | 0.8046 | 0.8058 | 0.9849 |
| NVFP4-GPTQ-imatrix-perf-blend | 0.7974 | 0.7914 | 0.7938 | 0.8070 | 0.9771 |
| NVFP4-GPTQ-BlockSize16 | 0.8026 | 0.8046 | 0.8010 | 0.8022 | 0.9835 |
| NVFP4-GPTQ-SmoothQuant | 0.7950 | 0.7926 | 0.8010 | 0.7914 | 0.9741 |
| NVFP4-GPTQ-fineweb | 0.8066 | 0.8010 | 0.8034 | 0.8153 | **0.9883** |
| NVFP4-GPTQ-red-pajama | 0.8034 | 0.8106 | 0.8058 | 0.7938 | 0.9844 |
| NVFP4-AutoRound | 0.7946 | 0.7842 | 0.7842 | 0.8153 | 0.9736 |
| NVFP4-AutoRound-Ultrachat | 0.7954 | 0.8034 | 0.7986 | 0.7842 | 0.9746 |
| NVFP4-AutoRound-AWQ | 0.8066 | 0.8153 | 0.7986 | 0.8058 | 0.9883 |
| NVFP4-AutoRound-AWQ-Ultrachat | 0.7902 | 0.7974 | 0.7866 | 0.7866 | 0.9683 |
| NVFP4-AutoRound-AWQ-Ultrachat-More-Iter | 0.8034 | 0.7962 | 0.8070 | 0.8070 | 0.9844 |
| NVFP4-AutoRound-SmoothQuant | 0.8006 | 0.7974 | 0.8010 | 0.8034 | 0.9810 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat | 0.7902 | 0.7938 | 0.7914 | 0.7854 | 0.9683 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat-More-Iter | 0.7990 | 0.7974 | 0.7890 | 0.8106 | 0.9790 |

## math_500

Metric: `pass@k:k=1&n=1`

| Model | Mean | Seed 1234 | Seed 2345 | Seed 3456 | Recovery |
|-------|------|-----------|-----------|-----------|----------|
| RedHatAI/Llama-3.1-8B-Instruct (FP16) | 0.4727 | 0.4700 | 0.4720 | 0.4760 | -  |
| NVFP4 | 0.3967 | 0.4000 | 0.4160 | 0.3740 | 0.8392 |
| NVFP4-AWQ | 0.3933 | 0.4140 | 0.3740 | 0.3920 | 0.8321 |
| NVFP4-SmoothQuant | 0.3973 | 0.3880 | 0.4040 | 0.4000 | 0.8406 |
| NVFP4-GPTQ | 0.4327 | 0.4500 | 0.4160 | 0.4320 | 0.9153 |
| NVFP4-GPTQ-perf-blend | 0.4313 | 0.4320 | 0.4340 | 0.4280 | 0.9125 |
| NVFP4-GPTQ-damp001-perf-blend | 0.4380 | 0.4340 | 0.4340 | 0.4460 | **0.9266** |
| NVFP4-GPTQ-damp05-perf-blend | 0.4440 | 0.4460 | 0.4460 | 0.4400 | **0.9393** |
| NVFP4-GPTQ-mse-perf-blend | 0.4260 | 0.4160 | 0.4320 | 0.4300 | 0.9012 |
| NVFP4-GPTQ-perf-blend-more-data | 0.4393 | 0.4300 | 0.4420 | 0.4460 | **0.9294** |
| NVFP4-GPTQ-AWQ | 0.3960 | 0.3860 | 0.4300 | 0.3720 | 0.8377 |
| NVFP4-GPTQ-AWQ-imatrix | 0.4040 | 0.3960 | 0.4180 | 0.3980 | 0.8547 |
| NVFP4-GPTQ-AWQ-perf-blend | 0.4220 | 0.4340 | 0.4300 | 0.4020 | 0.8927 |
| NVFP4-GPTQ-AWQ-imatrix-perf-blend | 0.4320 | 0.4380 | 0.4340 | 0.4240 | 0.9139 |
| NVFP4-GPTQ-imatrix | 0.4047 | 0.4020 | 0.4100 | 0.4020 | 0.8561 |
| NVFP4-GPTQ-imatrix-no-static | 0.4047 | 0.4100 | 0.4280 | 0.3760 | 0.8561 |
| NVFP4-GPTQ-imatrix-perf-blend | 0.4273 | 0.4120 | 0.4380 | 0.4320 | 0.9040 |
| NVFP4-GPTQ-BlockSize16 | 0.4153 | 0.4020 | 0.4080 | 0.4360 | 0.8786 |
| NVFP4-GPTQ-SmoothQuant | 0.4027 | 0.3960 | 0.4000 | 0.4120 | 0.8518 |
| NVFP4-GPTQ-fineweb | 0.4093 | 0.4100 | 0.4180 | 0.4000 | 0.8659 |
| NVFP4-GPTQ-red-pajama | 0.4013 | 0.4220 | 0.3720 | 0.4100 | 0.8490 |
| NVFP4-AutoRound | 0.4000 | 0.3900 | 0.3940 | 0.4160 | 0.8462 |
| NVFP4-AutoRound-Ultrachat | 0.3960 | 0.3900 | 0.4020 | 0.3960 | 0.8377 |
| NVFP4-AutoRound-AWQ | 0.4080 | 0.4040 | 0.4080 | 0.4120 | 0.8631 |
| NVFP4-AutoRound-AWQ-Ultrachat | 0.4240 | 0.4280 | 0.4260 | 0.4180 | 0.8970 |
| NVFP4-AutoRound-AWQ-Ultrachat-More-Iter | 0.4300 | 0.4380 | 0.4180 | 0.4340 | 0.9097 |
| NVFP4-AutoRound-SmoothQuant | 0.4133 | 0.4280 | 0.4100 | 0.4020 | 0.8744 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat | 0.3993 | 0.3840 | 0.4080 | 0.4060 | 0.8448 |
| NVFP4-AutoRound-SmoothQuant-Ultrachat-More-Iter | 0.4080 | 0.4000 | 0.4100 | 0.4140 | 0.8631 |

## Llama-3.1-8B-Instruct NVFP4 Long-Context Evaluation Results

- **Baseline**: RedHatAI/Llama-3.1-8B-Instruct (BF16)
- **Benchmark**: MRCR (openai/mrcr), 1,294 samples, AUC metric
- **Seeds**: 1234, 2345, 3456
- **Serving**: vLLM v0.25.1, TP=1, max-model-len=131072

### Summary

AutoRound with AWQ or SmoothQuant calibration achieves the best long-context recovery (94.6%), slightly ahead of GPTQ (94.1%). Ultrachat calibration consistently hurts all AutoRound variants by 1-3pp.

| Model | Task | Seed 1234 | Seed 2345 | Seed 3456 | Mean | Recovery |
|-------|------|-----------|-----------|-----------|------|----------|
| Baseline (BF16) | mrcr (AUC) | 0.3500996606 | 0.3506236038 | 0.3480719120 | 0.3495983921 | |
| NVFP4 | mrcr (AUC) | 0.3261855201 | 0.3217578310 | 0.3185279665 | 0.3221571059 | 92.15% |
| NVFP4-GPTQ | mrcr (AUC) | 0.3305336504 | 0.3283941261 | 0.3283568507 | 0.3290948757 | **94.14%** |
| NVFP4-GPTQ-BlockSize16 | mrcr (AUC) | 0.3285295664 | 0.3338950970 | 0.3237553741 | 0.3287266792 | 94.03% |
| NVFP4-GPTQ-AWQ | mrcr (AUC) | 0.3191868746 | 0.3223295168 | 0.3080259722 | 0.3165141212 | 90.54% |
| NVFP4-AutoRound | mrcr (AUC) | 0.3305330655 | 0.3244752919 | 0.3238901580 | 0.3262995051 | 93.34% |
| NVFP4-AutoRound-AWQ | mrcr (AUC) | 0.3381345503 | 0.3268904839 | 0.3270793821 | 0.3307014721 | **94.59%** |
| NVFP4-AutoRound-SmoothQuant | mrcr (AUC) | 0.3355041884 | 0.3284717061 | 0.3285328121 | 0.3308362355 | **94.63%** |
| NVFP4-AutoRound-Ultrachat | mrcr (AUC) | 0.3191376605 | 0.3330648224 | 0.3148907898 | 0.3223644242 | 92.21% |
| NVFP4-AutoRound-AWQ-Ultrachat | mrcr (AUC) | 0.3174015897 | 0.3263447104 | 0.3183634514 | 0.3207032505 | 91.73% |
| NVFP4-AutoRound-SmoothQuant-Ultrachat | mrcr (AUC) | 0.3256412505 | 0.3259602501 | 0.3065999423 | 0.3194004810 | 91.36% |

## Source

Ported from [issue #2922](https://github.com/vllm-project/llm-compressor/issues/2922),
part of the FP4 baseline benchmarking series tracked in
[#2921](https://github.com/vllm-project/llm-compressor/issues/2921). The MoE
half of that series is in
[the Qwen3-30B-A3B sweep](./nvfp4-moe-baseline-sweep-dsikka.md).
