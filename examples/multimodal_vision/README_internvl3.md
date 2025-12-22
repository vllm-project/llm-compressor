# Quantizing InternVL3-8B-hf
This file shows the example of quantizing InternVL3-8B-hf. 

## Step 1: Compressing Your Own Model

```python
model_id = "OpenGVLab/InternVL3-8B-hf"
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)
```

## Step 2: Load datasets
Use the `ultrachat_200k` datasets.
```python
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)
```

## Step 3: Preprocess and tokenize
```python
def preprocess_and_tokenize(example):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text":  example["messages"]
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    return inputs

ds = ds.map(preprocess_and_tokenize)
```
## Step 4: Adding Your Own Data Collator
We need custom data collation to satisfy the model-specific requirements.
```python
def data_collator(batch):
    assert len(batch) == 1
    item = {key: value for key, value in batch[0].items()}
    item["attention_mask"] = torch.tensor([item["attention_mask"]])
    item["input_ids"] = torch.LongTensor([item["input_ids"]])

    return item
```


## Step 5: Define the recipe
```python
recipe = GPTQModifier(
        targets="Linear",
        scheme="FP8", 
        ignore=["re:.*lm_head",  "re:.*vision_tower.*",  "re:.*multi_modal_projector.*"]
    )
```
Note: We also tried `ignore=["re:.*lm_head",  "re:.*multi_modal_projector.*"]`. However, this quantized model did not produce meaningful output for prompts with images. Therefore, we only quantize the LLM part.
## Step 6: Oneshot and save
```python
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator
)

SAVE_DIR = "OpenGVLab/InternVL3-8B-hf-FP8-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
```
## Step 7: Evaluate
With the model created, we can now load and run in vLLM.
### Accuracy
We can evaluate accuracy multimodal_vision model with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git)
```
torchrun --nproc-per-node=2 run.py --data MMStar --model InternVL3-8B-hf_FP8_GPTQ --verbose
```
### Performance
We can evaluate performance with vllm.
First, run in vllm.
```
vllm serve OpenGVLab/InternVL3-8B-hf-FP8-GPTQ \
        --served-model-name InternVL3-8B-hf-FP8-GPTQ \
        --gpu-memory-utilization 0.9 \
        --uvicorn_log_level error \
        --disable-log-stats \
        --trust-remote-code \
        --allowed-local-media-path /path/to/sharegpt4v/images \
        --limit-mm-per-prompt '{"image": 20}'  \
        --mm-processor-kwargs '{"max_dynamic_patch": 1}' \
        --no-enable-prefix-caching \
        --disable-mm-preprocessor-cache  \
        --max-model-len 6144
```
Second, use vllm bench serve.
```
vllm bench serve \
    --backend openai-chat \
    --dataset-name sharegpt \
    --dataset-path /path/to//ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco.json  \
    --num-prompts 500 \
    --endpoint /v1/chat/completions  \
    --max-concurrency 100 \
    --percentile-metrics='ttft,tpot,itl,e2el'  \
    --model InternVL3-8B-hf-FP8-GPTQ
```

The result of InternVL3-8B-hf:
```
============ Serving Benchmark Result ============
Successful requests:                     500       
Maximum request concurrency:             100       
Benchmark duration (s):                  251.15    
Total input tokens:                      6193      
Total generated tokens:                  30487     
Request throughput (req/s):              1.99      
Output token throughput (tok/s):         121.39    
Peak output token throughput (tok/s):    1055.00   
Peak concurrent requests:                107.00    
Total Token throughput (tok/s):          146.05    
---------------Time to First Token----------------
Mean TTFT (ms):                          25349.32  
Median TTFT (ms):                        25498.75  
P99 TTFT (ms):                           45969.02  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          415.96    
Median TPOT (ms):                        414.47    
P99 TPOT (ms):                           857.11    
---------------Inter-token Latency----------------
Mean ITL (ms):                           379.15    
Median ITL (ms):                         410.25    
P99 ITL (ms):                            524.16    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          48444.73  
Median E2EL (ms):                        50202.88  
P99 E2EL (ms):                           91705.52  
==================================================
```

The result of InternVL3-8B-hf-FP8-GPTQ:
```
============ Serving Benchmark Result ============
Successful requests:                     500       
Maximum request concurrency:             100       
Benchmark duration (s):                  163.36    
Total input tokens:                      6193      
Total generated tokens:                  34831     
Request throughput (req/s):              3.06      
Output token throughput (tok/s):         213.22    
Peak output token throughput (tok/s):    1787.00   
Peak concurrent requests:                109.00    
Total Token throughput (tok/s):          251.13    
---------------Time to First Token----------------
Mean TTFT (ms):                          14510.84  
Median TTFT (ms):                        14371.25  
P99 TTFT (ms):                           28978.21  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          257.52    
Median TPOT (ms):                        270.19    
P99 TPOT (ms):                           330.58    
---------------Inter-token Latency----------------
Mean ITL (ms):                           247.34    
Median ITL (ms):                         268.16    
P99 ITL (ms):                            386.99    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          31725.93  
Median E2EL (ms):                        32227.14  
P99 E2EL (ms):                           64293.40  
==================================================
```

