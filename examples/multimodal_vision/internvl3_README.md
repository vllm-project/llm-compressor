# Quantizing InternVL3-8B
It is recommended to use the local model as we need to modify the weights.

## Step 1: Prepare
- 1. Download OpenGVLab/InternVL3-8B from hf
- 2. Download  [chat_template.jinja](https://hf-mirror.com/OpenGVLab/InternVL3_5-8B/blob/main/chat_template.jinja), and place it in the local model dir of OpenGVLab/InternVL3-8B
- 3. Replace the `forward` function in  OpenGVLab/InternVL3-8B/modeling_internvl_chat.py with the code below:
```
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # vit_embeds = self.extract_feature(pixel_values)
        # vit_embeds = vit_embeds[image_flags == 1]
        # vit_batch_size = pixel_values.shape[0]

        # B, N, C = input_embeds.shape
        # input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        # input_ids = input_ids.reshape(B * N)
        # selected = (input_ids == self.img_context_token_id)
        # try:
        #     input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        # except Exception as e:
        #     vit_embeds = vit_embeds.reshape(-1, C)
        #     print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
        #           f'vit_embeds.shape={vit_embeds.shape}')
        #     n_token = min(selected.sum(), vit_embeds.size(0))
        #     input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        # input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```
## Step 2: Compressing Your Own Model

```
model_id = "OpenGVLab/InternVL3-8B"
model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
```

## Step 3: Load datasets
Use the `flickr30k` datasets.
```
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)
```

## Step 4: Preprocess and tokenize
```
def preprocess_and_tokenize(example):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": ""
                },
                {
                    "type": "text", 
                    "text":  "What does the image show?"
                },
            ],
        }
    ]
    text = tokenizer.apply_chat_template(messages)

    example["input_ids"] = text
    return example
```
## Step 5: Adding Your Own Data Collator
We need custom data collation to satisfy the model-specific requirements.
```
def data_collator(batch):
    assert len(batch) == 1
    item = {key: value for key, value in batch[0].items()}
    item["pixel_values"] = load_image_from_PIL(item["image"])
    item["labels"] = torch.LongTensor([item["input_ids"]])
    item["input_ids"] = torch.LongTensor([item["input_ids"]])
    return item
```


# Step 6: Define the recipe
- `ignore: ["re:.*lm_head",  "re:mlp1.*"]`: quantizing llm and vit.
- `ignore: ["re:.*lm_head",  "re:mlp1.*", "re:vision_model.*"]`: quantizing llm only.

```
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["re:.*lm_head",  "re:mlp1.*"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""
```

## Step 7: Evaluate
With the model created, we can now load and run in vLLM.
### Accuracy
We can evaluate accuracy multimodal_vision model with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git)
```
torchrun --nproc-per-node=2 run.py --data MMStar --model InternVL3-8B-FP8_W8A8-FP8_KV --verbose
```
### Performance
We can evaluate performance with vllm.
First, run in vllm.
```
vllm serve InternVL3-8B-INT8-W8A8-FP8-KV 
    --limit-mm-per-prompt '{"image": 1}' 
    --allowed-local-media-path /train2017/ 
    --trust-remote-code
```
Second, use vllm bench serve.
```
vllm bench serve 
    --backend openai-chat   
    --dataset-name sharegpt   
    --dataset-path /dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco.json  
    --num-prompts 500     
    --endpoint /v1/chat/completions  
    --max-concurrency 100 
    --percentile-metrics='ttft,tpot,itl,e2el'  
    --model InternVL3-8B-INT8-W8A8-FP8-KV
```
We can see the result:
```
============ Serving Benchmark Result ============
Successful requests:                     500
Maximum request concurrency:             100
Benchmark duration (s):                  138.02
Total input tokens:                      6193
Total generated tokens:                  32248
Request throughput (req/s):              3.62
Output token throughput (tok/s):         233.65
Peak output token throughput (tok/s):    1842.00
Peak concurrent requests:                110.00
Total Token throughput (tok/s):          278.52
---------------Time to First Token----------------
Mean TTFT (ms):                          14036.75
Median TTFT (ms):                        13397.58
P99 TTFT (ms):                           24056.63
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          206.22
Median TPOT (ms):                        227.34
P99 TPOT (ms):                           269.71
---------------Inter-token Latency----------------
Mean ITL (ms):                           194.09
Median ITL (ms):                         221.38
P99 ITL (ms):                            307.42
----------------End-to-end Latency----------------
Mean E2EL (ms):                          26493.33
Median E2EL (ms):                        19590.30
P99 E2EL (ms):                           71131.76
==================================================
```
