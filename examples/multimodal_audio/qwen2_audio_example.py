import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import soundfile as sf
from io import BytesIO
from urllib.request import urlopen

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import (
    TraceableQwen2AudioForConditionalGeneration,
)

# Select model and load it.
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

#model = TraceableQwen2AudioForConditionalGeneration.from_pretrained(
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# # Select calibration dataset.
# DATASET_ID = "MLCommons/peoples_speech"
# DATASET_SUBSET = "test"
# DATASET_SPLIT = "test"

# # Select number of samples. 512 samples is a good place to start.
# # Increasing the number of samples can improve accuracy.
# NUM_CALIBRATION_SAMPLES = 1 #512
# MAX_SEQUENCE_LENGTH = 2048

# # Load dataset and preprocess.
# ds = load_dataset(
#     DATASET_ID,
#     DATASET_SUBSET,
#     split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
#     trust_remote_code=True,
# )


# def preprocess(example):
#     messages = [
#         # {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": [{"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"},
#         # {"role": "user", "content": [{"type": "text", "text": "What does the person say?"}]},
#     ]}
#     ]

#     audio_data = example["audio"]["array"]
#     sample_rate = example["audio"]["sampling_rate"]

#     # import librosa
#     # new_sr = processor.feature_extractor.sampling_rate
#     # audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=new_sr)
#     # sample_rate = new_sr

#     #processor.feature_extractor.sampling_rate

#     # # Create an in-memory buffer
#     # import io
#     # buffer = io.BytesIO()

#     # # Write the audio data to the in-memory buffer in WAV format
#     # sf.write(buffer, audio_data, sample_rate, format='WAV')

#     # import librosa
#     # audio_data, sample_rate = librosa.load(buffer, sr=sample_rate)

#     import librosa
#     audio_data = librosa.load(
#         BytesIO(urlopen("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav").read()), 
#         sr=processor.feature_extractor.sampling_rate
#     )[0]

#     return {
#         "text": processor.apply_chat_template(
#             messages, add_generation_prompt=True, tokenize=False
#         ),
#         #"audios": [example["audio"]["array"]],
#         "audios": [audio_data],
#         #"array": example["audio"]["array"],
#         #"sampling_rate": example["audio"]["sampling_rate"],
#         "sampling_rate": sample_rate,
#         #"sampling_rate": processor.feature_extractor.sampling_rate
#     }


# ds = ds.map(preprocess, remove_columns=ds.column_names)


# # Tokenize inputs.
# def tokenize(sample):
#     return processor(**sample, return_tensors="pt")

# # Process inputs.
# def process(sample):

#     messages = [
#         {"role": "user", "content": [{"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"}]}
#     ]

#     # import librosa
#     # new_sr = processor.feature_extractor.sampling_rate
#     # audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=new_sr)
#     # sample_rate = new_sr

#     #processor.feature_extractor.sampling_rate

#     # # Create an in-memory buffer
#     # import io
#     # buffer = io.BytesIO()

#     # # Write the audio data to the in-memory buffer in WAV format
#     # sf.write(buffer, audio_data, sample_rate, format='WAV')

#     # import librosa
#     # audio_data, sample_rate = librosa.load(buffer, sr=sample_rate)

#     import librosa
#     audio_data = librosa.load(
#         BytesIO(urlopen("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav").read()), 
#         sr=processor.feature_extractor.sampling_rate
#     )[0]

#     return processor(
#         text=processor.apply_chat_template(
#             messages, add_generation_prompt=True, tokenize=False
#         ),
#         #audio=sample["array"],
#         audios=[audio_data],
#         #sampling_rate=sample["sampling_rate"],
#         #sampling_rate=sample["sampling_rate"],
#         #add_special_tokens=True,
#         return_tensors="pt",
#         padding=True
#     )




#     audio_inputs = processor(
#         text=sample["text"],
#         #audio=sample["array"],
#         audios=sample["audios"],
#         #sampling_rate=sample["sampling_rate"],
#         #sampling_rate=sample["sampling_rate"],
#         #add_special_tokens=True,
#         return_tensors="pt",
#         padding=True
#     )
#     return audio_inputs

#     text_inputs = processor(
#         text=sample["text"], add_special_tokens=True, return_tensors="pt"
#     )
#     text_inputs["decoder_input_ids"] = text_inputs["input_ids"]
#     del text_inputs["input_ids"]

#     return dict(**audio_inputs, **text_inputs)


# #ds = ds.map(tokenize, remove_columns=ds.column_names)
# ds = ds.map(process, remove_columns=ds.column_names)

messages = [
    {"role": "user", "content": [{"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"}]}
]

# import librosa
# new_sr = processor.feature_extractor.sampling_rate
# audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=new_sr)
# sample_rate = new_sr

#processor.feature_extractor.sampling_rate

# # Create an in-memory buffer
# import io
# buffer = io.BytesIO()

# # Write the audio data to the in-memory buffer in WAV format
# sf.write(buffer, audio_data, sample_rate, format='WAV')

# import librosa
# audio_data, sample_rate = librosa.load(buffer, sr=sample_rate)

import librosa
audio_data = librosa.load(
    BytesIO(urlopen("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav").read()), 
    sr=processor.feature_extractor.sampling_rate
)[0]

text = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)

breakpoint()
sample_input = processor(
    text=text,
    #audio=sample["array"],
    audios=[audio_data],
    #sampling_rate=sample["sampling_rate"],
    #sampling_rate=sample["sampling_rate"],
    #add_special_tokens=True,
    return_tensors="pt",
    padding=True
)
breakpoint()


# Define a oneshot data collator for multimodal inputs.
# def data_collator(batch):
#     assert len(batch) == 1
#     return {key: torch.tensor(value) for key, value in batch[0].items()}


# Configure the quantization algorithm to run.
# #   * quantize the weights to 4 bit with GPTQ with a group size 128
# recipe = GPTQModifier(
#     targets="Linear",
#     scheme="W4A16",
#     ignore=[
#         # "re:audio_tower.*",
#         # "re:multi_modal_projector.*",
#         "lm_head",
#     ],  # TODO: honestly, there's a decent number of parameters in the audio tower worth quantizing
# )

# Apply algorithms.
# oneshot(
#     model=model,
#     dataset=ds,
#     recipe=recipe,
#     max_seq_length=MAX_SEQUENCE_LENGTH,
#     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#     data_collator=data_collator,
# )

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
breakpoint()
#sample_input = data_collator([next(iter(ds))])
#sample_input = ds[0]
sample_input = {k: v.to(model.device) for k, v in sample_input.items()}
output = model.generate(**sample_input, max_new_tokens=256)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
print("==========================================\n\n")
# that's where you have a lot of windows in the south no actually that's passive solar
# and passive solar is something that was developed and designed in the 1960s and 70s
# and it was a great thing for what it was at the time but it's not a passive house

# Save to disk compressed.
# SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
# model.save_pretrained(SAVE_DIR, save_compressed=True)
# processor.save_pretrained(SAVE_DIR)
