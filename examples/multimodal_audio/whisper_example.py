import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.utils.data_collator import whisper_data_collator

# Select model and load it.
MODEL_ID = "openai/whisper-tiny"

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_SPLIT = f"validation[:512]"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, "clean", split=DATASET_SPLIT)


def preprocess(example):
    return {
        "array": example["audio"]["array"],
        "sampling_rate": example["audio"]["sampling_rate"],
    }


ds = ds.map(preprocess, remove_columns=ds.column_names)


# Tokenize inputs.
def tokenize(sample):
    generation_config = None
    return_token_timestamps = None
    logprob_threshold = None
    return_timestamps = None
    language = None
    task = None
    is_multilingual = None

    input_features = None,
    generation_config = None,
    logits_processor = None,
    stopping_criteria = None,
    prefix_allowed_tokens_fn = None,
    synced_gpus = False,
    return_timestamps = None,
    task = None,
    language = None,
    is_multilingual = None,
    prompt_ids = None,
    prompt_condition_type = None,  # first-segment, all-segments
    condition_on_prev_tokens = None,
    temperature = None,
    compression_ratio_threshold = None,
    logprob_threshold = None,
    no_speech_threshold = None,
    num_segment_frames = None,
    attention_mask = None,
    time_precision = 0.02,
    time_precision_features = 0.01,
    return_token_timestamps = None,
    return_segments = False,
    return_dict_in_generate = None,


    input_features = processor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
    ).input_features

    # 1. prepare generation config
    generation_config, kwargs = model._prepare_generation_config(generation_config, **kwargs)

    # 2. set global generate variables
    input_stride = model.model.encoder.conv1.stride[0] * model.model.encoder.conv2.stride[0]
    num_segment_frames = input_stride * model.config.max_source_positions
    batch_size, total_input_frames = model._retrieve_total_input_frames(
        input_features=input_features, input_stride=input_stride, kwargs=kwargs
    )
    is_shortform = total_input_frames <= num_segment_frames

    # 3. Make sure generation config is correctly set
    # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
    return_dict_in_generate = model._set_return_outputs(
        return_dict_in_generate=return_dict_in_generate,
        return_token_timestamps=return_token_timestamps,
        logprob_threshold=logprob_threshold,
        generation_config=generation_config,
    )
    timestamp_begin = model._set_return_timestamps(
        return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
    )
    model._set_language_and_task(
        language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
    )
    model._set_num_frames(
        return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
    )
    model._set_thresholds_and_condition(
        generation_config=generation_config,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_prev_tokens=condition_on_prev_tokens,
    )
    model._set_prompt_condition_type(
        generation_config=generation_config,
        prompt_condition_type=prompt_condition_type,
    )

    # pass self.config for backward compatibility
    init_tokens = model._retrieve_init_tokens(
        input_features,
        batch_size=batch_size,
        generation_config=generation_config,
        config=model.config,
        num_segment_frames=num_segment_frames,
        kwargs=kwargs,
    )
    # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
    # where the input ids are handled explicitly by the generate method
    model._check_decoder_input_ids(kwargs=kwargs)

    # 3. Retrieve logits processors
    device = kwargs["encoder_outputs"][0].device if "encoder_outputs" in kwargs else input_features.device
    begin_index = init_tokens.shape[1]
    logits_processor = model._retrieve_logit_processors(
        generation_config=generation_config,
        logits_processor=logits_processor,
        begin_index=begin_index,  # begin index is index of first generated decoder token
        num_beams=kwargs.get("num_beams", 1),
        device=device,
    )

    # 4 Set and retrieve global generation variables
    model._set_condition_on_prev_tokens(
        condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
    )

    temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
    temperature = temperatures[0]

    max_frames, seek = model._retrieve_max_frames_and_seek(
        batch_size=batch_size,
        attention_mask=attention_mask,
        total_input_frames=total_input_frames,
        is_shortform=is_shortform,
    )

    # 5 Prepare running variables, list for generation
    num_return_sequences = generation_config.num_return_sequences
    (
        batch_idx_map,
        cur_bsz,
        input_features,
        seek,
        max_frames,
        init_tokens,
        do_condition_on_prev_tokens,
    ) = model._expand_variables_for_generation(
        input_features=input_features,
        seek=seek,
        max_frames=max_frames,
        init_tokens=init_tokens,
        batch_size=batch_size,
        condition_on_prev_tokens=condition_on_prev_tokens,
        generation_config=generation_config,
    )

    current_segments = model._prepare_segments(
        prompt_ids=prompt_ids,
        batch_size=cur_bsz,
        generation_config=generation_config,
    )

    # 6 Transcribe audio until we reach the end of all input audios
    while (seek < max_frames).any():
        # 6.1 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
        # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
        # to know which original audio is being decoded
        # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
        input_features, cur_bsz, batch_idx_map = model._maybe_reduce_batch(
            input_features=input_features,
            seek=seek,
            max_frames=max_frames,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
        )
        time_offset = (
            seek.to(torch.float32 if device.type == "mps" else torch.float64) * time_precision / input_stride
        )
        seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

        # 6.2 cut out next 30s segment from input features
        segment_input = model._get_input_segment(
            input_features=input_features,
            seek=seek,
            seek_num_frames=seek_num_frames,
            num_segment_frames=num_segment_frames,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
        )

        # 6.3 prepare decoder input ids
        suppress_tokens = _get_attr_from_logit_processors(
            logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
        )

        decoder_input_ids, kwargs = model._prepare_decoder_input_ids(
            cur_bsz=cur_bsz,
            init_tokens=init_tokens,
            current_segments=current_segments,
            batch_idx_map=batch_idx_map,
            do_condition_on_prev_tokens=do_condition_on_prev_tokens,
            prompt_ids=prompt_ids,
            generation_config=generation_config,
            config=model.config,
            device=init_tokens.device,
            suppress_tokens=suppress_tokens,
            timestamp_begin=timestamp_begin,
            kwargs=kwargs,
        )

        # 6.4 set max new tokens or max length
        model._set_max_new_tokens_and_length(
            config=model.config,
            decoder_input_ids=decoder_input_ids,
            generation_config=generation_config,
        )

        # 6.5 Set current `begin_index` for all logit processors
        if logits_processor is not None:
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

        # 6.6 Run generate with fallback
        (
            seek_sequences,
            seek_outputs,
            should_skip,
            do_condition_on_prev_tokens,
            model_output_type,
        ) = model.generate_with_fallback(
            segment_input=segment_input,
            decoder_input_ids=decoder_input_ids,
            cur_bsz=cur_bsz,
            batch_idx_map=batch_idx_map,
            seek=seek,
            num_segment_frames=num_segment_frames,
            max_frames=max_frames,
            temperatures=temperatures,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            return_token_timestamps=return_token_timestamps,
            do_condition_on_prev_tokens=do_condition_on_prev_tokens,
            is_shortform=is_shortform,
            batch_size=batch_size,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )

    return segment_input["input_features"]


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
breakpoint()
sample_input = next(iter(ds))
output = model(**sample_input)


recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=whisper_data_collator,
)
breakpoint()

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
sample_input = next(iter(ds))
output = model.generate(sample_input)
print(processor.batch_decode(output, skip_special_tokens=True))
#[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)