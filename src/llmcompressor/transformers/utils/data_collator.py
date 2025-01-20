import torch

__all__ = [
    "mllama_data_collator",
    "pixtral_data_collator",
    "llava_data_collator",
    "qwen2_vl_data_collator",
    "phi3_vision_data_collator",
]


def mllama_data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),
        "aspect_ratio_ids": torch.tensor(batch[0]["aspect_ratio_ids"]),
        "aspect_ratio_mask": torch.tensor(batch[0]["aspect_ratio_mask"]),
        "cross_attention_mask": torch.tensor(batch[0]["cross_attention_mask"]),
    }


def pixtral_data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"])[0],
    }


def llava_data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),
    }


def qwen2_vl_data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),
        "image_grid_thw": torch.tensor(batch[0]["image_grid_thw"]),
    }


def phi3_vision_data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),
        "image_sizes": torch.tensor(batch[0]["image_sizes"]),
    }


def whisper_data_collator(batch):
    assert len(batch) == 1
    return {
        "input_features": torch.tensor(batch[0]["input_features"]),
        "decoder_input_ids": torch.tensor(batch[0]["decoder_input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
    }


def qwen2_audio_data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "input_features": torch.tensor(batch[0]["input_features"]),
        "feature_attention_mask": torch.tensor(batch[0]["feature_attention_mask"]),
    }
