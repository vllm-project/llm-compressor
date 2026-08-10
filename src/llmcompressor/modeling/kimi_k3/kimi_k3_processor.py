# Differences from https://huggingface.co/moonshotai/Kimi-K3/tree/main:
# - Formatting only (no functional changes)

"""Kimi-K3 processor: wraps vision processor + tokenizer into a single interface.

Chat rendering (including XTML tool-result ordering) is handled by the
tokenizer's Python encoder; this processor adds multimodal media preprocessing.
"""

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging

from .media_utils import ensure_media_type

logger = logging.get_logger(__name__)

# ── KimiK3Processor ───────────────────────────────────────────────────


class KimiK3Processor(ProcessorMixin):
    r"""
    Constructs a KimiK3 processor which wraps a KimiK3 image processor
    and a tokenizer into a single processor.

    [`KimiK3Processor`] offers all the functionalities of
    [`KimiK3VisionProcessor`] and [`TikTokenTokenizer`].

    Args:
        image_processor ([`KimiK3VisionProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`TikTokenTokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): Kept for ProcessorMixin
            compatibility. Kimi K3 chat encoding is implemented in Python by
            the tokenizer.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.media_processor = image_processor
        self.image_placeholder = "<|kimi_image_placeholder|>"

    # ── Media preprocessing ────────────────────────────────────────────

    def update_raw_text(self, text: str, image_prompts: list[str]) -> str:
        # Replace image placeholders
        image_count = text.count(self.image_placeholder)
        if image_count > 0:
            assert image_count == len(image_prompts), (
                f"image placeholder count {image_count} != "
                f"image_prompts count {len(image_prompts)}"
            )
            text_parts = text.split(self.image_placeholder)
            assert len(text_parts) == len(image_prompts) + 1
            text = "".join(
                [text_parts[i] + image_prompts[i] for i in range(len(image_prompts))]
            )
            text += text_parts[-1]

        return text

    def preprocess_medias(self, medias: list[dict]) -> tuple[list[dict], list[str]]:
        """Process media items and generate corresponding prompts.

        Returns:
            A tuple of (updated_medias, image_prompts).
        """
        updated_medias = []
        image_prompts = []
        for media in medias:
            if media["type"] == "image":
                updated_medias.append(media)
                img = ensure_media_type(
                    media,
                    transparent_bg_config=self.media_processor._transparent_bg_config,
                    transparent_bg_fill_stage=self.media_processor._transparent_bg_fill_stage,
                )["image"]
                w, h = img.size
                image_prompts.append(self.media_processor.make_image_prompt(w, h))
            else:
                raise ValueError(f"unsupported media type: {media['type']}")
        return updated_medias, image_prompts

    # ── Main entry points ──────────────────────────────────────────────

    def __call__(
        self,
        messages: list[dict] = None,
        medias: list[dict] = None,
        text: str = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Process multimodal inputs for Kimi-K3 model.

        Args:
            messages: List of message dicts with 'role' and 'content' fields.
                     If provided, medias and text will be extracted automatically.
            medias: Pre-extracted list of media dicts.
            text: Pre-formatted text string.
            return_tensors: Format of returned tensors. Default: 'pt'.
            **kwargs: Additional arguments passed to apply_chat_template.

        Returns:
            BatchFeature with fields: input_ids, attention_mask,
            pixel_values, grid_thws.
        """
        if messages is None and (medias is None or text is None):
            raise ValueError("Provide either 'messages' or both 'medias' and 'text'")

        if medias is not None and text is not None:
            updated_medias, image_prompts = self.preprocess_medias(medias)
            preprocessed = self.media_processor.preprocess(
                updated_medias, return_tensors=return_tensors
            )
            text = self.update_raw_text(text, image_prompts)
            text_inputs = self.tokenizer(text, return_tensors=return_tensors)
            return BatchFeature(data={**text_inputs, **preprocessed.data})

        if medias is None:
            medias = self._extract_medias_from_messages(messages)
        updated_medias, image_prompts = self.preprocess_medias(medias)
        preprocessed = self.media_processor.preprocess(
            updated_medias, return_tensors=return_tensors
        )

        if text is None:
            text_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors=return_tensors,
                return_dict=True,
                image_prompts=image_prompts,
                **kwargs,
            )
            return BatchFeature(data={**text_inputs, **preprocessed.data})

        text = self.update_raw_text(text, image_prompts)
        text_inputs = self.tokenizer(text, return_tensors=return_tensors)
        return BatchFeature(data={**text_inputs, **preprocessed.data})

    @staticmethod
    def _extract_medias_from_messages(messages: list[dict]) -> list[dict]:
        """Extract media items from messages in a single pass."""
        medias = []
        for msg in messages:
            if msg["role"] != "user" or not msg.get("content"):
                continue

            for content_part in msg["content"]:
                if not isinstance(content_part, dict):
                    continue

                content_type = content_part.get("type")
                if content_type in ["image_url", "image"]:
                    image_data = content_part.get(content_type)
                    assert (
                        image_data is not None
                    ), f"image data is missing for content part: {content_part}"
                    medias.append(
                        {
                            "type": "image",
                            "image": image_data,
                        }
                    )
        return medias

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "grid_thws"]
