from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "mistral-community/Pixtral-Large-Instruct-2411"
)
processor = AutoProcessor.from_pretrained("mgoin/pixtral-12b")
