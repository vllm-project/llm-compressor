from typing import TYPE_CHECKING, Dict

from compressed_tensors.registry import RegistryMixin

if TYPE_CHECKING:
    from llmcompressor.transformers.finetune.data.base import TextGenerationDataset


class PreprocessingFunctionRegistry(RegistryMixin):
    pass


@PreprocessingFunctionRegistry.register()
def custom_evolved_codealpaca_dataset(self: "TextGenerationDataset", data: Dict):
    PROMPT_DICT = """[Instruction]:\n{instruction}\n\n[Response]:"""
    data["prompt"] = PROMPT_DICT.format_map(data)
    data["text"] = data["prompt"] + data["output"]
    return data
