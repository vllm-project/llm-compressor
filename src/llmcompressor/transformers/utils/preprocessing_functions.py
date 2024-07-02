from typing import Dict

from compressed_tensors.registry import RegistryMixin


class PreprocessingFunctionRegistry(RegistryMixin):
    pass


@PreprocessingFunctionRegistry.register()
def custom_evolved_codealpaca_dataset(data: Dict):
    PROMPT_DICT = """[Instruction]:\n{instruction}\n\n[Response]:"""
    data["prompt"] = PROMPT_DICT.format_map(data)
    data["text"] = data["prompt"] + data["output"]
    return data
