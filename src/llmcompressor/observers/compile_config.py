_enable_torch_compile = False
_compile_chunk_size = 5


def set_torch_compile(enabled: bool, chunk_size: int = 5):
    global _enable_torch_compile, _compile_chunk_size
    _enable_torch_compile = enabled
    _compile_chunk_size = chunk_size


def get_torch_compile() -> bool:
    return _enable_torch_compile


def get_compile_chunk_size() -> int:
    return _compile_chunk_size
