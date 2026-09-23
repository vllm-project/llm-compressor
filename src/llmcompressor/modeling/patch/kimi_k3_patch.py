from contextlib import contextmanager
from functools import wraps

from compressed_tensors.utils import patch_attr
from transformers import CompressedTensorsConfig


@contextmanager
def patch_kimi_k3_ignore():
    """
    `moonshotai/Kimi-K3` has an incorrect ignore list. This context patches
    the qconfig on load so that the unquantized modules are properly ignored
    """
    original_init = CompressedTensorsConfig.__init__

    @wraps(original_init)
    def patched_init(self: CompressedTensorsConfig, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.quantization_config.ignore += [
            "re:.*mlp_res_proj.*",
            "re:.*self_attention_res_proj.*",
            "re:.*output_attn_res_proj.*",
            "re:.*routed_expert.*",
        ]

    with patch_attr(CompressedTensorsConfig, "__init__", patched_init):
        yield