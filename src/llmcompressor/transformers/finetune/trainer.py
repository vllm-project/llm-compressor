from loguru import logger
from transformers import Trainer as HFTransformersTrainer

from llmcompressor.transformers.finetune.session_mixin import SessionManagerMixIn

__all__ = ["Trainer"]


class Trainer(SessionManagerMixIn, HFTransformersTrainer):
    def __init__(self, **kwargs):
        # call SessionManagerMixIn, the downstream code will call
        # init of HFTransformersTrainer
        super().__init__(**kwargs)

        # if python files exist in the original model, transfer to the output directory
        # Requried for Phi-3-medium-128k-instruct when using lm_eval
        self._copy_python_files_from_model_cache(**kwargs)

    def _copy_python_files_from_model_cache(self, **kwargs):
        model = kwargs["model_init"]()
        config = model.config
        cache_dir = None
        if hasattr(config, "_name_or_path"):
            import os
            import shutil

            cache_dir = config._name_or_path
            output_dir = kwargs["args"].output_dir
            for file in os.listdir(cache_dir):
                full_file_name = os.path.join(cache_dir, file)
                if file.endswith(".py") and os.path.isfile(full_file_name):
                    logger.debug(f"Transferring {full_file_name} to {output_dir}")
                    shutil.copy(full_file_name, output_dir)
