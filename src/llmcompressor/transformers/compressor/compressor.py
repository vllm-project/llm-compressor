# from llmcompressor.transformers.train import Train
# from llmcompressor.transformers.stage import StageRunner
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.core.session import CompressionManager
from llmcompressor.transformers.calibration import Oneshot


class LLMCompressor:
    COMPRESSORS = {
        "oneshot": Oneshot,
        # "train": Train,
        # "stages": StageRunner,
    }

    def __init__(self):
        self.session = CompressionManager()

    def oneshot(self, **kwargs):
        self._run("oneshot", **kwargs)

    def train(self, **kwargs):
        self._run("train", **kwargs)

    def stages(self, **kwargs):
        self._run("stages", **kwargs)

    def _run(self, key: str, **kwargs):
        if key not in self.COMPRESSORS:
            raise ValueError(
                f"Invalid compressor key: {key}. Must be one of {list(self.COMPRESSORS.keys())}."
            )
        compressor = self._create(key, **kwargs)
        compressor.run()

    def _create(self, key: str, **kwargs):
        compressor = self.COMPRESSORS[key](**kwargs)
        self.session.add(compressor)
        return compressor


"""

compressor = LLMCompressor(model=model, recipe=recipe)

compressor.oneshot(**kwargs)
compressor.train(**kwargs)


compressor.model
compressor.tokenizer_or_processor
compressor.recipe
compressor.dataset
compressor.lifecycle


"""
