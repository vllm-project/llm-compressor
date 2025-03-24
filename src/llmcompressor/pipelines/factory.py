from llmcompressor.typing import PipelineFn

from llmcompressor.pipelines import sequential, layer_sequential, basic

class PipelineFactory:

    @staticmethod
    def from_modifiers(modifiers: List[Modifier]) -> PipelineFn:
        
        
        
        