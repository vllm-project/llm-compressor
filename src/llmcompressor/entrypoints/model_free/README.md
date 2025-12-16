# Quantizing models without a model definition 

`model_free_ptq` provides a PTQ pathway for data-free schemes (such as FP8 Dynamic Per Token or FP8 Block). Specifically, this pathway removes the requirement for a model definition or the need to load the model through transformers. If you are interested in applying a data-free scheme, there are two key scenarios in which applying this pathway may make sense for your model:

1. The model does not have a model definition available through transformers. This may be the case for a brand new model which has not landed in transformers.
2. The model is very large (such as Kimi K2 Thinking) and is running into issues with `oneshot`


`model_free_ptq` works directly with the safetensors in the checkpoint to which observers are applied, thereby removing the requirement for a model definition or transformers.