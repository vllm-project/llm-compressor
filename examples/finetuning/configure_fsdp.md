# Configuring FSDP for Sparse Finetuning

An example FSDP configuration file, `example_fsdp_config.yaml`, is provided in this
folder. It can be used out of the box by editing the `num_processes` parameter to 
fit the number of GPUs on your machine.

You can also customize your own config file by running the following prompt
```
accelerate config
```

An FSDP config file can be passed to the LLM Compressor finetuning script like this:
```
accelerate launch --config_file example_fsdp_config.yaml --no_python llmcompressor.transformers.text_generation.finetune
```
