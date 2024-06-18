<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Configuring FSDP for Sparse Finetuning

An example FSDP configuration file, `example_fsdp_config.yaml`, is provided in this
folder. It can be used out of the box by editting the `num_processes` parameter to 
fit the number of GPUs on your machine.

You can also customize your own config file by running the following prompt
```
accelerate config
```

An FSDP config file can be passed to the SparseML finetuning script like this:
```
accelerate launch --config_file example_fsdp_config.yaml --no_python sparseml.transformers.text_generation.finetune
```
