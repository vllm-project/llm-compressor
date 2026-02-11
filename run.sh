cd examples/awq
python fp8_dynamic_llama_example.py
mv Meta-Llama-3-8B-Instruct-awq-asym Meta-Llama-3-8B-Instruct-awq-asym-fp8-dynamic
python fp8_block_llama_example.py
mv Meta-Llama-3-8B-Instruct-awq-asym Meta-Llama-3-8B-Instruct-awq-asym-fp8-block
python gsm8k_eval.py ./Meta-Llama-3-8B-Instruct-awq-asym-fp8-dynamic
python gsm8k_eval.py ./Meta-Llama-3-8B-Instruct-awq-asym-fp8-block