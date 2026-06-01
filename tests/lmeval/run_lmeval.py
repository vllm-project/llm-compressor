import json
import sys

import lm_eval
import lm_eval.api.registry

# needed to populate model registry
import lm_eval.models  # noqa


def parse_args():
    """Parse model and config JSON arguments passed via command line."""
    if len(sys.argv) < 3:
        msg = "Usage: python script.py '<model>' '<config_json>'"
        raise ValueError(msg)

    model = sys.argv[1]

    try:
        config = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {e}")

    return model, config


def main():
    model, config = parse_args()
    lmeval_config = config["lmeval"]
    seed = config["seed"]

    model_args = (
        f"pretrained={model},"
        f"dtype={lmeval_config['dtype']},"
        f"add_bos_token={lmeval_config['add_bos_token']},"
        f"trust_remote_code={lmeval_config['trust_remote_code']},"
        f"max_model_len={lmeval_config['max_model_len']},"
        f"seed={seed},"
        f"gpu_memory_utilization={config['gpu_memory_utilization']},"
        f"pipeline_parallel_size="
        f"{config.get('num_gpus', 1) if config.get('pipeline_parallel', False) else 1},"
    )

    sampling_params = lmeval_config["sampling_params"]
    gen_kwargs = ",".join([f"{k}={v}" for k, v in sampling_params.items()])

    results = lm_eval.simple_evaluate(
        model=lmeval_config["model"],
        model_args=model_args,
        gen_kwargs=gen_kwargs,
        tasks=[lmeval_config["task"]],
        num_fewshot=lmeval_config["num_fewshot"],
        limit=lmeval_config["limit"],
        apply_chat_template=lmeval_config["apply_chat_template"],
        fewshot_as_multiturn=lmeval_config["fewshot_as_multiturn"],
        fewshot_random_seed=seed,
    )

    task = lmeval_config["task"]
    task_results = results["results"][task]
    metrics = {metric: task_results[metric] for metric in lmeval_config["metrics"]}
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
