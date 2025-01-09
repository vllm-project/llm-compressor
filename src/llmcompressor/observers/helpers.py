from collections import Counter

import torch

__all__ = ["get_observer_token_count"]


def get_observer_token_count(module: torch.nn.Module) -> Counter:
    """
    Parse the module and return the number of tokens observed by
    each module's observer.

    :param module: module to parse
    :return: counter with the number of tokens observed by each observer
    """
    token_counts = Counter()
    for name, module in module.named_modules():
        if name.endswith(".input_observer"):
            token_counts[name.replace(".input_observer", "")] = (
                module._num_observed_tokens
            )
    return token_counts
