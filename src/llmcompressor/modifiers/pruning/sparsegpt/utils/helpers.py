import operator
from collections import defaultdict
from math import ceil
from typing import List, Optional

import torch
from loguru import logger
from torch.nn.modules.sparse import Embedding

_DEFAULT_TARGET_IDS = ["attention_mask", "position_ids", "position_bias"]


class Catcher(torch.nn.Module):
    def __init__(self, module, target_keys: Optional[List[str]] = None):
        super().__init__()
        self.module = module
        self.target_keys = target_keys
        self.cache = defaultdict(list)
        self.cache["inputs"] = []

    def forward(self, *args, **kwargs):
        self.cache["inputs"].append(args)
        if self.target_keys is None:
            self.target_keys = self._get_target_keys(kwargs.keys())

        for key in self.target_keys:
            self.cache[key].append(kwargs[key])
        raise ValueError

    def get_cache(self):
        return self.cache

    def _get_target_keys(self, input_keys):
        target_keys = []
        for key in _DEFAULT_TARGET_IDS:
            if key in input_keys:
                target_keys.append(key)
        return target_keys


def replace_module(model, old_module, new_module):
    for module_name, module in model.named_modules():
        if old_module == module:
            break

    current_module = model
    module_name = module_name.split(".")
    for child_module in module_name[:-1]:
        current_module = getattr(current_module, child_module)
    setattr(current_module, module_name[-1], new_module)


def catch(model, attention_layer, target_keys, data_loader, nsamples):
    catcher_module = Catcher(attention_layer, target_keys)
    replace_module(model, attention_layer, catcher_module)
    device = next(attention_layer.parameters()).device
    for input_id, inp in enumerate(data_loader):
        if nsamples is not None and input_id == nsamples:
            break
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            model(inp.to(device), use_cache=False)
        except ValueError:
            pass
    replace_module(model, catcher_module, attention_layer)
    return catcher_module.get_cache()


def execute_offloaded_module(
    module,
    buffer,
    dev,
    nsamples=None,
    overwrite_buffer=True,
    cached_inputs=None,
    **kwargs,
):
    module.to(dev)
    if not overwrite_buffer:
        new_buffer = []
    for input_index, inp in enumerate(buffer):
        if nsamples is not None and input_index == nsamples:
            break
        if cached_inputs is None:
            module_kwargs = kwargs
        else:
            module_kwargs = {
                key: cached_inputs[key][input_index] for key in cached_inputs
            }
            module_kwargs.update(kwargs)
        if isinstance(inp, tuple):
            inp = inp[0]
        output = module(inp.to(dev), **module_kwargs)
        if overwrite_buffer:
            buffer[input_index] = output
        else:
            new_buffer.append(output)

    module.cpu()
    torch.cuda.empty_cache()
    if overwrite_buffer:
        return buffer
    else:
        del buffer
        torch.cuda.empty_cache()
        return new_buffer


def cache_attention_inputs(
    model, dataloader, device, nsamples, target_ids, layer_prefix
):
    if layer_prefix:  # get model-specific path to layers list
        split_prefix = layer_prefix.split(".")
        layers_name = split_prefix[-1]
        model_root_name = ".".join(split_prefix[:-1])
        model_root = operator.attrgetter(model_root_name)(model)
        first_layer = getattr(model_root, layers_name)[0]
    else:
        model_root = model.model
        layers_name = "layers"
        first_layer = model_root.layers[0]

    # send everything up to the first compressable layer to device
    pre_layers_modules = _get_pre_layer_modules(model_root, layers_name)
    for pre_layer in pre_layers_modules:
        pre_layer.to(device)
    first_layer.to(device)

    cached_inputs = catch(
        model=model,
        attention_layer=first_layer,
        target_keys=target_ids,
        data_loader=dataloader,
        nsamples=nsamples,
    )

    for pre_layer in pre_layers_modules:
        pre_layer.cpu()
    first_layer.cpu()
    torch.cuda.empty_cache()

    return cached_inputs


@torch.no_grad()
def ppl_eval_general(
    eval_logits, model, dataloader, dev, nsamples=None, max_samples_per_iteration=128
):
    logger.info("Evaluating perplexity...")

    if nsamples is None:
        nsamples = len(dataloader)

    number_iterations = int(ceil(nsamples / max_samples_per_iteration))
    neg_log_likelihood = 0.0
    number_tokens = 0
    for iteration in range(number_iterations):
        if iteration < number_iterations - 1:
            samples = dataloader[
                iteration * max_samples_per_iteration : (iteration + 1)
                * max_samples_per_iteration
            ]
        else:
            samples = dataloader[iteration * max_samples_per_iteration :]

        logits = eval_logits(model, samples, dev)

        vocabulary_size = logits[0].shape[-1]
        logits = [logit[:, :-1, :].view(-1, vocabulary_size) for logit in logits]
        logits = torch.cat(logits, dim=0).contiguous().to(torch.float32)

        labels = [sample[:, 1:].view(-1) for sample in samples]
        labels = torch.cat(labels, dim=0).to(dev)
        neg_log_likelihood += torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        number_tokens += labels.numel()
        perplexity = torch.exp(neg_log_likelihood / number_tokens)
        logger.debug(
            f"Processed iteration {iteration} of {number_iterations} with perplexity: "
            f"{perplexity}"
        )

    perplexity = torch.exp(neg_log_likelihood / number_tokens)
    logger.info(f"Evaluated perplexity: {perplexity}")

    return perplexity.item()


def _get_pre_layer_modules(model_root, layers_name):
    pre_layers_modules = []
    for name, layer in model_root.named_modules():
        if name.startswith(layers_name):
            break
        if isinstance(layer, Embedding):
            pre_layers_modules.append(layer)

    return pre_layers_modules
