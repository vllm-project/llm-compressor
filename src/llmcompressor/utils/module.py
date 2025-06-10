from typing import Callable, Union

import tqdm
from torch.nn import Module


def module_bfs(
    module: Module,
    func: Callable[[Module], Module],
    pre: bool = True,
    progress: Union[bool, tqdm.tqdm] = False,
) -> Module:
    if progress is True:
        total = len(list(module.modules()))
        progress = tqdm.tqdm(total=total)

    if pre:
        module = func(module)

    for name, child in list(module.named_children()):
        module.add_module(name, module_bfs(child, func, pre, progress))

    if not pre:
        module = func(module)

    if isinstance(progress, tqdm.tqdm):
        progress.update(1)

    return module
