try:
    from accelerate import Accelerator
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp._common_utils import TrainingState
except ImportError:
    FullyShardedDataParallel = None
    Accelerator = None

from contextlib import nullcontext

__all__ = [
    "summon_full_params_context",
    "main_process_first_context",
    "fix_fsdp_module_name",
]

FSDP_WRAPPER_NAME = "_fsdp_wrapped_module"


def summon_full_params_context(model, offload_to_cpu: bool = False):
    if FullyShardedDataParallel is not None:
        # avoid nested summon_full_param context
        if (
            hasattr(model, "training_state")
            and model.training_state is TrainingState.SUMMON_FULL_PARAMS
        ):
            return nullcontext()
        return FullyShardedDataParallel.summon_full_params(
            model, offload_to_cpu=offload_to_cpu
        )

    return nullcontext()


def main_process_first_context():
    """
    Creates a context manager where the main process runs the block before all other
    processes. Returns a nullcontext when called from a single process application.
    """
    if Accelerator is None:
        return nullcontext()

    return Accelerator().main_process_first()


def fix_fsdp_module_name(name: str) -> str:
    """
    Remove FSDP wrapper prefixes from a module name.
    Accounts for scenario where FSDP_WRAPPER_NAME is
    at the end of the name, as well as in the middle.

    :param name: name to strip
    :return: stripped name
    """
    return name.replace(FSDP_WRAPPER_NAME + ".", "").replace(
        "." + FSDP_WRAPPER_NAME, ""
    )
