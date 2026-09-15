import contextlib
from functools import wraps
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.utils import patch_attr

from llmcompressor.utils.dev import load_context, pin_checkpoint_revision

STUB = "nm-testing/tinysmokellama-3.2"
SHA = "0123456789abcdef0123456789abcdef01234567"


class DummyModel:
    """Stands in for a `PreTrainedModel` so that no checkpoint is actually loaded."""

    calls: list = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))


@contextlib.contextmanager
def distributed(world_size=2, initialized=True, rank=0, resolved=SHA):
    """Patches distributed and the hub so no network or process group is needed."""
    DummyModel.calls.clear()

    def broadcast_object_list(obj, src=0):
        obj[0] = resolved  # what rank `src` resolved, as seen by every rank

    api = patch("llmcompressor.utils.dev.HfApi")
    with (
        api as hf_api,
        patch.object(
            torch.distributed,
            "broadcast_object_list",
            side_effect=broadcast_object_list,
        ) as broadcast,
        patch.object(torch.distributed, "is_initialized", return_value=initialized),
        patch.object(torch.distributed, "get_world_size", return_value=world_size),
        patch.object(torch.distributed, "get_rank", return_value=rank),
    ):
        hf_api.return_value.model_info.return_value.sha = SHA
        yield hf_api, broadcast


@pytest.mark.unit
@pytest.mark.parametrize("initialized,world_size", [(False, 2), (True, 1)])
def test_noop_without_multi_rank_distributed(initialized, world_size):
    with distributed(world_size, initialized) as (hf_api, broadcast):
        with pin_checkpoint_revision(DummyModel):
            DummyModel.from_pretrained(STUB)

    hf_api.assert_not_called()
    broadcast.assert_not_called()
    assert DummyModel.calls[0][1].get("revision") is None


@pytest.mark.unit
def test_rank_zero_resolves_and_load_uses_the_commit():
    with distributed(rank=0) as (hf_api, broadcast):
        with pin_checkpoint_revision(DummyModel):
            DummyModel.from_pretrained(STUB, token="hf_secret")

    hf_api.assert_called_once_with(token="hf_secret")
    hf_api.return_value.model_info.assert_called_once_with(STUB, revision=None)
    broadcast.assert_called_once()
    assert DummyModel.calls[0][1]["revision"] == SHA


@pytest.mark.unit
def test_other_ranks_take_the_broadcast_commit_without_resolving():
    """Only rank zero queries the hub, so the branch cannot move between ranks."""
    with distributed(rank=1) as (hf_api, broadcast):
        with pin_checkpoint_revision(DummyModel):
            DummyModel.from_pretrained(STUB)

    hf_api.assert_not_called()
    broadcast.assert_called_once()
    assert DummyModel.calls[0][1]["revision"] == SHA


@pytest.mark.unit
def test_symbolic_revision_is_replaced():
    with distributed(rank=0) as (hf_api, _):
        with pin_checkpoint_revision(DummyModel):
            DummyModel.from_pretrained(STUB, revision="my-branch")

    hf_api.return_value.model_info.assert_called_once_with(STUB, revision="my-branch")
    assert DummyModel.calls[0][1]["revision"] == SHA


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs", [{"revision": SHA}, {}, {"pretrained_model_name_or_path": None}]
)
def test_skips_when_there_is_nothing_to_resolve(kwargs, tmp_path):
    """A commit hash is already immutable, local dirs have no hub cache, and a
    missing stub is left for transformers to report."""
    args = (
        ()
        if "pretrained_model_name_or_path" in kwargs
        else ((STUB,) if kwargs else (str(tmp_path),))
    )
    with distributed(rank=0) as (hf_api, broadcast):
        with pin_checkpoint_revision(DummyModel):
            DummyModel.from_pretrained(*args, **kwargs)

    hf_api.assert_not_called()
    broadcast.assert_not_called()
    assert len(DummyModel.calls) == 1


@pytest.mark.unit
def test_unresolvable_revision_loads_unpinned():
    """Offline or gated repos should surface their own error, not ours."""
    with distributed(rank=0, resolved=None) as (hf_api, broadcast):
        hf_api.return_value.model_info.side_effect = OSError("offline")
        with pin_checkpoint_revision(DummyModel):
            DummyModel.from_pretrained(STUB, revision="main")

    broadcast.assert_called_once()
    assert DummyModel.calls[0][1]["revision"] == "main"


@pytest.mark.unit
def test_restores_from_pretrained():
    original = DummyModel.__dict__["from_pretrained"]
    with distributed():
        with pin_checkpoint_revision(DummyModel):
            assert DummyModel.__dict__["from_pretrained"] is not original

    assert DummyModel.__dict__["from_pretrained"] is original


@pytest.mark.unit
def test_load_context_pins_before_inner_patches():
    """`pin_checkpoint_revision` must be the outermost patch, so the inner contexts
    resolve the config and estimate tensors against the pinned commit."""
    order = []

    def recorder(name):
        @contextlib.contextmanager
        def stub(model_cls=DummyModel):
            original = model_cls.from_pretrained

            @classmethod
            @wraps(original)
            def patched(cls, *args, **kwargs):
                order.append((name, kwargs.get("revision")))
                return original(*args, **kwargs)

            with patch_attr(model_cls, "from_pretrained", patched):
                yield

        return stub

    with distributed(rank=0):
        with (
            patch("llmcompressor.utils.dev.load_offloaded_model", recorder("offload")),
            patch(
                "llmcompressor.modeling.moe.linearize.load_quantizable_moe",
                recorder("moe"),
            ),
            load_context(DummyModel),
        ):
            DummyModel.from_pretrained(STUB)

    # both inner patches must already see the pinned commit
    assert order == [("moe", SHA), ("offload", SHA)]
