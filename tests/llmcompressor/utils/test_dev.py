import contextlib
import os
from functools import wraps
from unittest.mock import MagicMock, patch

import pytest
import torch
from compressed_tensors.utils import patch_attr

from llmcompressor.utils.dev import download_checkpoint_first, load_context

STUB = "nm-testing/tinysmokellama-3.2"


class DummyModel:
    """Stands in for a `PreTrainedModel` so that no checkpoint is actually loaded."""

    calls: list = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))


@contextlib.contextmanager
def distributed(world_size=2, initialized=True, local_rank="0", peer_failed=False):
    DummyModel.calls.clear()

    def all_reduce(tensor, op=None):
        if peer_failed:
            tensor.fill_(0)

    with (
        patch("llmcompressor.utils.dev.snapshot_download") as download,
        patch.object(torch.distributed, "all_reduce", side_effect=all_reduce) as reduce,
        patch.object(torch.distributed, "get_backend", return_value="gloo"),
        patch.object(torch.distributed, "is_initialized", return_value=initialized),
        patch.object(torch.distributed, "get_world_size", return_value=world_size),
        patch.dict(os.environ, {"LOCAL_RANK": local_rank}),
    ):
        yield download, reduce


@pytest.mark.unit
@pytest.mark.parametrize("initialized,world_size", [(False, 2), (True, 1)])
def test_noop_without_multi_rank_distributed(initialized, world_size):
    with distributed(world_size, initialized) as (download, reduce):
        with download_checkpoint_first(DummyModel):
            DummyModel.from_pretrained(STUB)

    download.assert_not_called()
    reduce.assert_not_called()
    assert len(DummyModel.calls) == 1


@pytest.mark.unit
def test_local_rank_zero_downloads_and_forwards_arguments():
    with distributed(local_rank="0") as (download, reduce):
        with download_checkpoint_first(DummyModel):
            DummyModel.from_pretrained(
                STUB, revision="abc123", cache_dir="/custom/cache", token="hf_secret"
            )

    assert download.call_args.args == (STUB,)
    assert download.call_args.kwargs["revision"] == "abc123"
    assert download.call_args.kwargs["cache_dir"] == "/custom/cache"
    assert download.call_args.kwargs["token"] == "hf_secret"
    reduce.assert_called_once()

    # the load must target the same revision and cache as the prefetch
    _, load_kwargs = DummyModel.calls[0]
    assert load_kwargs["revision"] == "abc123"
    assert load_kwargs["cache_dir"] == "/custom/cache"


@pytest.mark.unit
def test_other_ranks_wait_without_downloading():
    with distributed(local_rank="1") as (download, reduce):
        with download_checkpoint_first(DummyModel):
            DummyModel.from_pretrained(STUB)

    download.assert_not_called()
    reduce.assert_called_once()
    assert len(DummyModel.calls) == 1


@pytest.mark.unit
def test_force_download_is_disabled_after_prefetch():
    """Otherwise every rank re-downloads after the collective, restoring the race."""
    with distributed(local_rank="0") as (download, _):
        with download_checkpoint_first(DummyModel):
            DummyModel.from_pretrained(STUB, force_download=True)

    assert download.call_args.kwargs["force_download"] is True
    assert DummyModel.calls[0][1]["force_download"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs", [{"local_files_only": True}, {}, {"pretrained_model_name_or_path": None}]
)
def test_skips_when_cache_is_not_populated(kwargs, tmp_path):
    """Local directories, offline loads and a missing stub never write to the hub
    cache. The last is left for transformers to report."""
    args = (
        ()
        if "pretrained_model_name_or_path" in kwargs
        else ((STUB,) if kwargs else (str(tmp_path),))
    )
    with distributed(local_rank="0") as (download, reduce):
        with download_checkpoint_first(DummyModel):
            DummyModel.from_pretrained(*args, **kwargs)

    download.assert_not_called()
    reduce.assert_not_called()
    assert len(DummyModel.calls) == 1


@pytest.mark.unit
def test_downloader_failure_propagates_and_blocks_the_load():
    with distributed(local_rank="0", peer_failed=True) as (download, reduce):
        download.side_effect = OSError("gated repo")
        with download_checkpoint_first(DummyModel):
            with pytest.raises(OSError, match="gated repo"):
                DummyModel.from_pretrained(STUB)

    reduce.assert_called_once()
    assert len(DummyModel.calls) == 0


@pytest.mark.unit
def test_peer_failure_blocks_the_load():
    """A rank whose own download succeeded must not load against a partial cache."""
    with distributed(local_rank="1", peer_failed=True) as (_, reduce):
        with download_checkpoint_first(DummyModel):
            with pytest.raises(RuntimeError, match="failed on another rank"):
                DummyModel.from_pretrained(STUB)

    reduce.assert_called_once()
    assert len(DummyModel.calls) == 0


@pytest.mark.unit
def test_nccl_reduces_on_the_accelerator():
    """gloo reduces on cpu, but nccl requires the flag to live on the device."""
    flag = MagicMock()
    flag.item.return_value = 1

    with (
        distributed(local_rank="1"),
        patch.object(torch.distributed, "get_backend", return_value="nccl"),
        (
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("cuda"),
            )
        ),
        patch.object(torch.accelerator, "current_device_index", return_value=3),
        patch.object(torch, "tensor", return_value=flag) as tensor,
    ):
        with download_checkpoint_first(DummyModel):
            DummyModel.from_pretrained(STUB)

    assert tensor.call_args.kwargs["device"] == torch.device("cuda", 3)
    assert len(DummyModel.calls) == 1


@pytest.mark.unit
def test_missing_local_rank_raises_rather_than_downloading_everywhere():
    with distributed() as (download, reduce):
        with patch.dict(os.environ):
            os.environ.pop("LOCAL_RANK", None)
            with download_checkpoint_first(DummyModel):
                with pytest.raises(RuntimeError, match="LOCAL_RANK"):
                    DummyModel.from_pretrained(STUB)

    download.assert_not_called()
    reduce.assert_not_called()
    assert len(DummyModel.calls) == 0


@pytest.mark.unit
def test_restores_from_pretrained():
    original = DummyModel.__dict__["from_pretrained"]
    with distributed():
        with download_checkpoint_first(DummyModel):
            assert DummyModel.__dict__["from_pretrained"] is not original

    assert DummyModel.__dict__["from_pretrained"] is original


@pytest.mark.unit
def test_load_context_downloads_before_inner_patches():
    """`download_checkpoint_first` must be the outermost patch, so the checkpoint is
    fetched before `load_offloaded_model` estimates tensor counts and before
    `load_quantizable_moe` resolves the config."""
    order = []

    def recorder(name):
        @contextlib.contextmanager
        def stub(model_cls=DummyModel):
            original = model_cls.from_pretrained

            @classmethod
            @wraps(original)
            def patched(cls, *args, **kwargs):
                order.append(name)
                return original(*args, **kwargs)

            with patch_attr(model_cls, "from_pretrained", patched):
                yield

        return stub

    with distributed(local_rank="0") as (download, _):
        download.side_effect = lambda *a, **k: order.append("download")
        with (
            patch("llmcompressor.utils.dev.load_offloaded_model", recorder("offload")),
            patch(
                "llmcompressor.modeling.moe.linearize.load_quantizable_moe",
                recorder("moe"),
            ),
            load_context(DummyModel),
        ):
            DummyModel.from_pretrained(STUB)

    assert order == ["download", "moe", "offload"]
