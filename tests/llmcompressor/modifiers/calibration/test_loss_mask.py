from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
    initialize_qparams,
)
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import (
    calibrate_input_hook,
    calibrate_key_hook,
    calibrate_output_hook,
    calibrate_query_hook,
    calibrate_value_hook,
    initialize_observer,
    update_qparams,
)
from llmcompressor.pipelines.basic import BasicPipeline
from tests.testing_utils import torchrun

INPUT_ARGS = [
    QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor"),
    QuantizationArgs(num_bits=8, type="float", symmetric=True, strategy="tensor"),
    QuantizationArgs(
        num_bits=4,
        type="float",
        symmetric=True,
        strategy="tensor_group",
        group_size=4,
        dynamic="local",
    ),
]
ATTN_ARGS = [
    QuantizationArgs(num_bits=8, type="float", symmetric=True, strategy="tensor"),
    QuantizationArgs(num_bits=8, type="float", symmetric=True, strategy="attn_head"),
]
STATS = ("min_vals", "max_vals")


def _linear(args: QuantizationArgs, output: bool = False) -> torch.nn.Linear:
    linear = torch.nn.Linear(8, 8, bias=False)
    if output:
        scheme = QuantizationScheme(targets=[], output_activations=deepcopy(args))
    else:
        scheme = QuantizationScheme(targets=[], input_activations=deepcopy(args))
    initialize_module_for_quantization(linear, scheme)
    initialize_observer(linear, "output" if output else "input")
    return linear


def _attention(args: QuantizationArgs, num_heads: int, head_dim: int):
    attention = torch.nn.Module()
    attention.quantization_scheme = QuantizationScheme(
        targets=[], input_activations=deepcopy(args)
    )
    attention.quantization_status = QuantizationStatus.INITIALIZED
    for base_name in ("q", "k", "v"):
        # observed shape: (num_heads, seq_len, head_dim)
        initialize_qparams(
            attention, base_name, args, (num_heads, 1, head_dim), torch.float32
        )
        initialize_observer(attention, base_name)
    return attention


def _assert_same_stats(observer, reference):
    for name in STATS:
        torch.testing.assert_close(
            getattr(observer, name), getattr(reference, name), rtol=0, atol=0
        )


def _with_masked_outlier(value, mask, seq_dim):
    """Set every masked token to a large value"""
    value = value.clone()
    value.movedim(seq_dim, 1)[mask == 0] = 1000.0  # writes through the view
    return value


@pytest.mark.parametrize("args", INPUT_ARGS)
@pytest.mark.parametrize("output", [False, True])
def test_activation_hooks_observe_masked_tokens(args, output):
    torch.manual_seed(0)
    value = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 1]])
    value = _with_masked_outlier(value, mask, seq_dim=1)

    module = _linear(args, output)
    reference = _linear(args, output)
    observer = "output_observer" if output else "input_observer"

    with create_session() as session:
        session.state.loss_masks = [torch.ones(2, 5), mask]
        session.state.current_batch_idx = 1
        if output:
            calibrate_output_hook(module, (value,), value)
        else:
            calibrate_input_hook(module, (value,))

    # only the kept tokens were observed, in the shape the observer expects
    getattr(reference, observer)(value[mask.bool()].unsqueeze(0))
    _assert_same_stats(getattr(module, observer), getattr(reference, observer))
    assert getattr(module, observer).max_vals.max() < 1000.0


@pytest.mark.parametrize("args", INPUT_ARGS)
def test_input_hook_unchanged_without_mask(args):
    torch.manual_seed(0)
    value = torch.randn(2, 5, 8)
    expected = _linear(args)
    expected.input_observer(value)

    # use_loss_mask=False: no masks in the state
    module = _linear(args)
    with create_session():
        calibrate_input_hook(module, (value,))
    _assert_same_stats(module.input_observer, expected.input_observer)

    # an all-ones mask observes every token
    module = _linear(args)
    with create_session() as session:
        session.state.loss_masks = [torch.ones(2, 5, dtype=torch.long)]
        session.state.current_batch_idx = 0
        calibrate_input_hook(module, (value,))
    _assert_same_stats(module.input_observer, expected.input_observer)


@pytest.mark.parametrize("args", ATTN_ARGS)
def test_attention_hooks_observe_masked_tokens(args):
    torch.manual_seed(0)
    batch_size, num_heads, seq_len, head_dim = 2, 3, 5, 4
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    states = {
        name: _with_masked_outlier(
            torch.randn(batch_size, num_heads, seq_len, head_dim), mask, seq_dim=2
        )
        for name in ("q", "k", "v")
    }

    attention = _attention(args, num_heads, head_dim)
    reference = _attention(args, num_heads, head_dim)
    with create_session() as session:
        session.state.loss_masks = [mask]
        session.state.current_batch_idx = 0
        # the hooks only observe: returning None keeps the states unchanged
        assert calibrate_query_hook(attention, states["q"]) is None
        assert calibrate_key_hook(attention, states["k"]) is None
        assert calibrate_value_hook(attention, states["v"]) is None

    for name, value in states.items():
        # (num_kept_tokens, num_heads, head_dim) -> (1, num_heads, num_kept, head_dim)
        kept = value.transpose(1, 2)[mask.bool()].unsqueeze(0).transpose(1, 2)
        getattr(reference, f"{name}_observer")(kept)
        observer = getattr(attention, f"{name}_observer")
        _assert_same_stats(observer, getattr(reference, f"{name}_observer"))
        assert observer.max_vals.max() < 1000.0


def test_no_masking_outside_a_calibration_batch():
    value = torch.randn(1, 4, 8)
    expected = _linear(INPUT_ARGS[0])
    expected.input_observer(value)

    module = _linear(INPUT_ARGS[0])
    with create_session() as session:
        session.state.loss_masks = [torch.zeros(1, 4)]
        assert session.state.current_batch_idx == -1
        calibrate_input_hook(module, (value,))
    _assert_same_stats(module.input_observer, expected.input_observer)


def test_fully_masked_batch_is_skipped():
    module = _linear(INPUT_ARGS[0])
    value = torch.randn(1, 4, 8)
    with create_session() as session:
        session.state.loss_masks = [torch.zeros(1, 4), torch.ones(1, 4)]
        session.state.current_batch_idx = 0
        calibrate_input_hook(module, (value,))
        assert not module.input_observer.has_statistics

        session.state.current_batch_idx = 1
        calibrate_input_hook(module, (value,))
        assert module.input_observer.has_statistics


@pytest.mark.parametrize(
    "shape",
    [
        (3, 8),  # routed MoE experts see a flattened subset of the tokens
        (2, 7, 8),  # another sequence length, such as a vision encoder's
        (2, 5, 4, 8),  # not (batch, seq, hidden), such as a Conv2d input
    ],
)
def test_misaligned_input_raises(shape):
    module = _linear(INPUT_ARGS[0])
    with create_session() as session:
        session.state.loss_masks = [torch.ones(2, 5)]
        session.state.current_batch_idx = 0
        with pytest.raises(ValueError, match="cannot be aligned"):
            calibrate_input_hook(module, (torch.randn(shape),))


@pytest.mark.parametrize("shape", [(2, 3, 7, 4), (2, 5, 4)])
def test_misaligned_attention_states_raise(shape):
    attention = _attention(ATTN_ARGS[0], num_heads=3, head_dim=4)
    with create_session() as session:
        session.state.loss_masks = [torch.ones(2, 5)]
        session.state.current_batch_idx = 0
        with pytest.raises(ValueError, match="cannot be aligned"):
            calibrate_key_hook(attention, torch.randn(shape))


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config).eval()
    # token 63 is a spike in hidden dim 0, which q/k/v_proj of the first layer
    # amplify, so that its activations dominate the scales whenever it is observed
    model.model.embed_tokens.weight.data[63] = torch.nn.functional.one_hot(
        torch.tensor(0), config.hidden_size
    ).float()
    attention = model.model.layers[0].self_attn
    for proj in (attention.q_proj, attention.k_proj, attention.v_proj):
        proj.weight.data[:, 0] = 1.0
    return model


def _calibrated_qparams(samples, use_loss_mask) -> dict[str, torch.Tensor]:
    model = _tiny_llama()
    recipe = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=8, strategy="channel"),
                input_activations=QuantizationArgs(
                    num_bits=8, type="float", strategy="tensor", dynamic=False
                ),
            )
        },
        ignore=["lm_head"],
        kv_cache_scheme=QuantizationArgs(
            num_bits=8, type="float", strategy="tensor", dynamic=False
        ),
    )
    with create_session() as session:
        session.initialize(model=model, recipe=recipe, start=-1)
        dataloader = DataLoader(deepcopy(samples), batch_size=None)
        BasicPipeline()(
            model, dataloader, DatasetArguments(use_loss_mask=use_loss_mask)
        )
        session.finalize()

    qparams = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.endswith(("input_scale", "k_scale", "v_scale"))
    }
    assert any(name.endswith("k_scale") for name in qparams)
    assert any(name.endswith("input_scale") for name in qparams)
    return qparams


def _samples(last_token: int, mask: list[int]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.tensor([[1, 5, 9, 13, 17, last_token]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "loss_mask": torch.tensor([mask]),
        },
        {
            "input_ids": torch.tensor([[2, 6, 10, 14, 18, last_token]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "loss_mask": torch.tensor([mask]),
        },
    ]


def test_masked_tokens_do_not_affect_scales():
    # the samples differ only in the last token, which is masked out. Attention is
    # causal, so the kept tokens see the same activations in both runs
    mask = [1, 1, 1, 1, 1, 0]
    outlier = _calibrated_qparams(_samples(63, mask), use_loss_mask=True)
    plain = _calibrated_qparams(_samples(3, mask), use_loss_mask=True)
    for name, value in outlier.items():
        torch.testing.assert_close(value, plain[name], rtol=0, atol=0)

    # without masking, the outlier token changes the scales
    outlier = _calibrated_qparams(_samples(63, mask), use_loss_mask=False)
    plain = _calibrated_qparams(_samples(3, mask), use_loss_mask=False)
    changed = [name for name in outlier if not torch.equal(outlier[name], plain[name])]
    assert any(name.endswith("input_scale") for name in changed)
    assert any(name.endswith(("k_scale", "v_scale")) for name in changed)


def test_all_ones_mask_matches_no_masking():
    samples = _samples(63, [1, 1, 1, 1, 1, 1])
    masked = _calibrated_qparams(samples, use_loss_mask=True)
    unmasked = _calibrated_qparams(samples, use_loss_mask=False)
    assert masked.keys() == unmasked.keys()
    for name, value in masked.items():
        torch.testing.assert_close(value, unmasked[name], rtol=0, atol=0)


@torchrun(world_size=2)
def test_sync_with_a_fully_masked_rank():
    """
    With DDP, rank 1's samples are fully masked, so its observers hold no
    statistics. Both ranks must still join the same all-reduces (a mismatch hangs
    or fails), and both must end with rank 0's statistics and scales.
    """
    dist.init_process_group(
        backend="gloo", init_method="env://", timeout=timedelta(seconds=30)
    )
    try:
        rank = dist.get_rank()
        torch.manual_seed(rank)
        value = torch.randn(1, 4, 8)
        static = QuantizationArgs(
            num_bits=8, type="int", symmetric=True, observer="static_minmax"
        )
        moving = QuantizationArgs(num_bits=8, type="int", symmetric=True)
        masked = [_linear(static), _linear(moving)]
        unmasked, unused = _linear(static), _linear(static)

        with create_session() as session:
            session.state.loss_masks = [torch.full((1, 4), 1 - rank)]
            session.state.current_batch_idx = 0
            for module in masked:
                calibrate_input_hook(module, (value,))
        unmasked.input_observer(value)
        if rank == 0:
            unused.input_observer(value)  # e.g. an expert routed no tokens on rank 1
        assert masked[0].input_observer.has_statistics == (rank == 0)

        modules = masked + [unmasked, unused]
        QuantizationModifier(targets="Linear", scheme="W8A8").sync_obs_act_stats(
            modules
        )
        update_qparams(modules, "input")

        for module in modules:
            scale = module.input_scale.detach().clone()
            gathered = [torch.empty_like(scale) for _ in range(2)]
            dist.all_gather(gathered, scale)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)

        # the masked and unused modules take rank 0's statistics alone
        expected = _linear(static)
        rank0_value = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(0))
        expected.input_observer(rank0_value)
        update_qparams([expected], "input")
        for module in masked + [unused]:
            torch.testing.assert_close(
                module.input_scale, expected.input_scale, rtol=0, atol=0
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
