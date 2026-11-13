import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.entrypoints.higgs import HiggsMSECollectorConverter


def _w4(group_size):
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=group_size,
        ),
    )


def test_keeps_schemes_that_share_a_bitwidth_name():
    collector = HiggsMSECollectorConverter(
        candidate_schemes=[_w4(32), _w4(128)], ignore=[]
    )

    assert len(collector.candidate_schemes) == 2
    group_sizes = {
        scheme.weights.group_size for scheme in collector.candidate_schemes.values()
    }
    assert group_sizes == {32, 128}


def test_keeps_a_preset_alongside_a_matching_custom_scheme():
    collector = HiggsMSECollectorConverter(
        candidate_schemes=["W4A16", _w4(128)], ignore=[]
    )

    assert len(collector.candidate_schemes) == 2


def test_process_collects_mse_for_every_candidate():
    collector = HiggsMSECollectorConverter(
        candidate_schemes=[_w4(32), _w4(128)], ignore=[]
    )
    torch.manual_seed(0)
    tensors = {
        f"model.layers.{i}.mlp.down_proj.weight": torch.randn(128, 128)
        for i in range(2)
    }

    collector.process(tensors)

    assert len(collector.mse_matrix) == 2
    for layer_mse in collector.mse_matrix.values():
        assert len(layer_mse) == 2
