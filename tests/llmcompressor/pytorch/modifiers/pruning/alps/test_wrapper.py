import pytest
import torch

from llmcompressor.modifiers.pruning.alps.utils.alps_wrapper import ALPSWrapper
from llmcompressor.pytorch.utils import tensor_sparsity
from tests.llmcompressor.pytorch.helpers import DummyNetwork

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_alps_wrapper(
    dim: int,
    init_sparsity: float = 0,
) -> ALPSWrapper:
    layer = DummyNetwork(
        p=dim,
        init_sparsity=init_sparsity,
    )
    layer.to(DEV)
    wrapper = ALPSWrapper(name="test", layer=layer)
    out = torch.zeros(dim)  # dummy
    for i in range(dim):
        inp = torch.zeros((1, dim)).to(DEV)
        inp[0, i] = 1
        wrapper.add_batch(
            inp=inp, out=out
        )  # This ensures H = (2/nsamples) * X^T * X = (2/nsamples) * I
    return wrapper


def _project_nm(W: torch.Tensor, n: int, m: int) -> torch.Tensor:
    p, q = W.shape
    W = W.reshape(p * q // m, m)
    _, idx = torch.topk(-(W**2), m - n, dim=1)
    D = W.clone()
    D = D.scatter(
        src=torch.zeros((p * q // m, m - n)).to(W.device),
        dim=1,
        index=idx,
    )
    return D.reshape(p, q)


@pytest.mark.parametrize(
    "dim",
    [
        165,
        321,
        640,
        1000,
    ],
)
@pytest.mark.parametrize(
    "sparsity",
    [0.12, 0.2, 0.8],
)
@pytest.mark.parametrize(
    "seed",
    [0, 10, 123],
)
def test_alps_unstructured(dim, sparsity, seed):
    """
    This tests if the weights of ALPS are correctly pruned.
    This is done by considering a case where X^T*X = I, so the
    layerwise ALPS solution **should be close** to Magnitude Pruning.
    This is for unstructured pruning.
    """

    torch.manual_seed(seed)

    wrapper = _get_alps_wrapper(dim)

    W0 = wrapper.layer.weight.clone().reshape(-1)

    wrapper.compress(
        sparsity=sparsity,
    )
    W1 = wrapper.layer.weight.clone().reshape(-1)  # ALPS weights

    k_spar = int((1 - sparsity) * (dim**2))

    _, idx = torch.topk(W0**2, dim**2 - k_spar, largest=False)

    W2 = W0.clone()
    W2[idx] = torch.zeros_like(W2[idx]).to(DEV)  # MP weights

    # ALPS and MP should be close. We allow a 2% tolerance.
    assert torch.linalg.norm(W1 - W2).item() / torch.linalg.norm(W0).item() < 2e-2
    # ALPS solution should be sparse.
    assert tensor_sparsity(W1) >= sparsity

    idx = torch.where(torch.abs(W1) > 0)[0]
    # The non-zeros of ALPS solution should be the same as the original weights.
    # This is because X^T*X=I so pcg backsolve must return the original weights.
    assert torch.allclose(W1[idx], W0[idx])


@pytest.mark.parametrize(
    "dim",
    [
        165,
        321,
        640,
        1000,
    ],
)
@pytest.mark.parametrize(
    "init_sparsity",
    [0.03, 0.12, 0.2],
)
@pytest.mark.parametrize(
    "goal_sparsity",
    [0.3, 0.41, 0.52],
)
def test_alps_preserve_sparsity(dim, init_sparsity, goal_sparsity):
    """
    This tests if ALPS properly preserves the sparsity mask.
    """

    torch.manual_seed(0)

    wrapper = _get_alps_wrapper(dim, init_sparsity)
    W0 = wrapper.layer.weight.clone().reshape(-1)
    # The initial layer is sparse.
    assert tensor_sparsity(W0) >= init_sparsity

    wrapper.compress(
        sparsity=goal_sparsity,
    )

    W1 = wrapper.layer.weight.clone().reshape(-1)  # ALPS weights
    # Initial sparsity mask
    idx = torch.where(torch.abs(W0) == 0)[0]
    # Sparsity mask must be preserved.
    assert torch.linalg.norm(W1[idx]) < 1e-6

    # ALPS solution should be sparse.
    assert tensor_sparsity(W1) >= goal_sparsity

    idx = torch.where(torch.abs(W1) > 0)
    # The non-zeros of ALPS solution should be the same as the original weights.
    # This is because X^T*X=I so pcg backsolve must return the original weights.
    assert torch.allclose(W1[idx], W0[idx])


@pytest.mark.parametrize(
    "dim",
    [
        160,
        320,
        640,
        1000,
    ],
)
@pytest.mark.parametrize(
    "nm_n",
    [1, 2, 4],
)
def test_alps_semi_structured(dim, nm_n):
    """
    This tests if the weights of ALPS are correctly pruned for N:M
    semi-structured sparsity.
    This is done by considering a case where X^T*X = I.
    """
    nm_m = 2 * nm_n
    torch.manual_seed(0)

    wrapper = _get_alps_wrapper(dim)

    W0 = wrapper.layer.weight.clone()

    wrapper.compress(
        sparsity=0.5,
        prunen=nm_n,
        prunem=nm_m,
    )
    W1 = wrapper.layer.weight.clone()  # ALPS weights

    # Check ALPS mask has N:M structure
    assert torch.allclose(W1, _project_nm(W1, nm_n, nm_m))

    W0 = W0.reshape(-1)
    W1 = W1.reshape(-1)
    idx = torch.where(torch.abs(W1) > 0)[0]
    # The non-zeros of ALPS solution should be the same as the original weights.
    # This is because X^T*X=I so pcg backsolve must return the original weights.
    assert torch.allclose(W1[idx], W0[idx])
