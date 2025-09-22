import pytest

from llmcompressor.core import State
from tests.llmcompressor.pytorch.helpers import LinearNet


@pytest.fixture
def state():
    return State(model=LinearNet())
