from dataclasses import dataclass
from enum import Enum


# TODO: maybe test type as decorators?
class TestType(Enum):
    SANITY = "sanity"
    REGRESSION = "regression"
    SMOKE = "smoke"


class Cadence(Enum):
    COMMIT = "commit"
    WEEKLY = "weekly"
    NIGHTLY = "nightly"


@dataclass
class TestConfig:
    test_type: TestType
    cadence: Cadence


@dataclass
class CustomTestConfig(TestConfig):
    script_path: str
