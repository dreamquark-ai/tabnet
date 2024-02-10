import pytest
from pathlib import Path


@pytest.fixture
def context_path() -> Path:
    return Path(__file__).parent.parent.parent
