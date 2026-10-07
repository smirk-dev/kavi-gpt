import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kavi import backend as B  # noqa: E402


@pytest.fixture(autouse=True)
def float64():
    """Gradient checks need float64; restore float32 afterwards."""
    B.set_backend("numpy", precision="float64")
    yield
    B.set_precision("float32")
