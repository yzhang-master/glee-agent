import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _null_targets():
    """Keep every test hermetic: never let a real data/targets.json leak
    into strategy behavior. Tests that want targets call set_targets()."""
    from glee_agent.theory import targets

    targets.set_targets(targets.Targets.null())
    yield
    targets.set_targets(None)
