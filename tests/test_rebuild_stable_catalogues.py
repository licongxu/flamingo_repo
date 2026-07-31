import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/rebuild_stable_catalogues.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("rebuild_stable_catalogues", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inventory_lists_complete_matrix_without_writes(mod, tmp_path, capsys):
    """Inventory must expose missing scope without creating staging artifacts."""
    assert mod.main(["--root", str(tmp_path), "inventory"]) == 0

    output = capsys.readouterr().out
    assert "17 target(s), 68 canonical CSV(s)" in output
    assert "L1_m9/Jet_fgas-4sigma" in output
    assert "L2p8_m9/lightcone7" in output
    assert list(tmp_path.iterdir()) == []


def test_requested_snapshots_narrow_a_pilot_without_changing_defaults(mod, tmp_path):
    """A pilot must not accidentally start every remote snapshot."""
    target = mod.catalogue_targets(tmp_path)[0]

    assert mod._snapshot_numbers(target, (75,)) == (75,)
    assert mod._snapshot_numbers(target, ()) == tuple(range(18, 78))
