"""Make ``pytest`` import ``flamingo`` from this checkout, not from site-packages.

The package is usually pip-installed in editable mode pointing at another
working tree, which would otherwise shadow the ``src/`` here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
