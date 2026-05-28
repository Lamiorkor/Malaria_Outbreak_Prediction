"""
conftest.py
===========
Shared pytest configuration for the malaria CI test suite.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

# Compatibility shim: datetime.UTC was added in Python 3.11.
# main.py uses it; this lets the local 3.10 conda env import main.py too.
if not hasattr(_dt, "UTC"):
    _dt.UTC = _dt.timezone.utc

# Make project source folders importable from tests/
ROOT = Path(__file__).resolve().parent.parent
for sub in [
    ROOT,
    ROOT / "dockerisation_and_deployment" / "webservices",
    ROOT / "training_pipeline",
]:
    sub_str = str(sub)
    if sub_str not in sys.path:
        sys.path.insert(0, sub_str)
