"""Common pytest fixtures.

These tests are unit-level and intentionally avoid touching torch /
sentence-transformers; they exercise pure logic so they run in milliseconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the package importable when running ``pytest`` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Disable UI by default to keep tests focused on the API + core layers.
# We do NOT set ``LWE_MODELS_PRELOAD`` globally because individual settings
# tests assert default behaviour; tests that boot the real app set it
# themselves.
os.environ.setdefault("LWE_ENABLE_UI", "false")
