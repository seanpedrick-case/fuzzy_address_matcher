from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from gradio.blocks import Blocks

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.mark.smoke
def test_app_module_exposes_gradio_block() -> None:
    app_module = importlib.import_module("app")
    assert hasattr(app_module, "block")
    assert isinstance(app_module.block, Blocks)


@pytest.mark.smoke
def test_console_entrypoint_is_importable() -> None:
    gradio_app_module = importlib.import_module("tools.gradio_app")
    assert hasattr(gradio_app_module, "main")
    assert callable(gradio_app_module.main)
