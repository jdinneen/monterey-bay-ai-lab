from __future__ import annotations

import importlib
import sys


def test_forecast_import_does_not_load_gpu_analysis_module() -> None:
    sys.modules.pop("mbari_forecast_v2", None)
    sys.modules.pop("mbari_gpu_analysis", None)

    importlib.import_module("mbari_forecast_v2")

    assert "mbari_gpu_analysis" not in sys.modules


def test_deep_models_import_does_not_load_tensorboard() -> None:
    sys.modules.pop("mbari_deep_models", None)

    importlib.import_module("mbari_deep_models")

    assert "torch.utils.tensorboard" not in sys.modules
