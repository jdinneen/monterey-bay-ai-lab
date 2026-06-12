"""Smoke test for RTX 5090 local Monterey Bay AI Lab environments.

This script intentionally runs real CUDA work rather than only checking imports.
It exits non-zero if PyTorch CUDA or XGBoost GPU execution is unavailable.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass


@dataclass
class SmokeResult:
    python: str
    platform: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    device_name: str | None
    arch_list: list[str]
    pytorch_matmul_sum: float
    xgboost_version: str
    xgboost_device: str
    xgboost_rmse: float


def run_pytorch_check() -> tuple[str, str | None, bool, str | None, list[str], float]:
    import torch

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("PyTorch reports CUDA is not available.")

    device_name = torch.cuda.get_device_name(0)
    arch_list = list(torch.cuda.get_arch_list())

    if "sm_120" not in arch_list:
        raise RuntimeError(
            "PyTorch CUDA arch list does not include sm_120. "
            "RTX 5090 needs a Blackwell-compatible CUDA build."
        )

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    a = torch.randn((2048, 2048), device=device)
    b = torch.randn((2048, 2048), device=device)
    c = a @ b
    torch.cuda.synchronize()
    return torch.__version__, torch.version.cuda, cuda_available, device_name, arch_list, float(c.sum().item())


def run_xgboost_check() -> tuple[str, str, float]:
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    x = rng.normal(size=(5000, 16)).astype("float32")
    y = (
        0.8 * x[:, 0]
        - 0.4 * x[:, 1]
        + 0.2 * x[:, 2] * x[:, 3]
        + rng.normal(scale=0.05, size=x.shape[0])
    ).astype("float32")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, shuffle=True
    )

    model = xgb.XGBRegressor(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device="cuda",
        objective="reg:squarederror",
        random_state=42,
    )

    try:
        model.fit(x_train, y_train, verbose=False)
        booster_config = json.loads(model.get_booster().save_config())
        device = (
            booster_config.get("learner", {})
            .get("generic_param", {})
            .get("device", "unknown")
        )
        if "cuda" not in str(device).lower():
            raise RuntimeError(f"XGBoost trained but did not report CUDA device usage: {device}")
    except xgb.core.XGBoostError as exc:
        # Older XGBoost builds used gpu_hist. Try it only as a compatibility path.
        model = xgb.XGBRegressor(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            tree_method="gpu_hist",
            objective="reg:squarederror",
            random_state=42,
        )
        try:
            model.fit(x_train, y_train, verbose=False)
            device = "gpu_hist"
        except xgb.core.XGBoostError as fallback_exc:
            raise RuntimeError(
                f"XGBoost GPU failed with device=cuda ({exc}) and gpu_hist ({fallback_exc})."
            ) from fallback_exc

    pred = model.predict(x_test)
    rmse = float(mean_squared_error(y_test, pred) ** 0.5)

    if rmse > 0.25:
        raise RuntimeError(f"XGBoost smoke model RMSE is unexpectedly high: {rmse:.4f}")

    return xgb.__version__, device, rmse


def main() -> int:
    torch_version, torch_cuda_version, cuda_available, device_name, arch_list, matmul_sum = (
        run_pytorch_check()
    )
    xgb_version, xgb_device, xgb_rmse = run_xgboost_check()

    result = SmokeResult(
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        device_name=device_name,
        arch_list=arch_list,
        pytorch_matmul_sum=matmul_sum,
        xgboost_version=xgb_version,
        xgboost_device=xgb_device,
        xgboost_rmse=xgb_rmse,
    )

    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

