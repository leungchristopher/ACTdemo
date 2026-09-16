"""Check the LeRobot version required by the local integration."""


def require_training_dependencies():
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("lerobot")
    except PackageNotFoundError:
        raise SystemExit(
            "Install training dependencies: uv sync --extra train --extra dev"
        ) from None
    if installed != "0.6.1":
        raise SystemExit(
            f"This pipeline is tested with LeRobot 0.6.1, found {installed}; run uv sync --extra train"
        )
