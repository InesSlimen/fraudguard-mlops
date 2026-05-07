import importlib.util
from pathlib import Path


def import_check_metrics():
    path = Path("ci/quality-gates/check_metrics.py")
    spec = importlib.util.spec_from_file_location("check_metrics", path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not import {path}")

    spec.loader.exec_module(module)

    return module.check_metrics