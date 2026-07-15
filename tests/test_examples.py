from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


def load_example(name: str) -> ModuleType:
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_example_starts_only_from_raw_source_bases() -> None:
    example = load_example("train_example.py")

    assert example.DATASET_REVISION == "d35f1e8a813cfae662166164fc25965a31b01ae0"
    assert tuple(base.model_id for base in example.SOURCE_BASES) == (
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
    )
    assert example.TRAINING_CONFIG.epochs == 12
    assert example.TRAINING_CONFIG.source_steps_per_epoch == 500


def test_refit_example_uses_published_source_artifact_and_raw_target() -> None:
    example = load_example("refit_example.py")

    assert example.DATASET_REVISION == "d35f1e8a813cfae662166164fc25965a31b01ae0"
    assert example.SOURCE_ARTIFACT == "RampPublic/portal-qwen3-4b"
    assert example.SOURCE_ARTIFACT_REVISION == "v0.1.0"
    assert example.TARGET_BASE.model_id == "Qwen/Qwen3-8B"
    assert example.REFIT_MAX_EXAMPLES == 1000


def test_evaluation_example_uses_published_artifact_and_matching_base() -> None:
    example = load_example("evaluate_example.py")

    assert example.DATASET_REVISION == "d35f1e8a813cfae662166164fc25965a31b01ae0"
    assert example.PORTAL_ARTIFACT == "RampPublic/portal-qwen3-8b"
    assert example.PORTAL_ARTIFACT_REVISION == "v0.1.0"
    assert example.BASE.model_id == "Qwen/Qwen3-8B"
    assert example.EVAL_BATCH_SIZE == 8
