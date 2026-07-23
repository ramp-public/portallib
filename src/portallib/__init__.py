"""PorTAL: portable task adapters for language models."""

from importlib.metadata import PackageNotFoundError, version

from .config import PortalConfig, PortalProjectionTarget
from .data import ChoiceDataset, ChoiceExample
from .evaluation import EvaluationResult, PortalBase, PortalEvaluator, TaskEvaluation, collate_gold_batch
from .model import PortalModel
from .runtime import BaseModelSpec, load_base, load_dataset, runtime_device
from .training import (
    CoreTrainingResult,
    EpochMetrics,
    PortalAdapterRefitter,
    PortalCoreTrainer,
    PortalTrainingConfig,
    RefitResult,
)

try:
    __version__ = version("portallib")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "BaseModelSpec",
    "ChoiceDataset",
    "ChoiceExample",
    "CoreTrainingResult",
    "EpochMetrics",
    "EvaluationResult",
    "PortalAdapterRefitter",
    "PortalBase",
    "PortalConfig",
    "PortalCoreTrainer",
    "PortalEvaluator",
    "PortalModel",
    "PortalProjectionTarget",
    "PortalTrainingConfig",
    "RefitResult",
    "TaskEvaluation",
    "__version__",
    "collate_gold_batch",
    "load_base",
    "load_dataset",
    "runtime_device",
]
