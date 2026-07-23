"""PorTAL: portable task adapters for language models."""

from importlib.metadata import PackageNotFoundError, version

from .config import PortalConfig, PortalProjectionTarget
from .data import ChoiceDataset, ChoiceExample
from .decoder import PortalDecoder
from .evaluation import EvaluationResult, PortalBase, PortalEvaluator, TaskEvaluation, collate_gold_batch
from .model import PortalModel
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
    "ChoiceDataset",
    "ChoiceExample",
    "CoreTrainingResult",
    "EpochMetrics",
    "EvaluationResult",
    "PortalAdapterRefitter",
    "PortalBase",
    "PortalConfig",
    "PortalCoreTrainer",
    "PortalDecoder",
    "PortalEvaluator",
    "PortalModel",
    "PortalTrainingConfig",
    "PortalProjectionTarget",
    "RefitResult",
    "TaskEvaluation",
    "collate_gold_batch",
    "__version__",
]
