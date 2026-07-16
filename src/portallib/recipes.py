"""Strict, versioned TOML recipes for PorTAL workflows."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .config import SUPPORTED_MODULES
from .runtime import BaseRecipe
from .training import PortalTrainingConfig


class RecipeError(ValueError):
    """A structurally invalid CLI recipe."""


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonEmptyStr = Annotated[str, StringConstraints(min_length=1), AfterValidator(_non_blank)]
PositiveInt = Annotated[int, Field(gt=0)]
Limit = PositiveInt | None
DTypeName = Literal["bfloat16", "float16", "float32"]


def _parent(info: ValidationInfo) -> Path:
    if not info.context or "parent" not in info.context:
        raise ValueError("recipe path context is missing")
    return info.context["parent"]


def _resolve_path(value: Any, info: ValidationInfo) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_parent(info) / path).resolve()


def _resolve_location(value: str, info: ValidationInfo) -> str:
    if value.startswith(("./", "../", "~/")) or Path(value).is_absolute():
        return str(_resolve_path(value, info))
    return value


def _all_to_none(value: Any) -> Any:
    return None if value == "all" else value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetRecipe(_StrictModel):
    repo_id: NonEmptyStr | None = None
    path: Path | None = None
    revision: NonEmptyStr | None = None

    @field_validator("path", mode="before")
    @classmethod
    def resolve_path(cls, value: Any, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @model_validator(mode="after")
    def validate_source(self) -> "DatasetRecipe":
        if (self.repo_id is None) == (self.path is None):
            raise ValueError("define exactly one of repo_id or path")
        if self.path is not None and self.revision is not None:
            raise ValueError("revision is not allowed with path")
        return self

    @property
    def source(self) -> str:
        return str(self.path) if self.path is not None else self.repo_id or ""


class RuntimeRecipe(_StrictModel):
    device: NonEmptyStr = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value != "auto":
            try:
                torch.device(value)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid torch device {value!r}") from exc
        return value


class BaseModelRecipe(_StrictModel):
    model_id: NonEmptyStr
    revision: NonEmptyStr | None = None
    layer_path: NonEmptyStr = "model.layers"
    module_paths: dict[NonEmptyStr, NonEmptyStr] | None = None
    dtype: DTypeName | None = None
    device_map: NonEmptyStr | dict[NonEmptyStr, int | NonEmptyStr] | None = None
    attn_implementation: NonEmptyStr | None = None

    @field_validator("model_id")
    @classmethod
    def resolve_model_id(cls, value: str, info: ValidationInfo) -> str:
        return _resolve_location(value, info)

    @field_validator("dtype", mode="before")
    @classmethod
    def normalize_dtype(cls, value: Any) -> Any:
        return None if value == "auto" else value

    @model_validator(mode="after")
    def validate_runtime_recipe(self) -> "BaseModelRecipe":
        try:
            self.to_runtime()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def to_runtime(self) -> BaseRecipe:
        return BaseRecipe(**self.model_dump())


class _TrainingRecipe(_StrictModel):
    refit_max_examples: PositiveInt | None = None
    eval_max_examples: Limit = None
    eval_batch_size: PositiveInt | None = None
    epochs: PositiveInt | None = None
    batch_size: PositiveInt | None = None
    learning_rate: float | None = None
    lr_scheduler: Literal["constant", "linear"] | None = None
    warmup_ratio: float | None = None
    weight_decay: float | None = None
    grad_clip: float | None = None
    ema_decay: float | None = None
    ema_floor: float | None = None
    max_prompt: PositiveInt | None = None
    seed: int | None = None
    gradient_checkpointing: bool | None = None
    early_stopping_patience: PositiveInt | None = None
    task_regression_threshold: float | None = None

    @field_validator("eval_max_examples", mode="before")
    @classmethod
    def normalize_limits(cls, value: Any) -> Any:
        return _all_to_none(value)

    @model_validator(mode="after")
    def validate_training_config(self) -> "_TrainingRecipe":
        try:
            PortalTrainingConfig(**self.overrides())
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        return self

    def overrides(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class TrainSettings(_TrainingRecipe):
    modules: tuple[NonEmptyStr, ...] | None = None
    rank: PositiveInt | None = None
    alpha: PositiveInt | None = None
    d_z: PositiveInt | None = None
    d_layer: PositiveInt | None = None
    hidden: PositiveInt | None = None
    d_core: PositiveInt | None = None
    source_max_examples: Limit = None
    source_resample_each_epoch: bool | None = None
    source_steps_per_epoch: PositiveInt | None = None
    latent_learning_rate: float | None = None

    @field_validator("source_max_examples", mode="before")
    @classmethod
    def normalize_source_limit(cls, value: Any) -> Any:
        return _all_to_none(value)

    @field_validator("modules", mode="before")
    @classmethod
    def validate_modules(cls, value: Any) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, list):
            value = tuple(value)
        if not value:
            raise ValueError("must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("must not contain duplicates")
        unknown = sorted(set(value) - SUPPORTED_MODULES)
        if unknown:
            raise ValueError(f"unsupported module names: {', '.join(unknown)}")
        return value


class RefitSettings(_TrainingRecipe):
    pass


class CommonRecipe(_StrictModel):
    schema_version: Literal[1]
    kind: Literal["train", "refit", "evaluate"]
    dataset: DatasetRecipe
    runtime: RuntimeRecipe = Field(default_factory=RuntimeRecipe)
    tasks: tuple[NonEmptyStr, ...] | None = None
    result_path: Path | None = None

    @field_validator("result_path", mode="before")
    @classmethod
    def resolve_result_path(cls, value: Any, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @field_validator("tasks", mode="before")
    @classmethod
    def validate_tasks(cls, value: Any) -> tuple[str, ...] | None:
        if isinstance(value, list):
            value = tuple(value)
        if value is not None and (not value or len(value) != len(set(value))):
            raise ValueError("must be non-empty and contain no duplicates")
        return value


class TrainRecipe(CommonRecipe):
    kind: Literal["train"]
    output_dir: Path
    bases: tuple[BaseModelRecipe, ...]
    training: TrainSettings = Field(default_factory=TrainSettings)

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_output_dir(cls, value: Any, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)

    @field_validator("bases", mode="before")
    @classmethod
    def validate_bases(cls, value: Any) -> tuple[Any, ...]:
        if isinstance(value, list):
            value = tuple(value)
        if not value:
            raise ValueError("must contain at least one base")
        return value

    @model_validator(mode="after")
    def validate_unique_bases(self) -> "TrainRecipe":
        model_ids = [base.model_id for base in self.bases]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("bases model_id values must be unique")
        return self


class RefitRecipe(CommonRecipe):
    kind: Literal["refit"]
    tasks: None = None
    output_dir: Path
    source_artifact: NonEmptyStr
    source_artifact_revision: NonEmptyStr | None = None
    base: BaseModelRecipe
    training: RefitSettings = Field(default_factory=RefitSettings)

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_output_dir(cls, value: Any, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)

    @field_validator("source_artifact")
    @classmethod
    def resolve_source_artifact(cls, value: str, info: ValidationInfo) -> str:
        return _resolve_location(value, info)


class EvaluateRecipe(CommonRecipe):
    kind: Literal["evaluate"]
    artifact: NonEmptyStr
    artifact_revision: NonEmptyStr | None = None
    base: BaseModelRecipe
    max_examples: Limit = 1000
    max_prompt: PositiveInt = 768
    batch_size: PositiveInt = 8

    @field_validator("artifact")
    @classmethod
    def resolve_artifact(cls, value: str, info: ValidationInfo) -> str:
        return _resolve_location(value, info)

    @field_validator("max_examples", mode="before")
    @classmethod
    def normalize_max_examples(cls, value: Any) -> Any:
        return _all_to_none(value)


CliRecipe = Annotated[TrainRecipe | RefitRecipe | EvaluateRecipe, Field(discriminator="kind")]
_RECIPE_ADAPTER = TypeAdapter(CliRecipe)


def load_recipe(path: str | Path) -> TrainRecipe | RefitRecipe | EvaluateRecipe:
    """Parse and strictly validate one versioned TOML recipe without loading models."""
    config_path = Path(path).expanduser().resolve()
    try:
        value = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"invalid TOML in {config_path}: {exc}") from exc
    try:
        return _RECIPE_ADAPTER.validate_python(value, context={"parent": config_path.parent})
    except ValidationError as exc:
        raise RecipeError(str(exc)) from exc
