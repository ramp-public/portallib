"""Strict, versioned TOML recipes for PorTAL workflows."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .._topology import validate_modules
from ..runtime import BaseModelSpec
from ..training import PortalTrainingConfig


class RecipeError(ValueError):
    """A structurally invalid CLI recipe."""


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


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


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _unique_nonempty(value: tuple[str, ...]) -> tuple[str, ...]:
    if not value or len(value) != len(set(value)):
        raise ValueError("must be non-empty and contain no duplicates")
    return value


def _supported_modules(value: tuple[str, ...]) -> tuple[str, ...]:
    return validate_modules(value)


NonEmptyStr = Annotated[str, StringConstraints(min_length=1), AfterValidator(_non_blank)]
PositiveInt = Annotated[int, Field(gt=0)]
Limit = Annotated[PositiveInt | None, BeforeValidator(_all_to_none)]
ResolvedPath = Annotated[Path, BeforeValidator(_resolve_path)]
ResolvedLocation = Annotated[NonEmptyStr, BeforeValidator(_resolve_location)]
StringTuple = Annotated[tuple[NonEmptyStr, ...], BeforeValidator(_as_tuple)]
UniqueStringTuple = Annotated[StringTuple, AfterValidator(_unique_nonempty)]
ModuleTuple = Annotated[UniqueStringTuple, AfterValidator(_supported_modules)]
DTypeName = Literal["bfloat16", "float16", "float32"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetRecipe(_StrictModel):
    repo_id: NonEmptyStr | None = None
    path: ResolvedPath | None = None
    revision: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_source(self) -> DatasetRecipe:
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
    dtype: DTypeName | Literal["auto"] = "auto"

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
    model_id: ResolvedLocation
    revision: NonEmptyStr | None = None
    layer_path: NonEmptyStr = "model.layers"
    module_paths: dict[NonEmptyStr, NonEmptyStr] | None = None
    dtype: DTypeName | None = None
    device_map: NonEmptyStr | dict[NonEmptyStr, int | NonEmptyStr] | None = None
    attn_implementation: NonEmptyStr | None = None
    loader: Literal["causal_lm", "multimodal_lm"] = "causal_lm"
    allow_heterogeneous_targets: bool = False

    @field_validator("dtype", mode="before")
    @classmethod
    def normalize_dtype(cls, value: Any) -> Any:
        return None if value == "auto" else value

    @model_validator(mode="after")
    def validate_runtime_recipe(self) -> BaseModelRecipe:
        try:
            self.to_runtime()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def to_runtime(self) -> BaseModelSpec:
        return BaseModelSpec(**self.model_dump())


class RefitSettings(_StrictModel):
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

    @model_validator(mode="after")
    def validate_training_config(self) -> RefitSettings:
        try:
            PortalTrainingConfig(**self.overrides())
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        return self

    def overrides(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class TrainSettings(RefitSettings):
    modules: ModuleTuple | None = None
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


class CommonRecipe(_StrictModel):
    recipe_version: Literal[1]
    kind: Literal["train", "refit", "evaluate"]
    dataset: DatasetRecipe
    runtime: RuntimeRecipe = Field(default_factory=RuntimeRecipe)
    tasks: UniqueStringTuple | None = None
    result_path: ResolvedPath | None = None


class _OutputRecipe(CommonRecipe):
    output_dir: ResolvedPath


class TrainRecipe(_OutputRecipe):
    kind: Literal["train"]
    bases: Annotated[tuple[BaseModelRecipe, ...], BeforeValidator(_as_tuple)]
    training: TrainSettings = Field(default_factory=TrainSettings)

    @field_validator("bases")
    @classmethod
    def validate_bases(cls, value: tuple[BaseModelRecipe, ...]) -> tuple[BaseModelRecipe, ...]:
        if not value:
            raise ValueError("must contain at least one base")
        return value

    @model_validator(mode="after")
    def validate_unique_bases(self) -> TrainRecipe:
        model_ids = [base.model_id for base in self.bases]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("bases model_id values must be unique")
        return self


class RefitRecipe(_OutputRecipe):
    kind: Literal["refit"]
    tasks: None = None
    source_artifact: ResolvedLocation
    source_artifact_revision: NonEmptyStr | None = None
    base: BaseModelRecipe
    training: RefitSettings = Field(default_factory=RefitSettings)


class EvaluateRecipe(CommonRecipe):
    kind: Literal["evaluate"]
    artifact: ResolvedLocation
    artifact_revision: NonEmptyStr | None = None
    base: BaseModelRecipe
    max_examples: Limit = 1000
    max_prompt: PositiveInt = 768
    batch_size: PositiveInt = 8


CliRecipe = Annotated[TrainRecipe | RefitRecipe | EvaluateRecipe, Field(discriminator="kind")]
_RECIPE_ADAPTER = TypeAdapter(CliRecipe)


def _parse_recipe(
    text: str,
    *,
    parent: Path,
    source: str,
) -> TrainRecipe | RefitRecipe | EvaluateRecipe:
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"invalid TOML in {source}: {exc}") from exc
    try:
        return _RECIPE_ADAPTER.validate_python(value, context={"parent": parent})
    except ValidationError as exc:
        raise RecipeError(str(exc)) from exc


def load_recipe(path: str | Path) -> TrainRecipe | RefitRecipe | EvaluateRecipe:
    """Parse and strictly validate one versioned TOML recipe without loading models."""
    config_path = Path(path).expanduser().resolve()
    return _parse_recipe(
        config_path.read_text(encoding="utf-8"),
        parent=config_path.parent,
        source=str(config_path),
    )
