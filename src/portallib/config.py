"""Versioned configuration for canonical PorTAL artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ._paths import validate_dotted_path
from ._topology import (
    PortalProjectionTarget,
    alignment_group_dimensions,
    build_projection_targets,
    discover_projection_topology,
    require_supported_topology,
    resolve_module_paths,
)

_ARCHITECTURE_FIELDS = ("modules", "rank", "alpha", "d_z", "d_layer", "hidden", "d_core")


def _validate_common(config: PortalConfig) -> None:
    if config.format_version != 1:
        raise ValueError("PorTAL artifacts require format_version=1")
    if config.library_name != "portallib":
        raise ValueError(f"unsupported artifact library: {config.library_name!r}")
    if config.architecture != "canonical":
        raise ValueError("PorTAL artifacts support architecture='canonical' only")
    if config.task_type != "CAUSAL_LM":
        raise ValueError("PorTAL artifacts support task_type='CAUSAL_LM' only")
    if not config.base_model_name_or_path.strip():
        raise ValueError("base_model_name_or_path must not be empty")
    if not config.tasks or len(set(config.tasks)) != len(config.tasks):
        raise ValueError("tasks must be a non-empty list of unique names")
    if any(not isinstance(task, str) or not task.strip() for task in config.tasks):
        raise ValueError("task names must be non-empty strings")
    positive_fields = {
        "rank": config.rank,
        "alpha": config.alpha,
        "n_layers": config.n_layers,
        "d_z": config.d_z,
        "d_layer": config.d_layer,
        "hidden": config.hidden,
        "d_core": config.d_core,
    }
    invalid = [name for name, value in positive_fields.items() if not isinstance(value, int) or value <= 0]
    if invalid:
        raise ValueError(f"configuration values must be positive integers: {', '.join(invalid)}")


def _validate_projection_targets(config: PortalConfig) -> None:
    targets = config.projection_targets
    if not targets:
        raise ValueError("projection_targets must be non-empty")
    keys = [target.key for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("projection_targets contains duplicate layer/module targets")
    if any(target.layer_index >= config.n_layers for target in targets):
        raise ValueError("projection target layer index exceeds n_layers")
    alignment_group_dimensions(targets)


@dataclass(frozen=True)
class PortalConfig:
    """Everything required to reconstruct one canonical, base-specific PorTAL artifact."""

    base_model_name_or_path: str
    tasks: list[str]
    n_layers: int
    projection_targets: tuple[PortalProjectionTarget, ...]
    rank: int = 8
    alpha: int = 16
    d_z: int = 256
    d_layer: int = 32
    hidden: int = 512
    d_core: int = 1024
    layer_path: str = "model.layers"
    task_type: str = "CAUSAL_LM"
    architecture: str = "canonical"
    base_model_revision: str | None = None
    format_version: int = 1
    library_name: str = "portallib"

    def __post_init__(self) -> None:
        _validate_common(self)
        validate_dotted_path(self.layer_path, name="layer_path")
        _validate_projection_targets(self)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(target.module_name for target in self.projection_targets))

    @property
    def input_groups(self) -> dict[str, int]:
        """Return stable input-alignment groups and their projection widths."""
        return self.alignment_groups[0]

    @property
    def output_groups(self) -> dict[str, int]:
        """Return stable output-alignment groups and their projection widths."""
        return self.alignment_groups[1]

    @property
    def alignment_groups(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return input and output alignment groups resolved from the exact targets."""
        return alignment_group_dimensions(self.projection_targets)

    def shared_signature(self) -> tuple[Any, ...]:
        """Return fields that must match when a canonical core is shared across bases."""
        return (tuple(self.tasks), *self.architecture_kwargs().values())

    def architecture_kwargs(self) -> dict[str, Any]:
        """Return the canonical architecture fields shared by training and refitting."""
        return {name: getattr(self, name) for name in _ARCHITECTURE_FIELDS}

    def targets(self) -> Iterator[tuple[int, str, str]]:
        """Yield every configured target as ``(layer, short name, exact path)``."""
        for target, exact_path in self.resolved_targets():
            yield (*target.key, exact_path)

    def resolved_targets(self) -> Iterator[tuple[PortalProjectionTarget, str]]:
        """Yield configured targets with their exact base-model paths."""
        for target in self.projection_targets:
            yield target, target.exact_path(self.layer_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PortalConfig:
        if not isinstance(value, dict):
            raise TypeError("PorTAL configuration must be a JSON object")
        parsed = dict(value)
        if parsed.get("projection_targets") is not None:
            parsed["projection_targets"] = tuple(
                PortalProjectionTarget.from_dict(target) for target in parsed["projection_targets"]
            )
        return cls(**parsed)

    @classmethod
    def from_json(cls, path: str | Path) -> PortalConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        tasks: list[str],
        base_model_name_or_path: str | None = None,
        modules: tuple[str, ...] = ("q", "v"),
        layer_path: str = "model.layers",
        module_paths: dict[str, str] | None = None,
        allow_heterogeneous_targets: bool = False,
        **kwargs: Any,
    ) -> PortalConfig:
        """Build a config by resolving every requested projection path exactly."""
        paths = resolve_module_paths(modules, module_paths)
        discovery = discover_projection_topology(
            model,
            modules=modules,
            layer_path=layer_path,
            module_paths=paths,
        )
        require_supported_topology(
            discovery,
            layer_path=layer_path,
            module_paths=paths,
            allow_heterogeneous_targets=allow_heterogeneous_targets,
        )
        projection_targets = build_projection_targets(
            discovery,
            modules=modules,
        )
        model_name = base_model_name_or_path or getattr(getattr(model, "config", None), "_name_or_path", "")
        return cls(
            base_model_name_or_path=model_name,
            tasks=tasks,
            n_layers=discovery.n_layers,
            projection_targets=tuple(projection_targets),
            layer_path=layer_path,
            **kwargs,
        )
