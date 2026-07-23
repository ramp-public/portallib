"""Versioned configuration for canonical PorTAL artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._paths import validate_dotted_path
from ._topology import (
    PortalProjectionTarget,
    alignment_group_dimensions,
    build_artifact_topology,
    discover_projection_topology,
    require_supported_topology,
    resolve_module_paths,
)


_ARCHITECTURE_FIELDS = ("modules", "rank", "alpha", "d_z", "d_layer", "hidden", "d_core")


def _validate_common(config: PortalConfig) -> None:
    if config.library_name != "portallib":
        raise ValueError(f"unsupported artifact library: {config.library_name!r}")
    if config.architecture != "canonical":
        raise ValueError("portallib v1 supports architecture='canonical' only")
    if config.task_type != "CAUSAL_LM":
        raise ValueError("portallib v1 supports task_type='CAUSAL_LM' only")
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


def _normalize_projection_targets(
    config: PortalConfig,
) -> tuple[set[str], tuple[PortalProjectionTarget, ...]]:
    if config.schema_version == 1:
        if config.target_layout is not None:
            raise ValueError("schema version 1 does not support target_layout")
        module_names = tuple(config.in_dims)
        if set(module_names) != set(config.out_dims) or not module_names:
            raise ValueError("in_dims and out_dims must describe the same non-empty module set")
        if any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in (*config.in_dims.values(), *config.out_dims.values())
        ):
            raise ValueError("all projection dimensions must be positive integers")
    elif config.schema_version == 2:
        if not config.target_layout:
            raise ValueError("schema version 2 requires a non-empty target_layout")
        if config.in_dims or config.out_dims:
            raise ValueError("schema version 2 derives dimensions from target_layout")
        module_names = tuple(dict.fromkeys(target.module_name for target in config.target_layout))
        if config.module_paths is None:
            raise ValueError("module_paths must be provided for heterogeneous targets")
    else:
        raise ValueError(f"unsupported PorTAL schema version: {config.schema_version}")

    paths = resolve_module_paths(module_names, config.module_paths)
    if config.module_paths is None:
        object.__setattr__(config, "module_paths", paths)
    validate_dotted_path(config.layer_path, name="layer_path")
    return set(module_names), config.target_specs()


def _validate_projection_targets(
    config: PortalConfig,
    modules: set[str],
    targets: tuple[PortalProjectionTarget, ...],
) -> None:
    keys = [target.key for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("target_layout contains duplicate layer/module targets")
    if any(target.layer_index >= config.n_layers for target in targets):
        raise ValueError("target_layout layer index exceeds n_layers")
    if {target.module_name for target in targets} != modules:
        raise ValueError("every configured logical module must have at least one target")
    alignment_group_dimensions(targets)
    expected_targets = {target.module_path.rsplit(".", 1)[-1] for target in targets}
    if len(config.target_modules) != len(set(config.target_modules)) or set(config.target_modules) != expected_targets:
        raise ValueError(
            f"target_modules must correspond exactly to configured targets; expected {sorted(expected_targets)}"
        )


@dataclass(frozen=True)
class PortalConfig:
    """Everything required to reconstruct one canonical, base-specific PorTAL artifact."""

    base_model_name_or_path: str
    tasks: list[str]
    n_layers: int
    in_dims: dict[str, int]
    out_dims: dict[str, int]
    rank: int = 8
    alpha: int = 16
    d_z: int = 256
    d_layer: int = 32
    hidden: int = 512
    d_core: int = 1024
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    layer_path: str = "model.layers"
    module_paths: dict[str, str] | None = None
    target_layout: list[PortalProjectionTarget] | None = None
    task_type: str = "CAUSAL_LM"
    architecture: str = "canonical"
    base_model_revision: str | None = None
    schema_version: int = 1
    library_name: str = "portallib"

    def __post_init__(self) -> None:
        _validate_common(self)
        modules, targets = _normalize_projection_targets(self)
        _validate_projection_targets(self, modules, targets)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(self.module_paths)

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
        """Return input and output alignment groups resolved from one target layout."""
        return alignment_group_dimensions(self.target_specs())

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
        """Yield normalized targets with their exact base-model paths."""
        for target in self.target_specs():
            yield target, target.exact_path(self.layer_path)

    def target_specs(self) -> tuple[PortalProjectionTarget, ...]:
        """Return the ordered exact target topology, deriving uniform v1 targets when needed."""
        if self.target_layout is not None:
            return tuple(self.target_layout)
        return tuple(
            PortalProjectionTarget(
                layer_index=layer_index,
                module_name=module_name,
                module_path=module_path,
                in_features=self.in_dims[module_name],
                out_features=self.out_dims[module_name],
                input_group=module_name,
                output_group=module_name,
            )
            for layer_index in range(self.n_layers)
            for module_name, module_path in self.module_paths.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PortalConfig":
        if not isinstance(value, dict):
            raise TypeError("PorTAL configuration must be a JSON object")
        parsed = dict(value)
        if parsed.get("target_layout") is not None:
            parsed["target_layout"] = [
                PortalProjectionTarget.from_dict(target) for target in parsed["target_layout"]
            ]
        return cls(**parsed)

    @classmethod
    def from_json(cls, path: str | Path) -> "PortalConfig":
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
    ) -> "PortalConfig":
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
        topology = build_artifact_topology(
            discovery,
            modules=modules,
            schema_version=kwargs.pop("schema_version", None),
        )
        model_name = base_model_name_or_path or getattr(getattr(model, "config", None), "_name_or_path", "")
        return cls(
            base_model_name_or_path=model_name,
            tasks=tasks,
            n_layers=discovery.n_layers,
            in_dims=topology.in_dims,
            out_dims=topology.out_dims,
            target_modules=[path.rsplit(".", 1)[-1] for path in paths.values()],
            layer_path=layer_path,
            module_paths=paths,
            target_layout=topology.target_layout,
            schema_version=topology.schema_version,
            **kwargs,
        )
