"""Versioned configuration for canonical PorTAL artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ._paths import exact_module_path, validate_dotted_path


DEFAULT_MODULE_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
SUPPORTED_MODULES = frozenset(DEFAULT_MODULE_PATHS)
_ARCHITECTURE_FIELDS = ("modules", "rank", "alpha", "d_z", "d_layer", "hidden", "d_core")


@dataclass(frozen=True)
class PortalProjectionTarget:
    """One exact base-model projection generated from a logical canonical head."""

    layer_index: int
    module_name: str
    module_path: str
    in_features: int
    out_features: int
    input_group: str
    output_group: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_index, int) or self.layer_index < 0:
            raise ValueError("target layer_index must be a non-negative integer")
        if self.module_name not in SUPPORTED_MODULES:
            raise ValueError(f"unsupported target module name: {self.module_name!r}")
        validate_dotted_path(self.module_path, name="target module_path")
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("target projection dimensions must be positive")
        for name, value in (("input_group", self.input_group), ("output_group", self.output_group)):
            if not value.isidentifier():
                raise ValueError(f"target {name} must be a valid parameter identifier")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PortalProjectionTarget":
        if not isinstance(value, dict):
            raise TypeError("each target_layout entry must be an object")
        return cls(**value)


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
        modules = set(config.in_dims)
        if modules != set(config.out_dims) or not modules:
            raise ValueError("in_dims and out_dims must describe the same non-empty module set")
        if any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in (*config.in_dims.values(), *config.out_dims.values())
        ):
            raise ValueError("all projection dimensions must be positive integers")
        if config.module_paths is None:
            object.__setattr__(
                config,
                "module_paths",
                {name: DEFAULT_MODULE_PATHS[name] for name in config.in_dims},
            )
    elif config.schema_version == 2:
        if not config.target_layout:
            raise ValueError("schema version 2 requires a non-empty target_layout")
        if config.in_dims or config.out_dims:
            raise ValueError("schema version 2 derives dimensions from target_layout")
        modules = {target.module_name for target in config.target_layout}
    else:
        raise ValueError(f"unsupported PorTAL schema version: {config.schema_version}")

    unknown = sorted(modules - SUPPORTED_MODULES)
    if unknown:
        raise ValueError(f"unsupported module names: {unknown}")
    if config.module_paths is None:
        raise ValueError("module_paths must be provided for heterogeneous targets")
    if set(config.module_paths) != modules:
        raise ValueError("module_paths must describe exactly the configured modules")
    validate_dotted_path(config.layer_path, name="layer_path")
    for path in config.module_paths.values():
        validate_dotted_path(path, name="module_paths values")
    return modules, config.target_specs()


def _validate_alignment_groups(targets: tuple[PortalProjectionTarget, ...]) -> None:
    input_groups: dict[str, int] = {}
    output_groups: dict[str, int] = {}
    input_group_modules: dict[str, str] = {}
    output_group_modules: dict[str, str] = {}
    for target in targets:
        for groups, group_modules, group, size in (
            (input_groups, input_group_modules, target.input_group, target.in_features),
            (output_groups, output_group_modules, target.output_group, target.out_features),
        ):
            previous = groups.setdefault(group, size)
            if previous != size:
                raise ValueError(f"alignment group {group!r} has inconsistent dimensions")
            previous_module = group_modules.setdefault(group, target.module_name)
            if previous_module != target.module_name:
                raise ValueError(f"alignment group {group!r} cannot be shared across logical modules")


def _validate_projection_targets(
    config: PortalConfig,
    modules: set[str],
    targets: tuple[PortalProjectionTarget, ...],
) -> None:
    keys = [(target.layer_index, target.module_name) for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("target_layout contains duplicate layer/module targets")
    if any(target.layer_index >= config.n_layers for target in targets):
        raise ValueError("target_layout layer index exceeds n_layers")
    if {target.module_name for target in targets} != modules:
        raise ValueError("every configured logical module must have at least one target")
    _validate_alignment_groups(targets)
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
        groups: dict[str, int] = {}
        for target in self.target_specs():
            groups.setdefault(target.input_group, target.in_features)
        return groups

    @property
    def output_groups(self) -> dict[str, int]:
        """Return stable output-alignment groups and their projection widths."""
        groups: dict[str, int] = {}
        for target in self.target_specs():
            groups.setdefault(target.output_group, target.out_features)
        return groups

    def shared_signature(self) -> tuple[Any, ...]:
        """Return fields that must match when a canonical core is shared across bases."""
        return (tuple(self.tasks), *self.architecture_kwargs().values())

    def architecture_kwargs(self) -> dict[str, Any]:
        """Return the canonical architecture fields shared by training and refitting."""
        return {name: getattr(self, name) for name in _ARCHITECTURE_FIELDS}

    def targets(self) -> Iterator[tuple[int, str, str]]:
        """Yield every configured target as ``(layer, short name, exact path)``."""
        for target in self.target_specs():
            yield (
                target.layer_index,
                target.module_name,
                exact_module_path(self.layer_path, target.layer_index, target.module_path),
            )

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
        if not modules or len(set(modules)) != len(modules):
            raise ValueError("modules must be a non-empty tuple of unique short names")
        unknown = sorted(set(modules) - SUPPORTED_MODULES)
        if unknown:
            raise ValueError(f"unsupported module names: {unknown}")
        paths = module_paths or {name: DEFAULT_MODULE_PATHS[name] for name in modules}
        if set(paths) != set(modules):
            raise ValueError("module_paths must describe exactly the requested modules")
        try:
            layers = model.get_submodule(layer_path)
            n_layers = len(layers)
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(f"base model has no indexable exact layer path {layer_path!r}") from exc
        if n_layers == 0:
            raise ValueError("base model has no decoder layers")
        raw_targets: list[tuple[int, str, str, int, int]] = []
        dimensions_by_module: dict[str, set[tuple[int, int]]] = {name: set() for name in modules}
        missing_targets: list[tuple[int, str]] = []
        for short_name, relative_path in paths.items():
            for layer_index in range(n_layers):
                exact_path = exact_module_path(layer_path, layer_index, relative_path)
                try:
                    projection = model.get_submodule(exact_path)
                except (AttributeError, KeyError):
                    missing_targets.append((layer_index, short_name))
                    continue
                try:
                    in_features = projection.in_features
                    out_features = projection.out_features
                except AttributeError as exc:
                    raise ValueError(f"configured projection {exact_path!r} has no linear dimensions") from exc
                dimensions_by_module[short_name].add((in_features, out_features))
                raw_targets.append((layer_index, short_name, relative_path, in_features, out_features))
        empty_modules = [name for name, dimensions in dimensions_by_module.items() if not dimensions]
        if empty_modules:
            missing_module = empty_modules[0]
            layer_index = next(layer for layer, name in missing_targets if name == missing_module)
            exact_path = exact_module_path(layer_path, layer_index, paths[missing_module])
            raise ValueError(f"base model has no exact projection path {exact_path!r}")
        heterogeneous = bool(missing_targets) or any(len(dimensions) != 1 for dimensions in dimensions_by_module.values())
        if heterogeneous and not allow_heterogeneous_targets:
            if missing_targets:
                layer_index, short_name = missing_targets[0]
                exact_path = exact_module_path(layer_path, layer_index, paths[short_name])
                raise ValueError(f"base model has no exact projection path {exact_path!r}")
            name = next(name for name, dimensions in dimensions_by_module.items() if len(dimensions) != 1)
            raise ValueError(
                f"projection dimensions vary across layers for {name!r}: {sorted(dimensions_by_module[name])}"
            )
        explicit_schema = kwargs.pop("schema_version", None)
        schema_version = explicit_schema if explicit_schema is not None else (2 if heterogeneous else 1)
        if heterogeneous and schema_version != 2:
            raise ValueError("heterogeneous target layouts require schema_version=2")
        if not heterogeneous and schema_version != 1:
            raise ValueError("uniform target layouts require schema_version=1")
        in_dims: dict[str, int] = {}
        out_dims: dict[str, int] = {}
        target_layout: list[PortalProjectionTarget] | None = None
        if heterogeneous:
            in_sizes = {
                name: {in_features for in_features, _out_features in dimensions}
                for name, dimensions in dimensions_by_module.items()
            }
            out_sizes = {
                name: {out_features for _in_features, out_features in dimensions}
                for name, dimensions in dimensions_by_module.items()
            }
            module_order = {name: index for index, name in enumerate(modules)}
            raw_targets.sort(key=lambda target: (target[0], module_order[target[1]]))
            target_layout = [
                PortalProjectionTarget(
                    layer_index=layer_index,
                    module_name=short_name,
                    module_path=relative_path,
                    in_features=in_features,
                    out_features=out_features,
                    input_group=(
                        short_name if len(in_sizes[short_name]) == 1 else f"{short_name}__in_{in_features}"
                    ),
                    output_group=(
                        short_name if len(out_sizes[short_name]) == 1 else f"{short_name}__out_{out_features}"
                    ),
                )
                for layer_index, short_name, relative_path, in_features, out_features in raw_targets
            ]
        else:
            for short_name, dimensions in dimensions_by_module.items():
                in_dims[short_name], out_dims[short_name] = next(iter(dimensions))
        model_name = base_model_name_or_path or getattr(getattr(model, "config", None), "_name_or_path", "")
        return cls(
            base_model_name_or_path=model_name,
            tasks=tasks,
            n_layers=n_layers,
            in_dims=in_dims,
            out_dims=out_dims,
            target_modules=[path.rsplit(".", 1)[-1] for path in paths.values()],
            layer_path=layer_path,
            module_paths=paths,
            target_layout=target_layout,
            schema_version=schema_version,
            **kwargs,
        )
