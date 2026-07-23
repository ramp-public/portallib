"""Exact base-model projection topology discovery and normalization."""

from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def key(self) -> tuple[int, str]:
        return self.layer_index, self.module_name

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.in_features, self.out_features

    @property
    def alignment_groups(self) -> tuple[tuple[str, int], tuple[str, int]]:
        return (
            (self.input_group, self.in_features),
            (self.output_group, self.out_features),
        )

    def exact_path(self, layer_path: str) -> str:
        return exact_module_path(layer_path, self.layer_index, self.module_path)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PortalProjectionTarget:
        if not isinstance(value, dict):
            raise TypeError("each target_layout entry must be an object")
        return cls(**value)


@dataclass(frozen=True)
class _DiscoveredProjection:
    layer_index: int
    module_name: str
    module_path: str
    in_features: int
    out_features: int


@dataclass(frozen=True)
class _ProjectionDiscovery:
    n_layers: int
    projections: tuple[_DiscoveredProjection, ...]
    dimensions_by_module: dict[str, frozenset[tuple[int, int]]]
    missing_targets: tuple[tuple[int, str], ...]

    @property
    def heterogeneous(self) -> bool:
        return bool(self.missing_targets) or any(
            len(dimensions) != 1 for dimensions in self.dimensions_by_module.values()
        )


@dataclass(frozen=True)
class _ArtifactTopology:
    schema_version: int
    in_dims: dict[str, int]
    out_dims: dict[str, int]
    target_layout: list[PortalProjectionTarget] | None


def resolve_module_paths(
    modules: tuple[str, ...],
    module_paths: dict[str, str] | None,
) -> dict[str, str]:
    validate_modules(modules)
    paths = module_paths or {name: DEFAULT_MODULE_PATHS[name] for name in modules}
    if set(paths) != set(modules):
        raise ValueError("module_paths must describe exactly the configured modules")
    for path in paths.values():
        validate_dotted_path(path, name="module_paths values")
    return paths


def validate_modules(modules: tuple[str, ...]) -> tuple[str, ...]:
    if not modules or len(set(modules)) != len(modules):
        raise ValueError("modules must be a non-empty tuple of unique short names")
    unknown = sorted(set(modules) - SUPPORTED_MODULES)
    if unknown:
        raise ValueError(f"unsupported module names: {', '.join(unknown)}")
    return modules


def alignment_group_dimensions(
    targets: tuple[PortalProjectionTarget, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    dimensions: tuple[dict[str, int], dict[str, int]] = ({}, {})
    modules: tuple[dict[str, str], dict[str, str]] = ({}, {})
    for target in targets:
        for axis, (group, size) in enumerate(target.alignment_groups):
            previous_size = dimensions[axis].setdefault(group, size)
            if previous_size != size:
                raise ValueError(f"alignment group {group!r} has inconsistent dimensions")
            previous_module = modules[axis].setdefault(group, target.module_name)
            if previous_module != target.module_name:
                raise ValueError(f"alignment group {group!r} cannot be shared across logical modules")
    return dimensions


def discover_projection_topology(
    model: Any,
    *,
    modules: tuple[str, ...],
    layer_path: str,
    module_paths: dict[str, str],
) -> _ProjectionDiscovery:
    try:
        layers = model.get_submodule(layer_path)
        n_layers = len(layers)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"base model has no indexable exact layer path {layer_path!r}") from exc
    if n_layers == 0:
        raise ValueError("base model has no decoder layers")

    projections: list[_DiscoveredProjection] = []
    dimensions_by_module: dict[str, set[tuple[int, int]]] = {name: set() for name in modules}
    missing_targets: list[tuple[int, str]] = []
    for module_name, module_path in module_paths.items():
        for layer_index in range(n_layers):
            exact_path = exact_module_path(layer_path, layer_index, module_path)
            try:
                projection = model.get_submodule(exact_path)
            except (AttributeError, KeyError):
                missing_targets.append((layer_index, module_name))
                continue
            try:
                in_features = projection.in_features
                out_features = projection.out_features
            except AttributeError as exc:
                raise ValueError(f"configured projection {exact_path!r} has no linear dimensions") from exc
            dimensions_by_module[module_name].add((in_features, out_features))
            projections.append(
                _DiscoveredProjection(
                    layer_index=layer_index,
                    module_name=module_name,
                    module_path=module_path,
                    in_features=in_features,
                    out_features=out_features,
                )
            )

    empty_modules = [name for name, dimensions in dimensions_by_module.items() if not dimensions]
    if empty_modules:
        missing_module = empty_modules[0]
        layer_index = next(layer for layer, name in missing_targets if name == missing_module)
        exact_path = exact_module_path(layer_path, layer_index, module_paths[missing_module])
        raise ValueError(f"base model has no exact projection path {exact_path!r}")

    return _ProjectionDiscovery(
        n_layers=n_layers,
        projections=tuple(projections),
        dimensions_by_module={
            name: frozenset(dimensions) for name, dimensions in dimensions_by_module.items()
        },
        missing_targets=tuple(missing_targets),
    )


def require_supported_topology(
    discovery: _ProjectionDiscovery,
    *,
    layer_path: str,
    module_paths: dict[str, str],
    allow_heterogeneous_targets: bool,
) -> None:
    if not discovery.heterogeneous or allow_heterogeneous_targets:
        return
    if discovery.missing_targets:
        layer_index, module_name = discovery.missing_targets[0]
        exact_path = exact_module_path(layer_path, layer_index, module_paths[module_name])
        raise ValueError(f"base model has no exact projection path {exact_path!r}")
    module_name = next(
        name for name, dimensions in discovery.dimensions_by_module.items() if len(dimensions) != 1
    )
    raise ValueError(
        f"projection dimensions vary across layers for {module_name!r}: "
        f"{sorted(discovery.dimensions_by_module[module_name])}"
    )


def build_artifact_topology(
    discovery: _ProjectionDiscovery,
    *,
    modules: tuple[str, ...],
    schema_version: int | None,
) -> _ArtifactTopology:
    expected_schema = 2 if discovery.heterogeneous else 1
    if schema_version is not None and schema_version != expected_schema:
        topology = "heterogeneous" if discovery.heterogeneous else "uniform"
        raise ValueError(f"{topology} target layouts require schema_version={expected_schema}")

    if not discovery.heterogeneous:
        dimensions = {
            name: next(iter(module_dimensions))
            for name, module_dimensions in discovery.dimensions_by_module.items()
        }
        return _ArtifactTopology(
            schema_version=1,
            in_dims={name: sizes[0] for name, sizes in dimensions.items()},
            out_dims={name: sizes[1] for name, sizes in dimensions.items()},
            target_layout=None,
        )

    input_sizes = {
        name: {in_features for in_features, _out_features in dimensions}
        for name, dimensions in discovery.dimensions_by_module.items()
    }
    output_sizes = {
        name: {out_features for _in_features, out_features in dimensions}
        for name, dimensions in discovery.dimensions_by_module.items()
    }
    module_order = {name: index for index, name in enumerate(modules)}
    projections = sorted(
        discovery.projections,
        key=lambda projection: (projection.layer_index, module_order[projection.module_name]),
    )
    target_layout = [
        PortalProjectionTarget(
            layer_index=projection.layer_index,
            module_name=projection.module_name,
            module_path=projection.module_path,
            in_features=projection.in_features,
            out_features=projection.out_features,
            input_group=(
                projection.module_name
                if len(input_sizes[projection.module_name]) == 1
                else f"{projection.module_name}__in_{projection.in_features}"
            ),
            output_group=(
                projection.module_name
                if len(output_sizes[projection.module_name]) == 1
                else f"{projection.module_name}__out_{projection.out_features}"
            ),
        )
        for projection in projections
    ]
    return _ArtifactTopology(
        schema_version=2,
        in_dims={},
        out_dims={},
        target_layout=target_layout,
    )
