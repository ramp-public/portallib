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
    task_type: str = "CAUSAL_LM"
    architecture: str = "canonical"
    base_model_revision: str | None = None
    schema_version: int = 1
    library_name: str = "portallib"

    def __post_init__(self) -> None:
        if self.library_name != "portallib":
            raise ValueError(f"unsupported artifact library: {self.library_name!r}")
        if self.schema_version != 1:
            raise ValueError(f"unsupported PorTAL schema version: {self.schema_version}")
        if self.architecture != "canonical":
            raise ValueError("portallib v1 supports architecture='canonical' only")
        if self.task_type != "CAUSAL_LM":
            raise ValueError("portallib v1 supports task_type='CAUSAL_LM' only")
        if not self.base_model_name_or_path.strip():
            raise ValueError("base_model_name_or_path must not be empty")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be a non-empty list of unique names")
        if any(not isinstance(task, str) or not task.strip() for task in self.tasks):
            raise ValueError("task names must be non-empty strings")
        positive_fields = {
            "rank": self.rank,
            "alpha": self.alpha,
            "n_layers": self.n_layers,
            "d_z": self.d_z,
            "d_layer": self.d_layer,
            "hidden": self.hidden,
            "d_core": self.d_core,
        }
        invalid = [name for name, value in positive_fields.items() if not isinstance(value, int) or value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive integers: {', '.join(invalid)}")
        modules = set(self.in_dims)
        if modules != set(self.out_dims) or not modules:
            raise ValueError("in_dims and out_dims must describe the same non-empty module set")
        unknown = sorted(modules - SUPPORTED_MODULES)
        if unknown:
            raise ValueError(f"unsupported module names: {unknown}")
        if self.module_paths is None:
            object.__setattr__(self, "module_paths", {name: DEFAULT_MODULE_PATHS[name] for name in self.in_dims})
        if set(self.module_paths) != modules:
            raise ValueError("module_paths must describe exactly the configured modules")
        validate_dotted_path(self.layer_path, name="layer_path")
        for path in self.module_paths.values():
            validate_dotted_path(path, name="module_paths values")
        if any(not isinstance(dim, int) or dim <= 0 for dim in (*self.in_dims.values(), *self.out_dims.values())):
            raise ValueError("all projection dimensions must be positive integers")
        expected_targets = {path.rsplit(".", 1)[-1] for path in self.module_paths.values()}
        if len(self.target_modules) != len(set(self.target_modules)) or set(self.target_modules) != expected_targets:
            raise ValueError(
                f"target_modules must correspond exactly to module_paths; expected {sorted(expected_targets)}"
            )

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(self.in_dims)

    def shared_signature(self) -> tuple[Any, ...]:
        """Return fields that must match when a canonical core is shared across bases."""
        return (tuple(self.tasks), *self.architecture_kwargs().values())

    def architecture_kwargs(self) -> dict[str, Any]:
        """Return the canonical architecture fields shared by training and refitting."""
        return {name: getattr(self, name) for name in _ARCHITECTURE_FIELDS}

    def targets(self) -> Iterator[tuple[int, str, str]]:
        """Yield every configured target as ``(layer, short name, exact path)``."""
        for layer_index in range(self.n_layers):
            for short_name, module_path in self.module_paths.items():
                yield layer_index, short_name, exact_module_path(self.layer_path, layer_index, module_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PortalConfig":
        if not isinstance(value, dict):
            raise TypeError("PorTAL configuration must be a JSON object")
        return cls(**value)

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
        in_dims: dict[str, int] = {}
        out_dims: dict[str, int] = {}
        for short_name, relative_path in paths.items():
            dimensions: set[tuple[int, int]] = set()
            for layer_index in range(n_layers):
                exact_path = exact_module_path(layer_path, layer_index, relative_path)
                try:
                    projection = model.get_submodule(exact_path)
                    dimensions.add((projection.in_features, projection.out_features))
                except (AttributeError, KeyError) as exc:
                    raise ValueError(f"base model has no exact projection path {exact_path!r}") from exc
            if len(dimensions) != 1:
                raise ValueError(f"projection dimensions vary across layers for {short_name!r}: {sorted(dimensions)}")
            in_dims[short_name], out_dims[short_name] = dimensions.pop()
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
            **kwargs,
        )
