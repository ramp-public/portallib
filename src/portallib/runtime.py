"""Host-controlled loading helpers used by the CLI and runnable examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ._naming import model_slug
from ._paths import validate_dotted_path
from .data import ChoiceDataset
from .evaluation import PortalBase


@dataclass(frozen=True)
class BaseModelSpec:
    """One causal-language-model revision and its exact decoder paths."""

    model_id: str
    revision: str | None = None
    layer_path: str = "model.layers"
    module_paths: dict[str, str] | None = None
    dtype: torch.dtype | str | None = None
    device_map: str | dict[str, int | str | torch.device] | None = None
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("base model_id must not be empty")
        validate_dotted_path(self.layer_path, name="base layer_path")
        if self.module_paths is not None:
            if not self.module_paths or any(not name for name in self.module_paths):
                raise ValueError("base module_paths must map non-empty names to dotted paths")
            for path in self.module_paths.values():
                validate_dotted_path(path, name="base module_paths values")


def load_dataset(source: str, *, revision: str | None) -> ChoiceDataset:
    """Load the normalized dataset schema from a local JSON file or the Hub."""
    path = Path(source)
    if path.is_file():
        if revision is not None:
            raise ValueError("dataset revision must be omitted when source is a local JSON file")
        return ChoiceDataset.from_json(path)
    return ChoiceDataset.from_hub(source, revision=revision)


def runtime_device(device: str = "auto", dtype: str = "auto") -> tuple[torch.device, torch.dtype]:
    """Resolve host-selected device and floating-point dtype."""
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    selected_device = torch.device(device_name)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime requests CUDA, but CUDA is not available")
    if dtype == "auto":
        selected_dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    else:
        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtypes:
            raise ValueError("runtime dtype must be 'auto', 'bfloat16', 'float16', or 'float32'")
        selected_dtype = dtypes[dtype]
    return selected_device, selected_dtype


def load_base(recipe: BaseModelSpec, *, device: torch.device, dtype: torch.dtype) -> PortalBase:
    """Load one tokenizer/base pair and describe it as a :class:`PortalBase`."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("base-model loading requires `pip install portallib[training]`") from exc

    tokenizer = AutoTokenizer.from_pretrained(recipe.model_id, revision=recipe.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "revision": recipe.revision,
        "dtype": recipe.dtype or dtype,
    }
    if recipe.device_map is not None:
        model_kwargs["device_map"] = recipe.device_map
    if recipe.attn_implementation is not None:
        model_kwargs["attn_implementation"] = recipe.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(recipe.model_id, **model_kwargs)
    if recipe.device_map is None:
        model = model.to(device)
    return PortalBase(
        model_id=recipe.model_id,
        model=model,
        tokenizer=tokenizer,
        revision=recipe.revision,
        layer_path=recipe.layer_path,
        module_paths=recipe.module_paths,
    )
