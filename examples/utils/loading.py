"""Hugging Face loading helpers shared by the runnable examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from portallib import ChoiceDataset, PortalBase


@dataclass(frozen=True)
class BaseRecipe:
    """One exact Hugging Face causal-language-model revision and decoder path."""

    model_id: str
    revision: str
    layer_path: str = "model.layers"


def load_dataset(source: str, *, revision: str | None) -> ChoiceDataset:
    """Load the common normalized dataset schema from a local file or the Hub."""
    path = Path(source)
    if path.is_file():
        if revision is not None:
            raise ValueError("dataset revision must be None when source is a local JSON file")
        return ChoiceDataset.from_json(path)
    return ChoiceDataset.from_hub(source, revision=revision)


def runtime_device() -> tuple[torch.device, torch.dtype]:
    """Select the single-device dtype shared by the examples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return device, dtype


def load_base(recipe: BaseRecipe, *, device: torch.device, dtype: torch.dtype) -> PortalBase:
    """Load one pinned tokenizer/base pair and describe it as a PortalBase."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(recipe.model_id, revision=recipe.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        recipe.model_id,
        revision=recipe.revision,
        dtype=dtype,
    ).to(device)
    return PortalBase(
        model_id=recipe.model_id,
        model=model,
        tokenizer=tokenizer,
        revision=recipe.revision,
        layer_path=recipe.layer_path,
    )


def model_slug(model_id: str) -> str:
    """Return a stable filesystem name for a Hub model ID."""
    return model_id.rsplit("/", 1)[-1].lower().replace(".", "-")
