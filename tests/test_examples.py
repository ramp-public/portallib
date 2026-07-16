from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


def load_example(name: str) -> ModuleType:
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_training_example_starts_only_from_raw_source_bases() -> None:
    example = load_example("train_example.py")

    assert example.DATASET_REVISION == "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
    assert tuple(base.model_id for base in example.SOURCE_BASES) == (
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
    )
    assert example.TRAINING_CONFIG.epochs == 12
    assert example.TRAINING_CONFIG.source_steps_per_epoch == 500


def test_refit_example_uses_published_source_artifact_and_raw_target() -> None:
    example = load_example("refit_example.py")

    assert example.DATASET_REVISION == "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
    assert example.SOURCE_ARTIFACT == "RampPublic/portal-qwen3-4b"
    assert example.SOURCE_ARTIFACT_REVISION == "v0.1.0"
    assert example.TARGET_BASE.model_id == "Qwen/Qwen3-8B"
    assert example.REFIT_MAX_EXAMPLES == 1000


def test_evaluation_example_uses_published_artifact_and_matching_base() -> None:
    example = load_example("evaluate_example.py")

    assert example.DATASET_REVISION == "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
    assert example.PORTAL_ARTIFACT == "RampPublic/portal-qwen3-8b"
    assert example.PORTAL_ARTIFACT_REVISION == "v0.1.0"
    assert example.BASE.model_id == "Qwen/Qwen3-8B"
    assert example.EVAL_BATCH_SIZE == 8


def test_base_recipe_forwards_host_loading_controls_without_bulk_device_move(monkeypatch) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from portallib.runtime import BaseRecipe, load_base

    calls: dict[str, object] = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.to_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls += 1
            return self

    model = FakeModel()

    def load_tokenizer(model_id: str, **kwargs: object) -> FakeTokenizer:
        calls["tokenizer"] = (model_id, kwargs)
        return FakeTokenizer()

    def load_model(model_id: str, **kwargs: object) -> FakeModel:
        calls["model"] = (model_id, kwargs)
        return model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    recipe = BaseRecipe(
        "example/base",
        "exact-revision",
        module_paths={"q": "self_attn.q_proj"},
        dtype="float32",
        device_map="cuda",
        attn_implementation="sdpa",
    )

    base = load_base(recipe, device=torch.device("cpu"), dtype=torch.bfloat16)

    assert base.model is model
    assert calls["tokenizer"] == ("example/base", {"revision": "exact-revision"})
    assert calls["model"] == (
        "example/base",
        {
            "revision": "exact-revision",
            "dtype": "float32",
            "device_map": "cuda",
            "attn_implementation": "sdpa",
        },
    )
    assert model.to_calls == 0
    assert base.module_paths == {"q": "self_attn.q_proj"}
