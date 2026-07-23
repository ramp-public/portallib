from __future__ import annotations

import copy
import importlib.metadata
import json
import math
from pathlib import Path
from types import SimpleNamespace

import portallib
import pytest
import torch
from peft import PeftModel
from torch import nn
from torch.utils.checkpoint import checkpoint

from portallib import (
    ChoiceDataset,
    ChoiceExample,
    EpochMetrics,
    EvaluationResult,
    PortalAdapterRefitter,
    PortalBase,
    PortalConfig,
    PortalCoreTrainer,
    PortalDecoder,
    PortalEvaluator,
    PortalModel,
    PortalProjectionTarget,
    PortalTrainingConfig,
    TaskEvaluation,
    collate_gold_batch,
)
from portallib.evaluation import PortalInjector
from portallib.training import (
    _BatchCycle,
    _TrainingRunTracker,
    _equalize_gradients,
    _make_scheduler,
    _task_regressions,
    _update_ema,
)


def test_package_version_matches_distribution_metadata() -> None:
    assert portallib.__version__ == importlib.metadata.version("portallib")


class ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.q_proj(value) + self.v_proj(value)


class MetaPlaceholderLinear(nn.Linear):
    """Linear-shaped module whose public weight mimics an offloaded placeholder."""

    def __init__(self, width: int, *, dtype: torch.dtype = torch.float64):
        super().__init__(width, width, bias=False, device="meta", dtype=dtype)
        self.register_buffer("resident_weight", torch.eye(width, dtype=dtype))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.resident_weight)


class MetaOutputLinear(nn.Linear):
    """Linear-shaped module that simulates an invalid meta-only execution."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.empty((*value.shape[:-1], self.out_features), device="meta", dtype=value.dtype)


class ToyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = ToyAttention(width)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(width, width, bias=False)
        self.mlp.up_proj = nn.Linear(width, width, bias=False)
        self.mlp.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.tanh(self.self_attn(value))


class ToyBaseModel(nn.Module):
    def __init__(self, width: int = 4, n_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="toy/base", use_cache=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyLayer(width) for _ in range(n_layers)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            value = layer(value)
        return value

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs


class ToyMultimodalBaseModel(nn.Module):
    def __init__(self, width: int = 4, n_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="toy/multimodal", use_cache=True)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([ToyLayer(width) for _ in range(n_layers)])
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.layers = nn.ModuleList([ToyLayer(width) for _ in range(n_layers)])

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs


class HeterogeneousAttention(nn.Module):
    def __init__(self, width: int, q_width: int, v_width: int | None):
        super().__init__()
        self.q_proj = nn.Linear(width, q_width, bias=False)
        if v_width is not None:
            self.v_proj = nn.Linear(width, v_width, bias=False)


class HeterogeneousLayer(nn.Module):
    def __init__(self, width: int, q_width: int, v_width: int | None):
        super().__init__()
        self.self_attn = HeterogeneousAttention(width, q_width, v_width)


class HeterogeneousBaseModel(nn.Module):
    """Miniature global/local topology with later shared KV projections."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="toy/heterogeneous", use_cache=True)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [
                HeterogeneousLayer(4, 4, 2),
                HeterogeneousLayer(4, 6, 3),
                HeterogeneousLayer(4, 4, None),
                HeterogeneousLayer(4, 6, None),
            ]
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "yes": 2, "no": 3, "a": 4, "b": 5, "?": 6}

    def __call__(self, text: str, *, add_special_tokens: bool = True):
        tokens = [self.vocab.get(token.lower(), 4) for token in text.strip().split()]
        return SimpleNamespace(input_ids=([self.eos_token_id] if add_special_tokens else []) + tokens)


class ToyCausalLM(ToyBaseModel):
    def __init__(self, seed: int = 0, width: int = 4, vocab: int = 7):
        torch.manual_seed(seed)
        super().__init__(width=width, n_layers=2)
        self.embed = nn.Embedding(vocab, width)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.checkpointing = False
        self.checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs=None):
        self.checkpointing = True
        self.checkpointing_kwargs = gradient_checkpointing_kwargs

    def enable_input_require_grads(self):
        return None

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        value = self.embed(input_ids)
        for layer in self.model.layers:
            if self.checkpointing and self.training:
                value = checkpoint(layer, value, use_reentrant=False)
            else:
                value = layer(value)
        logits = self.lm_head(value)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class FixedChoiceLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.forward_calls = 0

    def forward(self, input_ids=None, attention_mask=None):
        self.forward_calls += 1
        logits = torch.full((*input_ids.shape, 7), -10.0, device=input_ids.device) + self.anchor
        logits[..., 2] = 0.0 + self.anchor
        logits[..., 3] = 0.7 + self.anchor
        return SimpleNamespace(logits=logits)


class FixedChoiceTokenizer(ToyTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool = True):
        stripped = text.strip()
        if stripped == "yyyyyyyyyy":
            tokens = [2]
        elif stripped == "n n":
            tokens = [3, 3]
        else:
            tokens = [4]
        return SimpleNamespace(input_ids=([1] if add_special_tokens else []) + tokens)


class BoundaryTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, *, add_special_tokens: bool = True):
        tokens = {
            "because": [4],
            "because ": [4, 7],
            "Ian": [8],
            "Dennis": [9],
            " Ian": [10],
            " Dennis": [11],
        }[text]
        return SimpleNamespace(input_ids=([self.eos_token_id] if add_special_tokens else []) + tokens)


class EmptyPromptTokenizer:
    pad_token_id = 0
    bos_token_id = None
    eos_token_id = 1

    def __call__(self, text: str, *, add_special_tokens: bool = True):
        return SimpleNamespace(input_ids=[] if text == "" else [2])


class RecordingChoiceLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.input_batches: list[torch.Tensor] = []

    def forward(self, input_ids=None, attention_mask=None):
        self.input_batches.append(input_ids.detach().cpu())
        logits = torch.zeros((*input_ids.shape, 12), device=input_ids.device) + self.anchor
        return SimpleNamespace(logits=logits)


def make_config(model: nn.Module | None = None, *, tasks: list[str] | None = None) -> PortalConfig:
    return PortalConfig.from_model(
        model or ToyBaseModel(),
        tasks=tasks or ["alpha", "beta"],
        base_model_name_or_path="toy/base",
        modules=("q", "v"),
        rank=2,
        alpha=4,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
    )


def make_portal(*, tasks: list[str] | None = None) -> PortalModel:
    torch.manual_seed(7)
    config = make_config(tasks=tasks)
    decoder = PortalDecoder(config)
    with torch.no_grad():
        for head in decoder.core.B.values():
            head.weight.normal_(std=0.02)
    latents = torch.randn(len(config.tasks), config.d_z)
    return PortalModel(config, latents, decoder)


def make_dataset() -> ChoiceDataset:
    train = []
    validation = []
    for task, gold in (("alpha", 0), ("beta", 1)):
        for index in range(4):
            train.append(ChoiceExample(task, f"a {index % 2} ?", (" yes", " no"), gold))
        validation.append(ChoiceExample(task, "a ?", (" yes", " no"), gold))
        validation.append(ChoiceExample(task, "b ?", (" yes", " no"), gold))
    return ChoiceDataset(train, validation)


def make_factor_parameters(
    config: PortalConfig,
    *,
    active_key: tuple[int, str],
    dtype: torch.dtype = torch.float32,
) -> dict[tuple[int, str], tuple[nn.Parameter, nn.Parameter]]:
    factors: dict[tuple[int, str], tuple[nn.Parameter, nn.Parameter]] = {}
    for layer_index, module_name, _exact_path in config.targets():
        key = (layer_index, module_name)
        active = key == active_key
        a = torch.full(
            (config.rank, config.in_dims[module_name]),
            0.25 if active else 0.0,
            dtype=dtype,
        )
        b = torch.full(
            (config.out_dims[module_name], config.rank),
            -0.2 if active else 0.0,
            dtype=dtype,
        )
        factors[key] = (nn.Parameter(a), nn.Parameter(b))
    return factors


def test_canonical_initialization_and_factor_shapes() -> None:
    config = make_config()
    decoder = PortalDecoder(config)
    factors = decoder(torch.randn(config.d_z))

    assert decoder.config.architecture == "canonical"
    assert torch.count_nonzero(decoder.core.film.weight) == 0
    assert torch.count_nonzero(decoder.core.film.bias) == 0
    assert all(torch.count_nonzero(head.weight) == 0 for head in decoder.core.B.values())
    assert factors[(0, "q")][0].shape == (2, 4)
    assert factors[(0, "q")][1].shape == (4, 2)
    assert all(torch.count_nonzero(b) == 0 for _a, b in factors.values())


def test_full_module_configuration_includes_attention_and_mlp_projections() -> None:
    config = PortalConfig.from_model(
        ToyBaseModel(),
        tasks=["alpha"],
        base_model_name_or_path="toy/base",
        modules=("q", "k", "v", "o", "gate", "up", "down"),
        rank=2,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
    )
    factors = PortalDecoder(config)(torch.randn(3))

    assert set(config.modules) == {"q", "k", "v", "o", "gate", "up", "down"}
    assert set(module for _layer, module in factors) == set(config.modules)


def test_refit_alignment_starts_with_zero_delta_after_trained_core() -> None:
    config = make_config()
    source = PortalDecoder(config)
    with torch.no_grad():
        for head in source.core.B.values():
            head.weight.normal_()
    refit = PortalDecoder(config, core=copy.deepcopy(source.core), refit_init=True)

    factors = refit(torch.randn(config.d_z))

    assert all(torch.count_nonzero(b) == 0 for _a, b in factors.values())


def test_native_artifact_round_trip(tmp_path: Path) -> None:
    original = make_portal()
    generated = original.generate("alpha")
    original.save_pretrained(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {"README.md", "config.json", "model.safetensors"}
    loaded = PortalModel.from_pretrained(tmp_path)

    assert loaded.config == original.config
    for key, (expected_a, expected_b) in generated.items():
        actual_a, actual_b = loaded.generate("alpha")[key]
        torch.testing.assert_close(actual_a, expected_a)
        torch.testing.assert_close(actual_b, expected_b)


def test_hub_download_uses_standard_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_portal().save_pretrained(tmp_path)
    calls: list[str] = []

    def fake_download(**kwargs):
        calls.append(kwargs["filename"])
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr("huggingface_hub.hub_mixin.hf_hub_download", fake_download)
    monkeypatch.setattr("portallib.model.hf_hub_download", fake_download)

    loaded = PortalModel.from_pretrained("example/portal", revision="release-1", local_files_only=True)

    assert loaded.tasks == ("alpha", "beta")
    assert calls == ["config.json", "model.safetensors"]


def test_populates_and_exports_normal_peft_lora(tmp_path: Path) -> None:
    portal = make_portal()
    generated = portal.generate("beta")
    peft_model = portal.get_peft_model("beta", ToyBaseModel())

    populated = 0
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        layer_index = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        short_name = name.rsplit(".", 1)[-1][0]
        expected_a, expected_b = generated[(layer_index, short_name)]
        torch.testing.assert_close(module.lora_A["default"].weight, expected_a)
        torch.testing.assert_close(module.lora_B["default"].weight, expected_b)
        populated += 1
    assert populated == 4

    portal.export_peft("beta", tmp_path)
    reloaded = PeftModel.from_pretrained(ToyBaseModel(), tmp_path)
    assert isinstance(reloaded, PeftModel)


def test_peft_targets_only_exact_language_model_paths(tmp_path: Path) -> None:
    base = ToyMultimodalBaseModel()
    config = PortalConfig.from_model(
        base,
        tasks=["alpha"],
        base_model_name_or_path="toy/multimodal",
        modules=("q", "v"),
        layer_path="model.language_model.layers",
        rank=2,
        alpha=4,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
    )
    decoder = PortalDecoder(config)
    with torch.no_grad():
        for head in decoder.core.B.values():
            head.weight.normal_(std=0.02)
    portal = PortalModel(config, torch.randn(1, config.d_z), decoder)

    peft_model = portal.get_peft_model("alpha", base)
    adapted = [name for name, module in peft_model.named_modules() if hasattr(module, "lora_A")]

    assert len(adapted) == config.n_layers * len(config.modules)
    assert all("model.language_model.layers" in name for name in adapted)
    assert all("vision_tower" not in name for name in adapted)

    portal.export_peft("alpha", tmp_path)
    adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert set(adapter_config["target_modules"]) == {
        f"model.language_model.layers.{layer_index}.{module_path}"
        for layer_index in range(config.n_layers)
        for module_path in config.module_paths.values()
    }
    reloaded = PeftModel.from_pretrained(ToyMultimodalBaseModel(), tmp_path)
    reloaded_adapted = [name for name, module in reloaded.named_modules() if hasattr(module, "lora_A")]
    assert reloaded_adapted == adapted


def test_config_rejects_direct_and_requires_exact_paths() -> None:
    with pytest.raises(ValueError, match="canonical"):
        PortalConfig(**{**make_config().to_dict(), "architecture": "direct"})
    with pytest.raises(ValueError, match="exact projection path"):
        PortalConfig.from_model(
            ToyBaseModel(),
            tasks=["alpha"],
            base_model_name_or_path="toy/base",
            modules=("q",),
            module_paths={"q": "self_attn.missing"},
        )


def test_config_enumerates_each_exact_target_once() -> None:
    config = make_config()

    assert all(isinstance(target, PortalProjectionTarget) for target in config.target_specs())
    assert list(config.targets()) == [
        (0, "q", "model.layers.0.self_attn.q_proj"),
        (0, "v", "model.layers.0.self_attn.v_proj"),
        (1, "q", "model.layers.1.self_attn.q_proj"),
        (1, "v", "model.layers.1.self_attn.v_proj"),
    ]


def make_heterogeneous_config() -> PortalConfig:
    return PortalConfig.from_model(
        HeterogeneousBaseModel(),
        tasks=["alpha", "beta"],
        base_model_name_or_path="toy/heterogeneous",
        modules=("q", "v"),
        layer_path="model.language_model.layers",
        allow_heterogeneous_targets=True,
        rank=2,
        alpha=4,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
    )


def make_heterogeneous_portal() -> PortalModel:
    source_config = PortalConfig.from_model(
        ToyBaseModel(width=4, n_layers=4),
        tasks=["alpha", "beta"],
        base_model_name_or_path="toy/source",
        modules=("q", "v"),
        rank=2,
        alpha=4,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
    )
    target_config = make_heterogeneous_config()
    source = PortalDecoder(source_config)
    with torch.no_grad():
        for head in source.core.B.values():
            head.weight.normal_(std=0.02)
        alignment = PortalDecoder(target_config, core=copy.deepcopy(source.core), refit_init=True).alignment
        for parameter in alignment.output.values():
            parameter.normal_(std=0.02)
    decoder = PortalDecoder(target_config, core=copy.deepcopy(source.core), alignment=alignment)
    return PortalModel(target_config, torch.randn(2, target_config.d_z), decoder)


def test_heterogeneous_target_layout_groups_shapes_and_absent_projections() -> None:
    config = make_heterogeneous_config()
    portal = make_heterogeneous_portal()
    factors = portal.generate("alpha")

    assert config.schema_version == 2
    assert all(isinstance(target, PortalProjectionTarget) for target in config.target_specs())
    assert config.in_dims == config.out_dims == {}
    assert config.input_groups == {"q": 4, "v": 4}
    assert config.output_groups == {
        "q__out_4": 4,
        "v__out_2": 2,
        "q__out_6": 6,
        "v__out_3": 3,
    }
    assert list(config.targets()) == [
        (0, "q", "model.language_model.layers.0.self_attn.q_proj"),
        (0, "v", "model.language_model.layers.0.self_attn.v_proj"),
        (1, "q", "model.language_model.layers.1.self_attn.q_proj"),
        (1, "v", "model.language_model.layers.1.self_attn.v_proj"),
        (2, "q", "model.language_model.layers.2.self_attn.q_proj"),
        (3, "q", "model.language_model.layers.3.self_attn.q_proj"),
    ]
    assert set(factors) == {(0, "q"), (0, "v"), (1, "q"), (1, "v"), (2, "q"), (3, "q")}
    for target in config.target_specs():
        a, b = factors[(target.layer_index, target.module_name)]
        assert a.shape == (config.rank, target.in_features)
        assert b.shape == (target.out_features, config.rank)


def test_heterogeneous_layout_is_opt_in_and_validates_exact_base_dimensions() -> None:
    with pytest.raises(ValueError, match="exact projection path|dimensions vary"):
        PortalConfig.from_model(
            HeterogeneousBaseModel(),
            tasks=["alpha"],
            modules=("q", "v"),
            layer_path="model.language_model.layers",
        )

    config = make_heterogeneous_config()
    base = HeterogeneousBaseModel()
    base.model.language_model.layers[0].self_attn.q_proj = nn.Linear(4, 5, bias=False)
    with pytest.raises(ValueError, match="configured dimensions"):
        PortalInjector(base, config)


def test_heterogeneous_refit_alignment_is_differentiable_and_core_compatible() -> None:
    source = make_portal()
    target_config = make_heterogeneous_config()
    decoder = PortalDecoder(target_config, core=copy.deepcopy(source.decoder.core), refit_init=True)

    assert all(torch.count_nonzero(parameter) == 0 for parameter in decoder.alignment.output.values())
    with torch.no_grad():
        for parameter in decoder.alignment.output.values():
            parameter.normal_(std=0.02)
    factors = decoder(torch.randn(target_config.d_z))
    sum(value.square().sum() for pair in factors.values() for value in pair).backward()

    assert all(parameter.grad is not None for parameter in decoder.alignment.parameters())


def test_heterogeneous_artifact_and_peft_round_trip(tmp_path: Path) -> None:
    portal = make_heterogeneous_portal()
    native = tmp_path / "native"
    peft = tmp_path / "peft"
    portal.save_pretrained(native)

    loaded = PortalModel.from_pretrained(native)
    assert loaded.config == portal.config
    for key, expected in portal.generate("alpha").items():
        actual = loaded.generate("alpha")[key]
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    adapted = portal.get_peft_model("alpha", HeterogeneousBaseModel())
    adapted_names = [name for name, module in adapted.named_modules() if hasattr(module, "lora_A")]
    assert len(adapted_names) == 6
    assert all("language_model.layers" in name for name in adapted_names)
    portal.export_peft("alpha", peft)
    reloaded = PeftModel.from_pretrained(HeterogeneousBaseModel(), peft)
    assert len([module for module in reloaded.modules() if hasattr(module, "lora_A")]) == 6


def test_uniform_alignment_state_names_remain_compatible() -> None:
    alignment = PortalDecoder(make_config()).alignment

    assert set(alignment.input) == {"q", "v"}
    assert set(alignment.output) == {"q", "v"}
    assert set(alignment.state_dict()) == {
        "layer_embeddings.weight",
        "input.q",
        "input.v",
        "output.q",
        "output.v",
    }


def test_injector_uses_live_output_placement_for_meta_weight() -> None:
    base = ToyBaseModel(width=4, n_layers=1)
    projection = MetaPlaceholderLinear(4)
    base.model.layers[0].self_attn.q_proj = projection
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    config = PortalConfig.from_model(
        base,
        tasks=["alpha"],
        base_model_name_or_path="toy/meta",
        modules=("q",),
        rank=2,
        alpha=4,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
    )
    factors = make_factor_parameters(config, active_key=(0, "q"))
    value = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    raw = projection(value).detach()
    a, b = factors[(0, "q")]
    expected = raw + torch.nn.functional.linear(
        torch.nn.functional.linear(value.detach(), a.detach().to(dtype=value.dtype)),
        b.detach().to(dtype=value.dtype),
    ) * config.scaling
    injector = PortalInjector(base, config)

    try:
        with injector.activate(factors):
            adapted = projection(value)
            adapted.square().mean().backward()
        after = projection(value.detach())
    finally:
        injector.close()

    assert projection.weight.device.type == "meta"
    assert adapted.dtype == torch.float64
    torch.testing.assert_close(adapted.detach(), expected)
    assert not torch.equal(adapted.detach(), raw)
    torch.testing.assert_close(after, raw)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert a.grad is not None and torch.isfinite(a.grad).all() and torch.count_nonzero(a.grad) > 0
    assert b.grad is not None and torch.isfinite(b.grad).all() and torch.count_nonzero(b.grad) > 0
    assert a.grad.dtype == torch.float32 and b.grad.dtype == torch.float32


def test_injector_rejects_missing_misshaped_and_meta_factors() -> None:
    base = ToyBaseModel(width=4, n_layers=1)
    config = PortalConfig.from_model(
        base,
        tasks=["alpha"],
        base_model_name_or_path="toy/base",
        modules=("q",),
        rank=2,
    )
    injector = PortalInjector(base, config)

    try:
        with pytest.raises(ValueError, match="missing configured target"):
            with injector.activate({}):
                pass
        with pytest.raises(ValueError, match="generated shape mismatch"):
            with injector.activate({(0, "q"): (torch.zeros(1, 4), torch.zeros(4, 2))}):
                pass
        with pytest.raises(ValueError, match="cannot be on the meta device"):
            with injector.activate(
                {
                    (0, "q"): (
                        torch.empty(2, 4, device="meta"),
                        torch.empty(4, 2, device="meta"),
                    )
                }
            ):
                pass
    finally:
        injector.close()


def test_injector_rejects_meta_runtime_output() -> None:
    base = ToyBaseModel(width=4, n_layers=1)
    projection = MetaOutputLinear(4, 4, bias=False)
    base.model.layers[0].self_attn.q_proj = projection
    config = PortalConfig.from_model(
        base,
        tasks=["alpha"],
        base_model_name_or_path="toy/meta-output",
        modules=("q",),
        rank=2,
    )
    factors = make_factor_parameters(config, active_key=(0, "q"))
    injector = PortalInjector(base, config)

    try:
        with pytest.raises(RuntimeError, match="projection output is on the meta device"):
            with injector.activate(factors):
                projection(torch.ones(1, 4))
    finally:
        injector.close()


def test_injector_is_differentiable_and_checkpoint_safe() -> None:
    base = ToyCausalLM()
    config = make_config(base, tasks=["alpha"])
    decoder = PortalDecoder(config)
    with torch.no_grad():
        for head in decoder.core.B.values():
            head.weight.normal_(std=0.1)
    latent = nn.Parameter(torch.randn(config.d_z))
    injector = PortalInjector(base, config)
    base.train()
    PortalBase("toy/base", base, ToyTokenizer()).freeze(gradient_checkpointing=True)
    assert base.checkpointing_kwargs == {"use_reentrant": False}
    ids = torch.tensor([[1, 4, 6, 2]])
    labels = torch.tensor([[-100, -100, -100, 2]])

    with injector.activate(decoder(latent)):
        base(input_ids=ids, labels=labels).loss.backward()
    injector.close()

    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_injector_caches_live_factor_placement_during_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ToyCausalLM()
    config = make_config(base, tasks=["alpha"])
    factors = PortalDecoder(config)(torch.randn(config.d_z))
    factor_ids = {id(factor) for pair in factors.values() for factor in pair}
    original_to = torch.Tensor.to
    conversion_calls = 0

    def counting_to(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal conversion_calls
        if id(tensor) in factor_ids:
            conversion_calls += 1
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counting_to)
    injector = PortalInjector(base, config)
    value = torch.randn(1, 2, base.model.layers[0].self_attn.q_proj.in_features)
    with torch.no_grad():
        with injector.activate(factors):
            base.model.layers[0](value)
            base.model.layers[0](value)
    injector.close()

    assert conversion_calls == 2 * len(config.modules)


def test_injector_matches_reference_with_disk_offloaded_layer(tmp_path: Path) -> None:
    from accelerate import dispatch_model

    torch.manual_seed(17)
    plain = ToyBaseModel(width=4, n_layers=2)
    offloaded = ToyBaseModel(width=4, n_layers=2)
    offloaded.load_state_dict(copy.deepcopy(plain.state_dict()))
    for model in (plain, offloaded):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    offloaded = dispatch_model(
        offloaded,
        device_map={"model.layers.0": "cpu", "model.layers.1": "disk"},
        offload_dir=tmp_path / "offload",
    )
    config = make_config(offloaded, tasks=["alpha"])
    plain_factors = make_factor_parameters(config, active_key=(1, "q"))
    offloaded_factors = make_factor_parameters(config, active_key=(1, "q"))
    plain_injector = PortalInjector(plain, config)
    offloaded_injector = PortalInjector(offloaded, config)
    target = offloaded.model.layers[1].self_attn.q_proj
    value = torch.randn(2, 3, 4)

    try:
        assert target.weight.device.type == "meta"
        with torch.no_grad():
            raw = offloaded(value)
            with offloaded_injector.activate(offloaded_factors):
                evaluated = offloaded(value)
        assert target.weight.device.type == "meta"
        assert (evaluated - raw).abs().max() > 1e-6

        plain_value = value.detach().clone().requires_grad_()
        offloaded_value = value.detach().clone().requires_grad_()
        with plain_injector.activate(plain_factors):
            plain_output = plain(plain_value)
            plain_output.square().mean().backward()
        with offloaded_injector.activate(offloaded_factors):
            offloaded_output = offloaded(offloaded_value)
            offloaded_output.square().mean().backward()
    finally:
        plain_injector.close()
        offloaded_injector.close()

    assert target.weight.device.type == "meta"
    torch.testing.assert_close(offloaded_output, plain_output)
    torch.testing.assert_close(offloaded_value.grad, plain_value.grad)
    for offloaded_factor, plain_factor in zip(offloaded_factors[(1, "q")], plain_factors[(1, "q")]):
        assert offloaded_factor.grad is not None and torch.isfinite(offloaded_factor.grad).all()
        assert torch.count_nonzero(offloaded_factor.grad) > 0
        torch.testing.assert_close(offloaded_factor.grad, plain_factor.grad)
    assert all(parameter.grad is None for parameter in plain.parameters())
    assert all(parameter.grad is None for parameter in offloaded.parameters())


def test_disk_offloaded_injector_is_checkpoint_safe(tmp_path: Path) -> None:
    from accelerate import dispatch_model

    model = ToyCausalLM()
    model = dispatch_model(
        model,
        device_map={
            "embed": "cpu",
            "model.layers.0": "cpu",
            "model.layers.1": "disk",
            "lm_head": "cpu",
        },
        offload_dir=tmp_path / "checkpoint-offload",
    )
    base = PortalBase("toy/base", model, ToyTokenizer())
    base.freeze(gradient_checkpointing=True)
    model.train()
    config = make_config(model, tasks=["alpha"])
    factors = make_factor_parameters(config, active_key=(1, "q"))
    injector = PortalInjector(model, config)
    ids = torch.tensor([[1, 4, 6, 2]])
    labels = torch.tensor([[-100, -100, -100, 2]])
    target = model.model.layers[1].self_attn.q_proj

    try:
        assert target.weight.device.type == "meta"
        with injector.activate(factors):
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
    finally:
        injector.close()

    assert model.checkpointing_kwargs == {"use_reentrant": False}
    assert model.config.use_cache is False
    assert target.weight.device.type == "meta"
    assert torch.isfinite(loss)
    for factor in factors[(1, "q")]:
        assert factor.grad is not None and torch.isfinite(factor.grad).all()
        assert torch.count_nonzero(factor.grad) > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluator_epoch_zero_matches_unadapted_base() -> None:
    model = ToyCausalLM()
    base = PortalBase("toy/base", model, ToyTokenizer())
    dataset = make_dataset()
    config = make_config(model)
    zero_portal = PortalModel(config, torch.randn(2, 3), PortalDecoder(config))
    evaluator = PortalEvaluator(max_prompt=32)

    unadapted = evaluator.evaluate(base, dataset)
    adapted = evaluator.evaluate(base, dataset, tasks=dataset.tasks, portal=zero_portal)

    assert model.config.use_cache is True
    assert adapted.macro_accuracy == unadapted.macro_accuracy
    assert adapted.macro_gold_nll == pytest.approx(unadapted.macro_gold_nll)
    assert math.isfinite(adapted.macro_gold_nll)


def test_evaluator_accepts_portal_task_subsets_and_rejects_unknown_tasks() -> None:
    model = ToyCausalLM()
    base = PortalBase("toy/base", model, ToyTokenizer())
    dataset = make_dataset()
    portal = PortalModel(make_config(model), torch.randn(2, 3), PortalDecoder(make_config(model)))

    result = PortalEvaluator(max_prompt=32).evaluate(base, dataset, tasks=("beta",), portal=portal)

    assert tuple(result.tasks) == ("beta",)
    with pytest.raises(ValueError, match="absent.*gamma"):
        PortalEvaluator(max_prompt=32).evaluate(base, dataset, tasks=("gamma",), portal=portal)


def test_evaluator_rejects_an_artifact_for_a_different_base() -> None:
    dataset = make_dataset()
    base = PortalBase("other/base", FixedChoiceLM(), FixedChoiceTokenizer())

    with pytest.raises(ValueError, match="artifact expects 'toy/base'.*'other/base'"):
        PortalEvaluator(max_prompt=32).evaluate(base, dataset, portal=make_portal())


def test_evaluator_uses_character_normalized_choice_score_and_token_nll() -> None:
    row = ChoiceExample("metric", "prompt", (" yyyyyyyyyy", " n n"), 0)
    dataset = ChoiceDataset([row], [row])
    base = PortalBase("toy/fixed", FixedChoiceLM(), FixedChoiceTokenizer())

    result = PortalEvaluator().evaluate(base, dataset)
    expected_nll = -float(torch.log_softmax(torch.tensor([-10.0, -10.0, 0.0, 0.7, -10.0, -10.0, -10.0]), 0)[2])

    assert result.tasks["metric"].accuracy == 1.0
    assert result.tasks["metric"].gold_nll == pytest.approx(expected_nll)


def test_evaluator_batches_choices_without_changing_metrics() -> None:
    rows = [ChoiceExample("metric", f"prompt {index}", (" yyyyyyyyyy", " n n"), index % 2) for index in range(5)]
    dataset = ChoiceDataset(rows, rows)
    serial_model = FixedChoiceLM()
    batched_model = FixedChoiceLM()

    serial = PortalEvaluator(batch_size=1).evaluate(
        PortalBase("toy/fixed", serial_model, FixedChoiceTokenizer()), dataset
    )
    batched = PortalEvaluator(batch_size=4).evaluate(
        PortalBase("toy/fixed", batched_model, FixedChoiceTokenizer()), dataset
    )

    assert batched == serial
    assert serial_model.forward_calls == 10
    assert batched_model.forward_calls == 3


def test_gold_batch_collator_masks_prompts_and_is_public() -> None:
    row = ChoiceExample("alpha", "a ?", (" yes", " no"), 0)
    input_ids, attention_mask, labels = collate_gold_batch(
        ToyTokenizer(),
        [row],
        max_prompt=32,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[1, 4, 6, 2]]
    assert attention_mask.tolist() == [[1, 1, 1, 1]]
    assert labels.tolist() == [[-100, -100, -100, 2]]


def test_training_and_evaluation_move_trailing_prompt_whitespace_to_continuations() -> None:
    row = ChoiceExample("boundary", "because ", ("Ian", "Dennis"), 0)
    input_ids, attention_mask, labels = collate_gold_batch(
        BoundaryTokenizer(),
        [row],
        max_prompt=32,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[1, 4, 10]]
    assert attention_mask.tolist() == [[1, 1, 1]]
    assert labels.tolist() == [[-100, -100, 10]]

    model = RecordingChoiceLM()
    dataset = ChoiceDataset([row], [row])
    PortalEvaluator(batch_size=2).evaluate(
        PortalBase("toy/boundary", model, BoundaryTokenizer()),
        dataset,
    )
    assert model.input_batches[0].tolist() == [[1, 4, 10], [1, 4, 11]]


def test_empty_prompts_use_the_tokenizer_end_token_as_context() -> None:
    row = ChoiceExample("boundary", "", (" answer", " alternative"), 0)
    input_ids, attention_mask, labels = collate_gold_batch(
        EmptyPromptTokenizer(),
        [row],
        max_prompt=32,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[1, 2]]
    assert attention_mask.tolist() == [[1, 1]]
    assert labels.tolist() == [[-100, 2]]


def test_ema_and_base_gradient_equalization_match_stage3_recipe() -> None:
    assert _update_ema(None, 4.0, 0.9) == 4.0
    assert _update_ema(4.0, 2.0, 0.9) == pytest.approx(3.8)
    combined = _equalize_gradients([torch.tensor([3.0, 0.0]), torch.tensor([0.0, 1.0])])
    torch.testing.assert_close(combined, torch.tensor([2.0, 2.0]))


def test_task_regression_diagnostics_compare_selected_epoch_to_epoch_zero() -> None:
    initial_result = EvaluationResult(
        "toy/base",
        {"alpha": TaskEvaluation("alpha", 0.8, 1.0, 10)},
        0.8,
        1.0,
    )
    selected_result = EvaluationResult(
        "toy/base",
        {"alpha": TaskEvaluation("alpha", 0.7, 0.8, 10)},
        0.7,
        0.8,
    )
    initial = EpochMetrics(0, {"toy/base": initial_result}, 0.8, 1.0)
    selected = EpochMetrics(2, {"toy/base": selected_result}, 0.7, 0.8)

    regressions = _task_regressions(initial, selected, 0.05)
    assert regressions["toy/base"]["alpha"] == pytest.approx(-0.1)


def test_choice_dataset_preserves_task_order_and_round_trips(tmp_path: Path) -> None:
    dataset = make_dataset()
    path = tmp_path / "tasks.json"
    dataset.save_json(path)

    loaded = ChoiceDataset.from_json(path)

    assert json.loads(path.read_text()) == dataset.to_dict()
    assert loaded.tasks == ("alpha", "beta")
    assert loaded.rows("train", "alpha", limit=2) == dataset.rows("train", "alpha", limit=2)


def test_evaluation_results_have_one_canonical_json_representation() -> None:
    task = TaskEvaluation("alpha", 0.75, 0.5, 4)
    result = EvaluationResult("toy/base", {"alpha": task}, 0.75, 0.5)

    assert task.to_dict() == {"accuracy": 0.75, "gold_nll": 0.5, "examples": 4}
    assert result.to_dict() == {
        "base_model": "toy/base",
        "macro_accuracy": 0.75,
        "macro_gold_nll": 0.5,
        "tasks": {"alpha": task.to_dict()},
    }


def test_training_config_reuses_artifact_architecture() -> None:
    config = make_config()
    recipe = PortalTrainingConfig.from_portal_config(config, epochs=7, refit_max_examples=100)

    assert (recipe.modules, recipe.rank, recipe.d_z, recipe.d_core) == (
        config.modules,
        config.rank,
        config.d_z,
        config.d_core,
    )
    assert recipe.epochs == 7
    assert recipe.refit_max_examples == 100
    with pytest.raises(ValueError, match="cannot be overridden"):
        PortalTrainingConfig.from_portal_config(config, rank=16)


def test_batch_cycle_recycles_deterministically_per_unit() -> None:
    first = ChoiceExample("alpha", "first", (" yes", " no"), 0)
    second = ChoiceExample("alpha", "second", (" yes", " no"), 0)
    cycle = _BatchCycle({0: [[first], [second]]}, lambda unit, pass_index: unit + pass_index)

    observed = [cycle.draw(10, 0)[0].prompt for _ in range(4)]
    replay = _BatchCycle({0: [[first], [second]]}, lambda unit, pass_index: unit + pass_index)

    assert observed == [replay.draw(10, 0)[0].prompt for _ in range(4)]
    assert cycle.passes[10] == 2


def test_training_run_tracker_centralizes_selection_and_early_stopping() -> None:
    callbacks: list[int] = []
    initial = EpochMetrics(0, {}, 0.5, 1.0)
    first = EpochMetrics(1, {}, 0.6, 0.9)
    tied_better_loss = EpochMetrics(2, {}, 0.6, 0.8)
    worse = EpochMetrics(3, {}, 0.5, 0.7)
    tracker = _TrainingRunTracker("initial", patience=1, on_epoch=lambda metrics: callbacks.append(metrics.epoch))

    tracker.record_initial(initial)
    assert tracker.record_epoch(first, "first") is False
    assert tracker.record_epoch(tied_better_loss, "second") is False
    assert tracker.record_epoch(worse, "third") is True

    assert callbacks == [0, 1, 2, 3]
    assert tracker.best_epoch == 2
    assert tracker.best_loss_epoch == 3
    assert tracker.best_state == "second"


def test_training_run_tracker_can_select_epoch_zero() -> None:
    initial = EpochMetrics(0, {}, 0.7, 0.6)
    worse = EpochMetrics(1, {}, 0.6, 0.8)
    tracker = _TrainingRunTracker("initial", patience=None, on_epoch=None)

    tracker.record_initial(initial)
    tracker.record_epoch(worse, "worse")

    assert tracker.best_epoch == 0
    assert tracker.best_loss_epoch == 0
    assert tracker.best_state == "initial"
    assert tracker.selected_metrics == initial


def test_core_trainer_rejects_duplicate_source_model_ids() -> None:
    duplicate_bases = [
        PortalBase("toy/duplicate", ToyCausalLM(seed=1), ToyTokenizer()),
        PortalBase("toy/duplicate", ToyCausalLM(seed=2), ToyTokenizer()),
    ]

    with pytest.raises(ValueError, match="model IDs must be unique"):
        PortalCoreTrainer(duplicate_bases, make_dataset())


def test_balanced_core_training_and_frozen_refit_contract() -> None:
    dataset = make_dataset()
    tokenizer = ToyTokenizer()
    first = PortalBase("toy/one", ToyCausalLM(seed=1), tokenizer)
    second = PortalBase("toy/two", ToyCausalLM(seed=2), ToyTokenizer())
    recipe = PortalTrainingConfig(
        rank=2,
        alpha=4,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
        source_max_examples=None,
        source_steps_per_epoch=5,
        refit_max_examples=1,
        epochs=1,
        batch_size=1,
        gradient_checkpointing=False,
    )

    trained = PortalCoreTrainer([first, second], dataset, config=recipe).train()

    assert trained.diagnostics["steps_per_epoch"] == 5
    assert trained.diagnostics["units_per_step"] == 4
    assert trained.diagnostics["first_decoder_grad_norm"] > 0
    assert len(trained.history) == 2 and trained.history[0].epoch == 0
    assert trained.best_epoch == trained.best_loss_epoch == 1
    assert set(trained.artifacts) == {"toy/one", "toy/two"}
    first_core = trained.artifacts["toy/one"].decoder.core.state_dict()
    second_core = trained.artifacts["toy/two"].decoder.core.state_dict()
    for name in first_core:
        torch.testing.assert_close(first_core[name], second_core[name])

    source = trained.artifacts["toy/two"]
    source_core = copy.deepcopy(source.decoder.core.state_dict())
    refit = PortalAdapterRefitter(
        source,
        PortalBase("toy/target", ToyCausalLM(seed=3), ToyTokenizer()),
        dataset,
        config=recipe,
    ).refit()

    assert refit.diagnostics["steps_per_epoch"] == 1
    assert refit.diagnostics["tasks_per_step"] == 2
    assert refit.diagnostics["first_alignment_grad_norm"] > 0
    assert len(refit.history) == 2 and refit.history[0].epoch == 0
    assert refit.best_epoch == refit.best_loss_epoch == 1
    for name, value in source_core.items():
        torch.testing.assert_close(refit.artifact.decoder.core.state_dict()[name], value)
    torch.testing.assert_close(refit.artifact.task_latents, source.task_latents)


def test_expanded_training_defaults() -> None:
    recipe = PortalTrainingConfig()

    assert recipe.source_max_examples == 2000
    assert recipe.eval_batch_size == 8
    assert recipe.source_resample_each_epoch is True
    assert recipe.source_steps_per_epoch is None
    assert recipe.refit_max_examples == 2000
    assert recipe.eval_max_examples == 1000
    assert recipe.epochs == 5
    assert recipe.batch_size == 4
    assert recipe.lr_scheduler == "constant"
    assert recipe.warmup_ratio == 0.0
    assert recipe.early_stopping_patience is None


def test_linear_warmup_decay_matches_stage3_recipe() -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=2.0)
    scheduler = _make_scheduler(
        optimizer,
        total_steps=10,
        scheduler="linear",
        warmup_ratio=0.2,
    )
    assert scheduler is not None
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates == pytest.approx([1.0, 2.0, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr_scheduler": "cosine"}, "lr_scheduler"),
        ({"warmup_ratio": -0.1}, "warmup_ratio"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
    ],
)
def test_training_schedule_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PortalTrainingConfig(**kwargs)


def test_source_rounds_are_derived_from_longest_capped_task() -> None:
    dataset = make_dataset()
    recipe = PortalTrainingConfig(
        rank=2,
        alpha=4,
        d_z=3,
        d_layer=2,
        hidden=5,
        d_core=3,
        source_max_examples=3,
        source_steps_per_epoch=None,
        refit_max_examples=1,
        eval_max_examples=1,
        epochs=1,
        batch_size=2,
        gradient_checkpointing=False,
    )

    result = PortalCoreTrainer(
        [PortalBase("toy/one", ToyCausalLM(seed=1), ToyTokenizer())],
        dataset,
        config=recipe,
    ).train()

    assert result.diagnostics["steps_per_epoch"] == 2
    assert result.diagnostics["source_examples_per_task"] == {"alpha": 3, "beta": 3}
    assert result.diagnostics["source_pool_examples_per_task"] == {"alpha": 4, "beta": 4}
    assert result.diagnostics["source_resample_each_epoch"] is True
    for epoch in result.history:
        for task in epoch.evaluations["toy/one"].tasks.values():
            assert task.examples == 1
