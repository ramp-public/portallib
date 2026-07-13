from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

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
    PortalTrainingConfig,
    TaskEvaluation,
)
from portallib.evaluation import PortalInjector
from portallib.training import _equalize_gradients, _task_regressions, _update_ema


class ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.q_proj(value) + self.v_proj(value)


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

    def forward(self, input_ids=None):
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


def test_evaluator_uses_character_normalized_choice_score_and_token_nll() -> None:
    row = ChoiceExample("metric", "prompt", (" yyyyyyyyyy", " n n"), 0)
    dataset = ChoiceDataset([row], [row])
    base = PortalBase("toy/fixed", FixedChoiceLM(), FixedChoiceTokenizer())

    result = PortalEvaluator().evaluate(base, dataset)
    expected_nll = -float(torch.log_softmax(torch.tensor([-10.0, -10.0, 0.0, 0.7, -10.0, -10.0, -10.0]), 0)[2])

    assert result.tasks["metric"].accuracy == 1.0
    assert result.tasks["metric"].gold_nll == pytest.approx(expected_nll)


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
    path.write_text(
        json.dumps(
            {
                "train": [row.to_dict() for row in dataset.train],
                "validation": [row.to_dict() for row in dataset.validation],
            }
        ),
        encoding="utf-8",
    )

    loaded = ChoiceDataset.from_json(path)

    assert loaded.tasks == ("alpha", "beta")
    assert loaded.rows("train", "alpha", limit=2) == dataset.rows("train", "alpha", limit=2)


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


def test_paper_training_defaults() -> None:
    recipe = PortalTrainingConfig()

    assert recipe.source_max_examples == 2000
    assert recipe.source_steps_per_epoch is None
    assert recipe.refit_max_examples == 2000
    assert recipe.eval_max_examples == 1000
    assert recipe.epochs == 5
    assert recipe.batch_size == 4
    assert recipe.early_stopping_patience is None


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
    for epoch in result.history:
        for task in epoch.evaluations["toy/one"].tasks.values():
            assert task.examples == 1
