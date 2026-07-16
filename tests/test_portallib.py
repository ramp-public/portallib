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


def test_injector_prepares_factors_once_per_activation(monkeypatch: pytest.MonkeyPatch) -> None:
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
    with injector.activate(factors):
        base.model.layers[0](value)
        base.model.layers[0](value)
    injector.close()

    assert conversion_calls == 2 * len(factors)


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
