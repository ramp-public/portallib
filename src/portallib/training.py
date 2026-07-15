"""Canonical PorTAL core training and adapter refitting."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import nn

from .config import PortalConfig
from .data import ChoiceDataset, ChoiceExample
from .decoder import PortalAlignment, PortalCore, PortalDecoder
from .evaluation import EvaluationResult, PortalBase, PortalEvaluator, PortalInjector, collate_gold_batch
from .model import PortalModel


@dataclass(frozen=True)
class PortalTrainingConfig:
    """Canonical architecture and optimizer recipe used by both training phases."""

    modules: tuple[str, ...] = ("q", "v")
    rank: int = 8
    alpha: int = 16
    d_z: int = 256
    d_layer: int = 32
    hidden: int = 512
    d_core: int = 1024
    source_max_examples: int | None = 2000
    source_resample_each_epoch: bool = True
    source_steps_per_epoch: int | None = None
    refit_max_examples: int = 2000
    eval_max_examples: int | None = 1000
    eval_batch_size: int = 8
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    latent_learning_rate: float = 2e-3
    lr_scheduler: Literal["constant", "linear"] = "constant"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.9
    ema_floor: float = 1e-3
    max_prompt: int = 768
    seed: int = 0
    gradient_checkpointing: bool = True
    early_stopping_patience: int | None = None
    task_regression_threshold: float = 0.05
    checkpoint_dir: Path | None = None

    @classmethod
    def from_portal_config(cls, config: PortalConfig, **training_overrides: Any) -> "PortalTrainingConfig":
        """Reuse an artifact's canonical architecture with caller-selected optimization settings."""
        architecture = {
            "modules": config.modules,
            "rank": config.rank,
            "alpha": config.alpha,
            "d_z": config.d_z,
            "d_layer": config.d_layer,
            "hidden": config.hidden,
            "d_core": config.d_core,
        }
        conflicting = architecture.keys() & training_overrides.keys()
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise ValueError(f"artifact architecture cannot be overridden: {names}")
        return cls(**architecture, **training_overrides)

    def __post_init__(self) -> None:
        positive = (self.rank, self.alpha, self.d_z, self.d_layer, self.hidden, self.d_core)
        counts = (self.refit_max_examples, self.eval_batch_size, self.epochs, self.batch_size, self.max_prompt)
        if any(value <= 0 for value in (*positive, *counts)):
            raise ValueError("architecture dimensions and training counts must be positive")
        if self.source_max_examples is not None and self.source_max_examples <= 0:
            raise ValueError("source_max_examples must be positive or None for all examples")
        if self.source_steps_per_epoch is not None and self.source_steps_per_epoch <= 0:
            raise ValueError("source_steps_per_epoch must be positive or None for a full epoch")
        if self.eval_max_examples is not None and self.eval_max_examples <= 0:
            raise ValueError("eval_max_examples must be positive or None for all examples")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        if self.task_regression_threshold < 0:
            raise ValueError("task_regression_threshold must be non-negative")
        if self.lr_scheduler not in {"constant", "linear"}:
            raise ValueError("lr_scheduler must be 'constant' or 'linear'")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.ema_decay < 1 or self.ema_floor <= 0:
            raise ValueError("ema_decay must be in [0, 1) and ema_floor must be positive")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    evaluations: dict[str, EvaluationResult]
    macro_accuracy: float
    macro_gold_nll: float


@dataclass(frozen=True)
class CoreTrainingResult:
    artifacts: dict[str, PortalModel]
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_loss_epoch: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefitResult:
    artifact: PortalModel
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_loss_epoch: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _model_config(base: PortalBase, tasks: tuple[str, ...], recipe: PortalTrainingConfig) -> PortalConfig:
    return PortalConfig.from_model(
        base.model,
        tasks=list(tasks),
        base_model_name_or_path=base.model_id,
        base_model_revision=base.revision,
        modules=recipe.modules,
        layer_path=base.layer_path,
        module_paths=base.module_paths,
        rank=recipe.rank,
        alpha=recipe.alpha,
        d_z=recipe.d_z,
        d_layer=recipe.d_layer,
        hidden=recipe.hidden,
        d_core=recipe.d_core,
    )


def _state_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    scheduler: Literal["constant", "linear"],
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build the step scheduler shared by source training and refitting."""
    warmup_steps = int(warmup_ratio * total_steps)
    if scheduler == "constant" and warmup_steps == 0:
        return None

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if scheduler == "linear":
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _artifact(
    config: PortalConfig,
    task_latents: torch.Tensor,
    core_state: dict[str, torch.Tensor],
    alignment_state: dict[str, torch.Tensor],
) -> PortalModel:
    decoder = PortalDecoder(config)
    decoder.core.load_state_dict(core_state, strict=True)
    decoder.alignment.load_state_dict(alignment_state, strict=True)
    decoder.eval()
    return PortalModel(config, task_latents.detach().cpu().clone(), decoder)


def _macro(evaluations: dict[str, EvaluationResult]) -> tuple[float, float]:
    return (
        sum(result.macro_accuracy for result in evaluations.values()) / len(evaluations),
        sum(result.macro_gold_nll for result in evaluations.values()) / len(evaluations),
    )


def _better(accuracy: float, nll: float, best_accuracy: float, best_nll: float) -> bool:
    return accuracy > best_accuracy or (accuracy == best_accuracy and nll < best_nll)


def _update_ema(previous: float | None, value: float, decay: float) -> float:
    return value if previous is None else decay * previous + (1 - decay) * value


def _equalize_gradients(gradients: list[torch.Tensor]) -> torch.Tensor:
    if not gradients:
        raise ValueError("at least one gradient is required")
    norms = [gradient.norm() for gradient in gradients]
    average = torch.stack(norms).mean()
    return sum(gradient / (norm + 1e-8) * average for gradient, norm in zip(gradients, norms))


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    gradients = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else 0.0


def _task_regressions(
    initial: EpochMetrics,
    selected: EpochMetrics,
    threshold: float,
) -> dict[str, dict[str, float]]:
    regressions: dict[str, dict[str, float]] = {}
    for base_name, result in selected.evaluations.items():
        baseline = initial.evaluations[base_name]
        for task, task_result in result.tasks.items():
            change = task_result.accuracy - baseline.tasks[task].accuracy
            if change < -threshold:
                regressions.setdefault(base_name, {})[task] = change
    return regressions


def _slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower().replace(".", "-")


def _write_metrics(path: Path, metrics: EpochMetrics) -> None:
    value = {
        "epoch": metrics.epoch,
        "macro_accuracy": metrics.macro_accuracy,
        "macro_gold_nll": metrics.macro_gold_nll,
        "bases": {
            name: {key: value for key, value in result.to_dict().items() if key != "base_model"}
            for name, result in metrics.evaluations.items()
        },
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_checkpoint(
    checkpoint_dir: Path | None,
    epoch: int,
    metrics: EpochMetrics,
    artifacts: dict[str, PortalModel],
    *,
    nest_by_base: bool,
) -> None:
    if checkpoint_dir is None:
        return
    root = Path(checkpoint_dir) / f"epoch-{epoch}"
    if nest_by_base:
        for name, artifact in artifacts.items():
            artifact.save_pretrained(root / _slug(name))
    else:
        if len(artifacts) != 1:
            raise ValueError("a flat checkpoint must contain exactly one artifact")
        next(iter(artifacts.values())).save_pretrained(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_metrics(root / "metrics.json", metrics)


class _BatchCycle:
    """Deterministically shuffle and recycle per-task batches for arbitrary training units."""

    def __init__(
        self,
        batches: dict[int, list[list[ChoiceExample]]],
        seed_for_pass: Callable[[Any, int], int],
        *,
        draw_from_end: bool = False,
    ) -> None:
        self.batches = batches
        self.seed_for_pass = seed_for_pass
        self.draw_from_end = draw_from_end
        self.passes: dict[Any, int] = {}
        self.remaining: dict[Any, list[list[ChoiceExample]]] = {}

    def draw(self, unit: Any, task_index: int) -> list[ChoiceExample]:
        remaining = self.remaining.get(unit)
        if not remaining:
            pass_index = self.passes.get(unit, 0)
            remaining = list(self.batches[task_index])
            random.Random(self.seed_for_pass(unit, pass_index)).shuffle(remaining)
            self.passes[unit] = pass_index + 1
            if not self.draw_from_end:
                remaining.reverse()
            self.remaining[unit] = remaining
        return remaining.pop()


class _TrainingRunTracker:
    """Track epoch history, best-state selection, callbacks, and early stopping."""

    def __init__(
        self,
        initial_state: Any,
        *,
        patience: int | None,
        on_epoch: Callable[[EpochMetrics], None] | None,
    ) -> None:
        self.history: list[EpochMetrics] = []
        self.best_accuracy = float("-inf")
        self.best_nll = float("inf")
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.best_loss_epoch = 0
        self.best_state = initial_state
        self.epochs_without_improvement = 0
        self.patience = patience
        self.on_epoch = on_epoch

    def record_initial(self, metrics: EpochMetrics) -> None:
        self.best_accuracy = metrics.macro_accuracy
        self.best_nll = metrics.macro_gold_nll
        self.best_epoch = metrics.epoch
        self.best_loss = metrics.macro_gold_nll
        self.best_loss_epoch = metrics.epoch
        self._append(metrics)

    def record_epoch(self, metrics: EpochMetrics, state: Any) -> bool:
        self._append(metrics)
        if _better(metrics.macro_accuracy, metrics.macro_gold_nll, self.best_accuracy, self.best_nll):
            self.best_accuracy = metrics.macro_accuracy
            self.best_nll = metrics.macro_gold_nll
            self.best_epoch = metrics.epoch
            self.best_state = state
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if metrics.macro_gold_nll < self.best_loss:
            self.best_loss = metrics.macro_gold_nll
            self.best_loss_epoch = metrics.epoch
        return self.patience is not None and self.epochs_without_improvement >= self.patience

    def _append(self, metrics: EpochMetrics) -> None:
        self.history.append(metrics)
        if self.on_epoch is not None:
            self.on_epoch(metrics)

    @property
    def selected_metrics(self) -> EpochMetrics:
        return next(metrics for metrics in self.history if metrics.epoch == self.best_epoch)


class PortalCoreTrainer:
    """Jointly train shared task latents/core and one alignment per source base."""

    def __init__(
        self,
        bases: list[PortalBase],
        dataset: ChoiceDataset,
        *,
        tasks: tuple[str, ...] | None = None,
        config: PortalTrainingConfig | None = None,
    ) -> None:
        if not bases:
            raise ValueError("at least one source base is required")
        model_ids = [base.model_id for base in bases]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("source base model IDs must be unique")
        self.bases = bases
        self.dataset = dataset
        self.tasks = tasks or dataset.tasks
        self.recipe = config or PortalTrainingConfig()
        if set(self.tasks) - set(dataset.tasks):
            raise ValueError("requested tasks are absent from the dataset")
        devices = {base.device for base in bases}
        if len(devices) != 1:
            raise ValueError("all source bases must be on the same device")
        self.device = devices.pop()
        for base in bases:
            base.freeze(gradient_checkpointing=self.recipe.gradient_checkpointing)
        self.configs = {base.model_id: _model_config(base, self.tasks, self.recipe) for base in bases}
        signatures = {config.shared_signature() for config in self.configs.values()}
        if len(signatures) != 1:
            raise ValueError("source bases do not share a compatible canonical architecture")
        first_config = self.configs[bases[0].model_id]
        self.task_latents = nn.Parameter(torch.randn(len(self.tasks), self.recipe.d_z, device=self.device) * 0.02)
        self.core = PortalCore(first_config).to(self.device)
        self.alignments = {
            base.model_id: PortalAlignment(self.configs[base.model_id]).to(self.device) for base in bases
        }

    def _portal(self, base: PortalBase) -> PortalModel:
        config = self.configs[base.model_id]
        decoder = PortalDecoder(config, core=self.core, alignment=self.alignments[base.model_id])
        return PortalModel(config, self.task_latents, decoder)

    def _evaluate(self, epoch: int) -> EpochMetrics:
        evaluator = PortalEvaluator(max_prompt=self.recipe.max_prompt, batch_size=self.recipe.eval_batch_size)
        evaluations = {
            base.model_id: evaluator.evaluate(
                base,
                self.dataset,
                tasks=self.tasks,
                portal=self._portal(base),
                max_examples=self.recipe.eval_max_examples,
            )
            for base in self.bases
        }
        accuracy, nll = _macro(evaluations)
        return EpochMetrics(epoch, evaluations, accuracy, nll)

    def _snapshot(self) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        return (
            self.task_latents.detach().cpu().clone(),
            _state_cpu(self.core),
            {name: _state_cpu(alignment) for name, alignment in self.alignments.items()},
        )

    def _artifacts(self, snapshot) -> dict[str, PortalModel]:
        latents, core_state, alignment_states = snapshot
        return {
            name: _artifact(self.configs[name], latents, core_state, alignment_states[name]) for name in self.configs
        }

    def _checkpoint(self, epoch: int, metrics: EpochMetrics, snapshot) -> None:
        _write_checkpoint(
            self.recipe.checkpoint_dir,
            epoch,
            metrics,
            self._artifacts(snapshot),
            nest_by_base=True,
        )

    def train(self, *, on_epoch: Callable[[EpochMetrics], None] | None = None) -> CoreTrainingResult:
        task_pools = {task_index: list(self.dataset.rows("train", task)) for task_index, task in enumerate(self.tasks)}
        task_sizes = {
            task_index: min(self.recipe.source_max_examples or len(rows), len(rows))
            for task_index, rows in task_pools.items()
        }
        rounds = self.recipe.source_steps_per_epoch or max(
            (size + self.recipe.batch_size - 1) // self.recipe.batch_size for size in task_sizes.values()
        )
        decoder_parameters = list(self.core.parameters()) + [
            parameter for alignment in self.alignments.values() for parameter in alignment.parameters()
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": [self.task_latents], "lr": self.recipe.latent_learning_rate},
                {"params": decoder_parameters, "lr": self.recipe.learning_rate},
            ],
            weight_decay=self.recipe.weight_decay,
        )
        scheduler = _make_scheduler(
            optimizer,
            total_steps=self.recipe.epochs * rounds,
            scheduler=self.recipe.lr_scheduler,
            warmup_ratio=self.recipe.warmup_ratio,
        )
        injectors = {base.model_id: PortalInjector(base.model, self.configs[base.model_id]) for base in self.bases}
        ema = {(base.model_id, task_index): None for base in self.bases for task_index in range(len(self.tasks))}
        tracker = _TrainingRunTracker(
            self._snapshot(),
            patience=self.recipe.early_stopping_patience,
            on_epoch=on_epoch,
        )
        first_decoder_grad_norm: float | None = None
        try:
            initial = self._evaluate(0)
            tracker.record_initial(initial)
            self._checkpoint(0, initial, tracker.best_state)
            for epoch in range(1, self.recipe.epochs + 1):
                epoch_rows: dict[int, list[ChoiceExample]] = {}
                for task_index, pool in task_pools.items():
                    size = task_sizes[task_index]
                    if size == len(pool):
                        epoch_rows[task_index] = list(pool)
                    elif self.recipe.source_resample_each_epoch:
                        epoch_rows[task_index] = random.Random(
                            self.recipe.seed * 1_000_003 + epoch * 10_007 + task_index * 131 + size
                        ).sample(pool, size)
                    else:
                        epoch_rows[task_index] = list(pool[:size])
                batches = {
                    task_index: [
                        rows[offset : offset + self.recipe.batch_size]
                        for offset in range(0, len(rows), self.recipe.batch_size)
                    ]
                    for task_index, rows in epoch_rows.items()
                }
                self.core.train()
                for alignment in self.alignments.values():
                    alignment.train()
                for base in self.bases:
                    base.model.train(self.recipe.gradient_checkpointing)
                batch_cycle = _BatchCycle(
                    batches,
                    lambda unit, pass_index: (
                        self.recipe.seed * 1_000_003 + epoch * 10_007 + unit[0] * 503 + unit[2] * 37 + pass_index
                    ),
                )

                for _round in range(rounds):
                    optimizer.zero_grad(set_to_none=True)
                    base_z_grads: list[torch.Tensor] = []
                    for base_index, base in enumerate(self.bases):
                        for task_index, task in enumerate(self.tasks):
                            batch = batch_cycle.draw((base_index, base.model_id, task_index), task_index)
                            input_ids, attention_mask, labels = collate_gold_batch(
                                base.tokenizer,
                                batch,
                                max_prompt=self.recipe.max_prompt,
                                device=self.device,
                            )
                            factors = self.alignments[base.model_id](self.core, self.task_latents[task_index])
                            with injectors[base.model_id].activate(factors):
                                loss = base.model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                ).loss
                                unit = (base.model_id, task_index)
                                value = float(loss.detach())
                                previous = ema[unit]
                                ema[unit] = _update_ema(previous, value, self.recipe.ema_decay)
                                (loss / max(ema[unit], self.recipe.ema_floor)).backward()
                        if self.task_latents.grad is not None:
                            base_z_grads.append(self.task_latents.grad.detach().clone())
                            self.task_latents.grad.zero_()
                    if base_z_grads:
                        self.task_latents.grad = _equalize_gradients(base_z_grads)
                    decoder_grad_norm = _gradient_norm(decoder_parameters)
                    if not torch.isfinite(torch.tensor(decoder_grad_norm)) or decoder_grad_norm == 0:
                        raise RuntimeError("canonical core/alignment received no finite gradient")
                    if first_decoder_grad_norm is None:
                        first_decoder_grad_norm = decoder_grad_norm
                    if self.recipe.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_([self.task_latents, *decoder_parameters], self.recipe.grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                metrics = self._evaluate(epoch)
                snapshot = self._snapshot()
                self._checkpoint(epoch, metrics, snapshot)
                if tracker.record_epoch(metrics, snapshot):
                    break
        finally:
            for injector in injectors.values():
                injector.close()
        artifacts = self._artifacts(tracker.best_state)
        if self.recipe.checkpoint_dir is not None:
            for name, artifact in artifacts.items():
                artifact.save_pretrained(Path(self.recipe.checkpoint_dir) / "best" / _slug(name))
        initial_metrics = tracker.history[0]
        return CoreTrainingResult(
            artifacts=artifacts,
            history=tuple(tracker.history),
            best_epoch=tracker.best_epoch,
            best_loss_epoch=tracker.best_loss_epoch,
            diagnostics={
                "steps_per_epoch": rounds,
                "units_per_step": len(self.bases) * len(self.tasks),
                "source_examples_per_task": {
                    task: task_sizes[task_index] for task_index, task in enumerate(self.tasks)
                },
                "source_pool_examples_per_task": {
                    task: len(task_pools[task_index]) for task_index, task in enumerate(self.tasks)
                },
                "source_resample_each_epoch": self.recipe.source_resample_each_epoch,
                "lr_scheduler": self.recipe.lr_scheduler,
                "warmup_ratio": self.recipe.warmup_ratio,
                "planned_optimizer_steps": self.recipe.epochs * rounds,
                "epochs_completed": tracker.history[-1].epoch,
                "task_regressions": _task_regressions(
                    initial_metrics,
                    tracker.selected_metrics,
                    self.recipe.task_regression_threshold,
                ),
                "first_decoder_grad_norm": first_decoder_grad_norm,
            },
        )


class PortalAdapterRefitter:
    """Freeze a source artifact's latents/core and fit one fresh target alignment."""

    def __init__(
        self,
        source: PortalModel,
        target: PortalBase,
        dataset: ChoiceDataset,
        *,
        config: PortalTrainingConfig | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.dataset = dataset
        self.tasks = tuple(source.config.tasks)
        self.recipe = config or PortalTrainingConfig(
            modules=source.config.modules,
            rank=source.config.rank,
            alpha=source.config.alpha,
            d_z=source.config.d_z,
            d_layer=source.config.d_layer,
            hidden=source.config.hidden,
            d_core=source.config.d_core,
        )
        if self.recipe.modules != source.config.modules:
            raise ValueError("refit module targets must match the source artifact")
        target.freeze(gradient_checkpointing=self.recipe.gradient_checkpointing)
        self.config = _model_config(target, self.tasks, self.recipe)
        if self.config.shared_signature() != source.config.shared_signature():
            raise ValueError("target refit architecture must match the source canonical core")
        self.device = target.device
        self.task_latents = source.task_latents.detach().clone().to(self.device)
        self.core = copy.deepcopy(source.decoder.core).to(self.device).eval()
        self.alignment = PortalAlignment(self.config, zero_output=True).to(self.device)
        self.task_latents.requires_grad_(False)
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)

    def _portal(self) -> PortalModel:
        return PortalModel(
            self.config,
            self.task_latents,
            PortalDecoder(self.config, core=self.core, alignment=self.alignment),
        )

    def _evaluate(self, epoch: int) -> EpochMetrics:
        result = PortalEvaluator(max_prompt=self.recipe.max_prompt, batch_size=self.recipe.eval_batch_size).evaluate(
            self.target,
            self.dataset,
            tasks=self.tasks,
            portal=self._portal(),
            max_examples=self.recipe.eval_max_examples,
        )
        return EpochMetrics(epoch, {self.target.model_id: result}, result.macro_accuracy, result.macro_gold_nll)

    def _artifact(self, alignment_state: dict[str, torch.Tensor]) -> PortalModel:
        return _artifact(
            self.config,
            self.task_latents,
            _state_cpu(self.core),
            alignment_state,
        )

    def _checkpoint(self, epoch: int, metrics: EpochMetrics, alignment_state: dict[str, torch.Tensor]) -> None:
        _write_checkpoint(
            self.recipe.checkpoint_dir,
            epoch,
            metrics,
            {self.target.model_id: self._artifact(alignment_state)},
            nest_by_base=False,
        )

    def refit(self, *, on_epoch: Callable[[EpochMetrics], None] | None = None) -> RefitResult:
        pools: dict[int, list[ChoiceExample]] = {}
        for task_index, task in enumerate(self.tasks):
            full = list(self.dataset.rows("train", task))
            size = min(self.recipe.refit_max_examples, len(full))
            pools[task_index] = (
                full
                if size == len(full)
                else random.Random(self.recipe.seed * 100_003 + task_index * 131 + size).sample(full, size)
            )
        batches = {
            task_index: [
                rows[offset : offset + self.recipe.batch_size] for offset in range(0, len(rows), self.recipe.batch_size)
            ]
            for task_index, rows in pools.items()
        }
        rounds = max(len(task_batches) for task_batches in batches.values())
        optimizer = torch.optim.AdamW(
            self.alignment.parameters(),
            lr=self.recipe.learning_rate,
            weight_decay=self.recipe.weight_decay,
        )
        scheduler = _make_scheduler(
            optimizer,
            total_steps=self.recipe.epochs * rounds,
            scheduler=self.recipe.lr_scheduler,
            warmup_ratio=self.recipe.warmup_ratio,
        )
        injector = PortalInjector(self.target.model, self.config)
        ema: dict[int, float | None] = {task_index: None for task_index in range(len(self.tasks))}
        tracker = _TrainingRunTracker(
            _state_cpu(self.alignment),
            patience=self.recipe.early_stopping_patience,
            on_epoch=on_epoch,
        )
        first_alignment_grad_norm: float | None = None
        try:
            initial = self._evaluate(0)
            tracker.record_initial(initial)
            self._checkpoint(0, initial, tracker.best_state)
            batch_cycle = _BatchCycle(
                batches,
                lambda task_index, pass_index: self.recipe.seed * 9_973 + task_index * 131 + pass_index,
                draw_from_end=True,
            )

            for epoch in range(1, self.recipe.epochs + 1):
                self.alignment.train()
                self.target.model.train(self.recipe.gradient_checkpointing)
                for _round in range(rounds):
                    optimizer.zero_grad(set_to_none=True)
                    for task_index, _task in enumerate(self.tasks):
                        batch = batch_cycle.draw(task_index, task_index)
                        input_ids, attention_mask, labels = collate_gold_batch(
                            self.target.tokenizer,
                            batch,
                            max_prompt=self.recipe.max_prompt,
                            device=self.device,
                        )
                        factors = self.alignment(self.core, self.task_latents[task_index])
                        with injector.activate(factors):
                            loss = self.target.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            ).loss
                            value = float(loss.detach())
                            previous = ema[task_index]
                            ema[task_index] = _update_ema(previous, value, self.recipe.ema_decay)
                            (loss / max(ema[task_index], self.recipe.ema_floor)).backward()
                    alignment_grad_norm = _gradient_norm(list(self.alignment.parameters()))
                    if not torch.isfinite(torch.tensor(alignment_grad_norm)) or alignment_grad_norm == 0:
                        raise RuntimeError("target alignment received no finite gradient")
                    if first_alignment_grad_norm is None:
                        first_alignment_grad_norm = alignment_grad_norm
                    if self.recipe.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.alignment.parameters(), self.recipe.grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                metrics = self._evaluate(epoch)
                alignment_state = _state_cpu(self.alignment)
                self._checkpoint(epoch, metrics, alignment_state)
                if tracker.record_epoch(metrics, alignment_state):
                    break
        finally:
            injector.close()
        artifact = self._artifact(tracker.best_state)
        if self.recipe.checkpoint_dir is not None:
            artifact.save_pretrained(Path(self.recipe.checkpoint_dir) / "best")
        initial_metrics = tracker.history[0]
        return RefitResult(
            artifact=artifact,
            history=tuple(tracker.history),
            best_epoch=tracker.best_epoch,
            best_loss_epoch=tracker.best_loss_epoch,
            diagnostics={
                "steps_per_epoch": rounds,
                "tasks_per_step": len(self.tasks),
                "refit_examples_per_task": {task: len(pools[task_index]) for task_index, task in enumerate(self.tasks)},
                "lr_scheduler": self.recipe.lr_scheduler,
                "warmup_ratio": self.recipe.warmup_ratio,
                "planned_optimizer_steps": self.recipe.epochs * rounds,
                "epochs_completed": tracker.history[-1].epoch,
                "task_regressions": _task_regressions(
                    initial_metrics,
                    tracker.selected_metrics,
                    self.recipe.task_regression_threshold,
                ),
                "first_alignment_grad_norm": first_alignment_grad_norm,
            },
        )
