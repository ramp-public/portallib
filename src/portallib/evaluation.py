"""Exact-path adapter injection and multiple-choice evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .config import PortalConfig
from .data import ChoiceDataset, ChoiceExample
from .decoder import GeneratedLora
from .model import PortalModel


@dataclass(frozen=True)
class PortalBase:
    """A caller-owned frozen base model and its matching tokenizer."""

    model_id: str
    model: nn.Module
    tokenizer: Any
    revision: str | None = None
    layer_path: str = "model.layers"
    module_paths: dict[str, str] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def freeze(self, *, gradient_checkpointing: bool = False) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            # Generated factors share one decoder graph. Reentrant checkpointing
            # would traverse and free it independently for each transformer segment.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()


@dataclass(frozen=True)
class TaskEvaluation:
    task: str
    accuracy: float
    gold_nll: float
    examples: int


@dataclass(frozen=True)
class EvaluationResult:
    base_model: str
    tasks: dict[str, TaskEvaluation]
    macro_accuracy: float
    macro_gold_nll: float


class PortalInjector:
    """Persistent exact-path hooks with context-local generated factors."""

    def __init__(self, base_model: nn.Module, config: PortalConfig) -> None:
        self.base_model = base_model
        self.config = config
        self._active: ContextVar[GeneratedLora | None] = ContextVar(
            f"portallib_lora_{id(self)}", default=None
        )
        self._checkpoint_active: GeneratedLora | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer_index in range(config.n_layers):
            for short_name, module_path in config.module_paths.items():
                exact_path = f"{config.layer_path}.{layer_index}.{module_path}"
                try:
                    module = base_model.get_submodule(exact_path)
                except (AttributeError, KeyError) as exc:
                    self.close()
                    raise ValueError(f"base model has no exact projection path {exact_path!r}") from exc
                if not isinstance(module, nn.Linear):
                    self.close()
                    raise TypeError(f"configured projection {exact_path!r} is not torch.nn.Linear")
                key = (layer_index, short_name)
                in_features = module.in_features
                out_features = module.out_features

                def hook(
                    _module: nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                    *,
                    factor_key: tuple[int, str] = key,
                    expected_in: int = in_features,
                    expected_out: int = out_features,
                ) -> torch.Tensor:
                    factors = self._active.get() or self._checkpoint_active
                    if factors is None:
                        return output
                    if factor_key not in factors:
                        raise ValueError(f"generated factors are missing configured target {factor_key}")
                    a, b = factors[factor_key]
                    x = inputs[0]
                    expected = ((config.rank, expected_in), (expected_out, config.rank))
                    if (tuple(a.shape), tuple(b.shape)) != expected:
                        raise ValueError(
                            f"generated shape mismatch at {factor_key}: A {tuple(a.shape)}, B {tuple(b.shape)}, "
                            f"expected {expected}"
                        )
                    delta = F.linear(
                        F.linear(x, a.to(device=x.device, dtype=x.dtype)),
                        b.to(device=x.device, dtype=x.dtype),
                    )
                    return output + delta.to(dtype=output.dtype) * config.scaling

                self._handles.append(module.register_forward_hook(hook))

    @contextmanager
    def activate(self, factors: GeneratedLora) -> Iterator[None]:
        token = self._active.set(factors)
        previous_checkpoint = self._checkpoint_active
        self._checkpoint_active = factors
        try:
            yield
        finally:
            self._checkpoint_active = previous_checkpoint
            self._active.reset(token)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "PortalInjector":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def gold_batch(
    tokenizer: Any,
    rows: list[ChoiceExample],
    *,
    max_prompt: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    for row in rows:
        prompt = tokenizer(row.prompt, add_special_tokens=True).input_ids[-max_prompt:]
        answer = tokenizer(row.choices[row.gold_idx], add_special_tokens=False).input_ids
        sequences.append(prompt + answer)
        prompt_lengths.append(len(prompt))
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(rows), max_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence)
        attention_mask[index, :length] = 1
        labels[index, prompt_length:length] = torch.tensor(sequence[prompt_length:])
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


class PortalEvaluator:
    """Evaluate base or PorTAL-adapted models with normalized choice metrics."""

    def __init__(self, *, max_prompt: int = 768) -> None:
        self.max_prompt = max_prompt

    @torch.no_grad()
    def evaluate(
        self,
        base: PortalBase,
        dataset: ChoiceDataset,
        *,
        tasks: tuple[str, ...] | None = None,
        portal: PortalModel | None = None,
        max_examples: int | None = None,
    ) -> EvaluationResult:
        task_names = tasks or dataset.tasks
        if portal is not None and tuple(portal.config.tasks) != tuple(task_names):
            raise ValueError("portal task order must match the requested evaluation tasks")
        was_training = base.model.training
        model_config = getattr(base.model, "config", None)
        previous_use_cache = getattr(model_config, "use_cache", None)
        if model_config is not None and previous_use_cache is not None:
            model_config.use_cache = False
        base.model.eval()
        injector = PortalInjector(base.model, portal.config) if portal is not None else None
        results: dict[str, TaskEvaluation] = {}
        try:
            for task in task_names:
                rows = dataset.rows("validation", task, limit=max_examples)
                factors = portal.generate(task) if portal is not None else None
                correct = 0
                gold_nll = 0.0
                gold_tokens = 0
                activation = injector.activate(factors) if injector is not None else _null_context()
                with activation:
                    for row in rows:
                        prompt = base.tokenizer(row.prompt, add_special_tokens=True).input_ids[-self.max_prompt:]
                        scores: list[float] = []
                        for choice_index, choice in enumerate(row.choices):
                            answer = base.tokenizer(choice, add_special_tokens=False).input_ids
                            if not answer:
                                scores.append(float("-inf"))
                                continue
                            ids = torch.tensor([prompt + answer], device=base.device)
                            logits = base.model(input_ids=ids).logits[:, :-1].float()
                            log_probs = logits.log_softmax(dim=-1)
                            selected = log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -len(answer):]
                            scores.append(float(selected.sum()) / max(len(choice), 1))
                            if choice_index == row.gold_idx:
                                gold_nll += -float(selected.sum())
                                gold_tokens += len(answer)
                        correct += int(max(range(len(scores)), key=scores.__getitem__) == row.gold_idx)
                results[task] = TaskEvaluation(
                    task=task,
                    accuracy=correct / len(rows),
                    gold_nll=gold_nll / max(gold_tokens, 1),
                    examples=len(rows),
                )
        finally:
            if injector is not None:
                injector.close()
            base.model.train(was_training)
            if model_config is not None and previous_use_cache is not None:
                model_config.use_cache = previous_use_cache
        return EvaluationResult(
            base_model=base.model_id,
            tasks=results,
            macro_accuracy=sum(result.accuracy for result in results.values()) / len(results),
            macro_gold_nll=sum(result.gold_nll for result in results.values()) / len(results),
        )


@contextmanager
def _null_context() -> Iterator[None]:
    yield
