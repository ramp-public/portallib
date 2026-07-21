"""Exact-path adapter injection and multiple-choice evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()


@dataclass(frozen=True)
class TaskEvaluation:
    task: str
    accuracy: float
    gold_nll: float
    examples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a stable JSON-ready representation."""
        return {
            "accuracy": self.accuracy,
            "gold_nll": self.gold_nll,
            "examples": self.examples,
        }


@dataclass(frozen=True)
class EvaluationResult:
    base_model: str
    tasks: dict[str, TaskEvaluation]
    macro_accuracy: float
    macro_gold_nll: float

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation."""
        return {
            "base_model": self.base_model,
            "macro_accuracy": self.macro_accuracy,
            "macro_gold_nll": self.macro_gold_nll,
            "tasks": {task: result.to_dict() for task, result in self.tasks.items()},
        }


@dataclass
class _ActiveFactors:
    """Shape-validated factors plus placements cached for one no-grad activation."""

    factors: GeneratedLora
    eval_cache: dict[
        tuple[tuple[int, str], torch.device, torch.dtype],
        tuple[torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict)


class PortalInjector:
    """Persistent exact-path hooks with context-local generated factors."""

    def __init__(self, base_model: nn.Module, config: PortalConfig) -> None:
        self.base_model = base_model
        self.config = config
        self._active: ContextVar[_ActiveFactors | None] = ContextVar(f"portallib_lora_{id(self)}", default=None)
        self._checkpoint_active: _ActiveFactors | None = None
        self._factor_specs: dict[tuple[int, str], tuple[int, int]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer_index, short_name, exact_path in config.targets():
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
            self._factor_specs[key] = (in_features, out_features)

            def hook(
                _module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                factor_key: tuple[int, str] = key,
            ) -> torch.Tensor:
                state = self._active.get() or self._checkpoint_active
                if state is None:
                    return output
                if factor_key not in state.factors:
                    raise ValueError(f"generated factors are missing configured target {factor_key}")
                a, b = self._place_factors(state, factor_key, output)
                x = inputs[0].to(device=output.device, dtype=output.dtype)
                delta = F.linear(F.linear(x, a), b)
                return output + delta.to(device=output.device, dtype=output.dtype) * config.scaling

            self._handles.append(module.register_forward_hook(hook))

    @staticmethod
    def _place_factors(
        state: _ActiveFactors,
        key: tuple[int, str],
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output.device.type == "meta":
            raise RuntimeError(
                f"cannot inject generated factors at {key}: projection output is on the meta device"
            )
        cache_key = (key, output.device, output.dtype)
        if not torch.is_grad_enabled() and cache_key in state.eval_cache:
            return state.eval_cache[cache_key]
        a, b = state.factors[key]
        placed = (
            a.to(device=output.device, dtype=output.dtype),
            b.to(device=output.device, dtype=output.dtype),
        )
        if not torch.is_grad_enabled():
            state.eval_cache[cache_key] = placed
        return placed

    def _prepare_factors(self, factors: GeneratedLora) -> _ActiveFactors:
        validated: GeneratedLora = {}
        for key, (expected_in, expected_out) in self._factor_specs.items():
            if key not in factors:
                raise ValueError(f"generated factors are missing configured target {key}")
            a, b = factors[key]
            expected = ((self.config.rank, expected_in), (expected_out, self.config.rank))
            if (tuple(a.shape), tuple(b.shape)) != expected:
                raise ValueError(
                    f"generated shape mismatch at {key}: A {tuple(a.shape)}, B {tuple(b.shape)}, expected {expected}"
                )
            if a.device.type == "meta" or b.device.type == "meta":
                raise ValueError(f"generated factors at {key} must contain data and cannot be on the meta device")
            validated[key] = (a, b)
        return _ActiveFactors(validated)

    @contextmanager
    def activate(self, factors: GeneratedLora) -> Iterator[None]:
        state = self._prepare_factors(factors)
        token = self._active.set(state)
        previous_checkpoint = self._checkpoint_active
        self._checkpoint_active = state
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


def collate_gold_batch(
    tokenizer: Any,
    rows: list[ChoiceExample],
    *,
    max_prompt: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize gold continuations and mask prompt tokens for causal-LM loss."""
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    for row in rows:
        prompt, boundary_whitespace = _encode_prompt(
            tokenizer,
            row.prompt,
            max_prompt=max_prompt,
        )
        answer, _continuation = _encode_continuation(
            tokenizer,
            boundary_whitespace,
            row.choices[row.gold_idx],
        )
        sequences.append(prompt + answer)
        prompt_lengths.append(len(prompt))
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(rows), max_length), _pad_token_id(tokenizer), dtype=torch.long)
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

    def __init__(self, *, max_prompt: int = 768, batch_size: int = 8) -> None:
        if max_prompt <= 0:
            raise ValueError("max_prompt must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.max_prompt = max_prompt
        self.batch_size = batch_size

    def _score_rows(
        self,
        base: PortalBase,
        rows: list[ChoiceExample],
    ) -> tuple[list[list[float]], float, int]:
        scores = [[float("-inf")] * len(row.choices) for row in rows]
        gold_nll = 0.0
        gold_tokens = 0
        pending: list[tuple[int, int, list[int], int, int]] = []

        def flush() -> None:
            nonlocal gold_nll, gold_tokens
            if not pending:
                return
            max_length = max(len(sequence) for _row, _choice, sequence, _answer_length, _chars in pending)
            input_ids = torch.full(
                (len(pending), max_length),
                _pad_token_id(base.tokenizer),
                dtype=torch.long,
                device=base.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for batch_index, (_row, _choice, sequence, _answer_length, _chars) in enumerate(pending):
                length = len(sequence)
                input_ids[batch_index, :length] = torch.tensor(sequence, device=base.device)
                attention_mask[batch_index, :length] = 1

            logits = base.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
            log_probs = logits.log_softmax(dim=-1)
            for batch_index, (row_index, choice_index, sequence, answer_length, chars) in enumerate(pending):
                answer_start = len(sequence) - answer_length
                token_positions = slice(answer_start - 1, len(sequence) - 1)
                targets = input_ids[batch_index, answer_start : len(sequence)]
                selected = log_probs[batch_index, token_positions].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                total_log_prob = float(selected.sum())
                scores[row_index][choice_index] = total_log_prob / max(chars, 1)
                if choice_index == rows[row_index].gold_idx:
                    gold_nll -= total_log_prob
                    gold_tokens += answer_length
            pending.clear()

        for row_index, row in enumerate(rows):
            prompt, boundary_whitespace = _encode_prompt(
                base.tokenizer,
                row.prompt,
                max_prompt=self.max_prompt,
            )
            for choice_index, choice in enumerate(row.choices):
                answer, continuation = _encode_continuation(
                    base.tokenizer,
                    boundary_whitespace,
                    choice,
                )
                if not answer:
                    continue
                pending.append((row_index, choice_index, prompt + answer, len(answer), len(continuation)))
                if len(pending) == self.batch_size:
                    flush()
        flush()
        return scores, gold_nll, gold_tokens

    def compare(
        self,
        base: PortalBase,
        dataset: ChoiceDataset,
        portal: PortalModel,
        *,
        tasks: tuple[str, ...] | None = None,
        max_examples: int | None = None,
    ) -> tuple[EvaluationResult, EvaluationResult]:
        """Evaluate the raw base and its PorTAL artifact over the same rows."""
        portal.validate_base_model(base.model_id)
        task_names = tasks or tuple(portal.config.tasks)
        return (
            self.evaluate(base, dataset, tasks=task_names, max_examples=max_examples),
            self.evaluate(base, dataset, tasks=task_names, portal=portal, max_examples=max_examples),
        )

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
        if portal is not None:
            portal.validate_base_model(base.model_id)
            missing_tasks = tuple(task for task in task_names if task not in portal.config.tasks)
            if missing_tasks:
                raise ValueError(f"requested tasks are absent from the PorTAL artifact: {missing_tasks}")
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
                activation = injector.activate(factors) if injector is not None else _null_context()
                with activation:
                    scores, gold_nll, gold_tokens = self._score_rows(base, rows)
                correct = sum(
                    max(range(len(row_scores)), key=row_scores.__getitem__) == row.gold_idx
                    for row, row_scores in zip(rows, scores)
                )
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


def _encode_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_prompt: int,
) -> tuple[list[int], str]:
    """Encode a prompt once and return whitespace that belongs with its continuations."""
    normalized_prompt = prompt.rstrip()
    boundary_whitespace = prompt[len(normalized_prompt) :]
    prompt_ids = tokenizer(normalized_prompt, add_special_tokens=True).input_ids[-max_prompt:]
    if not prompt_ids:
        start_token_id = getattr(tokenizer, "bos_token_id", None)
        if start_token_id is None:
            start_token_id = getattr(tokenizer, "eos_token_id", None)
        if start_token_id is None:
            raise ValueError("empty prompts require a tokenizer bos_token_id or eos_token_id")
        prompt_ids = [int(start_token_id)]
    return prompt_ids, boundary_whitespace


def _encode_continuation(
    tokenizer: Any,
    boundary_whitespace: str,
    continuation: str,
) -> tuple[list[int], str]:
    """Encode a continuation with trailing prompt whitespace moved to its natural boundary."""
    normalized_continuation = boundary_whitespace + continuation
    continuation_ids = tokenizer(normalized_continuation, add_special_tokens=False).input_ids
    return continuation_ids, normalized_continuation


def _pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer must define a pad_token_id or eos_token_id")
    return int(pad_token_id)
