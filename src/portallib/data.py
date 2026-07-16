"""Normalized multiple-choice task data for PorTAL training examples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChoiceExample:
    task: str
    prompt: str
    choices: tuple[str, ...]
    gold_idx: int

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if len(self.choices) < 2 or any(not choice for choice in self.choices):
            raise ValueError("choices must contain at least two non-empty strings")
        if not 0 <= self.gold_idx < len(self.choices):
            raise ValueError("gold_idx must index choices")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ChoiceExample":
        return cls(
            task=str(row["task"]),
            prompt=str(row["prompt"]),
            choices=tuple(str(choice) for choice in row["choices"]),
            gold_idx=int(row["gold_idx"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["choices"] = list(self.choices)
        return value


class ChoiceDataset:
    """Train/validation rows in the public PorTAL multiple-choice schema."""

    def __init__(self, train: list[ChoiceExample], validation: list[ChoiceExample]) -> None:
        if not train or not validation:
            raise ValueError("train and validation splits must both be non-empty")
        train_tasks = {row.task for row in train}
        validation_tasks = {row.task for row in validation}
        if train_tasks != validation_tasks:
            raise ValueError("train and validation splits must contain the same tasks")
        self.train = tuple(train)
        self.validation = tuple(validation)

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.task for row in self.train))

    def rows(self, split: str, task: str, *, limit: int | None = None) -> tuple[ChoiceExample, ...]:
        if split not in ("train", "validation"):
            raise KeyError("split must be 'train' or 'validation'")
        rows = tuple(row for row in getattr(self, split) if row.task == task)
        if not rows:
            raise KeyError(f"task {task!r} is absent from {split}")
        return rows[:limit]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return the canonical JSON-ready train/validation representation."""
        return {
            "train": [row.to_dict() for row in self.train],
            "validation": [row.to_dict() for row in self.validation],
        }

    def save_json(self, path: str | Path) -> None:
        """Write the canonical dataset representation to a local JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ChoiceDataset":
        """Load ``{'train': [...], 'validation': [...]}`` from a local JSON file."""
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"train", "validation"}:
            raise ValueError("dataset JSON must contain exactly 'train' and 'validation' arrays")
        return cls(
            [ChoiceExample.from_dict(row) for row in value["train"]],
            [ChoiceExample.from_dict(row) for row in value["validation"]],
        )

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
    ) -> "ChoiceDataset":
        """Load a normalized dataset repository with train/validation splits."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("ChoiceDataset.from_hub requires `pip install portallib[training]`") from exc
        dataset = load_dataset(repo_id, revision=revision, token=token)
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError("Hub dataset must contain train and validation splits")
        return cls(
            [ChoiceExample.from_dict(row) for row in dataset["train"]],
            [ChoiceExample.from_dict(row) for row in dataset["validation"]],
        )

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: str | bool | None = None,
    ) -> None:
        """Upload the normalized splits when explicitly requested by the caller."""
        try:
            from datasets import Dataset, DatasetDict
        except ImportError as exc:
            raise ImportError("ChoiceDataset.push_to_hub requires `pip install portallib[training]`") from exc
        dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in self.to_dict().items()})
        dataset.push_to_hub(repo_id, private=private, token=token)
