"""Prepare the normalized 14-task dataset used by the PorTAL training example.

The script downloads the original datasets from Hugging Face, applies the exact
multiple-choice prompts used by the portallib training recipes, and writes the
``ChoiceDataset`` JSON schema consumed by ``examples/train_example.py``.

Examples:

    python examples/prepare_dataset.py --output portal_tasks.json
    python examples/prepare_dataset.py --output portal_tasks.json --tasks rte,boolq
    python examples/prepare_dataset.py --output portal_tasks.json \
        --push-to-hub namespace/portal-tasks --private
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portallib import ChoiceDataset, ChoiceExample


@dataclass(frozen=True)
class TaskSource:
    repo_id: str
    revision: str
    config: str | None = None


# Revisions make the public recipe independent of later upstream dataset edits.
TASK_SOURCES: dict[str, TaskSource] = {
    "truthfulqa": TaskSource(
        "truthfulqa/truthful_qa",
        "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        "multiple_choice",
    ),
    "rte": TaskSource("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "rte"),
    "cb": TaskSource("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "cb"),
    "copa": TaskSource("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "copa"),
    "wic": TaskSource("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "wic"),
    "wsc": TaskSource("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "wsc.fixed"),
    "boolq": TaskSource("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "arc_easy": TaskSource("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Easy"),
    "arc_challenge": TaskSource(
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "ARC-Challenge",
    ),
    "hellaswag": TaskSource("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76"),
    "openbookqa": TaskSource("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", "main"),
    "winogrande": TaskSource(
        "allenai/winogrande",
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "winogrande_xl",
    ),
    "commonsense_qa": TaskSource(
        "tau/commonsense_qa",
        "94630fe30dad47192a8546eb75f094926d47e155",
    ),
    "sciq": TaskSource("allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815"),
}

DEFAULT_TASKS = tuple(TASK_SOURCES)
DATASET_CARD = Path(__file__).with_name("portal_tasks_dataset_card.md")


def _hellaswag_text(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def format_row(task: str, row: Mapping[str, Any]) -> ChoiceExample | None:
    """Normalize one upstream row, returning ``None`` for unlabeled rows."""
    try:
        if task == "truthfulqa":
            targets = row["mc1_targets"]
            labels = list(targets["labels"])
            if 1 not in labels:
                return None
            value = {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "choices": [" " + choice for choice in targets["choices"]],
                "gold_idx": labels.index(1),
            }
        elif task == "rte":
            value = {
                "prompt": f"{row['premise']}\nQuestion: {row['hypothesis']} True or False?\nAnswer:",
                "choices": [" True", " False"],
                "gold_idx": int(row["label"]),
            }
        elif task == "cb":
            value = {
                "prompt": f"{row['premise']}\nQuestion: {row['hypothesis']} True, False, or Neither?\nAnswer:",
                "choices": [" True", " False", " Neither"],
                "gold_idx": int(row["label"]),
            }
        elif task == "copa":
            connector = {"cause": "because", "effect": "therefore"}[str(row["question"])]
            choices = [str(row["choice1"]), str(row["choice2"])]
            value = {
                "prompt": str(row["premise"]).strip()[:-1] + f" {connector}",
                "choices": [" " + choice[0].lower() + choice[1:] for choice in choices],
                "gold_idx": int(row["label"]),
            }
        elif task == "wic":
            value = {
                "prompt": (
                    f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}\n"
                    f"Question: Is the word '{row['word']}' used the same way in both sentences?\nAnswer:"
                ),
                "choices": [" no", " yes"],
                "gold_idx": int(row["label"]),
            }
        elif task == "wsc":
            value = {
                "prompt": (
                    f"{row['text']}\nQuestion: In the passage above, does the pronoun "
                    f'"{row["span2_text"]}" refer to "{row["span1_text"]}"?\nAnswer:'
                ),
                "choices": [" no", " yes"],
                "gold_idx": int(row["label"]),
            }
        elif task == "boolq":
            value = {
                "prompt": f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:",
                "choices": [" no", " yes"],
                "gold_idx": int(row["answer"]),
            }
        elif task in ("arc_easy", "arc_challenge"):
            texts = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
            if row["answerKey"] not in labels:
                return None
            value = {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "choices": [" " + text for text in texts],
                "gold_idx": labels.index(row["answerKey"]),
            }
        elif task == "hellaswag":
            if row.get("label", "") == "":
                return None
            context = str(row["ctx_a"]) + " " + str(row["ctx_b"]).capitalize()
            value = {
                "prompt": _hellaswag_text(str(row["activity_label"]) + ": " + context),
                "choices": [" " + _hellaswag_text(ending) for ending in row["endings"]],
                "gold_idx": int(row["label"]),
            }
        elif task == "openbookqa":
            texts = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
            if row["answerKey"] not in labels:
                return None
            value = {
                "prompt": f"Question: {row['question_stem']}\nAnswer:",
                "choices": [" " + text for text in texts],
                "gold_idx": labels.index(row["answerKey"]),
            }
        elif task == "winogrande":
            if row.get("answer", "") not in ("1", "2") or "_" not in row["sentence"]:
                return None
            blank = str(row["sentence"]).index("_")
            prefix = str(row["sentence"])[:blank]
            suffix = str(row["sentence"])[blank + 1 :]
            value = {
                "prompt": prefix,
                "choices": [str(row["option1"]) + suffix, str(row["option2"]) + suffix],
                "gold_idx": int(row["answer"]) - 1,
            }
        elif task == "commonsense_qa":
            texts = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
            if row["answerKey"] not in labels:
                return None
            value = {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "choices": [" " + text for text in texts],
                "gold_idx": labels.index(row["answerKey"]),
            }
        elif task == "sciq":
            choices = [
                str(row["correct_answer"]),
                str(row["distractor1"]),
                str(row["distractor2"]),
                str(row["distractor3"]),
            ]
            if not all(choices):
                return None
            value = {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "choices": [" " + choice for choice in choices],
                "gold_idx": 0,
            }
        else:
            raise KeyError(f"unknown task: {task}")
    except (KeyError, TypeError, ValueError):
        return None
    if int(value["gold_idx"]) < 0:
        return None
    return ChoiceExample.from_dict({"task": task, **value})


def _normalize_rows(task: str, rows: Iterable[Mapping[str, Any]]) -> list[ChoiceExample]:
    normalized = []
    for row in rows:
        example = format_row(task, row)
        if example is not None:
            normalized.append(example)
    return normalized


def prepare_dataset(tasks: tuple[str, ...], *, cache_dir: str | None = None) -> ChoiceDataset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("dataset preparation requires `pip install portallib[training]`") from exc

    train: list[ChoiceExample] = []
    validation: list[ChoiceExample] = []
    loaded: dict[TaskSource, Any] = {}
    for task in tasks:
        source = TASK_SOURCES[task]
        if source not in loaded:
            loaded[source] = load_dataset(
                source.repo_id,
                source.config,
                revision=source.revision,
                cache_dir=cache_dir,
            )
        raw = loaded[source]
        if task == "truthfulqa":
            rows = list(raw[next(iter(raw))])
            cut = len(rows) - max(1, len(rows) // 4)
            source_rows, validation_rows = rows[:cut], rows[cut:]
        else:
            source_rows, validation_rows = raw["train"], raw["validation"]
        task_train = _normalize_rows(task, source_rows)
        task_validation = _normalize_rows(task, validation_rows)
        if not task_train or not task_validation:
            raise ValueError(f"task {task!r} produced an empty split")
        train.extend(task_train)
        validation.extend(task_validation)
        print(
            json.dumps(
                {"task": task, "train": len(task_train), "validation": len(task_validation)}
            ),
            flush=True,
        )
    return ChoiceDataset(train, validation)


def save_json(dataset: ChoiceDataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "train": [row.to_dict() for row in dataset.train],
                "validation": [row.to_dict() for row in dataset.validation],
            }
        ),
        encoding="utf-8",
    )


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(task.strip() for task in value.split(",") if task.strip())
    if not tasks:
        return DEFAULT_TASKS
    unknown = set(tasks) - set(TASK_SOURCES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown tasks: {', '.join(sorted(unknown))}")
    if len(set(tasks)) != len(tasks):
        raise argparse.ArgumentTypeError("tasks must be unique")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=parse_tasks,
        default=DEFAULT_TASKS,
        help="comma-separated task subset; defaults to the canonical 14 tasks",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--push-to-hub", default="", help="optional destination dataset repository")
    parser.add_argument("--private", action="store_true", help="make an explicitly uploaded dataset private")
    args = parser.parse_args()

    dataset = prepare_dataset(args.tasks, cache_dir=args.cache_dir)
    save_json(dataset, args.output)
    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub, private=args.private)
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=DATASET_CARD,
            path_in_repo="README.md",
            repo_id=args.push_to_hub,
            repo_type="dataset",
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tasks": list(dataset.tasks),
                "train": len(dataset.train),
                "validation": len(dataset.validation),
                "hub_repo": args.push_to_hub or None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
