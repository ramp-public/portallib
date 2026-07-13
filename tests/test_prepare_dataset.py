from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "examples" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_dataset
SPEC.loader.exec_module(prepare_dataset)


def test_canonical_task_order_and_pinned_revisions() -> None:
    assert prepare_dataset.DEFAULT_TASKS == (
        "truthfulqa",
        "rte",
        "cb",
        "copa",
        "wic",
        "wsc",
        "boolq",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "openbookqa",
        "winogrande",
        "commonsense_qa",
        "sciq",
    )
    assert all(len(source.revision) == 40 for source in prepare_dataset.TASK_SOURCES.values())


def test_rte_matches_the_training_prompt() -> None:
    example = prepare_dataset.format_row(
        "rte",
        {"premise": "A cat sleeps.", "hypothesis": "An animal rests.", "label": 0},
    )
    assert example is not None
    assert example.prompt == "A cat sleeps.\nQuestion: An animal rests. True or False?\nAnswer:"
    assert example.choices == (" True", " False")
    assert example.gold_idx == 0


def test_hellaswag_matches_the_training_preprocessing() -> None:
    example = prepare_dataset.format_row(
        "hellaswag",
        {
            "activity_label": "Cooking [title]",
            "ctx_a": "A person",
            "ctx_b": "mixes [noise]  batter",
            "endings": ["bakes  it", "throws it away"],
            "label": "0",
        },
    )
    assert example is not None
    assert example.prompt == "Cooking. : A person Mixes  batter"
    assert example.choices == (" bakes it", " throws it away")


def test_unlabeled_rows_are_ignored() -> None:
    assert prepare_dataset.format_row("hellaswag", {"label": ""}) is None
    assert prepare_dataset.format_row("rte", {"premise": "p", "hypothesis": "h", "label": -1}) is None


def test_task_parser_rejects_unknown_or_duplicate_tasks() -> None:
    assert prepare_dataset.parse_tasks("") == prepare_dataset.DEFAULT_TASKS
    assert prepare_dataset.parse_tasks("rte,boolq") == ("rte", "boolq")
    with pytest.raises(Exception, match="unknown tasks"):
        prepare_dataset.parse_tasks("rte,missing")
    with pytest.raises(Exception, match="unique"):
        prepare_dataset.parse_tasks("rte,rte")
