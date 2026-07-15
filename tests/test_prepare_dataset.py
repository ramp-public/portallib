from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dataset.py"
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


def test_dataset_and_provenance_card_are_uploaded_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = prepare_dataset.ChoiceDataset(
        [prepare_dataset.ChoiceExample("rte", "train", (" True", " False"), 0)],
        [prepare_dataset.ChoiceExample("rte", "validation", (" True", " False"), 1)],
    )
    uploads: list[dict[str, object]] = []
    repositories: list[dict[str, object]] = []

    class FakeDataset:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        @classmethod
        def from_list(cls, rows: list[dict[str, object]]) -> "FakeDataset":
            return cls(rows)

        def to_parquet(self, path: Path) -> None:
            path.write_text(str(self.rows), encoding="utf-8")

    class FakeApi:
        def create_repo(self, **kwargs: object) -> None:
            repositories.append(kwargs)

        def upload_folder(self, **kwargs: object) -> None:
            root = Path(str(kwargs["folder_path"]))
            uploads.append(
                {
                    **kwargs,
                    "card": (root / "README.md").read_bytes(),
                    "data_files": sorted(path.name for path in (root / "data").iterdir()),
                }
            )

    datasets_module = ModuleType("datasets")
    datasets_module.Dataset = FakeDataset
    hub_module = ModuleType("huggingface_hub")
    hub_module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    prepare_dataset._push_dataset_with_card(dataset, "owner/tasks", private=True)

    assert repositories == [
        {"repo_id": "owner/tasks", "repo_type": "dataset", "private": True, "exist_ok": True}
    ]
    assert len(uploads) == 1
    assert uploads[0]["repo_id"] == "owner/tasks"
    assert uploads[0]["card"] == prepare_dataset.DATASET_CARD.read_bytes()
    assert uploads[0]["data_files"] == [
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
    ]
