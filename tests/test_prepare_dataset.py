from __future__ import annotations

import importlib.util
import json
import random
import sys
from copy import deepcopy
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
    assert example.prompt == "Cooking. : A person Mixes batter"
    assert example.choices == (" bakes it", " throws it away")


def test_hellaswag_removes_markup_without_leaving_double_boundary_spaces() -> None:
    example = prepare_dataset.format_row(
        "hellaswag",
        {
            "activity_label": "Hunting",
            "ctx_a": "A hunter",
            "ctx_b": "waits",
            "endings": ["[substeps] Buy a vest.", "  Walk home.  "],
            "label": "0",
        },
    )

    assert example is not None
    assert example.choices == (" Buy a vest.", " Walk home.")


def test_winogrande_moves_the_blank_boundary_to_each_choice() -> None:
    example = prepare_dataset.format_row(
        "winogrande",
        {
            "sentence": "The bowl was already empty because   _ ate first.",
            "option1": "Ian",
            "option2": "Dennis",
            "answer": "1",
        },
    )

    assert example is not None
    assert example.prompt == "The bowl was already empty because"
    assert example.choices == (" Ian ate first.", " Dennis ate first.")


def test_winogrande_supports_a_blank_at_the_start_of_the_sentence() -> None:
    example = prepare_dataset.format_row(
        "winogrande",
        {
            "sentence": " _ likes chocolates.",
            "option1": "Patricia",
            "option2": "Carrie",
            "answer": "1",
        },
    )

    assert example is not None
    assert example.prompt == ""
    assert example.choices == (" Patricia likes chocolates.", " Carrie likes chocolates.")


def test_dynamic_choices_have_one_leading_space_and_no_trailing_whitespace() -> None:
    example = prepare_dataset.format_row(
        "commonsense_qa",
        {
            "question": "What form is it?",
            "choices": {"text": [" ice cream ", " cubes"], "label": ["A", "B"]},
            "answerKey": "B",
        },
    )

    assert example is not None
    assert example.choices == (" ice cream", " cubes")


def test_unlabeled_rows_are_ignored() -> None:
    assert prepare_dataset.format_row("hellaswag", {"label": ""}) is None
    assert prepare_dataset.format_row("rte", {"premise": "p", "hypothesis": "h", "label": -1}) is None


def test_seeded_sciq_choices_vary_gold_position_without_changing_the_answer() -> None:
    examples = [
        prepare_dataset.format_row(
            "sciq",
            {
                "question": f"Which element belongs to sample {index}?",
                "correct_answer": " helium ",
                "distractor1": "carbon",
                "distractor2": "oxygen",
                "distractor3": "nitrogen",
            },
            choice_seed=0,
        )
        for index in range(64)
    ]

    assert all(example is not None for example in examples)
    assert {example.gold_idx for example in examples} == {0, 1, 2, 3}
    for index, example in enumerate(examples):
        assert example.prompt == f"Question: Which element belongs to sample {index}?\nAnswer:"
        assert sorted(example.choices) == [" carbon", " helium", " nitrogen", " oxygen"]
        assert example.choices[example.gold_idx] == " helium"


def test_prepared_choices_are_stable_across_task_order_and_json_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sciq_rows = [
        {
            "question": f"Science question {index}",
            "correct_answer": "correct",
            "distractor1": "first distractor",
            "distractor2": "second distractor",
            "distractor3": "third distractor",
        }
        for index in range(32)
    ]
    truthfulqa_rows = [
        {
            "question": f"Truthful question {index}",
            "mc1_targets": {"choices": ["first distractor", "correct", "second distractor"], "labels": [0, 1, 0]},
        }
        for index in range(32)
    ]
    raw = {
        "allenai/sciq": {"train": sciq_rows[:24], "validation": sciq_rows[24:]},
        "truthfulqa/truthful_qa": {"validation": truthfulqa_rows},
    }
    original = deepcopy(raw)
    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda repo_id, *_args, **_kwargs: raw[repo_id]
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    dataset = prepare_dataset.prepare_dataset(("sciq", "truthfulqa"), choice_seed=0)
    reordered = prepare_dataset.prepare_dataset(("truthfulqa", "sciq"), choice_seed=0)
    subset = prepare_dataset.prepare_dataset(("sciq",), choice_seed=0)
    historical = prepare_dataset.prepare_dataset(("sciq", "truthfulqa"))
    dataset.save_json(tmp_path / "tasks.json")
    reloaded = prepare_dataset.ChoiceDataset.from_json(tmp_path / "tasks.json")

    assert reloaded.to_dict() == dataset.to_dict()
    for split in ("train", "validation"):
        assert dataset.rows(split, "sciq") == subset.rows(split, "sciq")
        assert {row.gold_idx for row in historical.rows(split, "sciq")} == {0}
        assert {row.gold_idx for row in historical.rows(split, "truthfulqa")} == {1}
        for task in ("sciq", "truthfulqa"):
            rows = dataset.rows(split, task)
            assert len(rows) == (24 if split == "train" else 8)
            assert rows == reordered.rows(split, task)
            assert len({example.gold_idx for example in rows}) > 1
            assert all(example.choices[example.gold_idx] == " correct" for example in rows)
    assert [example.prompt for example in dataset.rows("validation", "truthfulqa")] == [
        f"Question: Truthful question {index}\nAnswer:" for index in range(24, 32)
    ]
    assert raw == original


def test_truthfulqa_shuffle_preserves_duplicate_choice_indices() -> None:
    row = {
        "question": "Which occurrence is labelled?",
        "mc1_targets": {"choices": ["repeated", "distractor", "repeated"], "labels": [1, 0, 0]},
    }
    other_label = deepcopy(row)
    other_label["mc1_targets"]["labels"] = [0, 0, 1]

    first = prepare_dataset.format_row("truthfulqa", row, choice_seed=0)
    last = prepare_dataset.format_row("truthfulqa", other_label, choice_seed=0)

    assert first is not None and last is not None
    assert first.choices == last.choices
    assert sorted(first.choices) == [" distractor", " repeated", " repeated"]
    assert first.gold_idx != last.gold_idx
    assert first.choices[first.gold_idx] == last.choices[last.gold_idx] == " repeated"


def test_choice_seed_keeps_legacy_output_and_unaffected_tasks_unchanged() -> None:
    sciq_row = {
        "question": "Which element?",
        "correct_answer": "helium",
        "distractor1": "carbon",
        "distractor2": "oxygen",
        "distractor3": "nitrogen",
    }
    legacy = prepare_dataset.format_row("sciq", sciq_row)
    assert legacy is not None
    assert legacy.choices == (" helium", " carbon", " oxygen", " nitrogen")
    assert legacy.gold_idx == 0
    assert prepare_dataset.format_row("sciq", sciq_row, choice_seed=None) == legacy

    rte_row = {"premise": "A cat sleeps.", "hypothesis": "An animal rests.", "label": 0}
    rte = prepare_dataset.format_row("rte", rte_row, choice_seed=0)
    assert rte is not None
    assert rte.choices == (" True", " False")
    assert rte.gold_idx == 0
    assert rte == prepare_dataset.format_row("rte", rte_row)


def test_choice_seed_is_row_local_and_does_not_consume_global_randomness() -> None:
    rows = [
        {
            "question": f"Question {index}",
            "mc1_targets": {"choices": ["correct", "first distractor", "second distractor"], "labels": [1, 0, 0]},
        }
        for index in range(32)
    ]
    state = random.getstate()
    first = [prepare_dataset.format_row("truthfulqa", row, choice_seed=0) for row in rows]
    another_seed = [prepare_dataset.format_row("truthfulqa", row, choice_seed=1) for row in rows]
    reversed_rows = [prepare_dataset.format_row("truthfulqa", row, choice_seed=0) for row in reversed(rows)]

    assert first == list(reversed(reversed_rows))
    assert first != another_seed
    assert {example.gold_idx for example in first} == {0, 1, 2}
    assert all(example.choices[example.gold_idx] == " correct" for example in first)
    assert random.getstate() == state


@pytest.mark.parametrize("seed", [0, -7])
def test_preparation_cli_records_choice_seed_in_json_and_uploaded_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], seed: int
) -> None:
    import datasets
    import huggingface_hub
    import pyarrow.parquet as parquet

    rows = [
        {
            "question": f"Science question {index}",
            "correct_answer": "correct",
            "distractor1": "first distractor",
            "distractor2": "second distractor",
            "distractor3": "third distractor",
        }
        for index in range(32)
    ]
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *_args, **_kwargs: {"train": rows[:24], "validation": rows[24:]}
    )
    uploaded: dict[str, object] = {}

    class FakeApi:
        def create_repo(self, **_kwargs: object) -> None:
            pass

        def upload_folder(self, **kwargs: object) -> None:
            root = Path(str(kwargs["folder_path"]))
            uploaded["card"] = (root / "README.md").read_text(encoding="utf-8")
            uploaded["train"] = parquet.read_table(root / "data" / "train-00000-of-00001.parquet").to_pylist()
            uploaded["validation"] = parquet.read_table(root / "data" / "validation-00000-of-00001.parquet").to_pylist()

    output = tmp_path / "tasks.json"
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--output",
            str(output),
            "--tasks",
            "sciq",
            "--choice-seed",
            str(seed),
            "--push-to-hub",
            "owner/tasks",
            "--private",
        ],
    )

    prepare_dataset.main()

    local = json.loads(output.read_text(encoding="utf-8"))
    assert local == {"train": uploaded["train"], "validation": uploaded["validation"]}
    for split in ("train", "validation"):
        assert len({row["gold_idx"] for row in local[split]}) > 1
        assert all(row["choices"][row["gold_idx"]] == " correct" for row in local[split])
    assert f"--choice-seed {seed}" in uploaded["card"]
    assert "historical" in uploaded["card"]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["choice_seed"] == seed


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
        def from_list(cls, rows: list[dict[str, object]]) -> FakeDataset:
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

    assert repositories == [{"repo_id": "owner/tasks", "repo_type": "dataset", "private": True, "exist_ok": True}]
    assert len(uploads) == 1
    assert uploads[0]["repo_id"] == "owner/tasks"
    assert uploads[0]["card"] == prepare_dataset.DATASET_CARD.read_bytes()
    assert uploads[0]["data_files"] == [
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
    ]
