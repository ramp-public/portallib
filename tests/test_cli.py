from __future__ import annotations

import importlib.metadata
import io
import json
from pathlib import Path

import pytest

from portallib import cli


ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "examples" / "configs"


def test_console_entrypoint_targets_cli_main() -> None:
    entrypoint = next(
        entrypoint
        for entrypoint in importlib.metadata.entry_points(group="console_scripts")
        if entrypoint.name == "portallib"
    )
    assert entrypoint.value == "portallib.cli:main"


@pytest.mark.parametrize(
    ("name", "recipe_type", "kind"),
    [
        ("train.toml", cli.TrainRecipe, "train"),
        ("refit.toml", cli.RefitRecipe, "refit"),
        ("refit_gemma4_e2b.toml", cli.RefitRecipe, "refit"),
        ("refit_gemma4_e2b_smoke.toml", cli.RefitRecipe, "refit"),
        ("evaluate.toml", cli.EvaluateRecipe, "evaluate"),
    ],
)
def test_checked_in_cli_recipes_validate(name: str, recipe_type: type, kind: str) -> None:
    recipe = cli.load_recipe(CONFIGS / name)

    assert isinstance(recipe, recipe_type)
    assert recipe.kind == kind
    assert recipe.dataset.revision == "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"


def test_validate_command_does_not_run_recipe(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_run_train", lambda _recipe: pytest.fail("validate must not execute"))

    assert cli.main(["validate", "--config", str(CONFIGS / "train.toml")]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "event": "validated",
        "kind": "train",
        "schema_version": 1,
    }


def test_gemma4_recipe_selects_multimodal_heterogeneous_refit() -> None:
    recipe = cli.load_recipe(CONFIGS / "refit_gemma4_e2b.toml")

    assert isinstance(recipe, cli.RefitRecipe)
    assert recipe.base.model_id == "google/gemma-4-E2B"
    assert recipe.base.loader == "multimodal_lm"
    assert recipe.base.allow_heterogeneous_targets is True
    assert recipe.base.layer_path == "model.language_model.layers"
    assert recipe.training.refit_max_examples == 500
    assert recipe.training.refit_nested_prefix is True


def test_validate_reads_recipe_from_stdin_once(monkeypatch, capsys) -> None:
    class CountingInput(io.StringIO):
        reads = 0

        def read(self, *args, **kwargs):
            self.reads += 1
            return super().read(*args, **kwargs)

    stream = CountingInput(
        """
schema_version = 1
kind = "evaluate"
artifact = "example/artifact"

[dataset]
repo_id = "example/tasks"

[base]
model_id = "example/base"
""".strip()
    )
    monkeypatch.setattr("sys.stdin", stream)

    assert cli.main(["validate", "--config", "-"]) == 0
    assert stream.reads == 1
    assert json.loads(capsys.readouterr().out) == {
        "event": "validated",
        "kind": "evaluate",
        "schema_version": 1,
    }


def test_stdin_recipe_dispatches_and_resolves_paths_from_cwd(monkeypatch, tmp_path: Path, capsys) -> None:
    received: list[cli.EvaluateRecipe] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_run_evaluate", received.append)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            """
schema_version = 1
kind = "evaluate"
artifact = "./artifacts/portal"
result_path = "./results/evaluate.json"

[dataset]
path = "./data/tasks.json"

[base]
model_id = "./models/base"
""".strip()
        ),
    )

    assert cli.main(["evaluate", "--config", "-"]) == 0
    assert len(received) == 1
    recipe = received[0]
    assert recipe.artifact == str(tmp_path / "artifacts" / "portal")
    assert recipe.result_path == tmp_path / "results" / "evaluate.json"
    assert recipe.dataset.source == str(tmp_path / "data" / "tasks.json")
    assert recipe.base.model_id == str(tmp_path / "models" / "base")
    assert capsys.readouterr().err == ""


def test_stdin_train_recipe_resolves_output_from_cwd(monkeypatch, tmp_path: Path) -> None:
    received: list[cli.TrainRecipe] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_run_train", received.append)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            """
schema_version = 1
kind = "train"
output_dir = "./artifacts/source"

[dataset]
repo_id = "example/tasks"

[[bases]]
model_id = "example/base"
""".strip()
        ),
    )

    assert cli.main(["train", "--config", "-"]) == 0
    assert received[0].output_dir == tmp_path / "artifacts" / "source"


def test_invalid_stdin_toml_is_a_structured_config_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("kind = ["))

    assert cli.main(["validate", "--config", "-"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["event"] == "error"
    assert error["stage"] == "config"
    assert "invalid TOML in <stdin>" in error["message"]


def test_dot_slash_dash_remains_a_file_path(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "-").write_text(
        """
schema_version = 1
kind = "evaluate"
artifact = "example/artifact"

[dataset]
repo_id = "example/tasks"

[base]
model_id = "example/base"
""".strip(),
        encoding="utf-8",
    )

    assert cli.main(["validate", "--config", "./-"]) == 0
    assert json.loads(capsys.readouterr().out)["event"] == "validated"


def test_command_rejects_a_different_recipe_kind(capsys) -> None:
    assert cli.main(["train", "--config", str(CONFIGS / "evaluate.toml")]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["event"] == "error"
    assert error["stage"] == "config"
    assert "cannot run" in error["message"]


def test_recipe_rejects_unknown_keys_and_credentials(tmp_path: Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        """
schema_version = 1
kind = "evaluate"
artifact = "example/artifact"
token = "must-not-be-accepted"

[dataset]
repo_id = "example/tasks"

[base]
model_id = "example/base"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(cli.RecipeError, match="token"):
        cli.load_recipe(path)


def test_recipe_rejects_unknown_nested_keys(tmp_path: Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        """
schema_version = 1
kind = "evaluate"
artifact = "example/artifact"

[dataset]
repo_id = "example/tasks"

[base]
model_id = "example/base"
token = "must-not-be-accepted"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(cli.RecipeError, match="base.token"):
        cli.load_recipe(path)


def test_recipe_does_not_coerce_toml_field_types(tmp_path: Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        """
schema_version = 1
kind = "evaluate"
artifact = "example/artifact"
batch_size = "8"

[dataset]
repo_id = "example/tasks"

[base]
model_id = "example/base"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(cli.RecipeError, match="batch_size"):
        cli.load_recipe(path)


def test_validate_rejects_invalid_training_semantics(tmp_path: Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        """
schema_version = 1
kind = "train"
output_dir = "output"

[dataset]
repo_id = "example/tasks"

[training]
lr_scheduler = "cosine"

[[bases]]
model_id = "example/base"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(cli.RecipeError, match="lr_scheduler"):
        cli.load_recipe(path)


def test_local_dataset_path_is_explicit_and_relative_to_recipe(tmp_path: Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        """
schema_version = 1
kind = "evaluate"
artifact = "example/artifact"

[dataset]
path = "data/tasks.json"

[base]
model_id = "example/base"
""".strip(),
        encoding="utf-8",
    )

    recipe = cli.load_recipe(path)

    assert recipe.dataset.source == str(tmp_path / "data" / "tasks.json")
    assert recipe.dataset.revision is None


def test_local_artifact_and_base_paths_are_relative_to_recipe(tmp_path: Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        """
schema_version = 1
kind = "evaluate"
artifact = "./artifacts/portal"

[dataset]
repo_id = "example/tasks"

[base]
model_id = "../models/base"
""".strip(),
        encoding="utf-8",
    )

    recipe = cli.load_recipe(path)

    assert recipe.artifact == str(tmp_path / "artifacts" / "portal")
    assert recipe.base.model_id == str(tmp_path.parent / "models" / "base")


def test_refit_recipe_rejects_task_subsets(tmp_path: Path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        """
schema_version = 1
kind = "refit"
output_dir = "output"
source_artifact = "example/source"
tasks = ["rte"]

[dataset]
repo_id = "example/tasks"

[base]
model_id = "example/base"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(cli.RecipeError, match="refit.tasks"):
        cli.load_recipe(path)


def test_train_command_dispatches_the_parsed_recipe(monkeypatch, capsys) -> None:
    received: list[cli.TrainRecipe] = []
    monkeypatch.setattr(cli, "_run_train", received.append)

    assert cli.main(["train", "--config", str(CONFIGS / "train.toml")]) == 0
    assert len(received) == 1
    assert received[0].training.source_steps_per_epoch == 500
    assert capsys.readouterr().err == ""


def test_runtime_failure_is_structured(monkeypatch, capsys) -> None:
    def fail(_recipe) -> None:
        raise RuntimeError("model failed")

    monkeypatch.setattr(cli, "_run_train", fail)

    assert cli.main(["train", "--config", str(CONFIGS / "train.toml")]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "error_type": "RuntimeError",
        "event": "error",
        "message": "model failed",
        "stage": "runtime",
    }


def test_final_result_is_printed_and_persisted(tmp_path: Path, capsys) -> None:
    recipe = cli.load_recipe(CONFIGS / "evaluate.toml")
    recipe = recipe.model_copy(update={"result_path": tmp_path / "result.json"})
    result = {"event": "result", "kind": "evaluate", "score": 0.75}

    cli._write_result(recipe, result)

    assert json.loads(capsys.readouterr().out) == result
    assert json.loads(recipe.result_path.read_text(encoding="utf-8")) == result
