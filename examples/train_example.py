#!/usr/bin/env python3
"""Train canonical PorTAL source artifacts and optionally refit an unseen base.

Single-base training::

    python examples/train_example.py --dataset tasks.json --output portal-example

Shared-core training followed by frozen-core/latent refitting::

    python examples/train_example.py --dataset tasks.json --output portal-qwen \
      --base-model Qwen/Qwen3-1.7B --base-model Qwen/Qwen3-4B \
      --refit-base-model Qwen/Qwen3-8B

Shared-core source training can also save one artifact per source without refitting::

    python examples/train_example.py --dataset tasks.json --output portal-sources \
      --base-model Qwen/Qwen3-1.7B --base-model Qwen/Qwen3-4B

Cross-family refitting to Gemma 3 uses its exact decoder-layer path::

    python examples/train_example.py --dataset tasks.json --output portal-gemma3 \
      --base-model Qwen/Qwen3-1.7B --base-model Qwen/Qwen3-4B \
      --refit-base-model google/gemma-3-4b-pt \
      --refit-layer-path model.language_model.layers \
      --refit-max-train 2000 --no-early-stopping
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from portallib import (
    ChoiceDataset,
    PortalAdapterRefitter,
    PortalBase,
    PortalCoreTrainer,
    PortalTrainingConfig,
)

MODULES = {
    "qv": ("q", "v"),
    "full": ("q", "k", "v", "o", "gate", "up", "down"),
}


def load_dataset(source: str) -> ChoiceDataset:
    path = Path(source)
    return ChoiceDataset.from_json(path) if path.is_file() else ChoiceDataset.from_hub(source)


def load_base(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    layer_path: str = "model.layers",
) -> PortalBase:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    return PortalBase(model_id=model_id, model=model, tokenizer=tokenizer, layer_path=layer_path)


def slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower().replace(".", "-")


def print_epoch(phase: str, epoch) -> None:
    print(
        json.dumps(
            {
                "phase": phase,
                "epoch": epoch.epoch,
                "acc_norm": epoch.macro_accuracy,
                "gold_nll": epoch.macro_gold_nll,
                "bases": {
                    name: {"acc_norm": result.macro_accuracy, "gold_nll": result.macro_gold_nll}
                    for name, result in epoch.evaluations.items()
                },
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="normalized JSON path or Hugging Face dataset repo ID")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-model",
        action="append",
        dest="base_models",
        help="source base; repeat to share the canonical core and task latents",
    )
    parser.add_argument(
        "--base-layer-path",
        action="append",
        dest="base_layer_paths",
        help="exact decoder-layer path for the corresponding --base-model (default: model.layers)",
    )
    parser.add_argument("--refit-base-model", default="")
    parser.add_argument(
        "--refit-layer-path",
        default="model.layers",
        help="exact decoder-layer path in the refit base",
    )
    parser.add_argument("--tasks", default="", help="comma-separated subset in the desired stable order")
    parser.add_argument("--modules", choices=tuple(MODULES), default="qv")
    parser.add_argument(
        "--source-max-train",
        type=int,
        default=0,
        help="examples per source task; 0 uses all available examples",
    )
    parser.add_argument("--source-steps-per-epoch", type=int, default=1000)
    parser.add_argument("--refit-max-train", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="run every configured epoch and still restore the best held-out epoch",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push-dataset-to", default="")
    parser.add_argument("--private-dataset", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = load_dataset(args.dataset)
    if args.push_dataset_to:
        dataset.push_to_hub(args.push_dataset_to, private=args.private_dataset)
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip()) or dataset.tasks
    source_models = args.base_models or ["Qwen/Qwen3-4B"]
    if len(set(source_models)) != len(source_models):
        parser.error("--base-model values must be unique")
    if args.base_layer_paths and len(args.base_layer_paths) != len(source_models):
        parser.error("repeat --base-layer-path once for every --base-model")
    source_layer_paths = args.base_layer_paths or ["model.layers"] * len(source_models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    output = Path(args.output)
    recipe = PortalTrainingConfig(
        modules=MODULES[args.modules],
        source_max_examples=args.source_max_train or None,
        source_steps_per_epoch=args.source_steps_per_epoch,
        refit_max_examples=args.refit_max_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_learning_rate=args.latent_learning_rate,
        seed=args.seed,
        early_stopping_patience=None if args.no_early_stopping else 1,
        checkpoint_dir=output / "checkpoints" / "source",
    )
    source_bases = [
        load_base(model_id, device=device, dtype=dtype, layer_path=layer_path)
        for model_id, layer_path in zip(source_models, source_layer_paths, strict=True)
    ]
    source_result = PortalCoreTrainer(source_bases, dataset, tasks=tasks, config=recipe).train(
        on_epoch=lambda epoch: print_epoch("source", epoch)
    )

    if not args.refit_base_model:
        if len(source_models) == 1:
            artifact_outputs = {source_models[0]: output}
        else:
            output.mkdir(parents=True, exist_ok=True)
            artifact_outputs = {
                model_id: output / f"source-{slug(model_id)}"
                for model_id in source_models
            }
        for model_id, artifact_output in artifact_outputs.items():
            source_result.artifacts[model_id].save_pretrained(artifact_output)
        print(
            json.dumps(
                {
                    "best_epoch": source_result.best_epoch,
                    "epochs_completed": source_result.diagnostics["epochs_completed"],
                    "task_regressions": source_result.diagnostics["task_regressions"],
                    "source_examples_per_task": source_result.diagnostics["source_examples_per_task"],
                    "outputs": {
                        model_id: str(artifact_output)
                        for model_id, artifact_output in artifact_outputs.items()
                    },
                }
            ),
            flush=True,
        )
        return

    output.mkdir(parents=True, exist_ok=True)
    for model_id, artifact in source_result.artifacts.items():
        artifact.save_pretrained(output / f"source-{slug(model_id)}")
    source_artifact = source_result.artifacts[source_models[-1]]
    del source_bases
    torch.cuda.empty_cache()

    target = load_base(
        args.refit_base_model,
        device=device,
        dtype=dtype,
        layer_path=args.refit_layer_path,
    )
    refit_recipe = replace(recipe, checkpoint_dir=output / "checkpoints" / "refit")
    refit_result = PortalAdapterRefitter(source_artifact, target, dataset, config=refit_recipe).refit(
        on_epoch=lambda epoch: print_epoch("refit", epoch)
    )
    refit_output = output / f"refit-{slug(args.refit_base_model)}"
    refit_result.artifact.save_pretrained(refit_output)
    print(
        json.dumps(
            {
                "source_best_epoch": source_result.best_epoch,
                "source_best_loss_epoch": source_result.best_loss_epoch,
                "refit_best_epoch": refit_result.best_epoch,
                "refit_best_loss_epoch": refit_result.best_loss_epoch,
                "source_epochs_completed": source_result.diagnostics["epochs_completed"],
                "refit_epochs_completed": refit_result.diagnostics["epochs_completed"],
                "source_task_regressions": source_result.diagnostics["task_regressions"],
                "refit_task_regressions": refit_result.diagnostics["task_regressions"],
                "source_examples_per_task": source_result.diagnostics["source_examples_per_task"],
                "refit_examples_per_task": refit_result.diagnostics["refit_examples_per_task"],
                "output": str(refit_output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
