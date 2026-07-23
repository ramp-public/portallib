"""Configuration-driven command line interface for PorTAL workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import __version__
from .recipes import EvaluateRecipe, RecipeError, RefitRecipe, TrainRecipe, _parse_recipe, load_recipe
from .workflows import _emit, _run_evaluate, _run_refit, _run_train

__all__ = [
    "EvaluateRecipe",
    "RefitRecipe",
    "TrainRecipe",
    "build_parser",
    "load_recipe",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portallib", description="Run PorTAL workflows from strict TOML recipes.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "refit", "evaluate", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--config",
            required=True,
            help="Path to a versioned TOML recipe, or - to read it from stdin.",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        recipe = (
            _parse_recipe(sys.stdin.read(), parent=Path.cwd(), source="<stdin>")
            if args.config == "-"
            else load_recipe(args.config)
        )
    except (OSError, RecipeError) as exc:
        _emit({"event": "error", "stage": "config", "message": str(exc)}, stream=sys.stderr)
        return 2
    if args.command == "validate":
        _emit({"event": "validated", "kind": recipe.kind, "recipe_version": recipe.recipe_version})
        return 0
    if recipe.kind != args.command:
        _emit(
            {
                "event": "error",
                "stage": "config",
                "message": f"recipe kind {recipe.kind!r} cannot run with command {args.command!r}",
            },
            stream=sys.stderr,
        )
        return 2
    try:
        if isinstance(recipe, TrainRecipe):
            _run_train(recipe)
        elif isinstance(recipe, RefitRecipe):
            _run_refit(recipe)
        else:
            _run_evaluate(recipe)
    except Exception as exc:
        _emit(
            {
                "event": "error",
                "stage": "runtime",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            stream=sys.stderr,
        )
        return 1
    return 0
