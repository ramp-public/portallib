# Contributing to portallib

Thanks for helping improve `portallib`. Focused bug fixes, tests, documentation, compatibility
mappings, and carefully scoped API improvements are welcome.

## Before you start

- Search existing issues and pull requests before opening a duplicate.
- Open an issue before a large API, artifact-schema, or training-recipe change.
- Never include credentials, private data, trained checkpoints, or generated model artifacts in a
  pull request.

## Development setup

Fork and clone the repository, then install the package and development dependencies:

```bash
uv sync --all-extras --group dev
```

Run the release checks before submitting a change:

```bash
uv run ruff check src tests examples scripts
uv run pytest -q
uv run python -m build
uvx twine check dist/*
```

CPU tests should cover package contracts. GPU validation is appropriate when a change affects real
model loading, checkpointing, training stability, or artifact equivalence.

## Change guidelines

- Preserve exact model and projection paths; do not introduce fuzzy module matching.
- Add a regression test for bug fixes and tests for new public behavior.
- Update the README or reproduction guide when public usage or a pinned recipe changes.
- Keep examples thin and put reusable training or evaluation behavior in `portallib`.
- Keep Modal and other compute launchers outside the package runtime.
- Avoid unrelated formatting or refactors in the same pull request.

## Pull requests

Explain the problem, the behavior change, and how you validated it. Keep commits reviewable and
respond to review feedback with either a code change or concrete technical reasoning.

By contributing, you agree that your contribution is licensed under the repository's Apache-2.0
license.
