# Releasing `portallib`

External publication requires an explicit maintainer confirmation.

## Prepare

1. Replace `Unreleased` in `CHANGELOG.md` with the release date.
2. Confirm the version in `pyproject.toml` and `src/portallib/__init__.py`.
3. Run:

   ```bash
   uv run ruff check src tests examples
   uv run pytest -q
   rm -rf dist
   uv run python -m build
   uv run --with twine twine check dist/*
   ```

4. Inspect the wheel and source distribution in `dist/`.
5. Tag the exact reviewed commit as `v0.1.0`.

## Publish after confirmation

Publish the already-checked files without rebuilding them:

```bash
uv run --with twine twine upload dist/portallib-0.1.0*
```

Then create the matching GitHub release and verify installation from PyPI in a clean environment.

Before publishing native Qwen3-8B or Gemma-3-4B artifacts, confirm the final public Hugging Face
namespace, repository names, model revisions, and model-card metrics. Publish each native PorTAL
artifact separately from any generated task-specific PEFT adapters, and verify both layouts by
reloading them from the Hub before announcing the release.
