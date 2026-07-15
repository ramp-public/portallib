"""Run one editable PorTAL example on a persistent Modal GPU job.

Example:
    modal run examples/launchers/modal_launcher.py
"""

from __future__ import annotations

import subprocess

import modal

# Select train_example.py, refit_example.py, or evaluate_example.py.
EXAMPLE = "examples/train_example.py"

app = modal.App("portallib-training")
image = modal.Image.from_dockerfile("Dockerfile", context_dir=".").entrypoint([])
artifacts = modal.Volume.from_name("portallib-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("portallib-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("HF_TOKEN")],
    volumes={"/workspace/portallib/artifacts": artifacts, "/cache/huggingface": hf_cache},
)
def train() -> None:
    command = ["python", EXAMPLE]
    try:
        subprocess.run(command, cwd="/workspace/portallib", check=True)
    finally:
        artifacts.commit()


@app.local_entrypoint()
def main() -> None:
    train.remote()
