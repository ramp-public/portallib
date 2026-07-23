"""Run one configuration-driven PorTAL workflow on a persistent Modal GPU job.

Example:
    modal run examples/launchers/modal_launcher.py
"""

from __future__ import annotations

import subprocess

import modal

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
def run(command: str, config: str) -> None:
    if command not in {"train", "refit", "evaluate", "validate"}:
        raise ValueError(f"unsupported portallib command: {command!r}")
    try:
        subprocess.run(["portallib", command, "--config", config], cwd="/workspace/portallib", check=True)
    finally:
        artifacts.commit()


@app.local_entrypoint()
def main(
    command: str = "train",
    config: str = "examples/configs/train.toml",
    background: bool = False,
) -> None:
    if background:
        call = run.spawn(command, config)
        print(f"spawned function call {call.object_id}", flush=True)
        return
    run.remote(command, config)
