# Running PorTAL on GPU compute

PorTAL training and evaluation are single-process PyTorch workloads. The library is installed from
PyPI; the repository supplies strict TOML recipes, editable Python equivalents, and compute
launchers. A compute platform can run the Python workflows directly:

```bash
python examples/train_example.py
```

or invoke the equivalent configuration-driven CLI recipe:

```bash
portallib train --config examples/configs/train.toml
```

Use `examples/refit_example.py` or `examples/evaluate_example.py` for the other Python phases, or
the matching `refit` and `evaluate` TOML recipes. Both interfaces call the same package training,
refitting, and evaluation implementation.

The compute platform provisions the GPU, preserves the Hugging Face cache and `artifacts/`
directory, and supplies any required Hugging Face token. The release source and target-refit
settings are documented in [`REPRODUCING.md`](REPRODUCING.md).

## Hardware and execution model

Source training loads both source models in one process. Refitting loads one trained PorTAL artifact
and one raw target base; evaluation loads one artifact and its matching raw base. Use one high-memory
NVIDIA GPU; an H200-class GPU is recommended for the complete two-source recipe. A smaller model or
CPU can be used for development and contract tests.

The examples use one CUDA device and do not use multi-process or multi-node execution.

Each `BaseModelSpec` can override the automatic runtime defaults with Hugging Face loading controls:

```python
BaseModelSpec(
    "Qwen/Qwen3-8B",
    "<exact-revision>",
    dtype="float32",
    device_map="cuda",
    attn_implementation="sdpa",
)
```

When `device_map` is set, the loader preserves Hugging Face's placement and does not apply a bulk
`.to(device)`. Without these overrides, CUDA uses bf16 and one device while CPU uses fp32.

For frozen-base PorTAL evaluation and refitting, hybrid maps may offload decoder layers: generated
factors follow each target projection's live output device and dtype. Keep the model's primary
input placement on a real device. Whole-model disk offload that leaves the first parameter on
`meta` is not supported.

## Local Docker

The included [`Dockerfile`](Dockerfile) builds a CUDA training image containing the exact checked-out
package code and TOML recipes. This source-based image is useful for validating a release commit;
ordinary library installation should use PyPI.
The default base image can be overridden with `--build-arg PYTORCH_IMAGE=...` when the host requires
a different PyTorch/CUDA combination.

```bash
docker build -t portallib-training .
mkdir -p artifacts hf-cache
docker run --rm --gpus all \
  -e HF_TOKEN \
  -v "$PWD/artifacts:/workspace/portallib/artifacts" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  portallib-training
```

The image defaults to the source-training TOML recipe. Override its command for another stage:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN \
  -v "$PWD/artifacts:/workspace/portallib/artifacts" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  portallib-training portallib refit --config examples/configs/refit.toml
```

Choose a base image whose CUDA build supports the assigned GPU. This matters particularly for new
GPU architectures.

## Modal

[`examples/launchers/modal_launcher.py`](examples/launchers/modal_launcher.py) is one optional
compute wrapper. It builds the repository's Docker image, mounts persistent storage at the recipe's
`artifacts/` directory, and runs the CLI invocation selected by its `COMMAND` constant.

```bash
python -m pip install modal
modal setup
modal secret create HF_TOKEN HF_TOKEN=your_token
modal run examples/launchers/modal_launcher.py
```

Download a completed output directory with:

```bash
modal volume get portallib-artifacts portal-qwen-sources ./portal-qwen-sources
```

Use the selected example's `OUTPUT_DIR` name when downloading refit results.

Edit the launcher's `gpu` value if H200 is unavailable in the selected Modal workspace. The named
volumes are created automatically and preserve outputs and downloaded models across jobs.
