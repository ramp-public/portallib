# Running PorTAL on GPU compute

PorTAL training and evaluation are single-process PyTorch workloads. The library is installed from
PyPI; the repository supplies the editable recipe files and compute launchers. Select
`train_example.py`, `refit_example.py`, or `evaluate_example.py`, edit its recipe block, and run it.
Every compute platform follows the same pattern:

```bash
python examples/train_example.py
```

The compute platform provisions the GPU, preserves the Hugging Face cache and `artifacts/`
directory, and supplies any required Hugging Face token. The release source and both 1,000-example
refit settings are documented in [`REPRODUCING.md`](REPRODUCING.md).

## Hardware and execution model

Source training loads both source models in one process. Refitting loads one trained PorTAL artifact
and one raw target base; evaluation loads one artifact and its matching raw base. Use one high-memory
NVIDIA GPU; an H200-class GPU is recommended for the complete two-source recipe. A smaller model or
CPU can be used for development and contract tests.

The examples use one CUDA device and do not use multi-process or multi-node execution.

## Local Docker

The included [`Dockerfile`](Dockerfile) builds a CUDA training image containing the exact checked-out
package code and edited recipe. This source-based image is useful for validating a release commit;
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

The image defaults to `train_example.py`. Override its command for another stage:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN \
  -v "$PWD/artifacts:/workspace/portallib/artifacts" \
  -v "$PWD/hf-cache:/cache/huggingface" \
  portallib-training python examples/refit_example.py
```

Choose a base image whose CUDA build supports the assigned GPU. This matters particularly for new
GPU architectures.

## Modal

[`examples/launchers/modal_launcher.py`](examples/launchers/modal_launcher.py) is one optional
compute wrapper. It builds the repository's Docker image, mounts persistent storage at the recipe's
`artifacts/` directory, and runs the file selected by its `EXAMPLE` constant.

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
