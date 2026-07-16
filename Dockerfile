ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

WORKDIR /workspace/portallib

ENV HF_HOME=/cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md CLI.md LICENSE ./
COPY src ./src
COPY examples ./examples

RUN python -m pip install --no-cache-dir '.[training]'

CMD ["portallib", "train", "--config", "examples/configs/train.toml"]
