#!/usr/bin/env python3
"""Evaluate a trained PorTAL artifact from Hugging Face or a local directory.

Edit the recipe block, then run:

    python examples/evaluate_example.py

The artifact contains the task vectors, canonical hypernetwork, and matching base alignment. The
example loads the frozen raw base separately and reports both its floor and PorTAL-adapted scores.
"""

from __future__ import annotations

import json

from portallib import (
    BaseModelSpec,
    PortalEvaluator,
    PortalModel,
    load_base,
    load_dataset,
    runtime_device,
)

# ---------------------------------------------------------------------------
# Recipe: trained PorTAL artifact + its raw base -> held-out evaluation.
# ---------------------------------------------------------------------------

PORTAL_ARTIFACT = "RampPublic/portal-qwen3-8b"
PORTAL_ARTIFACT_REVISION: str | None = "v0.1.0"
BASE = BaseModelSpec(
    "Qwen/Qwen3-8B",
    "b968826d9c46dd6066d109eabc6255188de91218",
    # Optional host-specific controls:
    # dtype="float32",
    # device_map="cuda",
    # attn_implementation="sdpa",
)
DATASET = "RampPublic/portallib-tasks"
DATASET_REVISION = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
MAX_EXAMPLES = 1000
MAX_PROMPT = 768
EVAL_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# End recipe.
# ---------------------------------------------------------------------------


def main() -> None:
    dataset = load_dataset(DATASET, revision=DATASET_REVISION)
    portal = PortalModel.from_pretrained(PORTAL_ARTIFACT, revision=PORTAL_ARTIFACT_REVISION)
    portal.validate_base_model(BASE.model_id, BASE.revision)

    device, dtype = runtime_device()
    base = load_base(BASE, device=device, dtype=dtype)
    evaluator = PortalEvaluator(max_prompt=MAX_PROMPT, batch_size=EVAL_BATCH_SIZE)
    tasks = tuple(portal.config.tasks)
    base_result, portal_result = evaluator.compare(
        base,
        dataset,
        portal,
        tasks=tasks,
        max_examples=MAX_EXAMPLES,
    )
    print(
        json.dumps(
            {
                "artifact": PORTAL_ARTIFACT,
                "base": base_result.to_dict(),
                "portal": portal_result.to_dict(),
                "macro_accuracy_lift": portal_result.macro_accuracy - base_result.macro_accuracy,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
