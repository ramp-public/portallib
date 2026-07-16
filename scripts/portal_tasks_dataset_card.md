---
pretty_name: PorTAL 14-Task Multiple-Choice Suite
language:
- en
license: other
task_categories:
- multiple-choice
size_categories:
- 100K<n<1M
---

# PorTAL 14-Task Multiple-Choice Suite

This dataset is the normalized task suite used by the training examples in
[portallib](https://github.com/ramp-public/portallib). It contains 129,212 training examples and
19,548 validation examples. Each row has the following fields:

- `task`: stable task name
- `prompt`: language-model context with no trailing whitespace; it is empty only when a source
  sentence places its blank first
- `choices`: candidate continuations with exactly one leading space and no trailing whitespace
- `gold_idx`: zero-based index of the correct continuation

The validation metric in portallib is continuation log-probability normalized by character length
(`acc_norm`). Gold continuation token-mean NLL is reported separately.

## Recipe selection

The repository stores the complete normalized task pools. The checked-in PorTAL recipes select
examples deterministically from this dataset:

- Source training uses the leading 2,000 training examples for each task, or every available
  example when a task has fewer than 2,000. The subset is fixed across epochs
  (`source_resample_each_epoch=False`).
- Refitting uses a seeded sample of up to 1,000 training examples per task from the complete pool
  (`seed=0`).
- Evaluation uses the leading 1,000 validation examples per task, or every available example when
  a task has fewer than 1,000.

Keeping the complete pools here is necessary because source training and target refitting use
different deterministic selection policies. The exact recipes are recorded in
[`REPRODUCING.md`](https://github.com/ramp-public/portallib/blob/main/REPRODUCING.md).

## Sources and provenance

| Tasks | Upstream dataset | Pinned revision | License declared by upstream Hub card |
|---|---|---|---|
| TruthfulQA | `truthfulqa/truthful_qa` | `741b8276f2d1982aa3d5b832d3ee81ed3b896490` | Apache-2.0 |
| RTE, CB, COPA, WiC, WSC | `aps/super_glue` | `3de24cf8022e94f4ee4b9d55a6f539891524d646` | Other; consult task-specific terms |
| BoolQ | `google/boolq` | `35b264d03638db9f4ce671b711558bf7ff0f80d5` | CC BY-SA 3.0 |
| ARC Easy, ARC Challenge | `allenai/ai2_arc` | `210d026faf9955653af8916fad021475a3f00453` | CC BY-SA 4.0 |
| HellaSwag | `Rowan/hellaswag` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | Not declared in Hub metadata |
| OpenBookQA | `allenai/openbookqa` | `388097ea7776314e93a529163e0fea805b8a6454` | Unknown in Hub metadata |
| WinoGrande | `allenai/winogrande` | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | Not declared in Hub metadata |
| CommonsenseQA | `tau/commonsense_qa` | `94630fe30dad47192a8546eb75f094926d47e155` | MIT |
| SciQ | `allenai/sciq` | `2c94ad3e1aafab77146f384e23536f97a4849815` | CC BY-NC 3.0 |

The table reports upstream Hugging Face metadata and is not a legal interpretation. The collection
uses `license: other` because its components have different terms. The Apache-2.0 license of
portallib covers the software and does not relicense upstream benchmark content. Users are
responsible for complying with the terms of each source dataset.

## Reproduction

The complete deterministic normalization is in
[`scripts/prepare_dataset.py`](https://github.com/ramp-public/portallib/blob/main/scripts/prepare_dataset.py).
From a portallib checkout, install the training dependencies and run the preparation script:

```bash
pip install -e '.[training]'
python scripts/prepare_dataset.py --output portal_tasks.json
```

The canonical JSON serialization has SHA-256
`97ae9193a02b96daec13f7e21f56fbe7ed5102fd900e6c2093d9bbfc009f74cd`.

TruthfulQA exposes one labeled split, so the first 75% is used for training and the final 25% for
validation. Every other task uses its upstream `train` and `validation`
splits directly. No test labels are included.
