# Gemma 4 heterogeneous-target refit

PorTAL can refit its shared canonical core and task latents onto decoder
architectures whose target projections are sparse or change shape between
layers. The checked-in Gemma 4 E2B recipe demonstrates this support with
[`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) at revision
`d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`.

## Target topology

Gemma 4 E2B has 35 text-decoder layers at
`model.language_model.layers`. Sliding-attention and full-attention layers use
different projection widths, and the final 20 layers reuse KV states instead
of owning local `v_proj` modules:

| Logical module | Layer kind | Input width | Output width | Target count |
| --- | --- | ---: | ---: | ---: |
| `q` | sliding | 1,536 | 2,048 | 28 |
| `q` | full | 1,536 | 4,096 | 7 |
| `v` | sliding, locally owned KV | 1,536 | 256 | 12 |
| `v` | full, locally owned KV | 1,536 | 512 | 3 |

PorTAL records every target explicitly. Each entry contains the decoder layer,
logical canonical module, exact relative projection path, input and output
widths, and stable alignment-group identifiers. Injection and PEFT export use
only these declared exact paths; absent shared-KV projections are not invented.

Compatible targets share shape-grouped alignment parameters:

- one `Pin` per logical module and input-width group;
- one `Pout` per logical module and output-width group;
- one embedding per decoder layer.

The canonical core still owns one `q` head and one `v` head. Existing uniform
artifacts retain their original schema and parameter names.

## Loading

Gemma 4 is loaded through the explicit `multimodal_lm` recipe setting and
`AutoModelForMultimodalLM`. The task workload remains text-only, so
`AutoProcessor.tokenizer` is passed to `PortalBase`. This loader requires
Transformers 5 or newer.

The loader preserves an explicit Hugging Face `device_map`, validates every
declared text-decoder projection, and never applies a bulk device move to a
dispatched model.

## Refit recipe

The publication recipe is
[`examples/configs/refit_gemma4_e2b_1000.toml`](../examples/configs/refit_gemma4_e2b_1000.toml):

```bash
portallib refit --config examples/configs/refit_gemma4_e2b_1000.toml
```

It copies the frozen task latents and canonical core from
`RampPublic/portal-qwen3-4b@v0.1.0` and trains only a fresh Gemma 4 alignment:

- deterministic nested sample of up to 1,000 training examples per task;
- 5 epochs, batch size 4, alignment learning rate `1e-3`;
- linear schedule with 10% warmup, no weight decay, and gradient clipping at
  `1.0`;
- full pinned validation evaluation after epoch zero and every epoch;
- best macro `acc_norm` checkpoint, with lower gold NLL as the tie-breaker.

The smaller
[`examples/configs/refit_gemma4_e2b.toml`](../examples/configs/refit_gemma4_e2b.toml)
uses the first 500 examples from the same seeded per-task permutations.

## Validation result

The 1,000-example recipe selected epoch 4:

| Model | Macro `acc_norm` |
| --- | ---: |
| Frozen Gemma 4 E2B | 0.5729 |
| PorTAL-adapted | 0.7363 |
| Absolute lift | +0.1634 |

The selected native artifact is published as
[`RampPublic/portal-gemma-4-e2b@v0.1.0`](https://huggingface.co/RampPublic/portal-gemma-4-e2b/tree/v0.1.0).

Across the same 14 tasks, independently trained rank-16 full-text-decoder LoRA
baselines averaged 0.7084. Relative to the frozen base, the PorTAL refit
retained 120.6% of their aggregate lift. These are benchmark results for the
exact revisions and recipes above, not a general performance guarantee.

CPU contracts cover sparse-target discovery, shape grouping, zero-delta refit
initialization, differentiability, exact injection, native artifact round-trip,
and PEFT materialization/reload. A real Gemma 4 GPU smoke additionally verified
finite refit loss and reload parity.
