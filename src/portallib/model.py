"""High-level PorTAL artifact, Hugging Face, and PEFT integration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub import ModelCard, ModelHubMixin, constants, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .config import PortalConfig
from .decoder import GeneratedLora, PortalDecoder

if TYPE_CHECKING:
    from peft import LoraConfig

CONFIG_NAME = constants.CONFIG_NAME
WEIGHTS_NAME = constants.SAFETENSORS_SINGLE_FILE

_MODEL_CARD_TEMPLATE = """---
{{ card_data }}
base_model: "{{ base_model }}"
---

# PorTAL portable task adapters

This artifact generates task-specific LoRA adapters for `{{ base_model }}` with
[`portallib`]({{ repo_url }}).

## Available tasks

{% for task in tasks -%}
- `{{ task }}`
{% endfor %}
"""


class PortalModel(
    ModelHubMixin,
    library_name="portallib",
    license="apache-2.0",
    tags=["lora", "peft", "portal"],
    repo_url="https://github.com/ramp-public/portallib",
    docs_url="https://github.com/ramp-public/portallib#readme",
    model_card_template=_MODEL_CARD_TEMPLATE,
):
    """A set of shared task latents and one decoder for a supported base model."""

    def __init__(
        self,
        config: PortalConfig,
        task_latents: torch.Tensor,
        decoder: PortalDecoder,
        *,
        base_model: torch.nn.Module | None = None,
    ) -> None:
        if decoder.config != config:
            raise ValueError("decoder configuration does not match the artifact configuration")
        if task_latents.shape != (len(config.tasks), config.d_z):
            raise ValueError(
                f"expected task_latents shape {(len(config.tasks), config.d_z)}, got {tuple(task_latents.shape)}"
            )
        self.config = config
        self.task_latents = task_latents
        self.decoder = decoder
        self.base_model = base_model

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.config.tasks)

    def validate_base_model(self, model_id: str) -> None:
        """Require the base identity encoded by this base-specific artifact."""
        if self.config.base_model_name_or_path != model_id:
            raise ValueError(f"artifact expects {self.config.base_model_name_or_path!r}, but received {model_id!r}")

    def generate(self, task: str) -> GeneratedLora:
        """Generate LoRA A/B matrices for one named task."""
        try:
            index = self.config.tasks.index(task)
        except ValueError as exc:
            raise KeyError(f"unknown task {task!r}; available tasks: {', '.join(self.config.tasks)}") from exc
        device = next(self.decoder.parameters()).device
        latent = self.task_latents[index].to(device=device, dtype=next(self.decoder.parameters()).dtype)
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder(latent)

    def get_peft_model(
        self,
        task: str,
        base_model: torch.nn.Module | None = None,
        *,
        adapter_name: str = "default",
    ) -> torch.nn.Module:
        """Create a normal PEFT LoRA model populated with PorTAL-generated weights."""
        from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

        if base_model is None:
            base_model = self.base_model
        if base_model is None:
            raise ValueError("base_model must be passed here or to from_pretrained")
        peft_config = self._peft_config()
        peft_model = get_peft_model(base_model, peft_config, adapter_name=adapter_name)
        generated_state = self._peft_state_dict(task)
        template = get_peft_model_state_dict(
            peft_model,
            adapter_name=adapter_name,
            save_embedding_layers=False,
        )
        if set(generated_state) != set(template):
            missing = sorted(set(template) - set(generated_state))
            unexpected = sorted(set(generated_state) - set(template))
            raise ValueError(
                "configured PorTAL targets do not exactly match the PEFT adapter; "
                f"missing={missing[:4]}, unexpected={unexpected[:4]}"
            )
        adapter_state: dict[str, torch.Tensor] = {}
        for key, value in generated_state.items():
            if value.shape != template[key].shape:
                raise ValueError(
                    f"generated shape mismatch at {key}: {tuple(value.shape)} vs {tuple(template[key].shape)}"
                )
            adapter_state[key] = value.to(device=template[key].device, dtype=template[key].dtype)
        set_peft_model_state_dict(peft_model, adapter_state, adapter_name=adapter_name)
        peft_model.eval()
        return peft_model

    def _peft_config(self) -> "LoraConfig":
        """Build the standard PEFT configuration represented by this artifact."""
        from peft import LoraConfig

        target_paths = [exact_path for _layer, _name, exact_path in self.config.targets()]
        return LoraConfig(
            base_model_name_or_path=self.config.base_model_name_or_path,
            revision=self.config.base_model_revision,
            task_type=self.config.task_type,
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            target_modules=target_paths,
            lora_dropout=0.0,
            bias="none",
            inference_mode=True,
        )

    def _peft_state_dict(self, task: str) -> dict[str, torch.Tensor]:
        """Generate canonical PEFT adapter keys without loading the base model."""
        generated = self.generate(task)
        adapter_state: dict[str, torch.Tensor] = {}
        for layer_index, short_name, exact_path in self.config.targets():
            key = (layer_index, short_name)
            if key not in generated:
                raise ValueError(f"decoder did not generate configured PorTAL target {key}")
            a, b = generated[key]
            module_key = f"base_model.model.{exact_path}"
            adapter_state[f"{module_key}.lora_A.weight"] = a.detach().cpu().contiguous()
            adapter_state[f"{module_key}.lora_B.weight"] = b.detach().cpu().contiguous()
        return adapter_state

    def export_peft(
        self,
        task: str,
        output_dir: str | Path,
    ) -> Path:
        """Materialize a task as a standard PEFT adapter without loading the base model."""
        from peft.utils import SAFETENSORS_WEIGHTS_NAME

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._peft_config().save_pretrained(output)
        save_file(
            self._peft_state_dict(task),
            output / SAFETENSORS_WEIGHTS_NAME,
            metadata={"format": "pt"},
        )
        card = output / "README.md"
        card.write_text(
            f"---\nbase_model: {self.config.base_model_name_or_path}\nlibrary_name: peft\nlicense: apache-2.0\n"
            f"tags:\n- lora\n- portal\n---\n\n# PorTAL-generated adapter: {task}\n\n"
            "Generated with `portallib` from a portable task latent.\n",
            encoding="utf-8",
        )
        return output

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save PorTAL tensors; ModelHubMixin handles config and model-card files."""
        tensors: dict[str, torch.Tensor] = {"task_latents": self.task_latents.detach().cpu().contiguous()}
        tensors.update(
            {
                f"core.{name}": value.detach().cpu().contiguous()
                for name, value in self.decoder.core.state_dict().items()
            }
        )
        tensors.update(
            {
                f"alignment.{name}": value.detach().cpu().contiguous()
                for name, value in self.decoder.alignment.state_dict().items()
            }
        )
        save_file(
            tensors,
            save_directory / WEIGHTS_NAME,
            metadata={"format": "portallib", "schema_version": str(self.config.schema_version)},
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        config: PortalConfig | dict[str, Any],
        base_model: torch.nn.Module | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        **hub_kwargs: Any,
    ) -> "PortalModel":
        """Load PorTAL tensors; ModelHubMixin handles config and repository resolution."""
        if isinstance(config, dict):
            config = PortalConfig.from_dict(config)
        compatibility_values = {name: hub_kwargs.pop(name, None) for name in ("proxies", "resume_download")}
        path = Path(model_id)
        if path.is_dir():
            weights_path = path / WEIGHTS_NAME
        else:
            download_kwargs: dict[str, Any] = {
                "repo_id": model_id,
                "filename": WEIGHTS_NAME,
                "revision": revision,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "token": token,
                "local_files_only": local_files_only,
                "library_name": "portallib",
            }
            download_parameters = inspect.signature(hf_hub_download).parameters
            for compatibility_arg, value in compatibility_values.items():
                if value is not None and compatibility_arg in download_parameters:
                    download_kwargs[compatibility_arg] = value
            weights_path = Path(hf_hub_download(**download_kwargs))
        if hub_kwargs:
            names = ", ".join(sorted(hub_kwargs))
            raise TypeError(f"unexpected from_pretrained arguments: {names}")

        with safe_open(weights_path, framework="pt") as artifact:
            metadata = artifact.metadata() or {}
            if metadata.get("format") != "portallib":
                raise ValueError("invalid PorTAL weights: missing format='portallib' metadata")
            if metadata.get("schema_version") != str(config.schema_version):
                raise ValueError("PorTAL config and weights schema versions do not match")
        tensors = load_file(weights_path, device=str(device))
        if "task_latents" not in tensors:
            raise ValueError("invalid PorTAL weights: missing task_latents")
        task_latents = tensors.pop("task_latents")
        decoder = PortalDecoder(config).to(device=device, dtype=dtype)
        core_state = {name.removeprefix("core."): value for name, value in tensors.items() if name.startswith("core.")}
        alignment_state = {
            name.removeprefix("alignment."): value for name, value in tensors.items() if name.startswith("alignment.")
        }
        if len(core_state) + len(alignment_state) != len(tensors):
            unexpected = sorted(
                set(tensors)
                - {f"core.{name}" for name in core_state}
                - {f"alignment.{name}" for name in alignment_state}
            )
            raise ValueError(f"invalid canonical PorTAL weights; unexpected keys: {unexpected[:4]}")
        decoder.core.load_state_dict(core_state, strict=True)
        decoder.alignment.load_state_dict(alignment_state, strict=True)
        decoder.eval()
        return cls(config, task_latents, decoder, base_model=base_model)

    def generate_model_card(self, **kwargs: Any) -> ModelCard:
        """Generate the standard Hub card with artifact-specific base model and tasks."""
        kwargs.setdefault("base_model", self.config.base_model_name_or_path)
        kwargs.setdefault("tasks", self.config.tasks)
        return super().generate_model_card(**kwargs)

    def to_json(self) -> str:
        """Return a small human-readable artifact summary."""
        return json.dumps(
            {
                "base_model": self.config.base_model_name_or_path,
                "tasks": self.config.tasks,
                "rank": self.config.rank,
                "architecture": self.config.architecture,
            },
            indent=2,
        )
