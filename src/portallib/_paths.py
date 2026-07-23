"""Internal exact-path helpers for configured model modules."""


def model_slug(model_id: str) -> str:
    """Return a stable filesystem name for a Hub model ID."""
    return model_id.strip("/").lower().replace("/", "--").replace(".", "-")


def validate_dotted_path(path: str, *, name: str) -> None:
    if not path or any(not part for part in path.split(".")):
        raise ValueError(f"{name} must be a non-empty dotted path")


def exact_module_path(layer_path: str, layer_index: int, module_path: str) -> str:
    return f"{layer_path}.{layer_index}.{module_path}"
