"""Internal naming helpers shared by artifacts and host workflows."""


def model_slug(model_id: str) -> str:
    """Return a stable filesystem name for a Hub model ID."""
    return model_id.rsplit("/", 1)[-1].lower().replace(".", "-")
