"""Resolve local checkpoints before subprocesses change working directory."""
from pathlib import Path


def resolve_checkpoint(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    # Leave named pretrained models to SevenNet, including future aliases.
    if path.suffix or path.is_absolute() or any(c in value for c in ("/", "\\")) or value.startswith("~"):
        raise FileNotFoundError(
            f"Checkpoint file not found: {path.resolve()}. Relative checkpoint paths "
            f"are resolved from the controller working directory ({Path.cwd()}), "
            "not the iteration's finetune/md directory. Use an existing absolute path."
        )
    return value
