from pathlib import Path
from typing import Any, Mapping


def parse_scalar(value: str) -> Any:
    text = value.strip()
    normalized = text.lower()

    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if normalized in {"null", "none"}:
        return None

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]

    return text


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def read_flat_yaml(path: str | Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = parse_scalar(raw_value)

    return values


def write_flat_yaml(path: str | Path, values: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {format_scalar(value)}" for key, value in values.items()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_training_snapshot(
    source_config: str | Path,
    output_config: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = read_flat_yaml(source_config)
    snapshot["model_name_or_path"] = str(model_dir)
    snapshot["output_dir"] = str(output_dir)
    if overrides:
        snapshot.update(overrides)

    write_flat_yaml(output_config, snapshot)
    return snapshot


def build_llamafactory_command(config_path: str | Path) -> list[str]:
    return ["llamafactory-cli", "train", str(config_path)]
