"""Helpers for loading experiment configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without PyYAML
    yaml = None


def _parse_scalar(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if raw_value.startswith("[") or raw_value.startswith("{"):
        return json.loads(raw_value)

    try:
        if raw_value.startswith("0") and raw_value not in {"0", "0.0"} and not raw_value.startswith("0."):
            raise ValueError
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _preprocess_yaml_lines(text: str) -> list[tuple[int, str]]:
    processed = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        processed.append((indent, raw_line.strip()))
    return processed


def _parse_yaml_block(lines: list[tuple[int, str]], start_idx: int, indent: int) -> tuple[Any, int]:
    if start_idx >= len(lines):
        return {}, start_idx

    current_indent, content = lines[start_idx]
    if current_indent != indent:
        raise ValueError(f"Invalid indentation at line: {content}")

    if content.startswith("- "):
        return _parse_yaml_list(lines, start_idx, indent)
    return _parse_yaml_dict(lines, start_idx, indent)


def _parse_yaml_dict(lines: list[tuple[int, str]], start_idx: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    idx = start_idx

    while idx < len(lines):
        current_indent, content = lines[idx]
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"Unexpected indentation in mapping near: {content}")
        if content.startswith("- "):
            break

        key, sep, remainder = content.partition(":")
        if not sep:
            raise ValueError(f"Invalid mapping entry: {content}")

        key = key.strip()
        remainder = remainder.strip()
        idx += 1

        if remainder:
            result[key] = _parse_scalar(remainder)
            continue

        if idx >= len(lines) or lines[idx][0] <= current_indent:
            result[key] = {}
            continue

        nested_value, idx = _parse_yaml_block(lines, idx, lines[idx][0])
        result[key] = nested_value

    return result, idx


def _parse_yaml_list(lines: list[tuple[int, str]], start_idx: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    idx = start_idx

    while idx < len(lines):
        current_indent, content = lines[idx]
        if current_indent < indent:
            break
        if current_indent != indent or not content.startswith("- "):
            break

        item_content = content[2:].strip()
        idx += 1

        if not item_content:
            nested_value, idx = _parse_yaml_block(lines, idx, lines[idx][0])
            result.append(nested_value)
            continue

        if ":" in item_content:
            key, sep, remainder = item_content.partition(":")
            if not sep:
                raise ValueError(f"Invalid list mapping entry: {item_content}")

            item: dict[str, Any] = {}
            key = key.strip()
            remainder = remainder.strip()

            if remainder:
                item[key] = _parse_scalar(remainder)
            else:
                if idx >= len(lines) or lines[idx][0] <= current_indent:
                    item[key] = {}
                else:
                    nested_value, idx = _parse_yaml_block(lines, idx, lines[idx][0])
                    item[key] = nested_value

            while idx < len(lines):
                next_indent, next_content = lines[idx]
                if next_indent <= current_indent:
                    break
                if next_content.startswith("- "):
                    break
                if next_indent != current_indent + 2:
                    raise ValueError(f"Unexpected indentation in list item near: {next_content}")

                nested_key, sep, nested_remainder = next_content.partition(":")
                if not sep:
                    raise ValueError(f"Invalid nested mapping entry: {next_content}")

                nested_key = nested_key.strip()
                nested_remainder = nested_remainder.strip()
                idx += 1

                if nested_remainder:
                    item[nested_key] = _parse_scalar(nested_remainder)
                    continue

                if idx >= len(lines) or lines[idx][0] <= next_indent:
                    item[nested_key] = {}
                    continue

                nested_value, idx = _parse_yaml_block(lines, idx, lines[idx][0])
                item[nested_key] = nested_value

            result.append(item)
            continue

        result.append(_parse_scalar(item_content))

    return result, idx


def _load_without_pyyaml(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    lines = _preprocess_yaml_lines(text)
    if not lines:
        return {}

    parsed, next_idx = _parse_yaml_block(lines, 0, lines[0][0])
    if next_idx != len(lines):
        raise ValueError(f"Could not parse config beyond line index {next_idx}")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level config must be a mapping")
    return parsed


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    return _load_without_pyyaml(config_path)


def build_env_kwargs(
    environment_config: Mapping[str, Any],
    arrival_schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create environment kwargs from a config section and a schedule."""
    return {
        "arrival_schedule": arrival_schedule,
        "episode_length": int(environment_config.get("episode_length", 200)),
        "step_seconds": int(environment_config.get("step_seconds", 3)),
        "yellow_time": int(environment_config.get("yellow_time", 1)),
        "max_departures_per_step": int(environment_config.get("max_departures_per_step", 4)),
        "reward_mode": environment_config.get("reward_mode", "queue"),
        "switch_penalty": float(environment_config.get("switch_penalty", 2.0)),
    }
