"""Prompt dataset parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    generation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:100] or "prompt"


def load_prompts(path: str | Path) -> list[PromptRecord]:
    prompt_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(prompt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Prompt file not found: {prompt_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {prompt_path}: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("prompts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Prompt JSON must be a non-empty list or an object with a non-empty 'prompts' list")

    records: list[PromptRecord] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            prompt, prompt_id, generation, metadata = item, f"prompt-{index:05d}", {}, {}
        elif isinstance(item, dict):
            prompt = item.get("prompt")
            prompt_id = item.get("id", f"prompt-{index:05d}")
            generation = item.get("generation", {})
            metadata = item.get("metadata", {})
            if not isinstance(generation, dict):
                raise ValueError(f"Prompt {index}: generation must be an object")
            if not isinstance(metadata, dict):
                raise ValueError(f"Prompt {index}: metadata must be an object")
        else:
            raise ValueError(f"Prompt {index}: expected a string or object")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt {index}: prompt must be a non-empty string")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"Prompt {index}: id must be a non-empty string")
        safe_id = slugify(prompt_id)
        if safe_id in seen:
            raise ValueError(f"Duplicate prompt id after filename normalization: {safe_id}")
        seen.add(safe_id)
        records.append(PromptRecord(safe_id, prompt.strip(), dict(generation), dict(metadata)))
    return records
