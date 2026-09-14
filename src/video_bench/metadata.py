"""Metadata serialization and resumable shard merging."""

from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = (
    "status", "model", "adapter", "checkpoint", "prompt_id", "prompt",
    "prompt_metadata", "seed", "resolution", "frames", "fps",
    "inference_steps", "guidance_scale", "generation_parameters",
    "output_path", "generation_seconds",
)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected an object")
            records.append(record)
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV manifest, returning an empty list when it does not exist."""

    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            if "output_path" not in reader.fieldnames:
                raise ValueError(f"CSV manifest has no output_path column: {path}")
            return [dict(row) for row in reader]
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV manifest at {path}: {exc}") from exc


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if value is None:
        return ""
    return value


def upsert_csv(path: Path, record: dict[str, Any]) -> None:
    """Atomically insert or replace one manifest row, safely across GPU workers."""

    if not isinstance(record.get("output_path"), str):
        raise ValueError("CSV manifest records require a string output_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records: list[dict[str, Any]] = list(read_csv(path))
        records.append({key: _csv_value(value) for key, value in record.items()})
        deduplicated = {str(item["output_path"]): item for item in records}
        ordered = sorted(
            deduplicated.values(),
            key=lambda item: (
                str(item.get("model", "")), str(item.get("prompt_id", "")),
                int(item.get("seed") or 0),
            ),
        )
        extra_fields = sorted({key for item in ordered for key in item} - set(CSV_FIELDS))
        fieldnames = [*CSV_FIELDS, *extra_fields]
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(ordered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def merge_metadata(metadata_path: Path, shards: Iterable[Path], overwrite: bool) -> None:
    records = [] if overwrite else read_jsonl(metadata_path)
    for shard in shards:
        records.extend(read_jsonl(shard))

    # Output paths uniquely identify jobs. New records replace stale records.
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in records:
        deduplicated[str(record["output_path"])] = record
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (str(item.get("prompt_id", "")), int(item.get("seed", 0))),
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, metadata_path)
    for shard in shards:
        shard.unlink(missing_ok=True)
