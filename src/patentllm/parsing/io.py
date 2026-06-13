"""Input/output utilities for parser records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol


class JsonSerializable(Protocol):
    def to_dict(self) -> dict[str, object]: ...


def write_jsonl(path: Path, records: Iterable[JsonSerializable]) -> int:
    """Write records to JSON Lines and return the number of records written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def list_pdf_files(input_dir: Path) -> list[Path]:
    """Return PDF files in deterministic order."""

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".pdf")
