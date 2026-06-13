"""Inspect a few parsed PatentLLM chunks from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect parsed PatentLLM chunks.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/parsed_patents/chunks.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks_path}")

    with args.chunks_path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            row = json.loads(line)
            print("\n" + "-" * 100)
            print(f"chunk_id: {row['chunk_id']}")
            print(f"document_id: {row['document_id']}")
            print(f"section: {row['section']}")
            print(f"chunk_type: {row['chunk_type']}")
            print(f"retrieval_tier: {row['retrieval_tier']}")
            print(f"pages: {row['page_start']} - {row['page_end']}")
            print(f"tokens_estimate: {row['token_estimate']}")
            print("-" * 100)
            print(row["text"][:1_200])

            if index + 1 >= args.limit:
                break


if __name__ == "__main__":
    main()