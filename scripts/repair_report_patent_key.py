"""Repair a generated report by appending a deterministic patent key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patentllm.generation.patent_key import (
    append_or_replace_patent_key,
    build_patent_key_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair report patent key.")

    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/reference/patent_metadata.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path. Defaults to *_repaired.md beside report.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_text = args.report_path.read_text(encoding="utf-8")
    evidence_pack = json.loads(args.evidence_pack.read_text(encoding="utf-8"))

    patent_key = build_patent_key_markdown(
        evidence_pack=evidence_pack,
        metadata_csv_path=args.metadata_csv,
    )

    repaired = append_or_replace_patent_key(
        report_text=report_text,
        patent_key_markdown=patent_key,
    )

    output_path = args.output_path
    if output_path is None:
        output_path = args.report_path.with_name(
            args.report_path.stem + "_repaired.md"
        )

    output_path.write_text(repaired, encoding="utf-8")

    print(f"Wrote repaired report to: {output_path}")


if __name__ == "__main__":
    main()