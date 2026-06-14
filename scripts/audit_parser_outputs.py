"""Audit parser outputs before building embeddings.

Usage:
    python scripts/audit_parser_outputs.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
_CLAIM_CHUNK_TYPES = {"claim", "claim_clause_window"}


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_quality_summary(quality_rows: list[JsonDict], output_dir: Path) -> None:
    output_path = output_dir / "quality_summary.csv"
    fieldnames = [
        "document_id",
        "source_file",
        "page_count",
        "pages_extracted",
        "pages_with_text",
        "ocr_pages",
        "native_text_pages",
        "paragraph_count",
        "chunk_count",
        "embeddable_chunk_count",
        "title_found",
        "abstract_found",
        "claims_found",
        "claim_count",
        "claim_numbers_detected",
        "missing_claim_numbers",
        "low_confidence_claim_count",
        "examples_found",
        "tables_detected",
        "figures_detected",
        "suspicious_chunk_count",
        "detected_sections",
        "warnings",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in quality_rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames if key not in {"detected_sections", "warnings", "claim_numbers_detected", "missing_claim_numbers"}},
                    "claim_numbers_detected": " | ".join(str(v) for v in row.get("claim_numbers_detected", [])),
                    "missing_claim_numbers": " | ".join(str(v) for v in row.get("missing_claim_numbers", [])),
                    "detected_sections": " | ".join(row.get("detected_sections", [])),
                    "warnings": " | ".join(row.get("warnings", [])),
                }
            )


def write_metadata_audit(chunks: list[JsonDict], output_dir: Path) -> None:
    output_path = output_dir / "metadata_chunks.txt"
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in [c for c in chunks if c.get("chunk_type") == "metadata"]:
            file.write("\n\n" + "=" * 120 + "\n")
            file.write(f"{chunk.get('document_id')} | embeddable={chunk.get('embeddable')}\n")
            file.write("=" * 120 + "\n")
            file.write(chunk.get("text", ""))
            file.write("\n")


def write_claim_audit(chunks: list[JsonDict], output_dir: Path) -> None:
    output_path = output_dir / "claim_audit.txt"
    claim_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") in _CLAIM_CHUNK_TYPES]
    docs = sorted({chunk.get("document_id") for chunk in claim_chunks})
    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Claim Audit\n")
        file.write("# Uses claim_number metadata, so long claim_clause_window chunks are counted correctly.\n")
        for document_id in docs:
            doc_claims = [chunk for chunk in claim_chunks if chunk.get("document_id") == document_id]
            by_number: dict[int, list[JsonDict]] = collections.defaultdict(list)
            for chunk in doc_claims:
                number = chunk.get("metadata", {}).get("claim_number")
                if number is not None:
                    by_number[int(number)].append(chunk)
            numbers = sorted(by_number)
            missing = [value for value in range(1, numbers[-1] + 1) if value not in by_number] if numbers else []
            low_conf = [
                chunk for chunk in doc_claims
                if chunk.get("metadata", {}).get("claim_confidence") == "low"
                or chunk.get("metadata", {}).get("claim_warnings")
            ]
            file.write("\n" + str(document_id) + "\n")
            file.write("-" * len(str(document_id)) + "\n")
            file.write(f"claim_chunks: {len(doc_claims)}\n")
            file.write(f"claim_numbers: {numbers}\n")
            file.write(f"missing_claim_numbers: {missing}\n")
            file.write(f"low_or_warned_claim_chunks: {len(low_conf)}\n")


def write_section_audit(chunks: list[JsonDict], output_dir: Path) -> None:
    output_path = output_dir / "section_chunk_counts.txt"
    docs = sorted({chunk.get("document_id") for chunk in chunks})
    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Section and Chunk-Type Counts\n")
        for document_id in docs:
            file.write("\n" + str(document_id) + "\n")
            file.write("-" * len(str(document_id)) + "\n")
            counter = collections.Counter(
                (chunk.get("section", "unknown"), chunk.get("chunk_type", "unknown"), chunk.get("embeddable", True))
                for chunk in chunks
                if chunk.get("document_id") == document_id
            )
            for (section, chunk_type, embeddable), count in sorted(counter.items()):
                file.write(f"{section:35s} {chunk_type:35s} embeddable={str(embeddable):5s} {count:5d}\n")


def write_suspicious_chunk_audit(chunks: list[JsonDict], output_dir: Path) -> None:
    output_path = output_dir / "suspicious_chunks.txt"
    suspicious = [
        chunk for chunk in chunks
        if chunk.get("quality_flags") or chunk.get("metadata", {}).get("quality_flags")
    ]
    suspicious.sort(key=lambda chunk: (chunk.get("document_id", ""), chunk.get("page_start") or 0, chunk.get("chunk_type", "")))
    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Suspicious OCR / Chunk Audit\n")
        file.write(f"# suspicious_chunks: {len(suspicious)}\n")
        for chunk in suspicious:
            file.write("\n" + "=" * 120 + "\n")
            file.write(
                f"{chunk.get('document_id')} | {chunk.get('chunk_type')} | {chunk.get('section')} | "
                f"p{chunk.get('page_start')}-{chunk.get('page_end')} | embeddable={chunk.get('embeddable')} | "
                f"flags={chunk.get('quality_flags')}\n"
            )
            file.write("-" * 120 + "\n")
            file.write(chunk.get("text", "")[:1500])
            file.write("\n")


def write_parser_issues(quality_rows: list[JsonDict], chunks: list[JsonDict], output_dir: Path) -> None:
    output_path = output_dir / "parser_issues.md"
    p0: list[str] = []
    p1: list[str] = []
    p2: list[str] = []

    if any(chunk.get("chunk_type") == "metadata" and chunk.get("embeddable") for chunk in chunks):
        p0.append("Metadata chunks are embeddable; they should be filter/display records only.")

    for row in quality_rows:
        document_id = row.get("document_id")
        if not row.get("claims_found"):
            p0.append(f"{document_id}: no claim chunks produced.")
        if row.get("claim_numbers_detected") and row.get("claim_numbers_detected", [None])[0] != 1:
            p0.append(f"{document_id}: claim sequence does not start at claim 1.")
        if row.get("missing_claim_numbers"):
            p1.append(f"{document_id}: missing claim numbers {row.get('missing_claim_numbers')}.")
        if not row.get("abstract_found"):
            p1.append(f"{document_id}: abstract was not detected.")
        if row.get("low_confidence_claim_count", 0) > 0:
            p1.append(f"{document_id}: {row.get('low_confidence_claim_count')} claim chunks have extraction warnings.")
        if row.get("tables_detected"):
            p2.append(f"{document_id}: tables detected; use table enrichment later for exact values.")
        if row.get("figures_detected"):
            p2.append(f"{document_id}: figures detected; use figure/caption extraction later.")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Parser Issues\n\n")
        file.write("## P0 — Must fix before embeddings\n")
        file.write("- [x] No P0 issues detected.\n" if not p0 else "".join(f"- [ ] {item}\n" for item in sorted(set(p0))))
        file.write("\n## P1 — Should fix before report generation\n")
        file.write("- [x] No P1 issues detected.\n" if not p1 else "".join(f"- [ ] {item}\n" for item in sorted(set(p1))))
        file.write("\n## P2 — Later enrichment\n")
        file.write("- [x] No P2 issues detected.\n" if not p2 else "".join(f"- [ ] {item}\n" for item in sorted(set(p2))))


def write_audit_summary(quality_rows: list[JsonDict], chunks: list[JsonDict], output_dir: Path) -> None:
    output_path = output_dir / "audit_summary.txt"
    chunk_type_counter = collections.Counter(chunk.get("chunk_type", "unknown") for chunk in chunks)
    section_counter = collections.Counter(chunk.get("section", "unknown") for chunk in chunks)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("Parser Audit Summary\n====================\n\n")
        file.write(f"documents: {len(quality_rows)}\n")
        file.write(f"pages_extracted: {sum(int(row.get('pages_extracted') or 0) for row in quality_rows)}\n")
        file.write(f"paragraphs: {sum(int(row.get('paragraph_count') or 0) for row in quality_rows)}\n")
        file.write(f"chunks: {len(chunks)}\n")
        file.write(f"embeddable_chunks: {sum(1 for chunk in chunks if chunk.get('embeddable'))}\n")
        file.write(f"warnings: {sum(len(row.get('warnings', [])) for row in quality_rows)}\n")
        file.write(f"missing_abstracts: {[row.get('document_id') for row in quality_rows if not row.get('abstract_found')]}\n")
        file.write(f"missing_claims: {[row.get('document_id') for row in quality_rows if not row.get('claims_found')]}\n")
        file.write("\nChunk types\n-----------\n")
        for chunk_type, count in chunk_type_counter.most_common():
            file.write(f"{chunk_type}: {count}\n")
        file.write("\nSections\n--------\n")
        for section, count in section_counter.most_common():
            file.write(f"{section}: {count}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit parsed patent JSONL outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/parsed_patents"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/parser_audit"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = read_jsonl(args.input_dir / "quality_report.jsonl")
    chunks = read_jsonl(args.input_dir / "chunks.jsonl")

    write_quality_summary(quality_rows, args.output_dir)
    write_metadata_audit(chunks, args.output_dir)
    write_claim_audit(chunks, args.output_dir)
    write_section_audit(chunks, args.output_dir)
    write_suspicious_chunk_audit(chunks, args.output_dir)
    write_parser_issues(quality_rows, chunks, args.output_dir)
    write_audit_summary(quality_rows, chunks, args.output_dir)

    print(f"Wrote audit outputs to: {args.output_dir}")


def plausible_claim_numbers(numbers: list[int]) -> list[int]:
    """Remove OCR outlier claim numbers before computing missing claims."""
    if not numbers:
        return []

    sorted_numbers = sorted(set(numbers))

    # Most PCT claim sets in this project are below 50.
    # A sudden jump from 10 to 88 is almost certainly OCR noise.
    cleaned: list[int] = []
    for number in sorted_numbers:
        if number <= 50:
            cleaned.append(number)
            continue

        previous = max(cleaned) if cleaned else 0
        if number <= previous + 5:
            cleaned.append(number)

    return cleaned


if __name__ == "__main__":
    main()
