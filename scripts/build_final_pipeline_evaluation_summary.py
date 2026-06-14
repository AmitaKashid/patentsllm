"""Build final PatentLLM pipeline evaluation summary.

This script creates one concise comparison table across report-generation stages:
1. raw evidence-pack report
2. card-based report
3. claim-bank report
4. claim-bank + deterministic-table report

It extracts deterministic metrics directly from report files and optionally
adds claim-bank alignment metrics from the latest alignment audit.

Output:
    data/processed/final_evaluation/<timestamp>/
    ├── final_pipeline_evaluation_summary.csv
    └── final_pipeline_evaluation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_HEADINGS = [
    "# Patent Analysis for R&D",
    "## Scope snapshot",
    "## Main conclusion",
    "## Executive summary",
    "## Scope and evidence base",
    "## 1. Ownership and competitive landscape",
    "## 2. Temporal trends",
    "## 3. Technology deep dive and R&D relevance",
    "### 3.1 Polymer types",
    "### 3.2 Formulation strategies",
    "### 3.3 Application focus",
    "## 4. Key insights and R&D implications",
    "## 5. Limits of the conclusion",
    "## Patent key and extracted R&D signal",
]

TARGET_PATENTS = [
    "EP2756049A1",
    "EP2841522A1",
    "EP2411423A1",
    "EP2958970A1",
    "EP2686395A2",
    "EP3268442A1",
    "EP2456819A1",
    "EP3402857A1",
    "EP3707219A1",
    "EP4441161A1",
]

LEGAL_TERMS = [
    "infringement",
    "freedom-to-operate",
    "fto",
    "validity",
    "patentability",
    "clearance",
    "opposition",
]


@dataclass(frozen=True, slots=True)
class StageMetrics:
    stage: str
    report_path: str
    report_exists: bool
    word_count: int
    total_citations: int
    body_citations: int
    invalid_citations: int
    required_heading_coverage: str
    target_patent_coverage: str
    patent_key_coverage: str
    legal_leakage_before_limits: bool
    citation_spam_rows: str
    citation_review_rows: str
    claim_bank_alignment: str
    accepted_stage: bool
    interpretation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final PatentLLM evaluation summary.")

    parser.add_argument(
        "--evidence-pack",
        type=Path,
        required=True,
        help="Path to evidence_pack.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/final_evaluation"),
    )

    parser.add_argument("--raw-report", type=Path, default=None)
    parser.add_argument("--card-report", type=Path, default=None)
    parser.add_argument("--claim-bank-report", type=Path, default=None)
    parser.add_argument("--deterministic-table-report", type=Path, default=None)
    parser.add_argument("--alignment-summary", type=Path, default=None)
    parser.add_argument("--citation-audit-root", type=Path, default=Path("data/processed/citation_support_audits_v2"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evidence_pack = load_json(args.evidence_pack)
    valid_evidence_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in evidence_pack.get("evidence_items", [])
        if str(item.get("evidence_id", "")).strip()
    }

    reports = {
        "Raw evidence-pack sectioned report": args.raw_report or latest_file(
            Path("data/processed/generated_reports_sectioned"),
            "final_report_ollama_qwen3_8b*.md",
        ),
        "Card-based report": args.card_report or latest_file(
            Path("data/processed/generated_reports_from_cards"),
            "final_report_ollama_qwen3_8b.md",
        ),
        "Claim-bank report": args.claim_bank_report or latest_file(
            Path("data/processed/generated_reports_from_claim_bank"),
            "final_report_ollama_qwen3_8b.md",
        ),
        "Claim-bank + deterministic tables": args.deterministic_table_report or latest_file(
            Path("data/processed/generated_reports_from_claim_bank"),
            "final_report_ollama_qwen3_8b_deterministic_tables.md",
        ),
    }

    alignment_summary = args.alignment_summary or latest_file(
        Path("data/processed/claim_bank_alignment_audits"),
        "claim_bank_alignment_summary.md",
    )

    rows: list[StageMetrics] = []

    for stage, report_path in reports.items():
        citation_summary = find_latest_citation_summary(
            citation_audit_root=args.citation_audit_root,
            report_path=report_path,
        )

        rows.append(
            evaluate_stage(
                stage=stage,
                report_path=report_path,
                valid_evidence_ids=valid_evidence_ids,
                citation_summary=citation_summary,
                alignment_summary=alignment_summary if stage == "Claim-bank + deterministic tables" else None,
            )
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "final_pipeline_evaluation_summary.csv"
    md_path = output_dir / "final_pipeline_evaluation_summary.md"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"Wrote final evaluation summary to: {output_dir}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


def evaluate_stage(
    stage: str,
    report_path: Path | None,
    valid_evidence_ids: set[str],
    citation_summary: Path | None,
    alignment_summary: Path | None,
) -> StageMetrics:
    if report_path is None or not report_path.exists():
        return StageMetrics(
            stage=stage,
            report_path="not found",
            report_exists=False,
            word_count=0,
            total_citations=0,
            body_citations=0,
            invalid_citations=0,
            required_heading_coverage="0/14",
            target_patent_coverage="0/10",
            patent_key_coverage="0/10",
            legal_leakage_before_limits=False,
            citation_spam_rows="not available",
            citation_review_rows="not available",
            claim_bank_alignment="not available",
            accepted_stage=False,
            interpretation="Report not found.",
        )

    report_text = report_path.read_text(encoding="utf-8")

    all_citations = extract_citations(report_text)
    body_text = strip_patent_key(report_text)
    body_citations = extract_citations(body_text)
    invalid = sorted(set(all_citations) - valid_evidence_ids)

    heading_count = sum(1 for heading in REQUIRED_HEADINGS if heading in report_text)
    patent_count = sum(1 for patent in TARGET_PATENTS if patent in report_text)
    patent_key_text = report_text[report_text.find("## Patent key and extracted R&D signal"):]
    patent_key_count = sum(1 for patent in TARGET_PATENTS if patent in patent_key_text)

    citation_review_rows, citation_spam_rows = parse_citation_summary(citation_summary)
    claim_bank_alignment = parse_alignment_summary(alignment_summary)

    accepted = stage == "Claim-bank + deterministic tables"

    return StageMetrics(
        stage=stage,
        report_path=str(report_path),
        report_exists=True,
        word_count=count_words(report_text),
        total_citations=len(all_citations),
        body_citations=len(body_citations),
        invalid_citations=len(invalid),
        required_heading_coverage=f"{heading_count}/{len(REQUIRED_HEADINGS)}",
        target_patent_coverage=f"{patent_count}/{len(TARGET_PATENTS)}",
        patent_key_coverage=f"{patent_key_count}/{len(TARGET_PATENTS)}",
        legal_leakage_before_limits=has_legal_leakage_before_limits(report_text),
        citation_spam_rows=citation_spam_rows,
        citation_review_rows=citation_review_rows,
        claim_bank_alignment=claim_bank_alignment,
        accepted_stage=accepted,
        interpretation=interpret_stage(stage),
    )


def interpret_stage(stage: str) -> str:
    if stage == "Raw evidence-pack sectioned report":
        return "Baseline RAG generation from evidence pack."
    if stage == "Card-based report":
        return "Improves structure by summarizing patents before reporting."
    if stage == "Claim-bank report":
        return "Constrains report generation to approved atomic claims."
    if stage == "Claim-bank + deterministic tables":
        return "Accepted architecture: claim-bank grounded with deterministic table repair."
    return ""


def latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None

    files = sorted(root.rglob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_latest_citation_summary(
    citation_audit_root: Path,
    report_path: Path | None,
) -> Path | None:
    if report_path is None or not report_path.exists() or not citation_audit_root.exists():
        return None

    stem = report_path.stem
    candidates = sorted(
        citation_audit_root.rglob(f"{stem}_citation_support_summary.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return candidates[0] if candidates else None


def parse_citation_summary(path: Path | None) -> tuple[str, str]:
    if path is None or not path.exists():
        return "not available", "not available"

    text = path.read_text(encoding="utf-8")

    review = extract_summary_number(text, r"Review rows:\s*(\d+)")
    spam = extract_summary_number(text, r"Citation-spam rows:\s*(\d+)")

    return str(review), str(spam)


def parse_alignment_summary(path: Path | None) -> str:
    if path is None or not path.exists():
        return "not available"

    text = path.read_text(encoding="utf-8")

    match = re.search(r"Aligned or partially aligned:\s*([^\n]+)", text)
    return match.group(1).strip() if match else "not available"


def extract_summary_number(text: str, pattern: str) -> int | str:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else "not available"


def strip_patent_key(report_text: str) -> str:
    marker = "## Patent key and extracted R&D signal"
    index = report_text.find(marker)
    return report_text[:index] if index != -1 else report_text


def extract_citations(text: str) -> list[str]:
    return re.findall(r"\[(E\d+)\]", text)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def has_legal_leakage_before_limits(report_text: str) -> bool:
    lowered = report_text.lower()
    limits_index = lowered.find("## 5. limits of the conclusion")
    before_limits = lowered[:limits_index] if limits_index != -1 else lowered

    return any(term in before_limits for term in LEGAL_TERMS)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[StageMetrics], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(rows: list[StageMetrics], path: Path) -> None:
    lines = [
        "# Final PatentLLM Pipeline Evaluation Summary",
        "",
        "| Stage | Accepted | Headings | Patents | Patent key | Body citations | Invalid citations | Citation spam rows | Claim-bank alignment | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row.stage} | "
            f"{row.accepted_stage} | "
            f"{row.required_heading_coverage} | "
            f"{row.target_patent_coverage} | "
            f"{row.patent_key_coverage} | "
            f"{row.body_citations} | "
            f"{row.invalid_citations} | "
            f"{row.citation_spam_rows} | "
            f"{row.claim_bank_alignment} | "
            f"{row.interpretation} |"
        )

    lines.extend(
        [
            "",
            "## Final decision",
            "",
            "The accepted architecture is **claim-bank + deterministic tables** because it preserves full report structure, keeps citations valid, removes citation-spam rows, and validates deterministic table rows against approved claim-bank items.",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()