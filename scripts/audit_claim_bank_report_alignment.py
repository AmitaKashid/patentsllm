"""Audit deterministic report tables against the approved claim bank.

Purpose:
The normal citation-support audit checks report rows against evidence snippets.
For deterministic claim-bank tables, the stronger check is:
- Does the table row mention a patent from the claim bank?
- Do the cited evidence IDs belong to approved claims for that patent?
- Is the row aligned with at least one approved claim text?
- Are invalid or non-approved evidence IDs used?

Output:
    data/processed/claim_bank_alignment_audits/<timestamp>/
    ├── claim_bank_alignment_audit.csv
    └── claim_bank_alignment_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PATENT_KEY_HEADING = "## Patent key and extracted R&D signal"


@dataclass(frozen=True, slots=True)
class AlignmentRow:
    row_id: str
    section_heading: str
    report_row: str
    patent_publication: str
    cited_evidence_ids: str
    approved_claim_ids: str
    approved_claim_types: str
    alignment_label: str
    issue: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit report table rows against claim bank.")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--claim-bank-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/claim_bank_alignment_audits"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_text = args.report_path.read_text(encoding="utf-8")
    claim_bank = load_claim_bank(args.claim_bank_json)

    report_body = strip_patent_key(report_text)
    table_rows = extract_cited_table_rows(report_body)

    approved_claims = [
        claim for claim in claim_bank
        if claim.get("approved_for_report") is True
    ]

    index = build_claim_index(approved_claims)

    rows = []
    for idx, row in enumerate(table_rows, start=1):
        rows.append(
            audit_row(
                row_id=f"R{idx:03d}",
                table_row=row,
                claim_index=index,
            )
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "claim_bank_alignment_audit.csv"
    summary_path = output_dir / "claim_bank_alignment_summary.md"

    write_csv(rows, csv_path)
    write_summary(rows, summary_path, args.report_path)

    print(f"Wrote alignment audit to: {csv_path}")
    print(f"Wrote summary to: {summary_path}")


def load_claim_bank(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise TypeError("Expected claim-bank JSON to contain a list.")

    return data


def strip_patent_key(report_text: str) -> str:
    index = report_text.find(PATENT_KEY_HEADING)
    return report_text[:index].rstrip() if index != -1 else report_text


def extract_cited_table_rows(report_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_heading = ""

    for raw_line in report_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            current_heading = line
            continue

        if not is_markdown_table_row(line):
            continue

        if is_table_separator(line):
            continue

        if not extract_evidence_ids(line):
            continue

        rows.append(
            {
                "section_heading": current_heading,
                "row_text": line,
            }
        )

    return rows


def build_claim_index(
    approved_claims: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_patent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for claim in approved_claims:
        patent = str(claim.get("patent_publication", "")).strip()
        by_patent[patent].append(claim)

        evidence_ids = claim.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            continue

        for evidence_id in evidence_ids:
            by_evidence[str(evidence_id).strip()].append(claim)

    return {
        "by_patent": by_patent,
        "by_evidence": by_evidence,
    }


def audit_row(
    row_id: str,
    table_row: dict[str, str],
    claim_index: dict[str, dict[str, Any]],
) -> AlignmentRow:
    row_text = table_row["row_text"]
    section_heading = table_row["section_heading"]

    patent = extract_patent_publication(row_text)
    evidence_ids = extract_evidence_ids(row_text)

    by_evidence = claim_index["by_evidence"]
    by_patent = claim_index["by_patent"]

    matched_claims: list[dict[str, Any]] = []

    for evidence_id in evidence_ids:
        for claim in by_evidence.get(evidence_id, []):
            if patent and claim.get("patent_publication") != patent:
                continue
            matched_claims.append(claim)

    matched_claims = dedupe_claims(matched_claims)

    if not patent:
        label = "Needs review"
        issue = "No patent publication detected in row."
    elif patent not in by_patent:
        label = "Invalid"
        issue = "Patent not found in approved claim bank."
    elif not matched_claims:
        label = "Citation irrelevant"
        issue = "Cited evidence IDs do not map to approved claims for this patent."
    else:
        aligned = any(
            text_overlap_score(row_text, str(claim.get("claim_text", ""))) >= 0.35
            for claim in matched_claims
        )

        if aligned:
            label = "Aligned"
            issue = ""
        else:
            label = "Partially aligned"
            issue = "Evidence belongs to approved claim, but row text is broader than claim text."

    claim_ids = [str(claim.get("claim_id", "")) for claim in matched_claims]
    claim_types = [str(claim.get("claim_type", "")) for claim in matched_claims]

    return AlignmentRow(
        row_id=row_id,
        section_heading=section_heading,
        report_row=row_text,
        patent_publication=patent,
        cited_evidence_ids=", ".join(evidence_ids),
        approved_claim_ids=", ".join(claim_ids),
        approved_claim_types=", ".join(claim_types),
        alignment_label=label,
        issue=issue,
    )


def dedupe_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for claim in claims:
        claim_id = str(claim.get("claim_id", ""))
        if claim_id in seen:
            continue
        seen.add(claim_id)
        out.append(claim)

    return out


def extract_patent_publication(text: str) -> str:
    match = re.search(r"EP\d+[A-Z]\d?", text)
    return match.group(0) if match else ""


def extract_evidence_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []

    for evidence_id in re.findall(r"\[(E\d+)\]", text):
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        ids.append(evidence_id)

    return ids


def text_overlap_score(a: str, b: str) -> float:
    a_terms = important_terms(a)
    b_terms = important_terms(b)

    if not a_terms or not b_terms:
        return 0.0

    return len(a_terms & b_terms) / max(1, len(b_terms))


def important_terms(text: str) -> set[str]:
    stopwords = {
        "the", "and", "with", "for", "from", "that", "this", "into", "such",
        "using", "used", "uses", "where", "which", "than", "then", "their",
        "about", "claim", "patent", "evidence", "action", "risk", "test",
        "run", "compare", "measure", "screen", "composition", "adhesive",
    }

    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9°δ]+", text)
        if len(token) >= 4 and token.lower() not in stopwords
    }

    return terms


def is_markdown_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|")


def is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|", line))


def write_csv(rows: list[AlignmentRow], path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "row_id",
        "section_heading",
        "report_row",
        "patent_publication",
        "cited_evidence_ids",
        "approved_claim_ids",
        "approved_claim_types",
        "alignment_label",
        "issue",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def write_summary(rows: list[AlignmentRow], path: Path, report_path: Path) -> None:
    counts = Counter(row.alignment_label for row in rows)
    total = len(rows)

    with path.open("w", encoding="utf-8") as file:
        file.write("# Claim-Bank Alignment Audit Summary\n\n")
        file.write(f"Report: `{report_path}`\n\n")
        file.write(f"- Rows checked: {total}\n")
        for label, count in counts.most_common():
            percent = count / total * 100 if total else 0
            file.write(f"- {label}: {count} ({percent:.1f}%)\n")

        aligned = counts["Aligned"] + counts["Partially aligned"]
        percent_aligned = aligned / total * 100 if total else 0

        file.write("\n")
        file.write(f"Aligned or partially aligned: {aligned}/{total} ({percent_aligned:.1f}%)\n")


if __name__ == "__main__":
    main()