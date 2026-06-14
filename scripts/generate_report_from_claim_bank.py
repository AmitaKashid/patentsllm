"""Generate an R&D patent report from an approved atomic claim bank.

This generator uses only approved claim-bank items. The goal is to reduce
citation mismatch by forcing the model to write from pre-approved
claim/evidence pairs instead of free-form patent cards.

Input:
    claim_bank.json

Output:
    data/processed/generated_reports_from_claim_bank/<run_id>/
    ├── final_report_<model>.md
    ├── prompt_used.md
    ├── raw_response_<model>.json
    ├── metadata.json
    └── section_generation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


TARGET_PUBLICATIONS = [
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


PATENT_KEY_HEADING = "## Patent key and extracted R&D signal"


@dataclass(frozen=True, slots=True)
class ClaimReportRunRecord:
    run_id: str
    evidence_pack_id: str
    llm_spec: str
    section_id: str
    section_title: str
    success: bool
    latency_seconds: float
    prompt_path: str
    raw_response_path: str
    section_path: str
    prompt_eval_count: int | None
    eval_count: int | None
    done_reason: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patent R&D report from approved claim bank."
    )

    parser.add_argument("--claim-bank-json", type=Path, required=True)
    parser.add_argument("--llm", type=str, default="ollama:qwen3:8b")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/generated_reports_from_claim_bank"),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=12000)
    parser.add_argument("--num-predict", type=int, default=4200)
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--evidence-pack-id", type=str, default="claim_bank_report")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    claims = load_claim_bank(args.claim_bank_json)
    approved_claims = [claim for claim in claims if claim.get("approved_for_report") is True]

    if not approved_claims:
        raise RuntimeError("No approved claims found in claim bank.")

    backend, model_name = parse_llm_spec(args.llm)
    if backend != "ollama":
        raise ValueError("Only ollama:<model> is currently supported.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = safe_filename(args.llm)

    run_dir = args.output_dir / f"{run_id}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(approved_claims)

    prompt_path = run_dir / "prompt_used.md"
    raw_response_path = run_dir / f"raw_response_{safe_model}.json"
    final_report_path = run_dir / f"final_report_{safe_model}.md"
    summary_path = run_dir / "section_generation_summary.csv"

    prompt_path.write_text(prompt, encoding="utf-8")

    started = time.perf_counter()

    try:
        report_text, raw_data = call_ollama(
            base_url=args.ollama_base_url,
            model_name=model_name,
            prompt=prompt,
            raw_response_path=raw_response_path,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )

        latency_seconds = round(time.perf_counter() - started, 2)
        done_reason = str(raw_data.get("done_reason", ""))
        prompt_eval_count = safe_int(raw_data.get("prompt_eval_count"))
        eval_count = safe_int(raw_data.get("eval_count"))

        if done_reason.lower() == "length":
            raise RuntimeError(
                "Ollama stopped because output/context length was exhausted. "
                f"prompt_eval_count={prompt_eval_count}, eval_count={eval_count}."
            )

        final_report = append_or_replace_patent_key_from_claims(
            report_text=report_text,
            claims=approved_claims,
        )

        final_report_path.write_text(final_report.strip() + "\n", encoding="utf-8")

        record = ClaimReportRunRecord(
            run_id=run_id,
            evidence_pack_id=args.evidence_pack_id,
            llm_spec=args.llm,
            section_id="claim_bank_full_report",
            section_title="Claim-bank full report",
            success=True,
            latency_seconds=latency_seconds,
            prompt_path=str(prompt_path),
            raw_response_path=str(raw_response_path),
            section_path=str(final_report_path),
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            done_reason=done_reason,
            error="",
        )

    except Exception as exc:
        latency_seconds = round(time.perf_counter() - started, 2)
        raw_data = load_json_if_exists(raw_response_path)

        record = ClaimReportRunRecord(
            run_id=run_id,
            evidence_pack_id=args.evidence_pack_id,
            llm_spec=args.llm,
            section_id="claim_bank_full_report",
            section_title="Claim-bank full report",
            success=False,
            latency_seconds=latency_seconds,
            prompt_path=str(prompt_path),
            raw_response_path=str(raw_response_path),
            section_path=str(final_report_path),
            prompt_eval_count=safe_int(raw_data.get("prompt_eval_count")),
            eval_count=safe_int(raw_data.get("eval_count")),
            done_reason=str(raw_data.get("done_reason", "")),
            error=f"{type(exc).__name__}: {exc}",
        )

    write_summary_csv([record], summary_path)
    write_metadata(
        path=run_dir / "metadata.json",
        args=args,
        run_id=run_id,
        approved_claim_count=len(approved_claims),
        final_report_path=final_report_path,
        summary_path=summary_path,
        record=record,
    )

    print(f"\nWrote claim-bank report outputs to: {run_dir}")
    print(f"Success: {record.success}")
    print(f"Final report: {final_report_path}")
    print(f"Summary: {summary_path}")

    if record.error:
        print(f"Error: {record.error}")


def build_prompt(claims: list[dict[str, Any]]) -> str:
    claim_block = format_claims_for_prompt(claims)

    return f"""You are a senior patent strategy and R&D technology intelligence analyst.

You are generating an R&D-focused patent analysis report using an approved atomic claim bank.

Critical grounding rules:
- Use only the approved claims below.
- Every factual or technical sentence must be based on one or more claim IDs.
- Every factual or technical sentence must cite the exact evidence IDs from the claim used.
- Do not attach an evidence ID unless it appears in the same claim item.
- Do not cite multiple claims in one sentence unless the sentence explicitly compares those claims.
- Use no more than 2 evidence IDs per sentence.
- Do not invent owners, years, metrics, technical effects, applications, or trends.
- Do not write broad strategy claims unless they are directly supported by claim-bank items.
- Do not generate a patent-key table. It will be appended deterministically by code.
- Do not mention claim-bank IDs in the final report. Use evidence IDs only, like [E23].
- Do not write legal, infringement, validity, patentability, clearance, opposition, or freedom-to-operate conclusions.
- Write for senior R&D stakeholders.

Fixed dataset facts:
- The report is based on exactly 10 patent publications.
- Henkel owns 6 patents in this set.
- Bostik owns 4 patents in this set.
- Priority window: 2009-2021.
- Target publications: {", ".join(TARGET_PUBLICATIONS)}.

Required output structure:

# Patent Analysis for R&D
Technology intelligence brief on selected hot-melt adhesive patent publications

## Scope snapshot
Write exactly four bullets:
- Scope:
- Owners:
- Priority window:
- Main technology cluster:

## Main conclusion
Write one paragraph. Every technical claim must cite direct evidence.

## Executive summary
Write three short paragraphs:
1. Patent-set coverage.
2. Ownership and technology movement.
3. R&D next step.

## Scope and evidence base
Write three short paragraphs:
- Fact.
- Method.
- Interpretation.

## 1. Ownership and competitive landscape
Write:
- one Fact paragraph
- one Interpretation paragraph
- one markdown table:
Owner signal | Evidence in this set | What it means | R&D relevance

## 2. Temporal trends
Write:
- one Fact paragraph
- one Interpretation paragraph
- one markdown table:
Time signal | Evidence | Interpretation | R&D implication

## 3. Technology deep dive and R&D relevance

### 3.1 Polymer types
Group patents into chemistry routes. For each route:
Fact:
Interpretation:
R&D relevance:

### 3.2 Formulation strategies
Create a markdown table:
Repeated problem | Design lever | Evidence patents | R&D experiment or action

### 3.3 Application focus
Explain application areas and success metrics only when supported by claim-bank items.

## 4. Key insights and R&D implications
Create a markdown table:
Action | Evidence behind it | Technical risk to control

The risk column must contain only technical risks such as viscosity, odour, colour, creep, peel strength, wet adhesion, substrate compatibility, processability, thermal stability, coating quality, bleed-through, formulation stability, or sensory profile.

Then write one paragraph identifying the immediate technical next step for R&D.

## 5. Limits of the conclusion
Write 2 concise paragraphs:
- dataset and claim-bank limitations
- not legal advice / FTO limitation

Approved atomic claim bank:
{claim_block}

Generate the final report now.
"""


def format_claims_for_prompt(claims: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for claim in claims:
        grouped[str(claim.get("report_section", "other"))].append(claim)

    lines: list[str] = []

    for section in [
        "technology_deep_dive",
        "formulation_strategies",
        "application_focus",
        "rd_implications",
    ]:
        section_claims = grouped.get(section, [])
        if not section_claims:
            continue

        lines.append(f"\n[{section}]")

        for claim in section_claims:
            evidence_ids = claim.get("evidence_ids", [])
            if not isinstance(evidence_ids, list):
                evidence_ids = []

            evidence = ", ".join(f"[{eid}]" for eid in evidence_ids)

            lines.append(
                f"- {claim.get('claim_id')}: "
                f"patent={claim.get('patent_publication')}; "
                f"owner={claim.get('owner')}; "
                f"priority={claim.get('priority_year')}; "
                f"type={claim.get('claim_type')}; "
                f"claim={claim.get('claim_text')}; "
                f"evidence={evidence}"
            )

    return "\n".join(lines).strip()


def call_ollama(
    base_url: str,
    model_name: str,
    prompt: str,
    raw_response_path: Path,
    temperature: float,
    num_ctx: int,
    num_predict: int,
) -> tuple[str, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write grounded patent intelligence reports for R&D. "
                    "Use only approved atomic claims and their evidence IDs."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    response = requests.post(url, json=payload, timeout=1800)
    response.raise_for_status()

    data = response.json()

    raw_response_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    content = str(data.get("message", {}).get("content", "")).strip()

    if not content:
        raise RuntimeError(f"Ollama returned empty response. Raw: {raw_response_path}")

    return content, data


def append_or_replace_patent_key_from_claims(
    report_text: str,
    claims: list[dict[str, Any]],
) -> str:
    cleaned = remove_patent_key_artifacts(report_text).rstrip()
    patent_key = build_patent_key_from_claims(claims)

    return cleaned + "\n\n" + patent_key.strip() + "\n"


def build_patent_key_from_claims(claims: list[dict[str, Any]]) -> str:
    by_publication: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for claim in claims:
        by_publication[str(claim.get("patent_publication", ""))].append(claim)

    lines = [
        PATENT_KEY_HEADING,
        "",
        "| # | Patent | Owner | Priority | Core technology | Main R&D relevance | Evidence |",
        "|---:|---|---|---:|---|---|---|",
    ]

    for index, publication in enumerate(TARGET_PUBLICATIONS, start=1):
        patent_claims = by_publication.get(publication, [])

        owner = first_non_empty([str(claim.get("owner", "")) for claim in patent_claims])
        priority = first_non_empty([str(claim.get("priority_year", "")) for claim in patent_claims])

        core_claim = first_claim_by_type(patent_claims, "core_technology")
        rd_claim = first_claim_by_type(patent_claims, "rd_relevance")

        if not rd_claim:
            rd_claim = first_claim_by_type(patent_claims, "performance_evidence")

        evidence_refs = select_evidence_refs([core_claim, rd_claim])

        lines.append(
            "| "
            f"{index} | "
            f"{escape_md(publication)} | "
            f"{escape_md(owner)} | "
            f"{escape_md(priority)} | "
            f"{escape_md(core_claim.get('claim_text', 'not available'))} | "
            f"{escape_md(rd_claim.get('claim_text', 'not available'))} | "
            f"{escape_md(evidence_refs)} |"
        )

    return "\n".join(lines)


def first_claim_by_type(
    claims: list[dict[str, Any]],
    claim_type: str,
) -> dict[str, Any]:
    for claim in claims:
        if claim.get("claim_type") == claim_type:
            return claim
    return {}


def select_evidence_refs(claims: list[dict[str, Any]], max_ids: int = 4) -> str:
    selected: list[str] = []
    seen: set[str] = set()

    for claim in claims:
        if not claim:
            continue

        evidence_ids = claim.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            continue

        for evidence_id in evidence_ids:
            evidence_id = str(evidence_id).strip()
            if not evidence_id or evidence_id in seen:
                continue

            selected.append(evidence_id)
            seen.add(evidence_id)

            if len(selected) >= max_ids:
                break

    return ", ".join(f"[{eid}]" for eid in selected) if selected else "not available"


def remove_patent_key_artifacts(report_text: str) -> str:
    proper_patent_key_pattern = re.compile(
        r"\n##\s+Patent\s+key\s+and\s+extracted\s+R&D\s+signal\b.*\Z",
        flags=re.IGNORECASE | re.DOTALL,
    )

    malformed_key_pattern = re.compile(
        r"\n#\s*\|\s*Patent\s*\|\s*Owner\s*\|\s*Priority\s*\|\s*Core\s+technology\s*\|\s*Main\s+R&D\s+relevance\b.*?\Z",
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned = re.sub(proper_patent_key_pattern, "", report_text)
    cleaned = re.sub(malformed_key_pattern, "", cleaned)

    return cleaned.rstrip()


def load_claim_bank(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Claim-bank JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise TypeError("Expected claim-bank JSON to contain a list.")

    return data


def write_summary_csv(records: list[ClaimReportRunRecord], path: Path) -> None:
    fieldnames = list(asdict(records[0]).keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    run_id: str,
    approved_claim_count: int,
    final_report_path: Path,
    summary_path: Path,
    record: ClaimReportRunRecord,
) -> None:
    metadata = {
        "run_id": run_id,
        "llm": args.llm,
        "claim_bank_json": str(args.claim_bank_json),
        "approved_claim_count": approved_claim_count,
        "output_dir": str(final_report_path.parent),
        "final_report_path": str(final_report_path),
        "summary_path": str(summary_path),
        "success": record.success,
        "error": record.error,
    }

    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_llm_spec(raw_spec: str) -> tuple[str, str]:
    if ":" not in raw_spec:
        raise ValueError("Expected LLM spec format like ollama:qwen3:8b.")

    backend, model_name = raw_spec.split(":", 1)

    return backend.strip(), model_name.strip()


def first_non_empty(values: list[str]) -> str:
    for value in values:
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return "not available"


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def safe_filename(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


if __name__ == "__main__":
    main()