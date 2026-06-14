"""Generate an R&D patent report from repaired refined patent cards.

This script generates the final report from analyst-ready patent cards instead
of raw evidence chunks. The goal is to improve faithfulness and citation support
by giving the LLM cleaner, structured, evidence-backed intermediate facts.

Input:
    patent_cards_refined_repaired.json

Output:
    data/processed/generated_reports_from_cards/<run_id>/
    ├── final_report_<model>.md
    ├── draft_report_<model>.md
    ├── prompt_used.md
    ├── raw_response_<model>.json
    ├── metadata.json
    └── section_generation_summary.csv

The output folder intentionally contains section_generation_summary.csv so that
the existing benchmark_llm_reports.py script can benchmark card-based reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PATENT_KEY_HEADING = "## Patent key and extracted R&D signal"


@dataclass(frozen=True, slots=True)
class CardReportRunRecord:
    """Benchmark-compatible generation record."""

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
        description="Generate patent R&D report from repaired refined patent cards."
    )

    parser.add_argument("--cards-json", type=Path, required=True)
    parser.add_argument("--llm", type=str, default="ollama:qwen3:8b")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/generated_reports_from_cards"),
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-ctx", type=int, default=12000)
    parser.add_argument("--num-predict", type=int, default=4500)
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--evidence-pack-id", type=str, default="card_based_report")
    parser.add_argument("--max-card-chars", type=int, default=1200)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cards = load_cards(args.cards_json)
    validate_cards_for_generation(cards)

    backend, model_name = parse_llm_spec(args.llm)
    if backend != "ollama":
        raise ValueError("This script currently supports only ollama:<model> specs.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = safe_filename(args.llm)

    run_dir = args.output_dir / f"{run_id}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_report_prompt(cards=cards, max_card_chars=args.max_card_chars)

    prompt_path = run_dir / "prompt_used.md"
    raw_response_path = run_dir / f"raw_response_{safe_model}.json"
    draft_report_path = run_dir / f"draft_report_{safe_model}.md"
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
                f"prompt_eval_count={prompt_eval_count}, eval_count={eval_count}, "
                f"num_ctx={args.num_ctx}, num_predict={args.num_predict}."
            )

        draft_report_path.write_text(report_text.strip() + "\n", encoding="utf-8")

        final_report = append_or_replace_patent_key_from_cards(
            report_text=report_text,
            cards=cards,
        )

        final_report_path.write_text(final_report.strip() + "\n", encoding="utf-8")

        record = CardReportRunRecord(
            run_id=run_id,
            evidence_pack_id=args.evidence_pack_id,
            llm_spec=args.llm,
            section_id="card_based_full_report",
            section_title="Card-based full report",
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

        record = CardReportRunRecord(
            run_id=run_id,
            evidence_pack_id=args.evidence_pack_id,
            llm_spec=args.llm,
            section_id="card_based_full_report",
            section_title="Card-based full report",
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

    metadata = {
        "run_id": run_id,
        "llm": args.llm,
        "cards_json": str(args.cards_json),
        "cards_count": len(cards),
        "output_dir": str(run_dir),
        "prompt_path": str(prompt_path),
        "raw_response_path": str(raw_response_path),
        "draft_report_path": str(draft_report_path),
        "final_report_path": str(final_report_path),
        "summary_path": str(summary_path),
        "success": record.success,
        "error": record.error,
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote card-based report outputs to: {run_dir}")
    print(f"Success: {record.success}")
    print(f"Final report: {final_report_path}")
    print(f"Summary: {summary_path}")

    if record.error:
        print(f"Error: {record.error}")


def build_report_prompt(cards: list[dict[str, Any]], max_card_chars: int) -> str:
    cards_block = "\n\n".join(
        format_card_for_prompt(card, max_chars=max_card_chars)
        for card in cards
    )

    return f"""You are a senior patent strategy and R&D technology intelligence analyst.

You are generating an R&D-focused patent analysis report from refined, evidence-backed patent cards.

Rules:
- Use only the patent cards below.
- Every factual or technical paragraph must cite evidence IDs such as [E23], [E44].
- Cite only evidence IDs shown in the relevant patent card field_evidence_map or supporting_evidence_ids.
- Use no more than 3 evidence IDs per sentence or table cell.
- Separate fact from interpretation.
- Do not invent owners, years, metrics, applications, patent relationships, or technical effects.
- Do not provide legal advice, infringement analysis, validity analysis, patentability analysis, clearance analysis, or freedom-to-operate conclusions.
- Do not generate a patent-key table. A deterministic patent key will be added by code.
- Do not mention that you are using patent cards.
- Do not repeat these instructions.
- Write for senior R&D stakeholders.
- Use concise but specific technical language.

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
Write one analytical paragraph of 3-5 sentences with evidence citations.

## Executive summary
Write 3 concise paragraphs:
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
Group the patents into chemistry routes. For each route, write:
Fact:
Interpretation:
R&D relevance:

### 3.2 Formulation strategies
Create a markdown table:
Repeated problem | Design lever | Evidence patents | R&D experiment or action

### 3.3 Application focus
Explain application areas and success metrics. Focus on disposable products, elastic attachment, stretch laminates, paper products, nonwoven/film bonding, wet-out, coating quality, peel, creep, bleed-through, viscosity, and low-temperature processing only when supported by cards.

## 4. Key insights and R&D implications
Create a markdown table:
Action | Evidence behind it | Technical risk to control

The risk column must contain only technical risks such as viscosity, odour, colour, creep, peel strength, wet adhesion, substrate compatibility, processability, thermal stability, coating quality, bleed-through, formulation stability, or sensory profile.

Then write one paragraph identifying the immediate technical next step for R&D.

## 5. Limits of the conclusion
Write 2 concise paragraphs:
- dataset and evidence-card limitations
- not legal advice / FTO limitation

Patent cards:
{cards_block}

Generate the final report now.
"""


def format_card_for_prompt(card: dict[str, Any], max_chars: int) -> str:
    field_map = card.get("field_evidence_map", {})
    if not isinstance(field_map, dict):
        field_map = {}

    field_map_lines = []
    for field, ids in field_map.items():
        if isinstance(ids, list):
            evidence_refs = ", ".join(f"[{eid}]" for eid in ids)
        else:
            evidence_refs = str(ids)
        field_map_lines.append(f"- {field}: {evidence_refs}")

    supporting_ids = card.get("supporting_evidence_ids", [])
    if not isinstance(supporting_ids, list):
        supporting_ids = []

    block = f"""Patent: {card.get("patent_publication", "")}
Owner: {card.get("owner", "")}
Priority year: {card.get("priority_year", "")}
Core technology: {card.get("core_technology", "")}
Polymer route: {card.get("polymer_route", "")}
Formulation strategy: {card.get("formulation_strategy", "")}
Application focus: {card.get("application_focus", "")}
Performance evidence: {card.get("performance_evidence", "")}
R&D relevance: {card.get("rd_relevance", "")}
Missing or uncertain evidence: {"; ".join(card.get("missing_or_uncertain_evidence", []))}
Supporting evidence IDs: {", ".join(f"[{eid}]" for eid in supporting_ids)}
Field evidence map:
{chr(10).join(field_map_lines)}
"""

    return clip(block, max_chars)


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
                    "Use only supplied evidence-backed cards. Output markdown only."
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


def append_or_replace_patent_key_from_cards(
    report_text: str,
    cards: list[dict[str, Any]],
) -> str:
    cleaned = remove_patent_key_artifacts(report_text).rstrip()
    patent_key = build_patent_key_from_cards(cards)

    return cleaned + "\n\n" + patent_key.strip() + "\n"


def remove_patent_key_artifacts(report_text: str) -> str:
    cleaned = report_text

    proper_patent_key_pattern = re.compile(
        r"\n##\s+Patent\s+key\s+and\s+extracted\s+R&D\s+signal\b.*\Z",
        flags=re.IGNORECASE | re.DOTALL,
    )

    malformed_key_pattern = re.compile(
        r"\n#\s*\|\s*Patent\s*\|\s*Owner\s*\|\s*Priority\s*\|\s*Core\s+technology\s*\|\s*Main\s+R&D\s+relevance\b.*?\Z",
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned = re.sub(proper_patent_key_pattern, "", cleaned)
    cleaned = re.sub(malformed_key_pattern, "", cleaned)

    return cleaned.rstrip()


def build_patent_key_from_cards(cards: list[dict[str, Any]]) -> str:
    lines = [
        PATENT_KEY_HEADING,
        "",
        "| # | Patent | Owner | Priority | Core technology | Main R&D relevance | Evidence |",
        "|---:|---|---|---:|---|---|---|",
    ]

    for index, card in enumerate(cards, start=1):
        evidence_refs = select_card_evidence_refs(card, max_ids=5)

        lines.append(
            "| "
            f"{index} | "
            f"{escape_md(card.get('patent_publication', ''))} | "
            f"{escape_md(card.get('owner', ''))} | "
            f"{escape_md(card.get('priority_year', ''))} | "
            f"{escape_md(card.get('core_technology', ''))} | "
            f"{escape_md(card.get('rd_relevance', ''))} | "
            f"{escape_md(evidence_refs)} |"
        )

    return "\n".join(lines)


def select_card_evidence_refs(card: dict[str, Any], max_ids: int) -> str:
    selected: list[str] = []
    seen: set[str] = set()

    field_map = card.get("field_evidence_map", {})
    if isinstance(field_map, dict):
        for field in [
            "core_technology",
            "polymer_route",
            "formulation_strategy",
            "application_focus",
            "performance_evidence",
            "rd_relevance",
        ]:
            ids = field_map.get(field, [])
            if not isinstance(ids, list):
                continue

            for evidence_id in ids:
                evidence_id = str(evidence_id).strip()
                if not evidence_id or evidence_id in seen:
                    continue
                selected.append(evidence_id)
                seen.add(evidence_id)
                break

            if len(selected) >= max_ids:
                break

    supporting_ids = card.get("supporting_evidence_ids", [])
    if isinstance(supporting_ids, list):
        for evidence_id in supporting_ids:
            evidence_id = str(evidence_id).strip()
            if not evidence_id or evidence_id in seen:
                continue
            selected.append(evidence_id)
            seen.add(evidence_id)

            if len(selected) >= max_ids:
                break

    return ", ".join(f"[{eid}]" for eid in selected) if selected else "not available"


def load_cards(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cards file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise TypeError("Expected cards JSON to contain a list of card objects.")

    return data


def validate_cards_for_generation(cards: list[dict[str, Any]]) -> None:
    if len(cards) != 10:
        raise ValueError(f"Expected 10 patent cards, found {len(cards)}.")

    for card in cards:
        publication = card.get("patent_publication", "UNKNOWN")

        required_fields = [
            "patent_publication",
            "owner",
            "priority_year",
            "core_technology",
            "polymer_route",
            "formulation_strategy",
            "application_focus",
            "rd_relevance",
            "supporting_evidence_ids",
            "field_evidence_map",
        ]

        for field in required_fields:
            if field not in card:
                raise ValueError(f"{publication}: missing required field: {field}")

        supporting_ids = card.get("supporting_evidence_ids", [])
        if not isinstance(supporting_ids, list) or len(supporting_ids) < 3:
            raise ValueError(
                f"{publication}: expected at least 3 supporting evidence IDs."
            )


def write_summary_csv(records: list[CardReportRunRecord], path: Path) -> None:
    fieldnames = list(asdict(records[0]).keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def parse_llm_spec(raw_spec: str) -> tuple[str, str]:
    if ":" not in raw_spec:
        raise ValueError("Expected LLM spec format like ollama:qwen3:8b.")

    backend, model_name = raw_spec.split(":", 1)

    return backend.strip(), model_name.strip()


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


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + " [TRUNCATED]"


if __name__ == "__main__":
    main()