"""Evaluate generated PatentLLM reports with a local LLM judge.

This script adds semantic quality scoring on top of deterministic benchmarking.

It scores:
- faithfulness
- citation support
- answer relevance
- technical depth
- audience fit
- R&D usefulness

The judge is not treated as ground truth. It is a repeatable second-pass
quality signal that should later be compared with human review.
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


@dataclass(frozen=True, slots=True)
class JudgeResult:
    run_id: str
    report_path: str
    model_under_test: str
    judge_model: str

    faithfulness: int
    citation_support: int
    answer_relevance: int
    technical_depth: int
    audience_fit: int
    rd_usefulness: int

    overall_judge_score: float
    strengths: str
    weaknesses: str
    recommendation: str

    judge_latency_seconds: float
    judge_prompt_chars: int
    judge_raw_response_path: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PatentLLM reports with local LLM judge.")

    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        required=True,
        help="Path to llm_benchmark_summary.csv.",
    )
    parser.add_argument(
        "--evidence-pack",
        type=Path,
        required=True,
        help="Path to evidence_pack.json.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="qwen3:8b",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/judge_results"),
    )
    parser.add_argument(
        "--max-report-chars",
        type=int,
        default=18000,
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=16000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark_rows = read_csv(args.benchmark_summary)
    evidence_pack = load_json(args.evidence_pack)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    raw_dir = run_dir / "raw_responses"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[JudgeResult] = []

    for row in benchmark_rows:
        report_path = Path(row["report_path"])
        model_under_test = row["model_spec"]

        print(f"Judging: {model_under_test} | {report_path}")

        raw_response_path = raw_dir / f"{safe_filename(model_under_test)}_judge_raw.json"

        try:
            # 1. Load report first. This must happen before extract_citations().
            report_text = report_path.read_text(encoding="utf-8")

            # 2. Remove deterministic patent key before semantic judge scoring.
            report_text = strip_deterministic_sections(report_text)

            # 3. Extract only citations actually used in the judged report text.
            cited_ids = extract_citations(report_text)

            # 4. Build judge context only from cited evidence IDs.
            evidence_context = build_citation_targeted_evidence_context(
                evidence_pack=evidence_pack,
                cited_ids=cited_ids,
                max_chars=args.max_evidence_chars,
            )

            # 5. Clip report after citation extraction.
            report_text = clip(report_text, args.max_report_chars)

            prompt = build_judge_prompt(
                model_under_test=model_under_test,
                report_text=report_text,
                evidence_context=evidence_context,
            )

            started = time.perf_counter()

            judge_output = call_ollama_json(
                base_url=args.ollama_base_url,
                model=args.judge_model,
                prompt=prompt,
                raw_response_path=raw_response_path,
            )

            latency = time.perf_counter() - started
            parsed = parse_judge_json(judge_output)

            result = JudgeResult(
                run_id=run_id,
                report_path=str(report_path),
                model_under_test=model_under_test,
                judge_model=args.judge_model,
                faithfulness=clamp_score(parsed.get("faithfulness")),
                citation_support=clamp_score(parsed.get("citation_support")),
                answer_relevance=clamp_score(parsed.get("answer_relevance")),
                technical_depth=clamp_score(parsed.get("technical_depth")),
                audience_fit=clamp_score(parsed.get("audience_fit")),
                rd_usefulness=clamp_score(parsed.get("rd_usefulness")),
                overall_judge_score=calculate_overall_judge_score(parsed),
                strengths=str(parsed.get("strengths", "")).strip(),
                weaknesses=str(parsed.get("weaknesses", "")).strip(),
                recommendation=str(parsed.get("recommendation", "")).strip(),
                judge_latency_seconds=round(latency, 2),
                judge_prompt_chars=len(prompt),
                judge_raw_response_path=str(raw_response_path),
                error="",
            )

        except Exception as exc:
            result = JudgeResult(
                run_id=run_id,
                report_path=str(report_path),
                model_under_test=model_under_test,
                judge_model=args.judge_model,
                faithfulness=0,
                citation_support=0,
                answer_relevance=0,
                technical_depth=0,
                audience_fit=0,
                rd_usefulness=0,
                overall_judge_score=0.0,
                strengths="",
                weaknesses="",
                recommendation="",
                judge_latency_seconds=0.0,
                judge_prompt_chars=0,
                judge_raw_response_path=str(raw_response_path),
                error=f"{type(exc).__name__}: {exc}",
            )

        results.append(result)

    results = sorted(results, key=lambda item: item.overall_judge_score, reverse=True)

    csv_path = run_dir / "llm_judge_scores.csv"
    jsonl_path = run_dir / "llm_judge_scores.jsonl"
    md_path = run_dir / "llm_judge_report.md"

    write_csv(results, csv_path)
    write_jsonl(results, jsonl_path)
    write_markdown(results, md_path)

    print(f"\nWrote judge results to: {run_dir}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")

def build_judge_prompt(
    model_under_test: str,
    report_text: str,
    evidence_context: str,
) -> str:
    return f"""You are evaluating a patent-based R&D report generated by an LLM.

Model under test:
{model_under_test}

Evaluation task:
Score the report against the supplied evidence. Be strict. Do not reward polished writing if the report is generic or unsupported.

Scoring scale:
1 = poor
2 = weak
3 = acceptable
4 = strong
5 = excellent

Metrics:
- faithfulness: Are factual and technical claims supported by the evidence?
- citation_support: Do cited evidence IDs actually support the nearby statements?
- answer_relevance: Does the report answer the R&D patent-analysis task?
- technical_depth: Are polymer types, formulation strategies, application focus, and performance signals specific?
- audience_fit: Is the detail level appropriate for R&D stakeholders rather than legal or marketing readers?
- rd_usefulness: Are R&D implications actionable and technically meaningful?

Important:
- Penalize unsupported claims.
- Penalize generic R&D recommendations.
- Penalize legal/FTO language outside limitations.
- Penalize missing application or formulation detail.
- Reward clear separation of fact, interpretation, and R&D relevance.
- Output JSON only.

Required JSON schema:
{{
  "faithfulness": 1,
  "citation_support": 1,
  "answer_relevance": 1,
  "technical_depth": 1,
  "audience_fit": 1,
  "rd_usefulness": 1,
  "strengths": "short text",
  "weaknesses": "short text",
  "recommendation": "short text"
}}

<evidence_context>
{evidence_context}
</evidence_context>

<report_to_evaluate>
{report_text}
</report_to_evaluate>
"""


def build_evidence_context(evidence_pack: dict[str, Any], max_chars: int) -> str:
    items = evidence_pack.get("evidence_items", [])

    parts: list[str] = []
    for item in items:
        text = clean_text(str(item.get("text", "")))
        evidence_id = item.get("evidence_id", "")
        patent = item.get("patent_publication", "")
        owner = item.get("owner", "")
        priority = item.get("priority_year", "")
        category = item.get("selection_category", "")

        parts.append(
            f"[{evidence_id}] patent={patent}; owner={owner}; priority={priority}; "
            f"category={category}; text={clip(text, 700)}"
        )

        if sum(len(part) for part in parts) > max_chars:
            break

    return "\n\n".join(parts)


def call_ollama_json(
    base_url: str,
    model: str,
    prompt: str,
    raw_response_path: Path,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator. Return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": 0.0,
            "num_ctx": 12000,
            "num_predict": 1200,
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
        raise RuntimeError(f"Judge model returned empty response. Raw: {raw_response_path}")

    return content


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def calculate_overall_judge_score(parsed: dict[str, Any]) -> float:
    faithfulness = clamp_score(parsed.get("faithfulness"))
    citation_support = clamp_score(parsed.get("citation_support"))
    answer_relevance = clamp_score(parsed.get("answer_relevance"))
    technical_depth = clamp_score(parsed.get("technical_depth"))
    audience_fit = clamp_score(parsed.get("audience_fit"))
    rd_usefulness = clamp_score(parsed.get("rd_usefulness"))

    weighted = (
        0.22 * faithfulness
        + 0.20 * citation_support
        + 0.14 * answer_relevance
        + 0.16 * technical_depth
        + 0.12 * audience_fit
        + 0.16 * rd_usefulness
    )

    return round((weighted / 5.0) * 100.0, 2)


def clamp_score(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0

    return max(1, min(5, number))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(results: list[JudgeResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_jsonl(results: list[JudgeResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def write_markdown(results: list[JudgeResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# PatentLLM Local Judge Evaluation\n\n")
        file.write("| Rank | Model | Judge score | Faithfulness | Citation support | Relevance | Technical depth | Audience fit | R&D usefulness |\n")
        file.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")

        for index, result in enumerate(results, start=1):
            file.write(
                f"| {index} | `{result.model_under_test}` | {result.overall_judge_score:.2f} | "
                f"{result.faithfulness} | {result.citation_support} | {result.answer_relevance} | "
                f"{result.technical_depth} | {result.audience_fit} | {result.rd_usefulness} |\n"
            )

        file.write("\n## Notes\n\n")
        for result in results:
            file.write(f"### `{result.model_under_test}`\n\n")
            file.write(f"- Strengths: {result.strengths}\n")
            file.write(f"- Weaknesses: {result.weaknesses}\n")
            file.write(f"- Recommendation: {result.recommendation}\n")
            if result.error:
                file.write(f"- Error: `{result.error}`\n")
            file.write("\n")


def clean_text(text: str) -> str:
    return " ".join(text.split())


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + " [TRUNCATED]"


def safe_filename(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

def strip_deterministic_sections(report_text: str) -> str:
    """Remove deterministic appendix sections before semantic judge scoring."""

    patent_key_heading = "## Patent key and extracted R&D signal"
    index = report_text.find(patent_key_heading)

    if index == -1:
        return report_text

    return report_text[:index].rstrip()

def extract_citations(report_text: str) -> list[str]:
    """Extract cited evidence IDs from the report in first-seen order."""

    seen: set[str] = set()
    ordered: list[str] = []

    for evidence_id in re.findall(r"\[(E\d+)\]", report_text):
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        ordered.append(evidence_id)

    return ordered


def build_citation_targeted_evidence_context(
    evidence_pack: dict[str, Any],
    cited_ids: list[str],
    max_chars: int,
) -> str:
    """Build judge context from evidence IDs cited by the report."""

    evidence_by_id = {
        str(item.get("evidence_id", "")).strip(): item
        for item in evidence_pack.get("evidence_items", [])
    }

    parts: list[str] = []
    total_chars = 0

    for evidence_id in cited_ids:
        item = evidence_by_id.get(evidence_id)

        if item is None:
            block = f"[{evidence_id}] MISSING_FROM_EVIDENCE_PACK"
        else:
            text = clean_text(str(item.get("text", "")))
            snippet = clip(text, 850)

            block = (
                f"[{evidence_id}] "
                f"patent={item.get('patent_publication', '')}; "
                f"owner={item.get('owner', '')}; "
                f"priority={item.get('priority_year', '')}; "
                f"category={item.get('selection_category', '')}; "
                f"section={item.get('section', '')}; "
                f"text={snippet}"
            )

        if total_chars + len(block) > max_chars:
            parts.append(
                "[CONTEXT_TRUNCATED] Some later cited evidence IDs were omitted "
                "because judge context budget was reached."
            )
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


def strip_deterministic_sections(report_text: str) -> str:
    """Remove deterministic appendix sections before semantic judge scoring."""

    patent_key_heading = "## Patent key and extracted R&D signal"
    index = report_text.find(patent_key_heading)

    if index == -1:
        return report_text

    return report_text[:index].rstrip()


if __name__ == "__main__":
    main()