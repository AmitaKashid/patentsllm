"""Benchmark generated PatentLLM reports across LLMs.

This benchmark is intentionally split into deterministic metrics first:
- generation success
- latency
- token usage
- section completion
- report structure
- patent coverage
- citation validity
- citation density
- evidence utilization
- legal-term leakage
- technical specificity
- repetition

It does not claim to fully automate faithfulness or audience usefulness.
Those require a human rubric or a separate evaluator LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
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
    "legal risk",
]

TECHNICAL_TERMS = [
    "polymer",
    "polypropylene",
    "propylene",
    "ethylene",
    "copolymer",
    "olefin",
    "metallocene",
    "single-site",
    "wax",
    "tackifier",
    "resin",
    "plasticizer",
    "viscosity",
    "melt viscosity",
    "peel",
    "creep",
    "adhesion",
    "wet adhesion",
    "odour",
    "odor",
    "colour",
    "color",
    "nonwoven",
    "film",
    "substrate",
    "laminate",
    "elastic",
    "stretch",
    "coating",
    "spray",
    "thermal",
    "formulation",
]

AUDIENCE_TERMS = [
    "r&d",
    "experiment",
    "screening",
    "risk",
    "design lever",
    "application",
    "performance",
    "processability",
    "substrate",
    "technical next step",
]


@dataclass(slots=True)
class BenchmarkRow:
    run_dir: str
    report_path: str
    model_spec: str
    backend: str
    model_name: str

    section_success_rate: float
    length_stop_count: int
    total_latency_seconds: float
    p95_section_latency_seconds: float

    total_prompt_tokens: int
    total_output_tokens: int
    total_tokens: int
    output_tokens_per_second: float

    word_count: int
    required_heading_ratio: float
    missing_headings: str

    target_patent_coverage: float
    missing_target_patents: str

    patent_key_coverage: float
    malformed_patent_key_artifacts: int

    citation_count: int
    unique_citation_count: int
    invalid_citation_count: int
    citation_validity_rate: float
    citation_density_per_1000_words: float
    evidence_utilization_rate: float

    legal_leakage_before_limits: int
    technical_specificity_per_1000_words: float
    audience_fit_proxy_per_1000_words: float
    sentence_uniqueness_ratio: float

    quality_score: float
    efficiency_score: float
    final_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PatentLLM generated reports.")

    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("data/processed/generated_reports_sectioned"),
        help="Root folder containing sectioned report run directories.",
    )
    parser.add_argument(
        "--evidence-pack",
        type=Path,
        default=Path("data/processed/evidence_packs/20260613_155340/evidence_pack.json"),
        help="Evidence pack used to generate the reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/benchmark_results"),
    )
    parser.add_argument(
        "--only-clean",
        action="store_true",
        help="Prefer *_clean.md reports when present.",
    )
    parser.add_argument(
        "--dedupe-by-model",
        choices=["none", "latest", "best"],
        default="latest",
        help=(
            "How to handle multiple runs from the same model. "
            "'latest' keeps the most recent run, 'best' keeps highest final score, "
            "and 'none' keeps all runs."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evidence_pack = load_json(args.evidence_pack)
    valid_evidence_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in evidence_pack.get("evidence_items", [])
        if str(item.get("evidence_id", "")).strip()
    }

    run_dirs = find_run_dirs(args.reports_root)
    rows: list[BenchmarkRow] = []

    for run_dir in run_dirs:
        report_path = find_report_path(run_dir, only_clean=args.only_clean)
        summary_path = run_dir / "section_generation_summary.csv"
        metadata_path = run_dir / "metadata.json"

        if report_path is None or not summary_path.exists():
            continue

        report_text = report_path.read_text(encoding="utf-8")
        section_rows = read_csv(summary_path)
        metadata = load_json(metadata_path) if metadata_path.exists() else {}

        row = build_benchmark_row(
            run_dir=run_dir,
            report_path=report_path,
            report_text=report_text,
            section_rows=section_rows,
            metadata=metadata,
            valid_evidence_ids=valid_evidence_ids,
            total_evidence_items=len(valid_evidence_ids),
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No benchmarkable reports found under: {args.reports_root}")

    rows = add_efficiency_and_final_scores(rows)
    rows = dedupe_rows_by_model(rows, mode=args.dedupe_by_model)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "llm_benchmark_summary.csv"
    jsonl_path = output_dir / "llm_benchmark_summary.jsonl"
    md_path = output_dir / "llm_benchmark_report.md"

    write_csv(rows, csv_path)
    write_jsonl(rows, jsonl_path)
    write_markdown(rows, md_path)

    print(f"Benchmarked reports: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Markdown: {md_path}")


def find_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Reports root not found: {root}")

    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )


def find_report_path(run_dir: Path, only_clean: bool) -> Path | None:
    clean_reports = sorted(run_dir.glob("final_report_*_clean.md"))
    normal_reports = sorted(run_dir.glob("final_report_*.md"))

    if only_clean and clean_reports:
        return clean_reports[0]

    if clean_reports:
        return clean_reports[0]

    if normal_reports:
        # Avoid picking repaired/clean duplicates when a base final report exists.
        base_reports = [
            path for path in normal_reports
            if not path.name.endswith("_repaired.md") and not path.name.endswith("_clean.md")
        ]
        return base_reports[0] if base_reports else normal_reports[0]

    return None


def build_benchmark_row(
    run_dir: Path,
    report_path: Path,
    report_text: str,
    section_rows: list[dict[str, str]],
    metadata: dict[str, Any],
    valid_evidence_ids: set[str],
    total_evidence_items: int,
) -> BenchmarkRow:
    model_spec = str(metadata.get("llm", "") or infer_model_from_report_name(report_path))
    backend, model_name = split_model_spec(model_spec)

    section_success_rate = mean_bool([row.get("success", "").lower() == "true" for row in section_rows])
    length_stop_count = sum(1 for row in section_rows if row.get("done_reason", "").lower() == "length")

    latencies = [safe_float(row.get("latency_seconds")) for row in section_rows]
    latencies = [value for value in latencies if value is not None]

    total_latency_seconds = round(sum(latencies), 2)
    p95_section_latency_seconds = round(percentile(latencies, 95), 2) if latencies else 0.0

    prompt_tokens = [safe_int(row.get("prompt_eval_count")) for row in section_rows]
    output_tokens = [safe_int(row.get("eval_count")) for row in section_rows]

    total_prompt_tokens = sum(value for value in prompt_tokens if value is not None)
    total_output_tokens = sum(value for value in output_tokens if value is not None)
    total_tokens = total_prompt_tokens + total_output_tokens

    output_tokens_per_second = (
        round(total_output_tokens / total_latency_seconds, 2)
        if total_latency_seconds > 0
        else 0.0
    )

    word_count = count_words(report_text)

    required_heading_ratio, missing_headings = heading_metrics(report_text)
    target_patent_coverage, missing_target_patents = target_patent_metrics(report_text)
    patent_key_coverage = patent_key_coverage_metric(report_text)
    malformed_patent_key_artifacts = malformed_patent_key_count(report_text)

    citations = extract_citations(report_text)
    unique_citations = sorted(set(citations))
    invalid_citations = sorted(set(citations) - valid_evidence_ids)

    citation_count = len(citations)
    unique_citation_count = len(unique_citations)
    invalid_citation_count = len(invalid_citations)
    citation_validity_rate = (
        round(1.0 - (invalid_citation_count / max(1, unique_citation_count)), 4)
        if unique_citation_count
        else 0.0
    )

    citation_density = round(citation_count / max(1, word_count) * 1000, 2)
    evidence_utilization_rate = round(unique_citation_count / max(1, total_evidence_items), 4)

    legal_leakage = 1 if has_legal_leakage_before_limits(report_text) else 0

    technical_specificity = round(
        count_term_occurrences(report_text, TECHNICAL_TERMS) / max(1, word_count) * 1000,
        2,
    )

    audience_fit_proxy = round(
        count_term_occurrences(report_text, AUDIENCE_TERMS) / max(1, word_count) * 1000,
        2,
    )

    sentence_uniqueness_ratio = round(sentence_uniqueness(report_text), 4)

    quality_score = calculate_quality_score(
        required_heading_ratio=required_heading_ratio,
        target_patent_coverage=target_patent_coverage,
        patent_key_coverage=patent_key_coverage,
        citation_validity_rate=citation_validity_rate,
        citation_density_per_1000_words=citation_density,
        evidence_utilization_rate=evidence_utilization_rate,
        legal_leakage_before_limits=legal_leakage,
        section_success_rate=section_success_rate,
        length_stop_count=length_stop_count,
        technical_specificity_per_1000_words=technical_specificity,
        audience_fit_proxy_per_1000_words=audience_fit_proxy,
        sentence_uniqueness_ratio=sentence_uniqueness_ratio,
        malformed_patent_key_artifacts=malformed_patent_key_artifacts,
    )

    return BenchmarkRow(
        run_dir=str(run_dir),
        report_path=str(report_path),
        model_spec=model_spec,
        backend=backend,
        model_name=model_name,
        section_success_rate=round(section_success_rate, 4),
        length_stop_count=length_stop_count,
        total_latency_seconds=total_latency_seconds,
        p95_section_latency_seconds=p95_section_latency_seconds,
        total_prompt_tokens=total_prompt_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        output_tokens_per_second=output_tokens_per_second,
        word_count=word_count,
        required_heading_ratio=round(required_heading_ratio, 4),
        missing_headings="; ".join(missing_headings),
        target_patent_coverage=round(target_patent_coverage, 4),
        missing_target_patents="; ".join(missing_target_patents),
        patent_key_coverage=round(patent_key_coverage, 4),
        malformed_patent_key_artifacts=malformed_patent_key_artifacts,
        citation_count=citation_count,
        unique_citation_count=unique_citation_count,
        invalid_citation_count=invalid_citation_count,
        citation_validity_rate=citation_validity_rate,
        citation_density_per_1000_words=citation_density,
        evidence_utilization_rate=evidence_utilization_rate,
        legal_leakage_before_limits=legal_leakage,
        technical_specificity_per_1000_words=technical_specificity,
        audience_fit_proxy_per_1000_words=audience_fit_proxy,
        sentence_uniqueness_ratio=sentence_uniqueness_ratio,
        quality_score=quality_score,
        efficiency_score=0.0,
        final_score=0.0,
    )


def calculate_quality_score(
    required_heading_ratio: float,
    target_patent_coverage: float,
    patent_key_coverage: float,
    citation_validity_rate: float,
    citation_density_per_1000_words: float,
    evidence_utilization_rate: float,
    legal_leakage_before_limits: int,
    section_success_rate: float,
    length_stop_count: int,
    technical_specificity_per_1000_words: float,
    audience_fit_proxy_per_1000_words: float,
    sentence_uniqueness_ratio: float,
    malformed_patent_key_artifacts: int,
) -> float:
    """Calculate a deterministic quality score from 0 to 100.

    This is a proxy score. It checks structure, coverage, citations, safety,
    technical specificity, and repetition. It does not fully measure factual
    faithfulness or strategic usefulness.
    """

    citation_density_score = min(citation_density_per_1000_words / 25.0, 1.0)
    evidence_utilization_score = min(evidence_utilization_rate / 0.65, 1.0)
    technical_specificity_score = min(technical_specificity_per_1000_words / 45.0, 1.0)
    audience_fit_score = min(audience_fit_proxy_per_1000_words / 10.0, 1.0)

    legal_safety_score = 0.0 if legal_leakage_before_limits else 1.0
    length_score = 0.0 if length_stop_count else 1.0
    patent_key_artifact_score = 0.0 if malformed_patent_key_artifacts else 1.0

    # Weights sum exactly to 1.00.
    score = (
        0.11 * required_heading_ratio
        + 0.09 * target_patent_coverage
        + 0.09 * patent_key_coverage
        + 0.12 * citation_validity_rate
        + 0.07 * citation_density_score
        + 0.07 * evidence_utilization_score
        + 0.10 * legal_safety_score
        + 0.09 * section_success_rate
        + 0.08 * length_score
        + 0.08 * technical_specificity_score
        + 0.05 * audience_fit_score
        + 0.03 * sentence_uniqueness_ratio
        + 0.02 * patent_key_artifact_score
    )

    return round(min(score, 1.0) * 100, 2)


def add_efficiency_and_final_scores(rows: list[BenchmarkRow]) -> list[BenchmarkRow]:
    min_latency = min(row.total_latency_seconds for row in rows if row.total_latency_seconds > 0)
    min_tokens = min(row.total_tokens for row in rows if row.total_tokens > 0)

    for row in rows:
        latency_score = min_latency / row.total_latency_seconds if row.total_latency_seconds > 0 else 0.0
        token_score = min_tokens / row.total_tokens if row.total_tokens > 0 else 0.0

        row.efficiency_score = round((0.65 * latency_score + 0.35 * token_score) * 100, 2)
        row.final_score = round((0.85 * row.quality_score + 0.15 * row.efficiency_score), 2)

    return sorted(rows, key=lambda row: row.final_score, reverse=True)

def dedupe_rows_by_model(
    rows: list[BenchmarkRow],
    mode: str,
) -> list[BenchmarkRow]:
    """Collapse multiple runs of the same model into one row.

    latest = keep most recent run_dir path lexicographically, assuming timestamped folders
    best = keep highest final_score
    none = keep all rows
    """

    if mode == "none":
        return rows

    grouped: dict[str, list[BenchmarkRow]] = {}

    for row in rows:
        grouped.setdefault(row.model_spec, []).append(row)

    selected: list[BenchmarkRow] = []

    for model_spec, model_rows in grouped.items():
        if mode == "latest":
            chosen = sorted(model_rows, key=lambda row: row.run_dir)[-1]
        elif mode == "best":
            chosen = sorted(model_rows, key=lambda row: row.final_score)[-1]
        else:
            raise ValueError(f"Unsupported dedupe mode: {mode}")

        selected.append(chosen)

    return sorted(selected, key=lambda row: row.final_score, reverse=True)

def heading_metrics(report_text: str) -> tuple[float, list[str]]:
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in report_text]
    ratio = (len(REQUIRED_HEADINGS) - len(missing)) / len(REQUIRED_HEADINGS)
    return ratio, missing


def target_patent_metrics(report_text: str) -> tuple[float, list[str]]:
    missing = [patent for patent in TARGET_PATENTS if patent not in report_text]
    ratio = (len(TARGET_PATENTS) - len(missing)) / len(TARGET_PATENTS)
    return ratio, missing


def patent_key_coverage_metric(report_text: str) -> float:
    key_start = report_text.find("## Patent key and extracted R&D signal")
    patent_key = report_text[key_start:] if key_start != -1 else ""
    found = sum(1 for patent in TARGET_PATENTS if patent in patent_key)
    return found / len(TARGET_PATENTS)


def malformed_patent_key_count(report_text: str) -> int:
    malformed_headers = re.findall(
        r"(?m)^#\s*\|\s*Patent\s*\|\s*Owner\s*\|\s*Priority\s*\|",
        report_text,
        flags=re.IGNORECASE,
    )
    malformed_rows = re.findall(
        r"(?m)^#\s*\|\s*EP\d+",
        report_text,
        flags=re.IGNORECASE,
    )
    return len(malformed_headers) + len(malformed_rows)


def has_legal_leakage_before_limits(report_text: str) -> bool:
    lowered = report_text.lower()
    limits_index = lowered.find("## 5. limits of the conclusion")
    before_limits = lowered[:limits_index] if limits_index != -1 else lowered

    return any(term in before_limits for term in LEGAL_TERMS)


def extract_citations(report_text: str) -> list[str]:
    return re.findall(r"\[(E\d+)\]", report_text)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def count_term_occurrences(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    count = 0

    for term in terms:
        count += len(re.findall(rf"\b{re.escape(term.lower())}\b", lowered))

    return count


def sentence_uniqueness(text: str) -> float:
    sentences = [
        normalize_sentence(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) > 30
    ]

    if not sentences:
        return 0.0

    return len(set(sentences)) / len(sentences)


def normalize_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.lower().strip())


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = math.ceil((percentile_value / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def mean_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_model_from_report_name(path: Path) -> str:
    name = path.stem
    if name.startswith("final_report_"):
        model = name.replace("final_report_", "")
        return "unknown:" + model
    return "unknown:unknown"


def split_model_spec(model_spec: str) -> tuple[str, str]:
    if ":" not in model_spec:
        return "unknown", model_spec

    backend, model_name = model_spec.split(":", 1)
    return backend, model_name


def write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def write_jsonl(rows: list[BenchmarkRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_markdown(rows: list[BenchmarkRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# PatentLLM LLM Benchmark Report\n\n")
        file.write("This benchmark compares generated reports using deterministic metrics.\n\n")

        file.write("## Ranking\n\n")
        file.write("| Rank | Model | Final score | Quality | Efficiency | Latency s | Output tokens | Citation validity | Patent coverage | Legal leakage |\n")
        file.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for rank, row in enumerate(rows, start=1):
            file.write(
                f"| {rank} | `{row.model_spec}` | {row.final_score:.2f} | "
                f"{row.quality_score:.2f} | {row.efficiency_score:.2f} | "
                f"{row.total_latency_seconds:.2f} | {row.total_output_tokens} | "
                f"{row.citation_validity_rate:.2f} | {row.target_patent_coverage:.2f} | "
                f"{row.legal_leakage_before_limits} |\n"
            )

        file.write("\n## Detailed notes\n\n")
        for row in rows:
            file.write(f"### `{row.model_spec}`\n\n")
            file.write(f"- Report: `{row.report_path}`\n")
            file.write(f"- Section success rate: `{row.section_success_rate}`\n")
            file.write(f"- Length stops: `{row.length_stop_count}`\n")
            file.write(f"- Word count: `{row.word_count}`\n")
            file.write(f"- Unique citations: `{row.unique_citation_count}`\n")
            file.write(f"- Evidence utilization rate: `{row.evidence_utilization_rate}`\n")
            file.write(f"- Technical specificity per 1000 words: `{row.technical_specificity_per_1000_words}`\n")
            file.write(f"- Audience-fit proxy per 1000 words: `{row.audience_fit_proxy_per_1000_words}`\n")
            file.write(f"- Sentence uniqueness ratio: `{row.sentence_uniqueness_ratio}`\n")

            if row.missing_headings:
                file.write(f"- Missing headings: `{row.missing_headings}`\n")
            if row.missing_target_patents:
                file.write(f"- Missing patents: `{row.missing_target_patents}`\n")

            file.write("\n")


if __name__ == "__main__":
    main()