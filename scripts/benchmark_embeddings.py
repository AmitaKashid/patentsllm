"""Benchmark multiple patent embedding indexes on the same retrieval query suite.

This script compares already-built Chroma collections, for example:
- patent_chunks_bge_m3
- patent_chunks_patentsberta

It writes:
- retrieval_results.jsonl        full results, one row per query/model
- retrieval_results.csv          flattened ranking table
- query_model_scores.csv         one row per query/model with metrics
- model_summary.csv              aggregated model-level metrics
- benchmark_report.md            readable benchmark report

Usage:
    python scripts/benchmark_embeddings.py

Optional:
    python scripts/benchmark_embeddings.py `
      --persist-dir data/processed/vector_store/chroma `
      --top-k 5 `
      --output-dir data/processed/retrieval_benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from patentllm.indexing.config import IndexingConfig
from patentllm.indexing.indexer import PatentChunkIndexer


Scalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class EmbeddingIndexSpec:
    """One embedding/index configuration to benchmark."""

    label: str
    collection_name: str
    embedding_backend: str
    model_name: str


@dataclass(frozen=True, slots=True)
class QueryCase:
    """One retrieval benchmark query with weak relevance labels."""

    query_id: str
    category: str
    query: str
    expected_document_ids: tuple[str, ...]
    expected_chunk_types: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    where: dict[str, Scalar] | None = None
    notes: str = ""


@dataclass(slots=True)
class RetrievedChunk:
    """One retrieved chunk result."""

    rank: int
    chunk_id: str
    distance: float | None
    document_id: str
    chunk_type: str
    section: str
    page_start: int | None
    page_end: int | None
    claim_number: int | None
    text_preview: str


@dataclass(slots=True)
class QueryModelResult:
    """Benchmark result for one query against one embedding index."""

    run_id: str
    query_id: str
    category: str
    query: str
    model_label: str
    collection_name: str
    model_name: str
    top_k: int
    latency_ms: float
    expected_document_ids: list[str]
    expected_chunk_types: list[str]
    expected_sections: list[str]
    where: dict[str, Scalar] | None
    metrics: dict[str, float | int | bool]
    results: list[RetrievedChunk] = field(default_factory=list)


DEFAULT_INDEX_SPECS: tuple[EmbeddingIndexSpec, ...] = (
    EmbeddingIndexSpec(
        label="bge_m3",
        collection_name="patent_chunks_bge_m3",
        embedding_backend="sentence_transformer",
        model_name="BAAI/bge-m3",
    ),
    EmbeddingIndexSpec(
        label="patentsberta",
        collection_name="patent_chunks_patentsberta",
        embedding_backend="sentence_transformer",
        model_name="AI-Growth-Lab/PatentSBERTa",
    ),
)


QUERY_SUITE: tuple[QueryCase, ...] = (
    QueryCase(
        query_id="Q01_claim_ingredients",
        category="claim_scope",
        query="hot melt adhesive composition comprising polymer tackifier plasticizer stabilizer wax",
        expected_document_ids=(
            "WO_2011011729_A1",
            "WO_2014130180_A1",
            "WO_2013162059_A1",
            "WO_2017123874_A1",
            "WO_2023099461_A1",
        ),
        expected_chunk_types=("claim", "claim_clause_window"),
        where={"chunk_type": "claim"},
        notes="Checks whether claim-focused formulation queries return claim chunks.",
    ),
    QueryCase(
        query_id="Q02_obc_route",
        category="technology_route",
        query="olefin block copolymer hot melt adhesive nonwoven polyethylene film",
        expected_document_ids=("WO_2011011729_A1",),
        expected_chunk_types=(
            "description_paragraph_window",
            "claim",
            "evidence_paragraph_window",
        ),
        notes="OBC should strongly retrieve the olefin block copolymer patent.",
    ),
    QueryCase(
        query_id="Q03_single_site_propylene",
        category="technology_route",
        query="propylene polymer hot melt adhesive prepared using single site catalysts",
        expected_document_ids=(
            "WO_2013039261_A1",
            "WO_2017123874_A1",
            "WO_2019094659_A1",
            "WO_2023099461_A1",
        ),
        expected_chunk_types=("claim", "description_paragraph_window"),
        notes="Checks metallocene / single-site catalyst propylene chemistry.",
    ),
    QueryCase(
        query_id="Q04_experimental_creep_peel",
        category="experimental_evidence",
        query="examples experimental peel strength creep retention disposable absorbent article",
        expected_document_ids=(
            "WO_2011011729_A1",
            "WO_2014130180_A1",
            "WO_2016140830_A1",
        ),
        expected_chunk_types=("evidence_paragraph_window", "description_paragraph_window"),
        expected_sections=("examples", "experimental_part", "detailed_description"),
        notes="Checks whether examples and test-method evidence are retrieved.",
    ),
    QueryCase(
        query_id="Q05_evidence_only_performance",
        category="experimental_evidence",
        query="experimental examples peel strength viscosity open time heat resistance",
        expected_document_ids=(
            "WO_2012092233_A2",
            "WO_2013162059_A1",
            "WO_2016140830_A1",
            "WO_2017123874_A1",
            "WO_2023099461_A1",
        ),
        expected_chunk_types=("evidence_paragraph_window",),
        expected_sections=("examples", "experimental_part"),
        where={"chunk_type": "evidence_paragraph_window"},
        notes="Checks metadata filtering and evidence-only retrieval.",
    ),
    QueryCase(
        query_id="Q06_wet_adhesion",
        category="application_need",
        query="wet adhesion hydrophilic nonwoven material based products acid grafted polyolefin",
        expected_document_ids=("WO_2023099461_A1",),
        expected_chunk_types=("description_paragraph_window", "claim", "evidence_paragraph_window"),
        notes="Checks wet-adhesion and acid-grafted polyolefin retrieval.",
    ),
    QueryCase(
        query_id="Q07_stretch_laminate",
        category="application_need",
        query="stretch laminate hot melt adhesive nonwoven film absorbent article",
        expected_document_ids=("WO_2016140830_A1",),
        expected_chunk_types=("description_paragraph_window", "claim", "evidence_paragraph_window"),
        notes="Checks stretch-laminate application retrieval.",
    ),
    QueryCase(
        query_id="Q08_polymer_systems_low_viscosity",
        category="technology_route",
        query="adhesives made from polymer systems reduced melt viscosity shear stress radical donor",
        expected_document_ids=("WO_2010109018_A1",),
        expected_chunk_types=("description_paragraph_window", "claim"),
        notes="Checks polymer-system and viscosity-reduction patent retrieval.",
    ),
    QueryCase(
        query_id="Q09_polar_polyester",
        category="technology_route",
        query="polar functional group modified polymer aliphatic polyester resin tackifier thermoplastic elastomer",
        expected_document_ids=("WO_2013162059_A1",),
        expected_chunk_types=("claim", "description_paragraph_window"),
        notes="Checks functionalized polymer / polyester resin patent retrieval.",
    ),
    QueryCase(
        query_id="Q10_document_filter_ssc",
        category="metadata_filter",
        query="propylene copolymers single-site catalysts low surface energy substrates",
        expected_document_ids=("WO_2017123874_A1",),
        expected_chunk_types=("description_paragraph_window", "claim"),
        where={"document_id": "WO_2017123874_A1"},
        notes="Checks document_id metadata filtering.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PatentLLM embedding indexes.")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/processed/vector_store/chroma"),
        help="Path to persistent Chroma directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/retrieval_benchmark"),
        help="Directory where benchmark run outputs are written.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=None,
        help=(
            "Optional model spec in the format "
            "label|collection_name|embedding_backend|model_name. "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--fail-on-missing-collection",
        action="store_true",
        help="Fail immediately if a configured Chroma collection is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    index_specs = parse_model_specs(args.model_spec)
    validate_collections(
        persist_dir=args.persist_dir,
        specs=index_specs,
        fail_on_missing=args.fail_on_missing_collection,
    )

    results: list[QueryModelResult] = []

    for spec in index_specs:
        print(f"\nLoading index: {spec.label} | {spec.collection_name} | {spec.model_name}")

        config = IndexingConfig(
            persist_dir=args.persist_dir,
            collection_name=spec.collection_name,
            embedding_backend=spec.embedding_backend,  # type: ignore[arg-type]
            model_name=spec.model_name,
            top_k=args.top_k,
        )
        indexer = PatentChunkIndexer(config)

        for query_case in QUERY_SUITE:
            print(f"  Running {query_case.query_id}: {query_case.query[:70]}...")

            start = time.perf_counter()
            raw_result = indexer.query(
                query=query_case.query,
                top_k=args.top_k,
                where=query_case.where,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0

            retrieved_chunks = normalize_chroma_result(raw_result)
            metrics = compute_metrics(query_case, retrieved_chunks, args.top_k)

            results.append(
                QueryModelResult(
                    run_id=run_id,
                    query_id=query_case.query_id,
                    category=query_case.category,
                    query=query_case.query,
                    model_label=spec.label,
                    collection_name=spec.collection_name,
                    model_name=spec.model_name,
                    top_k=args.top_k,
                    latency_ms=latency_ms,
                    expected_document_ids=list(query_case.expected_document_ids),
                    expected_chunk_types=list(query_case.expected_chunk_types),
                    expected_sections=list(query_case.expected_sections),
                    where=query_case.where,
                    metrics=metrics,
                    results=retrieved_chunks,
                )
            )

    write_jsonl(results, run_dir / "retrieval_results.jsonl")
    write_flat_csv(results, run_dir / "retrieval_results.csv")
    write_query_scores_csv(results, run_dir / "query_model_scores.csv")
    write_model_summary_csv(results, run_dir / "model_summary.csv")
    write_markdown_report(results, run_dir / "benchmark_report.md", run_id)

    print("\nBenchmark completed.")
    print(f"Wrote outputs to: {run_dir}")
    print("Important files:")
    print(f"  {run_dir / 'benchmark_report.md'}")
    print(f"  {run_dir / 'model_summary.csv'}")
    print(f"  {run_dir / 'query_model_scores.csv'}")
    print(f"  {run_dir / 'retrieval_results.jsonl'}")
    print(f"  {run_dir / 'retrieval_results.csv'}")


def parse_model_specs(model_specs: list[str] | None) -> tuple[EmbeddingIndexSpec, ...]:
    if not model_specs:
        return DEFAULT_INDEX_SPECS

    parsed_specs: list[EmbeddingIndexSpec] = []

    for raw_spec in model_specs:
        parts = raw_spec.split("|")
        if len(parts) != 4:
            raise ValueError(
                "Invalid --model-spec. Expected: "
                "label|collection_name|embedding_backend|model_name"
            )

        label, collection_name, embedding_backend, model_name = parts
        parsed_specs.append(
            EmbeddingIndexSpec(
                label=label.strip(),
                collection_name=collection_name.strip(),
                embedding_backend=embedding_backend.strip(),
                model_name=model_name.strip(),
            )
        )

    return tuple(parsed_specs)


def validate_collections(
    persist_dir: Path,
    specs: tuple[EmbeddingIndexSpec, ...],
    fail_on_missing: bool,
) -> None:
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required. Install with: pip install -e '.[indexing]'"
        ) from exc

    client = chromadb.PersistentClient(path=str(persist_dir))
    existing = {collection.name for collection in client.list_collections()}
    required = {spec.collection_name for spec in specs}
    missing = sorted(required - existing)

    if missing:
        message = (
            f"Missing Chroma collections: {missing}. "
            f"Existing collections: {sorted(existing)}"
        )
        if fail_on_missing:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


def normalize_chroma_result(raw_result: dict[str, Any]) -> list[RetrievedChunk]:
    ids = first_result_list(raw_result, "ids")
    documents = first_result_list(raw_result, "documents")
    metadatas = first_result_list(raw_result, "metadatas")
    distances = first_result_list(raw_result, "distances")

    retrieved: list[RetrievedChunk] = []

    for index, chunk_id in enumerate(ids):
        metadata = safe_get(metadatas, index, default={}) or {}
        document = safe_get(documents, index, default="") or ""
        distance = safe_get(distances, index, default=None)

        retrieved.append(
            RetrievedChunk(
                rank=index + 1,
                chunk_id=str(chunk_id),
                distance=float(distance) if distance is not None else None,
                document_id=str(metadata.get("document_id", "")),
                chunk_type=str(metadata.get("chunk_type", "")),
                section=str(metadata.get("section", "")),
                page_start=int_or_none(metadata.get("page_start")),
                page_end=int_or_none(metadata.get("page_end")),
                claim_number=int_or_none(metadata.get("claim_number")),
                text_preview=clean_preview(document, max_chars=900),
            )
        )

    return retrieved


def first_result_list(raw_result: dict[str, Any], key: str) -> list[Any]:
    value = raw_result.get(key, [])
    if not value:
        return []
    first = value[0]
    return first if isinstance(first, list) else []


def safe_get(values: list[Any], index: int, default: Any) -> Any:
    if index >= len(values):
        return default
    return values[index]


def int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return None if number == -1 else number


def clean_preview(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars]


def compute_metrics(
    query_case: QueryCase,
    retrieved_chunks: list[RetrievedChunk],
    top_k: int,
) -> dict[str, float | int | bool]:
    expected_docs = set(query_case.expected_document_ids)
    expected_chunk_types = set(query_case.expected_chunk_types)
    expected_sections = set(query_case.expected_sections)

    doc_hit_ranks = [
        chunk.rank for chunk in retrieved_chunks if chunk.document_id in expected_docs
    ]
    type_hit_ranks = [
        chunk.rank for chunk in retrieved_chunks if chunk.chunk_type in expected_chunk_types
    ]
    section_hit_ranks = [
        chunk.rank for chunk in retrieved_chunks if chunk.section in expected_sections
    ]

    relevant_doc_hits = sum(
        1 for chunk in retrieved_chunks if chunk.document_id in expected_docs
    )

    expected_doc_count = max(1, len(expected_docs))
    retrieved_count = max(1, len(retrieved_chunks))

    distances = [
        chunk.distance for chunk in retrieved_chunks if chunk.distance is not None
    ]

    noisy_hits = count_noisy_hits(retrieved_chunks)

    return {
        "hit_at_k_doc": bool(doc_hit_ranks),
        "rank_first_expected_doc": min(doc_hit_ranks) if doc_hit_ranks else 0,
        "mrr_doc": reciprocal_rank(doc_hit_ranks),
        "precision_at_k_doc": relevant_doc_hits / retrieved_count,
        "recall_at_k_doc": min(relevant_doc_hits / expected_doc_count, 1.0),
        "hit_at_k_chunk_type": bool(type_hit_ranks) if expected_chunk_types else True,
        "rank_first_expected_chunk_type": min(type_hit_ranks) if type_hit_ranks else 0,
        "hit_at_k_section": bool(section_hit_ranks) if expected_sections else True,
        "rank_first_expected_section": min(section_hit_ranks) if section_hit_ranks else 0,
        "noisy_hits_at_k": noisy_hits,
        "top1_distance": distances[0] if distances else -1.0,
        "mean_distance": statistics.mean(distances) if distances else -1.0,
        "latency_safe_placeholder": 0.0,
        "top_k": top_k,
    }


def reciprocal_rank(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return 1.0 / min(ranks)


def count_noisy_hits(retrieved_chunks: list[RetrievedChunk]) -> int:
    noisy_chunk_types = {"metadata", "title_abstract_claims"}
    noisy_sections = {"search_report", "metadata", "title_abstract_claims"}

    count = 0

    for chunk in retrieved_chunks:
        if chunk.chunk_type in noisy_chunk_types:
            count += 1
            continue

        if chunk.section in noisy_sections:
            count += 1
            continue

        text = chunk.text_preview.lower()
        if "international search report" in text or "documents considered to be relevant" in text:
            count += 1

    return count


def write_jsonl(results: list[QueryModelResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            row = asdict(result)
            row["results"] = [asdict(chunk) for chunk in result.results]
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_flat_csv(results: list[QueryModelResult], path: Path) -> None:
    fieldnames = [
        "run_id",
        "query_id",
        "category",
        "query",
        "model_label",
        "collection_name",
        "model_name",
        "where",
        "rank",
        "chunk_id",
        "distance",
        "document_id",
        "chunk_type",
        "section",
        "page_start",
        "page_end",
        "claim_number",
        "expected_document_ids",
        "expected_chunk_types",
        "expected_sections",
        "text_preview",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            for chunk in result.results:
                writer.writerow(
                    {
                        "run_id": result.run_id,
                        "query_id": result.query_id,
                        "category": result.category,
                        "query": result.query,
                        "model_label": result.model_label,
                        "collection_name": result.collection_name,
                        "model_name": result.model_name,
                        "where": json.dumps(result.where, ensure_ascii=False, sort_keys=True),
                        "rank": chunk.rank,
                        "chunk_id": chunk.chunk_id,
                        "distance": chunk.distance,
                        "document_id": chunk.document_id,
                        "chunk_type": chunk.chunk_type,
                        "section": chunk.section,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "claim_number": chunk.claim_number,
                        "expected_document_ids": " | ".join(result.expected_document_ids),
                        "expected_chunk_types": " | ".join(result.expected_chunk_types),
                        "expected_sections": " | ".join(result.expected_sections),
                        "text_preview": chunk.text_preview,
                    }
                )


def write_query_scores_csv(results: list[QueryModelResult], path: Path) -> None:
    metric_names = sorted({key for result in results for key in result.metrics})

    fieldnames = [
        "run_id",
        "query_id",
        "category",
        "query",
        "model_label",
        "collection_name",
        "model_name",
        "latency_ms",
        "expected_document_ids",
        "expected_chunk_types",
        "expected_sections",
        "where",
        *metric_names,
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "run_id": result.run_id,
                "query_id": result.query_id,
                "category": result.category,
                "query": result.query,
                "model_label": result.model_label,
                "collection_name": result.collection_name,
                "model_name": result.model_name,
                "latency_ms": round(result.latency_ms, 2),
                "expected_document_ids": " | ".join(result.expected_document_ids),
                "expected_chunk_types": " | ".join(result.expected_chunk_types),
                "expected_sections": " | ".join(result.expected_sections),
                "where": json.dumps(result.where, ensure_ascii=False, sort_keys=True),
            }
            row.update(result.metrics)
            writer.writerow(row)


def write_model_summary_csv(results: list[QueryModelResult], path: Path) -> None:
    summaries = build_model_summaries(results)

    fieldnames = [
        "model_label",
        "collection_name",
        "model_name",
        "query_count",
        "mean_mrr_doc",
        "mean_precision_at_k_doc",
        "mean_recall_at_k_doc",
        "doc_hit_rate",
        "chunk_type_hit_rate",
        "section_hit_rate",
        "mean_noisy_hits_at_k",
        "mean_latency_ms",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for summary in summaries:
            writer.writerow(summary)


def build_model_summaries(results: list[QueryModelResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[QueryModelResult]] = {}

    for result in results:
        grouped.setdefault(result.model_label, []).append(result)

    summaries: list[dict[str, Any]] = []

    for model_label, model_results in sorted(grouped.items()):
        first = model_results[0]

        summaries.append(
            {
                "model_label": model_label,
                "collection_name": first.collection_name,
                "model_name": first.model_name,
                "query_count": len(model_results),
                "mean_mrr_doc": round(mean_metric(model_results, "mrr_doc"), 4),
                "mean_precision_at_k_doc": round(
                    mean_metric(model_results, "precision_at_k_doc"), 4
                ),
                "mean_recall_at_k_doc": round(
                    mean_metric(model_results, "recall_at_k_doc"), 4
                ),
                "doc_hit_rate": round(bool_rate(model_results, "hit_at_k_doc"), 4),
                "chunk_type_hit_rate": round(
                    bool_rate(model_results, "hit_at_k_chunk_type"), 4
                ),
                "section_hit_rate": round(bool_rate(model_results, "hit_at_k_section"), 4),
                "mean_noisy_hits_at_k": round(mean_metric(model_results, "noisy_hits_at_k"), 4),
                "mean_latency_ms": round(
                    statistics.mean(result.latency_ms for result in model_results), 2
                ),
            }
        )

    return summaries


def mean_metric(results: list[QueryModelResult], metric_name: str) -> float:
    values = [float(result.metrics.get(metric_name, 0.0)) for result in results]
    return statistics.mean(values) if values else 0.0


def bool_rate(results: list[QueryModelResult], metric_name: str) -> float:
    values = [bool(result.metrics.get(metric_name, False)) for result in results]
    return sum(values) / len(values) if values else 0.0


def write_markdown_report(
    results: list[QueryModelResult],
    path: Path,
    run_id: str,
) -> None:
    summaries = build_model_summaries(results)

    with path.open("w", encoding="utf-8") as file:
        file.write(f"# PatentLLM Embedding Benchmark Report\n\n")
        file.write(f"Run ID: `{run_id}`\n\n")

        file.write("## Purpose\n\n")
        file.write(
            "This benchmark compares multiple embedding indexes on the same patent "
            "retrieval query suite. It uses weak relevance labels based on expected "
            "patent documents, expected chunk types, and expected sections. This is "
            "not a substitute for final human legal or R&D review, but it is a "
            "repeatable engineering gate before LLM report generation.\n\n"
        )

        file.write("## Important interpretation notes\n\n")
        file.write(
            "- Lower Chroma distance is better within the same collection, but distances "
            "should not be compared directly across different embedding models.\n"
        )
        file.write(
            "- `MRR` rewards models that return the first expected document earlier.\n"
        )
        file.write(
            "- `Precision@k` measures how much of the returned top-k belongs to expected documents.\n"
        )
        file.write(
            "- `Recall@k` measures whether expected documents were recovered in the top-k.\n"
        )
        file.write(
            "- `noisy_hits_at_k` counts metadata, title-abstract-claims, and search-report-like chunks.\n\n"
        )

        file.write("## Model summary\n\n")
        file.write(
            "| Model | Collection | Mean MRR | Precision@k | Recall@k | Doc Hit Rate | "
            "Chunk-Type Hit Rate | Section Hit Rate | Noisy Hits@k | Mean Latency ms |\n"
        )
        file.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for summary in summaries:
            file.write(
                f"| {summary['model_label']} "
                f"| `{summary['collection_name']}` "
                f"| {summary['mean_mrr_doc']} "
                f"| {summary['mean_precision_at_k_doc']} "
                f"| {summary['mean_recall_at_k_doc']} "
                f"| {summary['doc_hit_rate']} "
                f"| {summary['chunk_type_hit_rate']} "
                f"| {summary['section_hit_rate']} "
                f"| {summary['mean_noisy_hits_at_k']} "
                f"| {summary['mean_latency_ms']} |\n"
            )

        file.write("\n## Query-by-query results\n\n")

        for query_id in sorted({result.query_id for result in results}):
            query_results = [result for result in results if result.query_id == query_id]
            query_case = query_results[0]

            file.write(f"### {query_id} — {query_case.category}\n\n")
            file.write(f"Query: `{query_case.query}`\n\n")
            file.write(
                f"Expected documents: `{', '.join(query_case.expected_document_ids)}`\n\n"
            )
            if query_case.where:
                file.write(
                    f"Metadata filter: `{json.dumps(query_case.where, sort_keys=True)}`\n\n"
                )

            file.write(
                "| Model | MRR | Precision@k | Recall@k | First Expected Rank | "
                "Noisy Hits | Latency ms | Top Retrieved Documents |\n"
            )
            file.write("|---|---:|---:|---:|---:|---:|---:|---|\n")

            for result in sorted(query_results, key=lambda item: item.model_label):
                top_docs = " → ".join(chunk.document_id for chunk in result.results[:5])
                file.write(
                    f"| {result.model_label} "
                    f"| {round(float(result.metrics['mrr_doc']), 4)} "
                    f"| {round(float(result.metrics['precision_at_k_doc']), 4)} "
                    f"| {round(float(result.metrics['recall_at_k_doc']), 4)} "
                    f"| {result.metrics['rank_first_expected_doc']} "
                    f"| {result.metrics['noisy_hits_at_k']} "
                    f"| {round(result.latency_ms, 2)} "
                    f"| {top_docs} |\n"
                )

            file.write("\nTop chunks:\n\n")

            for result in sorted(query_results, key=lambda item: item.model_label):
                file.write(f"**{result.model_label}**\n\n")
                for chunk in result.results[:3]:
                    file.write(
                        f"- Rank {chunk.rank}: `{chunk.document_id}` | "
                        f"`{chunk.chunk_type}` | `{chunk.section}` | "
                        f"pages {chunk.page_start}-{chunk.page_end} | "
                        f"distance={chunk.distance}\n"
                    )
                    file.write(f"  - Preview: {chunk.text_preview[:350]}\n")
                file.write("\n")

        file.write("## Recommended decision rule\n\n")
        file.write(
            "Use the model with the best combination of high MRR, high expected-document "
            "hit rate, low noisy hits, and acceptable latency. If one model performs better "
            "on claims and another performs better on examples/descriptions, keep both and "
            "route queries by category in the next retrieval layer.\n"
        )


if __name__ == "__main__":
    main()
