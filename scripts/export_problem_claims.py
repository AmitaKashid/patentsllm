import json
from pathlib import Path

problem_docs = {
    "WO_2012092233_A2",
    "WO_2013039261_A1",
    "WO_2013162059_A1",
    "WO_2017123874_A1",
}

paragraphs_path = Path("data/processed/parsed_patents/paragraphs.jsonl")
out_path = Path("data/processed/parser_audit/problem_claim_paragraphs.txt")

rows = [json.loads(line) for line in paragraphs_path.open(encoding="utf-8")]

with out_path.open("w", encoding="utf-8") as f:
    for doc in sorted(problem_docs):
        f.write("\n" + "=" * 120 + "\n")
        f.write(doc + "\n")
        f.write("=" * 120 + "\n")

        doc_rows = [
            r for r in rows
            if r.get("document_id") == doc and r.get("section") == "claims"
        ]

        for r in doc_rows:
            f.write(
                f"\n--- page={r.get('page_number')} "
                f"paragraph_id={r.get('paragraph_id')} "
                f"tokens={r.get('token_estimate')} ---\n"
            )
            f.write(r.get("text", "")[:2000])
            f.write("\n")

print(f"Wrote {out_path}")
