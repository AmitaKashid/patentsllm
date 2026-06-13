"""Find claim-like text in parsed patent pages.

Use this as a diagnostic script when the quality report says claims_found=False.
It scans OCR page text for numbered claim patterns near the end of each document.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


CLAIM_START_RE = re.compile(
    r"(?m)^\s*(?:claim\s*)?\d{1,3}\s*[\.)]\s+"
    r"(?:A|An|The|Use|Method|Composition|Adhesive|Polymer|Hot|Process)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find claim-like candidate pages.")
    parser.add_argument(
        "--pages-path",
        type=Path,
        default=Path("data/processed/parsed_patents/pages.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pages_by_doc: dict[str, list[dict]] = defaultdict(list)

    with args.pages_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            pages_by_doc[row["document_id"]].append(row)

    for document_id, pages in sorted(pages_by_doc.items()):
        total_pages = len(pages)
        start_page = max(1, int(total_pages * 0.55))

        candidates = []
        for page in pages:
            page_number = int(page["page_number"])
            if page_number < start_page:
                continue

            text = page.get("text", "")
            matches = list(CLAIM_START_RE.finditer(text))
            if matches:
                snippet_start = max(0, matches[0].start() - 120)
                snippet_end = min(len(text), matches[0].start() + 500)
                candidates.append(
                    {
                        "page_number": page_number,
                        "match_count": len(matches),
                        "snippet": text[snippet_start:snippet_end].replace("\n", " "),
                    }
                )

        print("\n" + "=" * 100)
        print(document_id)
        print(f"total_pages={total_pages}")
        print(f"claim_candidate_pages={len(candidates)}")

        for candidate in candidates[:8]:
            print("-" * 100)
            print(f"page={candidate['page_number']} matches={candidate['match_count']}")
            print(candidate["snippet"][:700])


if __name__ == "__main__":
    main()