\# Embedding Benchmark Report



\## 1. Objective



This report documents the embedding-model evaluation performed for the PatentLLM retrieval pipeline.



The goal of this benchmark is to compare multiple embedding models on the same parsed patent corpus before connecting retrieval results to LLM-based report generation. The benchmark is designed as an engineering quality gate: it checks whether the retrieval layer returns the correct patent documents, the correct chunk types, and usable evidence passages for downstream report generation.



This benchmark does not replace expert legal, patent, or R\&D review. It validates whether the retrieval system is reliable enough to supply grounded context to an LLM.



\---



\## 2. Retrieval Setup



The benchmark was run on parsed and quality-filtered patent chunks generated from 10 patent PDF documents.



The indexed chunks represent evidence-bearing parts of the patents, including:



| Chunk Type                     | Purpose                                                                 |

| ------------------------------ | ----------------------------------------------------------------------- |

| `claim`                        | Legal and technical claim scope                                         |

| `claim\_clause\_window`          | Long claim fragments split into smaller retrieval units                 |

| `description\_paragraph\_window` | Technical description, background, summary, and implementation passages |

| `evidence\_paragraph\_window`    | Examples, test methods, experimental passages, and performance evidence |



Chunks that are not reliable as primary evidence are excluded or flagged during indexing. These include metadata-only chunks, low-confidence claims, search-report-like chunks, and table-heavy OCR regions.



The vector store used for this benchmark is Chroma, with each indexed record storing:



| Field                     | Purpose                                     |

| ------------------------- | ------------------------------------------- |

| `chunk\_id`                | Stable unique identifier for the chunk      |

| `document\_id`             | Patent-level identifier                     |

| `chunk\_type`              | Type of parsed chunk                        |

| `section`                 | Patent section detected during parsing      |

| `page\_start` / `page\_end` | Source-page traceability                    |

| `text`                    | Chunk text used for embedding and retrieval |



\---



\## 3. Compared Embedding Models



Two local embedding models were compared.



| Model                        | Chroma Collection            | Role in Benchmark                                           |

| ---------------------------- | ---------------------------- | ----------------------------------------------------------- |

| `BAAI/bge-m3`                | `patent\_chunks\_bge\_m3`       | General technical retrieval model for mixed patent evidence |

| `AI-Growth-Lab/PatentSBERTa` | `patent\_chunks\_patentsberta` | Patent-specialized semantic retrieval baseline              |



Both models were evaluated on the same query suite, same top-k setting, same vector-store backend, and same parsed patent chunk corpus.



\---



\## 4. Query Suite Design



The benchmark uses 10 manually defined patent-retrieval queries. The query suite was designed to reflect the types of retrieval needed for R\&D and innovation-report generation.



| Query Category                  | What It Tests                                                                                                                                     |

| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |

| Claim-scope retrieval           | Whether formulation and composition questions retrieve claim chunks                                                                               |

| Technology-route retrieval      | Whether polymer-route concepts retrieve the right patent families                                                                                 |

| Experimental-evidence retrieval | Whether examples, test methods, peel strength, creep retention, viscosity, open time, and heat-resistance passages are retrieved                  |

| Application-need retrieval      | Whether use-case-specific concepts such as wet adhesion, stretch laminates, absorbent articles, and hydrophilic nonwoven substrates are retrieved |

| Metadata-filter retrieval       | Whether Chroma metadata filters such as `document\_id` and `chunk\_type` work correctly                                                             |



Each query contains weak relevance labels:



| Label Type                | Purpose                                                     |

| ------------------------- | ----------------------------------------------------------- |

| Expected document IDs     | Checks whether the correct patent documents appear in top-k |

| Expected chunk types      | Checks whether the correct evidence type appears            |

| Expected sections         | Checks whether the expected patent section is retrieved     |

| Optional metadata filters | Checks whether filtered retrieval works correctly           |



These labels are intentionally lightweight. They are used to validate retrieval behavior before LLM generation, not to create a final legal relevance judgment.



\---



\## 5. Evaluation Metrics



The benchmark computes both model-level and query-level metrics.



| Metric                | Interpretation                                                                                            |

| --------------------- | --------------------------------------------------------------------------------------------------------- |

| `MRR`                 | Rewards models that return the first expected patent earlier in the ranking                               |

| `Precision@k`         | Measures how many of the top-k chunks come from expected patent documents                                 |

| `Recall@k`            | Measures whether expected patent documents were recovered in top-k                                        |

| `Doc Hit Rate`        | Measures whether at least one expected document appeared in top-k                                         |

| `Chunk-Type Hit Rate` | Measures whether the expected chunk type appeared in top-k                                                |

| `Section Hit Rate`    | Measures whether the expected section appeared in top-k                                                   |

| `Noisy Hits@k`        | Counts undesirable chunks such as metadata, title/abstract summary chunks, and search-report-like content |

| `Mean Latency ms`     | Average query latency for the embedding collection                                                        |



Chroma distances are interpreted only within the same embedding collection. Distances are not compared directly across BGE-M3 and PatentSBERTa because each embedding model creates a different vector space.



\---



\## 6. Benchmark Results



\### 6.1 Model-Level Summary



| Model          | Collection                   | Mean MRR | Precision@k | Recall@k | Doc Hit Rate | Chunk-Type Hit Rate | Section Hit Rate | Noisy Hits@k | Mean Latency ms |

| -------------- | ---------------------------- | -------: | ----------: | -------: | -----------: | ------------------: | ---------------: | -----------: | --------------: |

| `bge\_m3`       | `patent\_chunks\_bge\_m3`       |     1.00 |        0.82 |     0.98 |         1.00 |                1.00 |             1.00 |         0.30 |          125.02 |

| `patentsberta` | `patent\_chunks\_patentsberta` |     0.90 |        0.82 |     0.98 |         1.00 |                1.00 |             1.00 |         0.30 |           41.45 |



\### 6.2 Interpretation



BGE-M3 achieved the stronger ranking result with a Mean MRR of 1.00. This means the first expected patent document appeared earlier in the ranked results across the query suite.



PatentSBERTa was faster, with a mean latency of 41.45 ms compared with 125.02 ms for BGE-M3. It also matched BGE-M3 on Precision@k, Recall@k, document hit rate, chunk-type hit rate, section hit rate, and noisy-hit rate.



The main difference is therefore not whether the models can retrieve relevant documents at all. Both can. The difference is how reliably the model ranks the most expected patent earlier.



\---



\## 7. Query-Level Observations



\### 7.1 Claim-Scope Retrieval



For the query:



```text

hot melt adhesive composition comprising polymer tackifier plasticizer stabilizer wax

```



Both models retrieved claim chunks successfully. BGE-M3 ranked a claim from `WO\_2013162059\_A1` first, while PatentSBERTa ranked a claim from `WO\_2023099461\_A1` first.



This confirms that both models can support claim-focused retrieval when `chunk\_type = claim` filtering is applied.



\### 7.2 Olefin Block Copolymer Route



For the query:



```text

olefin block copolymer hot melt adhesive nonwoven polyethylene film

```



BGE-M3 retrieved `WO\_2011011729\_A1` at rank 1. PatentSBERTa retrieved the same expected document, but at rank 2.



This is an important result because `WO\_2011011729\_A1` is the expected patent for the olefin block copolymer route. BGE-M3 performed better for this technology-route query.



\### 7.3 Single-Site Catalyst Propylene Systems



For the query:



```text

propylene polymer hot melt adhesive prepared using single site catalysts

```



Both models performed well. BGE-M3 retrieved `WO\_2013039261\_A1`, `WO\_2023099461\_A1`, and `WO\_2017123874\_A1`. PatentSBERTa strongly retrieved `WO\_2017123874\_A1` passages related to SSC-PP polymer blends.



This suggests that PatentSBERTa remains useful for patent-specific similarity and claim-oriented comparison, while BGE-M3 provides broader technical coverage.



\### 7.4 Experimental Evidence Retrieval



For the query:



```text

experimental examples peel strength viscosity open time heat resistance

```



Both models retrieved evidence chunks successfully when `chunk\_type = evidence\_paragraph\_window` filtering was applied.



BGE-M3 ranked `WO\_2013162059\_A1` first, while PatentSBERTa ranked `WO\_2016140830\_A1` first. Both retrieved expected documents and example/test-method passages.



This validates the evidence-only retrieval mode.



\### 7.5 Wet Adhesion Application Query



For the query:



```text

wet adhesion hydrophilic nonwoven material based products acid grafted polyolefin

```



BGE-M3 retrieved `WO\_2023099461\_A1` at rank 1 with claims relating to acid-grafted polyolefin copolymers. PatentSBERTa retrieved the expected document at rank 2.



This supports selecting BGE-M3 as the default retriever for application-specific R\&D queries.



\### 7.6 Metadata-Filtered Retrieval



For the query:



```text

propylene copolymers single-site catalysts low surface energy substrates

```



with the metadata filter:



```json

{"document\_id": "WO\_2017123874\_A1"}

```



both models returned only chunks from `WO\_2017123874\_A1`.



This validates the metadata-filtering layer and confirms that document-specific retrieval can be used in downstream report-generation workflows.



\---



\## 8. Selected Retrieval Strategy



Based on the benchmark, `BAAI/bge-m3` is selected as the default embedding model for the primary PatentLLM evidence retriever.



The decision is based on the following results:



1\. BGE-M3 achieved the highest Mean MRR.

2\. BGE-M3 matched PatentSBERTa on Precision@k, Recall@k, document hit rate, chunk-type hit rate, section hit rate, and noisy-hit rate.

3\. BGE-M3 ranked expected documents earlier for key R\&D-oriented queries, including olefin block copolymer retrieval, wet adhesion, stretch laminates, and polymer-system viscosity reduction.

4\. BGE-M3 performed consistently across mixed chunk types: claims, descriptions, examples, and application-specific passages.



PatentSBERTa is retained as a secondary benchmark. It may be useful for future claim-similarity experiments, patent-to-patent similarity, or routed retrieval where the query is specifically claim-heavy.



\---



\## 9. Known Limitations



The benchmark uses weak relevance labels rather than expert-annotated relevance judgments. This is acceptable for the current engineering stage but should be improved later with human-validated relevance labels.



Some retrieved chunks still include OCR artifacts because the original patents are scanned/image-style PDFs. The parser and indexing pipeline reduce this risk through chunk filtering, but OCR noise is not fully eliminated.



The current benchmark evaluates dense vector retrieval only. BGE-M3 also supports sparse and multi-vector retrieval modes, but those are not yet implemented in this pipeline.



Some table-heavy evidence is still not structurally extracted. For exact formulation tables, peel-strength values, viscosity values, or performance comparisons, a later table-extraction module is required.



\---



\## 10. Benchmark Output Files



Each benchmark run writes a timestamped output directory:



```text

data/processed/retrieval\_benchmark/<run\_id>/

├── benchmark\_report.md

├── model\_summary.csv

├── query\_model\_scores.csv

├── retrieval\_results.csv

└── retrieval\_results.jsonl

```



| File                      | Description                                             |

| ------------------------- | ------------------------------------------------------- |

| `benchmark\_report.md`     | Human-readable benchmark summary                        |

| `model\_summary.csv`       | Aggregated model-level metrics                          |

| `query\_model\_scores.csv`  | Query-level scores for every model                      |

| `retrieval\_results.csv`   | Flattened top-k retrieval results for manual inspection |

| `retrieval\_results.jsonl` | Full structured result output for downstream analysis   |



Generated benchmark outputs are not committed to Git. The report under `docs/` summarizes the benchmark results in a stable and reviewable format.



\---



\## 11. How to Reproduce



Build the BGE-M3 index:



```powershell

python scripts/build\_index.py `

&#x20; --chunks-path data/processed/parsed\_patents/chunks.jsonl `

&#x20; --persist-dir data/processed/vector\_store/chroma `

&#x20; --collection-name patent\_chunks\_bge\_m3 `

&#x20; --embedding-backend sentence\_transformer `

&#x20; --model-name BAAI/bge-m3 `

&#x20; --batch-size 8

```



Build the PatentSBERTa index:



```powershell

python scripts/build\_index.py `

&#x20; --chunks-path data/processed/parsed\_patents/chunks.jsonl `

&#x20; --persist-dir data/processed/vector\_store/chroma `

&#x20; --collection-name patent\_chunks\_patentsberta `

&#x20; --embedding-backend sentence\_transformer `

&#x20; --model-name AI-Growth-Lab/PatentSBERTa `

&#x20; --batch-size 16

```



Run the embedding benchmark:



```powershell

python scripts/benchmark\_embeddings.py `

&#x20; --persist-dir data/processed/vector\_store/chroma `

&#x20; --top-k 5 `

&#x20; --output-dir data/processed/retrieval\_benchmark

```



\---



\## 12. Current Decision



The current retrieval configuration is:



| Component                 | Selection                                      |

| ------------------------- | ---------------------------------------------- |

| Primary embedding model   | `BAAI/bge-m3`                                  |

| Primary Chroma collection | `patent\_chunks\_bge\_m3`                         |

| Secondary benchmark model | `AI-Growth-Lab/PatentSBERTa`                   |

| Vector store              | Chroma                                         |

| Evaluation top-k          | 5                                              |

| Retrieval strategy        | Dense vector retrieval with metadata filtering |



The retrieval layer is considered ready for the next stage: connecting retrieved patent chunks to a controlled LLM report-generation pipeline.



Every generated report insight should be grounded in retrieved chunks and should preserve source traceability through document ID, section, chunk type, and page range.



