# DocBenchTR

**A pilot benchmark for retrieval on long, structured Turkish documents**<br>
Ümit Altınöz · Beykent University

DocBenchTR is a research proposal comparing retrieval methods for retrieval-augmented generation (RAG) over Turkish documents with explicit articles, sections, and cross-references. Turkish is underrepresented in common retrieval evaluations, while legal and institutional documents often require finding a provision by meaning and structure rather than by a repeated keyword.

This public pilot reports retrieval only. It evaluates physical PDF pages as retrieval units so results are comparable across methods with different internal indexes. It does not generate answers or assess answer faithfulness.

## Methods

- **BM25:** lexical page ranking.
- **BM25 (Turkish stemmed):** Snowball Turkish stemming before lexical ranking.
- **Dense:** `intfloat/multilingual-e5-base`, with query and passage prefixes.
- **Hybrid (RRF):** reciprocal rank fusion of the unstemmed BM25 and dense rankings.
- **PageIndex agentic (default instructions):** the [PageIndex](https://github.com/VectifyAI/PageIndex) local SDK uses its built-in instructions with `qwen2.5:7b-instruct` through LiteLLM's `ollama_chat/` route for tool-using chat/search.
- **PageIndex agentic (read-pages instruction):** the same SDK with our additional retrieval instruction requiring a successful `get_page_content` result before answering. This is an explicit modification of the default agentic method. For both rows, `tool_call_count` counts every tool call; only successful `get_page_content` results contribute pages to its ranking.
- **PageIndex LLM Tree Search:** gives the local model the cached PageIndex tree (titles, node IDs, inclusive physical page ranges, and summaries) plus the question. It requests `{"thinking": ..., "node_list": [...]}` with Ollama JSON-schema output. The ranking expands each selected node's inclusive `start_index..end_index` range in returned `node_list` order and deduplicates pages by first occurrence. Parse failures remain in the per-query results as failed rows.

All methods return ranked, one-based physical PDF page numbers. The dense model uses CUDA when available and otherwise runs on CPU; the actual device is written to the generated results.

## Pilot setup

The pipeline downloads three openly published Turkish legal documents, checks each file against its pinned SHA-256 snapshot, and records the fetched source URL and digest in [`data/sources.json`](data/sources.json). It extracts text by physical page, drafts questions from article text, and checks every answer span against the cited extracted page before retrieval evaluation. The kept questions are in [`data/pilot_questions.json`](data/pilot_questions.json); the per-question article and page verification log is in [`data/question_verification.json`](data/question_verification.json). Failed drafts are excluded. The question set is labelled **pilot set, auto-drafted from source text, span-verified; not yet human-reviewed**.

The question types are direct fact, paraphrase, cross-section (two articles), and distractor (overlapping wording elsewhere in the document). Each method is evaluated within the question's source document. The candidate relevance set is the exact source page or pages in `source_pages`.

Source PDFs are downloaded to `data/raw/` and are intentionally excluded from Git. Their legal texts are officially published and may be reused under the official-text provision of the Turkish Copyright Act; `data/sources.json` retains each source's reuse note and links. The PDFs are snapshots, not a claim that the text reflects current law. The archived KVKK PDF reflects the text available in two thousand twenty, and the Consumer Protection PDF is an agency-hosted January edition from two thousand twenty-one. Consult the source agency for authoritative current wording.

### Reproduce

Requires Python three point twelve, `uv`, an Xpdf or Poppler `pdftotext` executable on `PATH`, an internet connection for source PDFs and the embedding model, and a CUDA-capable PyTorch install for GPU embedding. The project pins runtime dependencies and selects the CUDA twelve point six PyTorch wheel. The actual extractor version is stored in `data/sources.json`.

```powershell
uv sync --group dev
uv run python -m docbenchtr.run_pilot
```

The run regenerates the verified question set, CSV and JSON results, the chart, and the results block in this README. To download and hash only the source documents, run `uv run python scripts/download_docs.py`.

For PageIndex, install and start Ollama locally and pull `qwen2.5:7b-instruct` before running the pilot. Both agentic variants use the SDK's tool definitions through LiteLLM's `ollama_chat/` route and share the local cached trees under `.pageindex/docbenchtr-pilot/`. PageIndex LLM Tree Search sends the cached tree and question to Ollama's `/api/chat` endpoint with a JSON schema. These methods make no cloud calls. The pipeline checks the local tags endpoint and records per-query failures.

## Results

The generated table and chart below are populated from the run's `results/summary.json`. Per-query values are in [`results/results.csv`](results/results.csv) and [`results/per_query.json`](results/per_query.json).

## How to read the PageIndex rows

These rows are **not a verdict on PageIndex**. Every PageIndex variant here runs on `qwen2.5:7b-instruct`, a small local model that fits an 8 GB laptop GPU, while PageIndex's own documentation recommends the strongest chat model you can afford for tree search. What the pilot does show, at this scale:

- **Agentic mode never reached a page.** With both the default and our stricter instruction, the model requested the document structure but never called `get_page_content`, so all 36 rankings are empty and score zero. A direct request confirmed that the tool definitions reach the model, so this is model behaviour, not a wiring failure.
- **LLM Tree Search works but trails the classical baselines** (Recall@5 0.403 vs 0.792 for Hybrid), with 12 of 36 queries producing no usable node selection, and it is roughly four orders of magnitude slower.
- Queries a method attempted but could not answer are **counted as zero, not dropped**; the coverage columns show how many.

Whether PageIndex closes this gap with a frontier model is exactly the question the full DocBenchTR study is designed to answer.

<!-- GENERATED:START -->
## Pilot results

This generated section reads its metrics from `results/summary.json` (per-question records: `results/results.csv` and `results/per_query.json`).

| Method | Attempted | Scored | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 36 | 36 | 0 | 0 | 0.278 | 0.653 | 0.764 | 0.532 | 1.325 ms |
| BM25 (Turkish stemmed) | 36 | 36 | 0 | 0 | 0.250 | 0.708 | 0.722 | 0.526 | 2.367 ms |
| Dense (multilingual-e5-base) | 36 | 36 | 0 | 0 | 0.403 | 0.667 | 0.750 | 0.627 | 384.373 ms |
| Hybrid (RRF) | 36 | 36 | 0 | 0 | 0.333 | 0.708 | 0.792 | 0.612 | 14.965 ms |
| PageIndex agentic (default instructions) | 36 | 36 | 36 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 14365.587 ms |
| PageIndex agentic (read-pages instruction) | 36 | 36 | 36 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 9251.201 ms |
| PageIndex LLM Tree Search | 36 | 36 | 12 | 0 | 0.236 | 0.347 | 0.403 | 0.365 | 76580.507 ms |

![Recall@5 by method and question type](results/recall_at_5_by_question_type.png)

### Metrics by question type

| Method | Question type | Attempted | Scored | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | cross section | 9 | 9 | 0 | 0 | 0.111 | 0.500 | 0.611 | 0.500 | 1.389 ms |
| BM25 | direct fact | 9 | 9 | 0 | 0 | 0.444 | 0.667 | 0.667 | 0.564 | 1.202 ms |
| BM25 | distractor | 9 | 9 | 0 | 0 | 0.222 | 0.667 | 0.889 | 0.488 | 1.400 ms |
| BM25 | paraphrase | 9 | 9 | 0 | 0 | 0.333 | 0.778 | 0.889 | 0.578 | 1.308 ms |
| BM25 (Turkish stemmed) | cross section | 9 | 9 | 0 | 0 | 0.111 | 0.722 | 0.778 | 0.574 | 2.946 ms |
| BM25 (Turkish stemmed) | direct fact | 9 | 9 | 0 | 0 | 0.444 | 0.667 | 0.667 | 0.582 | 1.796 ms |
| BM25 (Turkish stemmed) | distractor | 9 | 9 | 0 | 0 | 0.111 | 0.778 | 0.778 | 0.452 | 2.469 ms |
| BM25 (Turkish stemmed) | paraphrase | 9 | 9 | 0 | 0 | 0.333 | 0.667 | 0.667 | 0.495 | 2.257 ms |
| Dense (multilingual-e5-base) | cross section | 9 | 9 | 0 | 0 | 0.278 | 0.667 | 0.778 | 0.741 | 479.459 ms |
| Dense (multilingual-e5-base) | direct fact | 9 | 9 | 0 | 0 | 0.444 | 0.556 | 0.778 | 0.563 | 285.518 ms |
| Dense (multilingual-e5-base) | distractor | 9 | 9 | 0 | 0 | 0.333 | 0.667 | 0.667 | 0.524 | 395.568 ms |
| Dense (multilingual-e5-base) | paraphrase | 9 | 9 | 0 | 0 | 0.556 | 0.778 | 0.778 | 0.682 | 376.947 ms |
| Hybrid (RRF) | cross section | 9 | 9 | 0 | 0 | 0.222 | 0.500 | 0.833 | 0.667 | 15.230 ms |
| Hybrid (RRF) | direct fact | 9 | 9 | 0 | 0 | 0.444 | 0.778 | 0.778 | 0.614 | 14.463 ms |
| Hybrid (RRF) | distractor | 9 | 9 | 0 | 0 | 0.222 | 0.667 | 0.667 | 0.488 | 14.594 ms |
| Hybrid (RRF) | paraphrase | 9 | 9 | 0 | 0 | 0.444 | 0.889 | 0.889 | 0.678 | 15.573 ms |
| PageIndex agentic (default instructions) | cross section | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 14812.862 ms |
| PageIndex agentic (default instructions) | direct fact | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 15086.221 ms |
| PageIndex agentic (default instructions) | distractor | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 14628.899 ms |
| PageIndex agentic (default instructions) | paraphrase | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 12934.369 ms |
| PageIndex agentic (read-pages instruction) | cross section | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 8635.745 ms |
| PageIndex agentic (read-pages instruction) | direct fact | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 9121.433 ms |
| PageIndex agentic (read-pages instruction) | distractor | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 8918.523 ms |
| PageIndex agentic (read-pages instruction) | paraphrase | 9 | 9 | 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 10329.105 ms |
| PageIndex LLM Tree Search | cross section | 9 | 9 | 3 | 0 | 0.278 | 0.389 | 0.500 | 0.578 | 76242.011 ms |
| PageIndex LLM Tree Search | direct fact | 9 | 9 | 3 | 0 | 0.222 | 0.333 | 0.333 | 0.278 | 75764.868 ms |
| PageIndex LLM Tree Search | distractor | 9 | 9 | 3 | 0 | 0.333 | 0.333 | 0.444 | 0.371 | 75916.688 ms |
| PageIndex LLM Tree Search | paraphrase | 9 | 9 | 3 | 0 | 0.111 | 0.333 | 0.333 | 0.235 | 78398.461 ms |

### Run metadata

- Documents: 3 — Kişisel Verilerin Korunması Kanunu, Tüketicinin Korunması Hakkında Kanun ve İkincil Mevzuatı (Ocak 2021), Yükseköğretim Kanunu.
- Span-verified pilot questions: 36; excluded after verification: 0.
- Physical PDF pages extracted: 500 across kvkk_6698: 18, tuketicinin_korunmasi_6502: 340, yuksekogretim_2547: 142.
- Embedding model: `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a` on `cuda`.
- Question set label: **pilot set, auto-drafted from source text, span-verified; not yet human-reviewed**.
- PageIndex agentic (default instructions) queries: 36 attempted, 36 scored, 36 zero-ranking, 0 excluded.
- PageIndex agentic (read-pages instruction) queries: 36 attempted, 36 scored, 36 zero-ranking, 0 excluded.
- PageIndex agentic (default instructions) token counts: available for 36 queries; 140728 input and 15000 output tokens total.
- PageIndex agentic (read-pages instruction) token counts: available for 36 queries; 143824 input and 15789 output tokens total.
- PageIndex LLM Tree Search queries: 36 attempted, 36 scored, 12 zero-ranking, 0 excluded; 12 parse failures.
- PageIndex LLM HTTP safety net: retry a request after 30 minutes without response progress, up to 3 retries; 0 retries recorded; there is no total index or query deadline.

### PageIndex index builds

- `kvkk_6698`: reused; build cached; prior build time unavailable; `toc_source=detected`; cache validation 15.4 s
- `tuketicinin_korunmasi_6502`: reused; build cached; prior build time unavailable; `toc_source=detected`; cache validation 3.8 s
- `yuksekogretim_2547`: reused; build cached; prior build time unavailable; `toc_source=detected`; cache validation 3.9 s

PageIndex agentic default-instruction and read-pages-instruction variants use `ollama_chat/qwen2.5:7b-instruct` through Ollama and share the cached trees. The read-pages instruction is an explicit modification to the SDK default. `tool_call_count` counts every agent tool invocation; page ranking uses only successful `get_page_content` results, in tool-call order and each result's `returned_pages` order, retaining each physical page's first occurrence. Outline-only calls count as tool calls but do not enter the ranking. PageIndex LLM Tree Search asks the same local model to select nodes from the cached tree using Ollama's JSON-schema output. It expands each selected node's inclusive one-based `start_index..end_index` range in returned `node_list` order and deduplicates pages by first occurrence. Attempted queries without a usable ranking, including parse failures, score zero; infrastructure exclusions remain outside the means. Both methods return no PageIndex page scores; per-query errors and traces are preserved in `results/per_query.json` and `results/results.csv`.
<!-- GENERATED:END -->

## Limitations

- This is a small pilot set, not a statistically powered comparison.
- Questions are auto-drafted from source text and span-verified, but have not yet been reviewed by human annotators. A matching text span confirms provenance, not question quality or answer completeness.
- Evaluation measures retrieval only; generation, answer correctness, and faithfulness are out of scope.
- The dense comparison uses a single multilingual embedding model.
- PDF pagination, extraction quality, edition dates, and the balance of document topics can affect retrieval scores.
- PageIndex agentic page rankings are reconstructed from successful `get_page_content` tool results, ordered by the agent's tool-call sequence and each result's `returned_pages`; repeated pages keep their first rank. Outline-only calls count in `tool_call_count` but do not rank pages. PageIndex returns no page scores. Per-query traces and SDK token usage, when present, are retained.
- A method that was attempted but produced no usable ranking receives recall@k = 0 and reciprocal rank = 0 and is included in the overall and per-question-type means. This includes completed agentic queries with no successful page reads and LLM Tree Search JSON/schema parse failures. Queries excluded because the document was not indexed or an infrastructure/tool error prevented the attempt remain outside the means. Each method and question type reports attempted, scored, zero-ranking, and excluded counts so coverage is visible.
- PageIndex agentic page rankings are reconstructed from successful `get_page_content` tool results, ordered by the agent's tool-call sequence and each result's `returned_pages`; repeated pages keep their first rank. Outline-only calls count in `tool_call_count` but do not rank pages. PageIndex returns no page scores. Per-query traces and SDK token usage, when present, are retained.
- PageIndex LLM Tree Search is a separate non-agentic method. It ranks inclusive one-based page ranges from model-selected cached-tree nodes in returned node order, deduplicating by first occurrence. JSON parse or schema failures are recorded per query and are not silently dropped.
- PageIndex used the small local `qwen2.5:7b-instruct` model for both indexing and search. Its developers say a basic model is sufficient for indexing but recommend the [best model you can afford for chat/search](https://github.com/VectifyAI/PageIndex#model-recommendations); the 7B search model may disadvantage its results against that recommendation.
- PageIndex's tree `toc_source`, per-document build time, and any document or query failures are listed in the generated run metadata and per-query files.

## Roadmap to the full study

The proposed full study will expand to approximately one thousand human-reviewed questions, compare nine retrieval methods plus two controls (an agentic hybrid control and a long-context baseline), and use one fixed generator LLM for any answer-generation evaluation. The final protocol will preregister document snapshots, question sampling, model versions, and scoring rules before the full comparison.

## Kısa Türkçe özet

DocBenchTR, uzun ve maddeli Türkçe belgelerde RAG için farklı erişim yöntemlerini karşılaştırmayı amaçlayan bir araştırma önerisidir. Bu herkese açık pilot; üç resmî mevzuat PDF'inden otomatik taslak sorular üretir, yanıt parçalarını belirtilen sayfada programatik olarak doğrular ve BM25, Türkçe köklendirmeli BM25, çok dilli yoğun erişim, hibrit RRF ve çalıştırılabildiğinde yerel PageIndex sonuçlarını sayfa düzeyinde karşılaştırır. Küçük ve henüz insan incelemesinden geçmemiş soru kümesi yalnızca ilk sonuçları gösterir; yanıt üretimi ve doğruluğu bu pilotun kapsamı dışındadır.

## License

The code is licensed under the MIT License. The included legal texts remain subject to their source and publication notes; no PDFs or model weights are committed.
