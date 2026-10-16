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

<!-- GENERATED:START -->
## Pilot results

No benchmark numbers are published yet. The maintainer should run the full pipeline to generate the results table and chart. The agentic and LLM tree-search methods will appear as separate rows.
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
