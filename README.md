# DocBenchTR

**A pilot benchmark for retrieval on long, structured Turkish documents**<br>
Ümit Altınöz · Beykent University

DocBenchTR is a research proposal comparing retrieval methods for retrieval-augmented generation (RAG) over Turkish documents with explicit articles, sections, and cross-references. Turkish is underrepresented in common retrieval evaluations, while legal and institutional documents often require finding a provision by meaning and structure rather than by a repeated keyword.

This public pilot reports retrieval only. It evaluates physical PDF pages as retrieval units so results are comparable across methods with different internal indexes. It does not generate answers or assess answer faithfulness.

**What the 36-question pilot supports:** PageIndex with a 7B model is clearly and significantly behind Hybrid. With the 27B model at 64k, Tree Search scores numerically higher than Hybrid, but the difference is not statistically distinguishable on these 36 questions (Recall@5 Δ +0.125, paired 95% CI [−0.028, +0.278], exact sign-test p = 0.227).

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

The pipeline reuses or downloads three openly published Turkish legal documents, checks each file against its pinned SHA-256 snapshot, and records the source URL and digest in [`data/sources.json`](data/sources.json). It extracts text by physical page and evaluates 36 hand-written question stems with source spans selected from the article text. The validator checks span provenance and every planned term. Four questions fail the planned-term check; they remain in the published results and are listed in [`data/question_verification.json`](data/question_verification.json). The question set is not yet human-reviewed.

The question types are direct fact, paraphrase, cross-section (two articles), and distractor (overlapping wording elsewhere in the document). Each method is evaluated within the question's source document. The candidate relevance set is the exact source page or pages in `source_pages`.

The three pinned source PDFs in `data/raw/` and the PageIndex cache in `.pageindex/` are committed to Git so Colab can reuse the same documents and trees. Their legal texts are officially published and may be reused under the official-text provision of the Turkish Copyright Act; `data/sources.json` retains each source's reuse note and links. The PDFs are snapshots, not a claim that the text reflects current law. The archived KVKK PDF reflects the text available in 2020, and the Consumer Protection PDF is an agency-hosted January 2021 edition. Consult the source agency for authoritative current wording.

### Reproduce

Requires Python 3.12, `uv`, an Xpdf or Poppler `pdftotext` executable on `PATH`, an internet connection for source PDFs and the embedding model, and a CUDA-capable PyTorch install for GPU embedding. The project pins runtime dependencies and selects the CUDA 12.6 PyTorch wheel. The actual extractor version is stored in `data/sources.json`.

```powershell
uv sync --group dev
uv run python -m docbenchtr.run_pilot
```

The run regenerates the question and verification files, CSV and JSON results, the chart, and the generated results block in this README. The saved-ranking analysis can be regenerated without model calls with `uv run python -m docbenchtr.reanalyze`. To download and hash only the source documents, run `uv run python scripts/download_docs.py`.

For PageIndex, install and start Ollama locally and pull `qwen2.5:7b-instruct` before running the pilot. Both agentic variants use the SDK's tool definitions through LiteLLM's `ollama_chat/` route and share the local cached trees under `.pageindex/docbenchtr-pilot/`. PageIndex LLM Tree Search sends the cached tree and question to Ollama's `/api/chat` endpoint with a JSON schema. These methods make no cloud calls. The pipeline checks the local tags endpoint and records per-query failures.

## Results

The generated table and chart below are populated from the run's `results/summary.json`. Per-query values are in [`results/results.csv`](results/results.csv) and [`results/per_query.json`](results/per_query.json).

> **Correction history.** An earlier version of this table (2026-10-16) published PageIndex rows produced while Ollama served the model with its default 4,096-token context window, which silently truncated PageIndex prompts. Those rows were invalid and have been replaced. The current run serves the model with a 32,768-token context (`OLLAMA_CONTEXT_LENGTH=32768`, `num_ctx 32768`) and caps each response at 4,096 tokens; the PageIndex trees and node summaries were rebuilt under the same setting. Recorded PageIndex inputs now exceed 4,096 tokens (up to ~16,400 for one tree-search prompt and ~76,000 cumulative for one agentic query). In the first pass, 8 LLM Tree Search queries hit a 180 s client HTTP timeout; they were re-run with an 1,800 s timeout (`python -m docbenchtr.rerun_failed`) and the re-run is recorded in `results/summary.json`. The BM25 / Dense / Hybrid rows are unchanged.

## How to read the PageIndex rows

Every PageIndex variant in the generated local table runs on `qwen2.5:7b-instruct`, a small local model, on an 8 GB laptop GPU (RTX 3070 Laptop) where the model plus a 32k context does not fully fit, so part of it runs on the CPU and long agentic queries take minutes. PageIndex's own documentation recommends the strongest chat model you can afford. At this scale:

- **Agentic mode reads pages only some of the time.** In 14 of 36 queries (default instructions) and 12 of 36 (read-pages instruction), the model finished without calling `get_page_content`, so those rankings are empty and score zero. Where it did read pages, its rankings still trail the classical baselines.
- **With 7B, PageIndex is significantly behind Hybrid.** Tree Search has Recall@5 0.472 vs 0.792 locally; the paired-comparison table below gives the question-level intervals and exact tests.
- Queries a method attempted but could not answer are **counted as zero, not dropped**; the coverage columns show how many.
- The classical results match on laptop and A100; 7B PageIndex remains significantly behind Hybrid in both runs (see [Hardware control](#hardware-control-colab-a100)). The 27B, 64k comparison is numerically higher but statistically indistinguishable on 36 questions (see [Current models](#current-models-colab-a100)).

**The generated table below is the 7B pilot.** With `qwen3.8:27b` at 64k, Tree Search has Recall@5 0.917 against 0.792 for Hybrid, but the paired 95% CI includes zero ([Current models](#current-models-colab-a100)). The pilot does not establish a difference across more documents or repeated runs.

<!-- GENERATED:START -->
## Pilot results

This generated section reads its metrics from `results/summary.json` (per-question records: `results/results.csv` and `results/per_query.json`).

Reported means count every failed query as zero. Infrastructure rows marked not-run are excluded. Scored-only values are kept under `methods[].scored_only` in `summary.json`.

| Method | Attempted | Scored | Failed as zero | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 36 | 36 | 0 | 0 | 0 | 0.278 | 0.653 | 0.764 | 0.532 | 2.370 ms |
| BM25 (Turkish stemmed) | 36 | 36 | 0 | 0 | 0 | 0.250 | 0.708 | 0.722 | 0.526 | 3.233 ms |
| Dense (multilingual-e5-base) | 36 | 36 | 0 | 0 | 0 | 0.403 | 0.667 | 0.750 | 0.627 | 652.390 ms |
| Hybrid (RRF) | 36 | 36 | 0 | 0 | 0 | 0.333 | 0.708 | 0.792 | 0.612 | 17.549 ms |
| PageIndex agentic (default instructions) | 36 | 36 | 0 | 14 | 0 | 0.167 | 0.222 | 0.222 | 0.218 | 273792.557 ms |
| PageIndex agentic (read-pages instruction) | 36 | 36 | 0 | 12 | 0 | 0.097 | 0.125 | 0.153 | 0.131 | 234038.412 ms |
| PageIndex LLM Tree Search | 36 | 36 | 0 | 0 | 0 | 0.278 | 0.375 | 0.472 | 0.396 | 48581.440 ms |

![Recall@5 by method and question type](results/recall_at_5_by_question_type.png)

### Metrics by question type

| Method | Question type | Attempted | Scored | Failed as zero | Zero ranking | Excluded | Recall@1 | Recall@3 | Recall@5 | MRR | Mean query latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | direct fact | 9 | 9 | 0 | 0 | 0 | 0.444 | 0.667 | 0.667 | 0.564 | 2.898 ms |
| BM25 | paraphrase | 9 | 9 | 0 | 0 | 0 | 0.333 | 0.778 | 0.889 | 0.578 | 2.035 ms |
| BM25 | distractor | 9 | 9 | 0 | 0 | 0 | 0.222 | 0.667 | 0.889 | 0.488 | 2.387 ms |
| BM25 | cross section | 9 | 9 | 0 | 0 | 0 | 0.111 | 0.500 | 0.611 | 0.500 | 2.159 ms |
| BM25 (Turkish stemmed) | direct fact | 9 | 9 | 0 | 0 | 0 | 0.444 | 0.667 | 0.667 | 0.582 | 3.324 ms |
| BM25 (Turkish stemmed) | paraphrase | 9 | 9 | 0 | 0 | 0 | 0.333 | 0.667 | 0.667 | 0.495 | 2.822 ms |
| BM25 (Turkish stemmed) | distractor | 9 | 9 | 0 | 0 | 0 | 0.111 | 0.778 | 0.778 | 0.452 | 3.508 ms |
| BM25 (Turkish stemmed) | cross section | 9 | 9 | 0 | 0 | 0 | 0.111 | 0.722 | 0.778 | 0.574 | 3.278 ms |
| Dense (multilingual-e5-base) | direct fact | 9 | 9 | 0 | 0 | 0 | 0.444 | 0.556 | 0.778 | 0.563 | 442.623 ms |
| Dense (multilingual-e5-base) | paraphrase | 9 | 9 | 0 | 0 | 0 | 0.556 | 0.778 | 0.778 | 0.682 | 766.519 ms |
| Dense (multilingual-e5-base) | distractor | 9 | 9 | 0 | 0 | 0 | 0.333 | 0.667 | 0.667 | 0.524 | 697.825 ms |
| Dense (multilingual-e5-base) | cross section | 9 | 9 | 0 | 0 | 0 | 0.278 | 0.667 | 0.778 | 0.741 | 702.593 ms |
| Hybrid (RRF) | direct fact | 9 | 9 | 0 | 0 | 0 | 0.444 | 0.778 | 0.778 | 0.614 | 15.268 ms |
| Hybrid (RRF) | paraphrase | 9 | 9 | 0 | 0 | 0 | 0.444 | 0.889 | 0.889 | 0.678 | 17.258 ms |
| Hybrid (RRF) | distractor | 9 | 9 | 0 | 0 | 0 | 0.222 | 0.667 | 0.667 | 0.488 | 17.423 ms |
| Hybrid (RRF) | cross section | 9 | 9 | 0 | 0 | 0 | 0.222 | 0.500 | 0.833 | 0.667 | 20.249 ms |
| PageIndex agentic (default instructions) | direct fact | 9 | 9 | 0 | 3 | 0 | 0.222 | 0.222 | 0.222 | 0.222 | 239089.147 ms |
| PageIndex agentic (default instructions) | paraphrase | 9 | 9 | 0 | 4 | 0 | 0.222 | 0.222 | 0.222 | 0.222 | 172961.116 ms |
| PageIndex agentic (default instructions) | distractor | 9 | 9 | 0 | 4 | 0 | 0.111 | 0.333 | 0.333 | 0.204 | 285612.757 ms |
| PageIndex agentic (default instructions) | cross section | 9 | 9 | 0 | 3 | 0 | 0.111 | 0.111 | 0.111 | 0.222 | 397507.210 ms |
| PageIndex agentic (read-pages instruction) | direct fact | 9 | 9 | 0 | 3 | 0 | 0.000 | 0.000 | 0.111 | 0.022 | 185928.940 ms |
| PageIndex agentic (read-pages instruction) | paraphrase | 9 | 9 | 0 | 3 | 0 | 0.222 | 0.222 | 0.222 | 0.222 | 137041.937 ms |
| PageIndex agentic (read-pages instruction) | distractor | 9 | 9 | 0 | 3 | 0 | 0.111 | 0.111 | 0.111 | 0.111 | 299557.438 ms |
| PageIndex agentic (read-pages instruction) | cross section | 9 | 9 | 0 | 3 | 0 | 0.056 | 0.167 | 0.167 | 0.167 | 313625.332 ms |
| PageIndex LLM Tree Search | direct fact | 9 | 9 | 0 | 0 | 0 | 0.333 | 0.444 | 0.556 | 0.417 | 61338.217 ms |
| PageIndex LLM Tree Search | paraphrase | 9 | 9 | 0 | 0 | 0 | 0.111 | 0.222 | 0.333 | 0.201 | 68379.981 ms |
| PageIndex LLM Tree Search | distractor | 9 | 9 | 0 | 0 | 0 | 0.444 | 0.444 | 0.556 | 0.467 | 32511.824 ms |
| PageIndex LLM Tree Search | cross section | 9 | 9 | 0 | 0 | 0 | 0.222 | 0.389 | 0.444 | 0.500 | 32095.740 ms |

### PageIndex vs Hybrid: paired question-level comparison

Each row uses the same questions for PageIndex and Hybrid. Intervals are percentile 95% paired-bootstrap CIs (10,000 resamples; fixed seed); p values are exact two-sided sign tests. Win/loss/tie counts are shown for each metric. Original `source_pages` are the primary labels; expanded-gold results treat documented equivalent pages as alternatives within the same provision, so the number of targets per question stays fixed. Full results and validator-pass subsets are in [`results/paired_vs_hybrid.json`](results/paired_vs_hybrid.json).

| Run | PageIndex variant | Recall@5 Δ [95% CI] | W/L/T | p | MRR Δ [95% CI] | W/L/T | p | Expanded-gold Recall@5 Δ [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| colab_a100_models/gpt-oss-20b_ctx32k | agentic (default instructions) | -0.264 [-0.458, -0.056] | 5/15/16 | 0.041 | -0.159 [-0.335, +0.028] | 9/20/7 | 0.061 | -0.306 [-0.514, -0.097] |
| colab_a100_models/gpt-oss-20b_ctx32k | agentic (read-pages instruction) | -0.306 [-0.500, -0.097] | 4/16/16 | 0.012 | -0.153 [-0.333, +0.034] | 8/21/7 | 0.024 | -0.347 [-0.542, -0.139] |
| colab_a100_models/gpt-oss-20b_ctx32k | LLM Tree Search | -0.264 [-0.458, -0.056] | 4/14/18 | 0.031 | -0.139 [-0.305, +0.028] | 9/19/8 | 0.087 | -0.306 [-0.500, -0.097] |
| colab_a100_models/gpt-oss-20b_ctx64k | agentic (default instructions) | +0.000 [-0.181, +0.181] | 7/7/22 | 1.000 | -0.022 [-0.204, +0.159] | 12/15/9 | 0.701 | -0.042 [-0.208, +0.125] |
| colab_a100_models/gpt-oss-20b_ctx64k | agentic (read-pages instruction) | -0.083 [-0.292, +0.111] | 7/10/19 | 0.629 | +0.017 [-0.175, +0.200] | 13/13/10 | 1.000 | -0.125 [-0.319, +0.069] |
| colab_a100_models/gpt-oss-20b_ctx64k | LLM Tree Search | +0.014 [-0.167, +0.194] | 7/7/22 | 1.000 | +0.113 [-0.074, +0.293] | 16/10/10 | 0.327 | +0.000 [-0.167, +0.181] |
| colab_a100_models/qwen3.8-27b_ctx32k | agentic (default instructions) | -0.208 [-0.417, +0.000] | 6/13/17 | 0.167 | -0.108 [-0.284, +0.073] | 10/19/7 | 0.136 | -0.250 [-0.472, -0.028] |
| colab_a100_models/qwen3.8-27b_ctx32k | agentic (read-pages instruction) | -0.236 [-0.444, -0.014] | 5/14/17 | 0.064 | -0.144 [-0.325, +0.047] | 9/19/8 | 0.087 | -0.278 [-0.486, -0.056] |
| colab_a100_models/qwen3.8-27b_ctx32k | LLM Tree Search | -0.139 [-0.333, +0.056] | 6/11/19 | 0.332 | -0.056 [-0.222, +0.114] | 10/17/9 | 0.248 | -0.181 [-0.389, +0.028] |
| colab_a100_models/qwen3.8-27b_ctx64k | agentic (default instructions) | +0.042 [-0.125, +0.209] | 7/5/24 | 0.774 | +0.090 [-0.096, +0.264] | 15/10/11 | 0.424 | +0.000 [-0.167, +0.167] |
| colab_a100_models/qwen3.8-27b_ctx64k | agentic (read-pages instruction) | +0.000 [-0.181, +0.194] | 7/8/21 | 1.000 | +0.144 [-0.042, +0.323] | 17/8/11 | 0.108 | -0.042 [-0.222, +0.153] |
| colab_a100_models/qwen3.8-27b_ctx64k | LLM Tree Search | +0.125 [-0.028, +0.278] | 8/3/25 | 0.227 | +0.126 [-0.045, +0.296] | 15/9/12 | 0.307 | +0.083 [-0.069, +0.236] |
| colab_a100_qwen2.5-7b | agentic (default instructions) | -0.639 [-0.792, -0.472] | 1/25/10 | <0.001 | -0.513 [-0.649, -0.372] | 3/33/0 | <0.001 | -0.681 [-0.833, -0.514] |
| colab_a100_qwen2.5-7b | agentic (read-pages instruction) | -0.569 [-0.736, -0.403] | 1/22/13 | <0.001 | -0.440 [-0.580, -0.291] | 3/31/2 | <0.001 | -0.611 [-0.764, -0.444] |
| colab_a100_qwen2.5-7b | LLM Tree Search | -0.347 [-0.542, -0.139] | 4/17/15 | 0.007 | -0.242 [-0.402, -0.079] | 7/22/7 | 0.008 | -0.389 [-0.583, -0.181] |
| local | agentic (default instructions) | -0.569 [-0.708, -0.417] | 0/23/13 | <0.001 | -0.394 [-0.554, -0.226] | 4/29/3 | <0.001 | -0.611 [-0.764, -0.458] |
| local | agentic (read-pages instruction) | -0.639 [-0.778, -0.486] | 0/25/11 | <0.001 | -0.481 [-0.641, -0.311] | 5/31/0 | <0.001 | -0.681 [-0.819, -0.528] |
| local | LLM Tree Search | -0.319 [-0.528, -0.097] | 5/17/14 | 0.017 | -0.216 [-0.392, -0.029] | 8/23/5 | 0.011 | -0.361 [-0.569, -0.153] |

### Original and expanded gold: absolute scores by run

Each pair shows the primary pre-registered score followed by the expanded-gold sensitivity score. Equivalent pages are alternatives within the same provision, so the target count per question is unchanged.

| Run | Method | Recall@5 primary → expanded | MRR primary → expanded |
|---|---|---:|---:|
| colab_a100_models/gpt-oss-20b_ctx32k | Hybrid (RRF) | 0.792 → 0.833 | 0.612 → 0.636 |
| colab_a100_models/gpt-oss-20b_ctx32k | PageIndex agentic (default instructions) | 0.528 → 0.528 | 0.452 → 0.452 |
| colab_a100_models/gpt-oss-20b_ctx32k | PageIndex agentic (read-pages instruction) | 0.486 → 0.486 | 0.458 → 0.458 |
| colab_a100_models/gpt-oss-20b_ctx32k | PageIndex LLM Tree Search | 0.528 → 0.528 | 0.472 → 0.472 |
| colab_a100_models/gpt-oss-20b_ctx64k | Hybrid (RRF) | 0.792 → 0.833 | 0.612 → 0.636 |
| colab_a100_models/gpt-oss-20b_ctx64k | PageIndex agentic (default instructions) | 0.792 → 0.792 | 0.590 → 0.590 |
| colab_a100_models/gpt-oss-20b_ctx64k | PageIndex agentic (read-pages instruction) | 0.708 → 0.708 | 0.628 → 0.628 |
| colab_a100_models/gpt-oss-20b_ctx64k | PageIndex LLM Tree Search | 0.806 → 0.833 | 0.724 → 0.733 |
| colab_a100_models/qwen3.8-27b_ctx32k | Hybrid (RRF) | 0.792 → 0.833 | 0.612 → 0.636 |
| colab_a100_models/qwen3.8-27b_ctx32k | PageIndex agentic (default instructions) | 0.583 → 0.583 | 0.504 → 0.504 |
| colab_a100_models/qwen3.8-27b_ctx32k | PageIndex agentic (read-pages instruction) | 0.556 → 0.556 | 0.468 → 0.468 |
| colab_a100_models/qwen3.8-27b_ctx32k | PageIndex LLM Tree Search | 0.653 → 0.653 | 0.556 → 0.556 |
| colab_a100_models/qwen3.8-27b_ctx64k | Hybrid (RRF) | 0.792 → 0.833 | 0.612 → 0.636 |
| colab_a100_models/qwen3.8-27b_ctx64k | PageIndex agentic (default instructions) | 0.833 → 0.833 | 0.701 → 0.701 |
| colab_a100_models/qwen3.8-27b_ctx64k | PageIndex agentic (read-pages instruction) | 0.792 → 0.792 | 0.756 → 0.756 |
| colab_a100_models/qwen3.8-27b_ctx64k | PageIndex LLM Tree Search | 0.917 → 0.917 | 0.738 → 0.756 |
| colab_a100_qwen2.5-7b | Hybrid (RRF) | 0.792 → 0.833 | 0.612 → 0.636 |
| colab_a100_qwen2.5-7b | PageIndex agentic (default instructions) | 0.153 → 0.153 | 0.098 → 0.098 |
| colab_a100_qwen2.5-7b | PageIndex agentic (read-pages instruction) | 0.222 → 0.222 | 0.171 → 0.171 |
| colab_a100_qwen2.5-7b | PageIndex LLM Tree Search | 0.444 → 0.444 | 0.370 → 0.370 |
| local | Hybrid (RRF) | 0.792 → 0.833 | 0.612 → 0.636 |
| local | PageIndex agentic (default instructions) | 0.222 → 0.222 | 0.218 → 0.218 |
| local | PageIndex agentic (read-pages instruction) | 0.153 → 0.153 | 0.131 → 0.131 |
| local | PageIndex LLM Tree Search | 0.472 → 0.472 | 0.396 → 0.396 |

Validator-pass sensitivity for qwen3.8 27B at 64k (32 of 36 questions; Tree Search vs Hybrid):

| Metric | Δ [95% CI] | W/L/T | Exact p |
|---|---:|---:|---:|
| Recall@5 | +0.156 [+0.000, +0.328] | 7/2/23 | 0.180 |
| MRR | +0.168 [-0.015, +0.346] | 15/8/9 | 0.210 |

Expanded-gold Recall@5 for that validator-pass subset: +0.109 [-0.047, +0.266].

### Simple baselines (original gold)

The first-five baseline always returns physical pages 1–5. Random-order values are the analytic expected Recall@5 for a uniformly random page ranking. The 6502 equivalent-page additions are outside page 5, so the first-five baseline is unchanged under expanded gold.

| Scope | First 5 pages Recall@5 | Random order expected Recall@5 |
|---|---:|---:|
| Overall | 0.361 | 0.109 |
| KVKK | 1.000 | 0.278 |
| Consumer law | 0.000 | 0.015 |
| Higher education law | 0.083 | 0.035 |

### Agentic citation pages (secondary score)

This row scores `cited_pages` rather than reconstructed page-read order. Failed queries count as zero under the same rule.

| Run | Agentic variant | Citation Recall@5, original | Citation MRR, original | Citation Recall@5, expanded |
|---|---|---:|---:|---:|
| colab_a100_models/gpt-oss-20b_ctx32k | PageIndex agentic (default instructions) | 0.444 | 0.444 | 0.444 |
| colab_a100_models/gpt-oss-20b_ctx32k | PageIndex agentic (read-pages instruction) | 0.347 | 0.389 | 0.347 |
| colab_a100_models/gpt-oss-20b_ctx64k | PageIndex agentic (default instructions) | 0.500 | 0.509 | 0.500 |
| colab_a100_models/gpt-oss-20b_ctx64k | PageIndex agentic (read-pages instruction) | 0.486 | 0.514 | 0.486 |
| colab_a100_models/qwen3.8-27b_ctx32k | PageIndex agentic (default instructions) | 0.569 | 0.542 | 0.569 |
| colab_a100_models/qwen3.8-27b_ctx32k | PageIndex agentic (read-pages instruction) | 0.528 | 0.514 | 0.528 |
| colab_a100_models/qwen3.8-27b_ctx64k | PageIndex agentic (default instructions) | 0.833 | 0.819 | 0.833 |
| colab_a100_models/qwen3.8-27b_ctx64k | PageIndex agentic (read-pages instruction) | 0.861 | 0.833 | 0.861 |
| colab_a100_qwen2.5-7b | PageIndex agentic (default instructions) | 0.083 | 0.069 | 0.083 |
| colab_a100_qwen2.5-7b | PageIndex agentic (read-pages instruction) | 0.139 | 0.125 | 0.139 |
| local | PageIndex agentic (default instructions) | 0.083 | 0.111 | 0.083 |
| local | PageIndex agentic (read-pages instruction) | 0.083 | 0.083 | 0.083 |

### Run metadata

- Documents: 3 — Kişisel Verilerin Korunması Kanunu, Tüketicinin Korunması Hakkında Kanun ve İkincil Mevzuatı (Ocak 2021), Yükseköğretim Kanunu.
- Retained pilot questions: 36; failed planned-term validator checks retained in results: 4; dropped: 0.
- Physical PDF pages extracted: 500 across kvkk_6698: 18, tuketicinin_korunmasi_6502: 340, yuksekogretim_2547: 142.
- Embedding model: `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a` on `cuda`.
- Question set label: **pilot set; hand-written question stems and source spans checked; 4 planned-term validator failures retained; not yet human-reviewed**.
- PageIndex agentic (default instructions) queries: 36 attempted, 36 scored, 0 failed-as-zero, 14 zero-ranking, 0 excluded.
- PageIndex agentic (read-pages instruction) queries: 36 attempted, 36 scored, 0 failed-as-zero, 12 zero-ranking, 0 excluded.
- PageIndex agentic (default instructions) token counts: available for 36 queries; 1012135 input and 27926 output tokens total.
- PageIndex agentic (read-pages instruction) token counts: available for 36 queries; 982424 input and 25060 output tokens total.
- PageIndex LLM Tree Search queries: 36 attempted, 36 scored, 0 failed-as-zero, 0 zero-ranking, 0 excluded; 0 parse failures.
- PageIndex LLM HTTP safety net: retry a request after 30 minutes without response progress, up to 3 retries; 0 retries recorded; there is no total index or query deadline.

### PageIndex index builds

- `kvkk_6698`: reused; build cached; prior build time unavailable; `toc_source=detected`; cache validation 12.1 s
- `tuketicinin_korunmasi_6502`: reused; build cached; prior build time unavailable; `toc_source=detected`; cache validation 4.5 s
- `yuksekogretim_2547`: reused; build cached; prior build time unavailable; `toc_source=detected`; cache validation 4.0 s

PageIndex agentic default-instruction and read-pages-instruction variants use `ollama_chat/qwen2.5:7b-instruct` through Ollama and share the cached trees. The read-pages instruction is an explicit modification to the SDK default. `tool_call_count` counts every agent tool invocation; page ranking uses only successful `get_page_content` results, in tool-call order and each result's `returned_pages` order, retaining each physical page's first occurrence. Outline-only calls count as tool calls but do not enter the ranking. PageIndex LLM Tree Search asks the same local model to select nodes from the cached tree using Ollama's JSON-schema output. It expands each selected node's inclusive one-based `start_index..end_index` range in returned `node_list` order and deduplicates pages by first occurrence. Attempted queries without a usable ranking, including parse failures, score zero; infrastructure exclusions remain outside the means. Both methods return no PageIndex page scores; per-query errors and traces are preserved in `results/per_query.json` and `results/results.csv`.
<!-- GENERATED:END -->

## Hardware control (Colab A100)

To check whether the laptop GPU (RTX 3070 Laptop, 8 GB, ~9% of the model on CPU) shaped the PageIndex numbers, the pilot was re-run on a Colab **NVIDIA A100-SXM4-40GB** with everything else held fixed: the same `qwen2.5:7b-instruct` weights (same Ollama model ID `ad3d627188ab`, base blob `sha256-2bada8a7…`), Ollama 0.40.5, 32,768-token context, 4,096-token output cap, temperature 0, the same 36 questions, the same source PDFs (SHA-256 verified) and the **same cached PageIndex trees**. Notebook: [`colab/run_a100_control.ipynb`](colab/run_a100_control.ipynb) (the repository ships the cached trees and the pinned PDFs). Full outputs: [`results/colab_a100_qwen2.5-7b/`](results/colab_a100_qwen2.5-7b/).

| Method | Recall@5 laptop | Recall@5 A100 | MRR laptop | MRR A100 | Zero rankings laptop / A100 | Mean latency laptop → A100 |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid (RRF) | 0.792 | 0.792 | 0.612 | 0.612 | 0 / 0 | unchanged |
| BM25 | 0.764 | 0.764 | 0.532 | 0.532 | 0 / 0 | unchanged |
| Dense (multilingual-e5-base) | 0.750 | 0.750 | 0.627 | 0.627 | 0 / 0 | unchanged |
| BM25 (Turkish stemmed) | 0.722 | 0.722 | 0.526 | 0.526 | 0 / 0 | unchanged |
| PageIndex LLM Tree Search | 0.472 | 0.444 | 0.396 | 0.370 | 0 / 0 | 48.6 s → 3.0 s |
| PageIndex agentic (default instructions) | 0.222 | 0.153 | 0.218 | 0.098 | 14 / 12 | 273.8 s → 12.6 s |
| PageIndex agentic (read-pages instruction) | 0.153 | 0.222 | 0.131 | 0.171 | 12 / 13 | 234.0 s → 8.8 s |

- **Classical baselines are identical**, confirming the two environments match.
- **The 7B comparison is consistent across hardware.** Tree Search is significantly below Hybrid on both machines; the A100 is 15–30× faster per query. No Tree Search query timed out on the A100, so the 8 laptop timeouts were a local artefact.
- **PageIndex outputs are not reproducible across GPUs even at temperature 0.** The ranked pages are identical in only 11/36 (Tree Search), 16/36 (agentic default) and 20/36 (agentic read-pages) queries. Floating-point differences between GPUs change token choices, and long multi-turn agent runs amplify them. The two agentic variants swap places between the runs, so **their difference is within run-to-run noise**. The full study should run each PageIndex configuration several times and report the mean and spread.

## Current models (Colab A100)

With the 27B model at 64k, Tree Search scores numerically higher than Hybrid, but the difference is not statistically distinguishable on these 36 questions. The same questions, PDFs and cached trees (built with `qwen2.5:7b-instruct`) were run with two newer open models on a Colab A100: `qwen3.8:27b` (Alibaba) and `gpt-oss:20b` (OpenAI). Output cap 32,768 tokens, temperature 0, Ollama 0.40.5. Notebook: [`colab/run_a100_models.ipynb`](colab/run_a100_models.ipynb); run logs with the full tables: [`results/colab_a100_models/logs/`](results/colab_a100_models/logs/).

Recall@5 over all 36 questions. Failed agentic queries count as zero; scored-only values are in parentheses. The classical baselines are identical in every run (Hybrid 0.792, MRR 0.612).

| PageIndex variant | qwen2.5:7b, 32k | qwen3.8:27b, 32k | **qwen3.8:27b, 64k** | gpt-oss:20b, 32k | **gpt-oss:20b, 64k** |
|---|---:|---:|---:|---:|---:|
| LLM Tree Search | 0.444 | 0.653 | **0.917** | 0.528 | **0.806** |
| Agentic, default instructions | 0.153 | 0.583 (0.913, 23 q) | **0.833** (0.909, 33 q) | 0.528 (0.576, 33 q) | **0.792** (0.814, 35 q) |
| Agentic, read-pages instruction | 0.222 | 0.556 (0.909, 22 q) | **0.792** (0.891, 32 q) | 0.486 | **0.708** (0.750, 34 q) |

| 64k, all 36 questions | Recall@1 | Recall@3 | Recall@5 | MRR | Mean latency |
|---|---:|---:|---:|---:|---:|
| Hybrid (RRF) | 0.333 | 0.708 | 0.792 | 0.612 | 13 ms |
| Dense (multilingual-e5-base) | 0.403 | 0.667 | 0.750 | 0.627 | 14 ms |
| PageIndex LLM Tree Search, qwen3.8:27b | 0.500 | 0.847 | **0.917** | **0.738** | 27.3 s |
| PageIndex LLM Tree Search, gpt-oss:20b | **0.528** | 0.778 | 0.806 | 0.724 | 19.3 s |

- **Scores vary with model and context.** This one-run comparison does not establish that model size or context caused the observed score differences.
- **Context length matters for long documents.** At 32k, the 340-page consumer-protection law's document structure plus tool results overflowed the window: Ollama truncated the prompt, dropping the user's question ("no user query found in messages"), and 13–14 agentic queries failed. At 64k, failures were fewer but did not disappear.
- **Paired test.** qwen3.8 Tree Search vs Hybrid: Recall@5 Δ +0.125, 95% CI [−0.028, +0.278], 8/3/25 wins/losses/ties, exact sign-test p = 0.227. MRR Δ +0.126, 95% CI [−0.045, +0.296]. The difference is not statistically distinguishable on 36 questions.
- **Caveats.** One run per configuration (see the hardware-control section: rankings vary across GPUs at temperature 0). gpt-oss Tree Search vs Hybrid (0.806 vs 0.792) is also not statistically distinguishable; see the paired table. PageIndex costs 15–47 s per query against milliseconds. Trees were built by the 7B model. The 64k Modelfile override changes the Ollama model ID (`472dc…` vs `56ead…` at 32k); the weights are the same.
- **Per-document scores (descriptive).** Recall@5 at 64k (12 questions each):

  | Document | Hybrid | Tree Search, qwen3.8 | Tree Search, gpt-oss |
  |---|---:|---:|---:|
  | 6698 KVKK (18 pages) | 1.000 | 1.000 | 1.000 |
  | 6502 Consumer protection (340 pages) | 0.792 | 0.833 | 0.750 |
  | 2547 Higher education (142 pages) | 0.583 | **0.917** | 0.667 |

  Scores differ most on the higher-education law; on the longest document the methods are close. These per-document values are descriptive for this small pilot.
- **Why some agentic queries failed at 64k.** qwen3.8: 7 queries with `litellm.APIConnectionError: Extra data` (LiteLLM could not parse Ollama's response). gpt-oss: 3 queries with `KeyError: 'message', Got unexpected response from Ollama`. These are LiteLLM/Ollama integration errors with reasoning models, not model answers; they are counted as zero above. At 32k, gpt-oss also tried to call tools that do not exist (`repo_browser.search`, `repo_browser.print_tree`), its built-in harmony tool names.

Per-query outputs for all four runs: [`results/colab_a100_models/`](results/colab_a100_models/).

## Limitations

- This is a small pilot set, not a statistically powered comparison.
- Question stems are hand-written; four fail the planned-term check. The set has not yet been reviewed by human annotators. Matching source text confirms provenance, not question quality or answer completeness.
- Evaluation measures retrieval only; generation, answer correctness, and faithfulness are out of scope.
- The dense comparison uses a single multilingual embedding model.
- PDF pagination, extraction quality, edition dates, and the balance of document topics can affect retrieval scores.
- Every attempted query that fails receives Recall@k = 0 and reciprocal rank = 0 and is included in overall and per-question-type means. Completed queries with no usable ranking also receive zero. A query that never starts because its document was not indexed is marked not-run and excluded. Summaries report failed-as-zero, zero-ranking, scored, and excluded counts; scored-only values are separate.
- PageIndex agentic page rankings are reconstructed from successful `get_page_content` tool results, ordered by the agent's tool-call sequence and each result's `returned_pages`; repeated pages keep their first rank. Outline-only calls count in `tool_call_count` but do not rank pages. PageIndex returns no page scores. Per-query traces and SDK token usage, when present, are retained.
- PageIndex LLM Tree Search is a separate non-agentic method. It ranks inclusive one-based page ranges from model-selected cached-tree nodes in returned node order, deduplicating by first occurrence. JSON parse or schema failures are recorded per query and are not silently dropped.
- PageIndex used the small local `qwen2.5:7b-instruct` model for both indexing and search. Its developers say a basic model is sufficient for indexing but recommend the [best model you can afford for chat/search](https://github.com/VectifyAI/PageIndex#model-recommendations); the 7B search model may disadvantage its results against that recommendation.
- PageIndex's tree `toc_source`, per-document build time, and any document or query failures are listed in the generated run metadata and per-query files.

## Roadmap to the full study

The proposed full study will expand to approximately one thousand human-reviewed questions, compare nine retrieval methods plus two controls (an agentic hybrid control and a long-context baseline), and use one fixed generator LLM for any answer-generation evaluation. The final protocol will preregister document snapshots, question sampling, model versions, and scoring rules before the full comparison.

## External review

The [17 October 2026 review by Barış Şavk](docs/reviews/2026-10-17_review_baris_savk.md) is included unchanged.

**Fixed in this revision:** softened the headline and added paired bootstrap intervals and exact sign tests; made failed-query-as-zero the reported rule; added the 6502 equivalent gold pages and original-versus-expanded rescoring; retained and listed the four planned-term validator failures; added first-five and random-order baselines, citation-page scores, saved-ranking analysis, CI tests, and pinned Colab checkouts.

**Still open for the full study:** compare Hybrid after reranking its top 20 with the same LLM; repeat each configuration; rebuild the trees with the 27B model; expand to a larger human-reviewed question set; and use unified text extraction across methods.

## Kısa Türkçe özet

DocBenchTR, uzun ve maddeli Türkçe belgelerde RAG erişim yöntemlerini karşılaştıran bir pilot çalışmadır. Bu herkese açık pilot; üç mevzuat PDF'inde 36 elle yazılmış soruyu değerlendirir. Dört soru planlanan terim kontrolünden geçmemiştir; bu sorular sonuçlarda tutulur ve ayrı duyarlılık analizinde incelenir. Küçük ve henüz insan incelemesinden geçmemiş soru kümesi yalnızca ilk sonuçları gösterir; yanıt üretimi ve doğruluğu kapsam dışındadır.

## License

The code is licensed under the MIT License. The included legal texts remain subject to their source and publication notes. No model weights are committed. The three source PDFs (SHA-256 pinned) and cached PageIndex trees are committed because Colab hosts cannot reach the `.gov.tr` download URLs and the A100 runs reuse the same trees.
