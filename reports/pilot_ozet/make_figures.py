"""Figures for the short Turkish pilot report (reports/pilot_ozet).

Usage: python reports/pilot_ozet/make_figures.py
Every number is read from the committed result files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

RUNS = {
    "local_7b": RES / "summary.json",
    "a100_7b": RES / "colab_a100_qwen2.5-7b" / "summary.json",
    "qwen_32k": RES / "colab_a100_models" / "qwen3.8-27b_ctx32k" / "summary.json",
    "qwen_64k": RES / "colab_a100_models" / "qwen3.8-27b_ctx64k" / "summary.json",
    "oss_32k": RES / "colab_a100_models" / "gpt-oss-20b_ctx32k" / "summary.json",
    "oss_64k": RES / "colab_a100_models" / "gpt-oss-20b_ctx64k" / "summary.json",
}
TREE = "PageIndex LLM Tree Search"
HYBRID = "Hybrid (RRF)"

CLASSIC = "#8fa3b8"
HYB = "#2f5d8a"
PI = "#e07a2f"
PI_LIGHT = "#f2b483"
GREEN = "#2e9e6b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
})


def summary(run: str) -> dict:
    return json.loads(RUNS[run].read_text(encoding="utf-8"))


def metric(run: str, method: str, key: str = "recall@5") -> float:
    for m in summary(run)["methods"]:
        if m["method"] == method:
            return m["overall"][key]
    raise KeyError((run, method))


def fmt(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def fig_main() -> None:
    labels = ["BM25 (kök)", "Dense (e5)", "BM25", "Hybrid (BM25+Dense)",
              "PageIndex · gpt-oss 20B", "PageIndex · Qwen 3.8 27B"]
    values = [metric("qwen_64k", "BM25 (Turkish stemmed)"), metric("qwen_64k", "Dense (multilingual-e5-base)"),
              metric("qwen_64k", "BM25"), metric("qwen_64k", HYBRID),
              metric("oss_64k", TREE), metric("qwen_64k", TREE)]
    colors = [CLASSIC, CLASSIC, CLASSIC, HYB, PI_LIGHT, PI]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for bar, v in zip(bars, values):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2, fmt(v), va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Recall@5  (doğru sayfa ilk 5 sonuçta mı?  1,0 = her soruda)")
    ax.set_title("Son durum (64k bağlam, 36 soru)", loc="left")
    ax.axvline(metric("qwen_64k", HYBRID), color=HYB, ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_final.png", dpi=200)
    plt.close(fig)


def fig_journey() -> None:
    steps = [
        ("Qwen2.5 7B\ndizüstü · 32k", metric("local_7b", TREE), PI_LIGHT),
        ("Qwen2.5 7B\nA100 · 32k", metric("a100_7b", TREE), PI_LIGHT),
        ("gpt-oss 20B\nA100 · 32k", metric("oss_32k", TREE), PI_LIGHT),
        ("gpt-oss 20B\nA100 · 64k", metric("oss_64k", TREE), PI),
        ("Qwen 3.8 27B\nA100 · 32k", metric("qwen_32k", TREE), PI_LIGHT),
        ("Qwen 3.8 27B\nA100 · 64k", metric("qwen_64k", TREE), PI),
    ]
    hybrid = metric("qwen_64k", HYBRID)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    xs = range(len(steps))
    bars = ax.bar(xs, [s[1] for s in steps], color=[s[2] for s in steps], width=0.62)
    for bar, (_, v, _) in zip(bars, steps):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, fmt(v), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(hybrid, color=HYB, ls="--", lw=1.6)
    ax.text(-0.45, hybrid + 0.02, f"Hybrid = {fmt(hybrid)}", color=HYB, ha="left", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs), [s[0] for s in steps], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall@5")
    ax.set_title("PageIndex Tree Search, koşu koşu", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_journey.png", dpi=200)
    plt.close(fig)


def fig_documents() -> None:
    docs = [("kvkk_6698", "KVKK\n(18 sayfa)"), ("tuketicinin_korunmasi_6502", "Tüketici K.\n(340 sayfa)"),
            ("yuksekogretim_2547", "Yükseköğretim K.\n(142 sayfa)")]
    series = [("Hybrid", "qwen_64k", HYBRID, HYB), ("PageIndex · gpt-oss 20B", "oss_64k", TREE, PI_LIGHT),
              ("PageIndex · Qwen 3.8 27B", "qwen_64k", TREE, PI)]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.26
    for i, (label, run, method, color) in enumerate(series):
        rows = json.loads((RUNS[run].parent / "per_query.json").read_text(encoding="utf-8"))
        vals = []
        for doc, _ in docs:
            r = [x["recall@5"] or 0 for x in rows if x["method"] == method and x["document_id"] == doc]
            vals.append(sum(r) / len(r))
        bars = ax.bar([j + (i - 1) * width for j in range(len(docs))], vals, width, color=color, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, fmt(v), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(docs)), [d[1] for d in docs])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall@5 (12 soru)")
    ax.legend(frameon=False, fontsize=8.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Kanun bazında Recall@5 (64k)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_documents.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_main()
    fig_journey()
    fig_documents()
    print("figures written to", OUT)
