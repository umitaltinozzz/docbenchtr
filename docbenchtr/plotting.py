from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import RESULTS_DIR


def plot_recall_by_type(summary: dict[str, Any]) -> Path:
    methods = [method for method in summary["methods"] if method["status"] == "ran"]
    question_types = sorted(summary["question_type_counts"])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(question_types))
    width = 0.8 / max(1, len(methods))
    palette = ["#1f4e79", "#2e8b57", "#dd8452", "#8172b2", "#64b5cd"]
    for index, method in enumerate(methods):
        values = [method["by_question_type"].get(qtype, {}).get("recall@5", float("nan")) for qtype in question_types]
        ax.bar(x - 0.4 + width * (index + 0.5), values, width, label=method["method"], color=palette[index % len(palette)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Page-level Recall@5")
    ax.set_xlabel("Pilot question type")
    ax.set_xticks(x, [item.replace("_", " ").title() for item in question_types])
    ax.grid(axis="y", color="#d9dfe5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Recall@5 by retrieval method and question type", fontsize=15, y=0.975)
    fig.legend(
        handles, labels, frameon=False, ncols=min(2, len(methods)),
        loc="upper center", bbox_to_anchor=(0.5, 0.91), columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.72)
    path = RESULTS_DIR / "recall_at_5_by_question_type.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path
