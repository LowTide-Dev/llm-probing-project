"""
visualize_results.py
--------------------
Generates layer-wise probe accuracy and selectivity figures from
the CSV results produced by train_probes.py.

Saves figures to results/figures/:
  - probe_accuracy_{task}.png   (probe vs control, CLS vs mean)
  - selectivity_{task}.png      (selectivity curves, CLS vs mean)
  - summary_heatmap.png         (selectivity across both tasks and poolings)

Usage:
    python src/visualize_results.py
    python src/visualize_results.py --tasks convergence   # single task
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "cls_probe":        "#2563EB",
    "cls_control":      "#93C5FD",
    "mean_probe":       "#DC2626",
    "mean_control":     "#FCA5A5",
    "selectivity_cls":  "#1D4ED8",
    "selectivity_mean": "#B91C1C",
}

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})


# ── helpers ───────────────────────────────────────────────────────────────────

def load_results(task, results_dir="results"):
    out = {}
    for pooling in ("cls", "mean"):
        path = os.path.join(results_dir, f"probe_results_{task}_{pooling}.csv")
        if os.path.exists(path):
            out[pooling] = pd.read_csv(path)
        else:
            print(f"  [warn] missing: {path}")
    return out


def plot_accuracy(task, data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.suptitle(
        f"Layer-wise Probe Accuracy — {task.capitalize()} (MatSciBERT)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, pooling, title in zip(
        axes,
        ("cls", "mean"),
        ("[CLS] Pooling", "Mean Pooling"),
    ):
        if pooling not in data:
            ax.set_visible(False)
            continue

        df = data[pooling]
        layers = df["layer"].values

        ax.plot(layers, df["probe_acc"], marker="o", linewidth=2,
                color=COLORS[f"{pooling}_probe"], label="Probe (true labels)")
        ax.fill_between(layers, df["probe_acc"], df["control_acc"],
                        alpha=0.15, color=COLORS[f"{pooling}_probe"])
        ax.plot(layers, df["control_acc"], marker="s", linewidth=1.5,
                linestyle="--", color=COLORS[f"{pooling}_control"],
                label="Control (shuffled labels)")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_xticks(layers)
        ax.set_ylim(0.3, 1.08)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":",
                   alpha=0.6, label="Chance (0.5)")
        ax.legend(fontsize=9, framealpha=0.8)

        peak_idx = df["probe_acc"].idxmax()
        ax.annotate(
            f"peak\n{df.loc[peak_idx, 'probe_acc']:.3f}",
            xy=(df.loc[peak_idx, "layer"], df.loc[peak_idx, "probe_acc"]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    plt.tight_layout()
    path = os.path.join(out_dir, f"probe_accuracy_{task}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_selectivity(task, data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title(
        f"Selectivity by Layer — {task.capitalize()} (MatSciBERT)",
        fontsize=13, fontweight="bold",
    )

    for pooling, label in (("cls", "[CLS] pooling"), ("mean", "Mean pooling")):
        if pooling not in data:
            continue
        df = data[pooling]
        ax.plot(df["layer"], df["selectivity"], marker="o", linewidth=2,
                color=COLORS[f"selectivity_{pooling}"], label=label)
        ax.fill_between(df["layer"], 0, df["selectivity"],
                        alpha=0.08, color=COLORS[f"selectivity_{pooling}"])

        peak_idx = df["selectivity"].idxmax()
        ax.annotate(
            f"L{df.loc[peak_idx, 'layer']}\n{df.loc[peak_idx, 'selectivity']:+.3f}",
            xy=(df.loc[peak_idx, "layer"], df.loc[peak_idx, "selectivity"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, color=COLORS[f"selectivity_{pooling}"],
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Selectivity (probe - control)", fontsize=11)
    ax.set_xticks(data[next(iter(data))]["layer"].values)
    ax.legend(fontsize=10, framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"selectivity_{task}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_summary_heatmap(all_data, tasks, out_dir):
    rows, row_labels = [], []
    for task in tasks:
        for pooling in ("cls", "mean"):
            if task in all_data and pooling in all_data[task]:
                rows.append(all_data[task][pooling]["selectivity"].values)
                row_labels.append(f"{task} ({pooling.upper()})")

    if not rows:
        return

    matrix = np.array(rows)
    n_layers = matrix.shape[1]

    fig, ax = plt.subplots(figsize=(10, 0.9 * len(rows) + 1.5))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=0.6)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(i + 1) for i in range(n_layers)])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title("Selectivity Heatmap — MatSciBERT", fontsize=13, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if matrix[i, j] > 0.35 else "black")

    plt.colorbar(im, ax=ax, label="Selectivity", shrink=0.6)
    plt.tight_layout()
    path = os.path.join(out_dir, "summary_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+",
        default=["convergence", "stability"],
        choices=["convergence", "stability"],
    )
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    all_data = {}
    for task in args.tasks:
        print(f"\n-- {task.upper()} --")
        data = load_results(task, args.results_dir)
        all_data[task] = data
        if data:
            plot_accuracy(task, data, out_dir)
            plot_selectivity(task, data, out_dir)

    print("\n-- SUMMARY HEATMAP --")
    plot_summary_heatmap(all_data, args.tasks, out_dir)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()