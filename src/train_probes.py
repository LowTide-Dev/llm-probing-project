"""
train_probes.py
---------------
Trains logistic regression probes on each layer's embeddings for MatSciBERT.
Uses leave-one-simulation-out (LOSO) cross-validation to respect the
paraphrase structure of the dataset (5 paraphrases per simulation).

For each layer:
  - Reduces 768-dim embeddings to 32 dims via PCA (fit on train only)
  - Trains a real probe on true labels        → probe_acc
  - Trains a control probe on shuffled labels → control_acc
  - Selectivity = probe_acc - control_acc

Saves results to:
  results/probe_results_{task}_{pooling}.csv

Usage:
    python src/train_probes.py --task convergence
    python src/train_probes.py --task stability
    python src/train_probes.py --task convergence --pooling cls
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_simulation_groups(df):
    return df["source"].values


def probe_one_layer(X, y, groups, shuffle_seed=None, n_components=32):
    """
    Leave-one-simulation-out CV for a single layer's embeddings.

    X            : (n_examples, hidden_size)
    y            : (n_examples,) int labels
    groups       : (n_examples,) simulation folder names
    shuffle_seed : if not None, shuffle y before training (control probe)
    n_components : PCA dims before logistic regression
    """
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        y = rng.permutation(y)

    unique_groups = np.unique(groups)
    fold_accs = []

    for held_out in unique_groups:
        test_mask  = groups == held_out
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        # Scale
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # PCA: reduce 768 → n_components (fit on train split only)
        n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)

        # liblinear: fast and reliable on small low-dim data
        clf = LogisticRegression(
            C=1.0,
            max_iter=200,
            random_state=42,
            solver="liblinear",
        )
        clf.fit(X_train, y_train)
        fold_accs.append(accuracy_score(y_test, clf.predict(X_test)))

    return float(np.mean(fold_accs)) if fold_accs else float("nan")


def run_probes(embeddings, labels, groups, n_control_seeds=3):
    """
    Run real + control probes across all layers.

    embeddings : (n_examples, n_layers, hidden_size)
    labels     : (n_examples,)
    groups     : (n_examples,) simulation folder names
    """
    n_layers = embeddings.shape[1]
    rows = []

    for layer_idx in range(n_layers):
        layer_num = layer_idx + 1
        X = embeddings[:, layer_idx, :]

        probe_acc = probe_one_layer(X, labels, groups)

        control_accs = [
            probe_one_layer(X, labels, groups, shuffle_seed=seed)
            for seed in range(n_control_seeds)
        ]
        control_acc = float(np.mean(control_accs))
        selectivity = probe_acc - control_acc

        rows.append({
            "layer":       layer_num,
            "probe_acc":   round(probe_acc,   4),
            "control_acc": round(control_acc, 4),
            "selectivity": round(selectivity, 4),
        })

        print(
            f"  Layer {layer_num:2d} | "
            f"probe={probe_acc:.3f}  "
            f"control={control_acc:.3f}  "
            f"selectivity={selectivity:+.3f}"
        )

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    choices=["convergence", "stability"], required=True)
    parser.add_argument("--model",   default="matscibert")
    parser.add_argument("--pooling", choices=["cls", "mean", "both"], default="both")
    args = parser.parse_args()

    emb_dir   = f"data/embeddings/{args.model}/{args.task}"
    data_path = f"data/processed/{args.task}_labels.csv"
    out_dir   = "results"
    os.makedirs(out_dir, exist_ok=True)

    df     = pd.read_csv(data_path)
    labels = df["label"].values
    groups = get_simulation_groups(df)

    print(f"\n{'='*60}")
    print(f"Task : {args.task}  |  Model : {args.model}")
    print(f"N    : {len(labels)}  |  Groups : {len(np.unique(groups))}")
    print(f"Label balance: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"{'='*60}\n")

    pooling_modes = ["cls", "mean"] if args.pooling == "both" else [args.pooling]

    for pooling in pooling_modes:
        emb_file   = os.path.join(emb_dir, f"{pooling}_embeddings.npy")
        embeddings = np.load(emb_file)

        print(f"── Pooling: {pooling.upper()}  shape={embeddings.shape} ──")

        results_df = run_probes(embeddings, labels, groups)

        out_path = os.path.join(out_dir, f"probe_results_{args.task}_{pooling}.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")

        best = results_df.loc[results_df["selectivity"].idxmax()]
        print(
            f"Peak selectivity: layer {int(best.layer)} "
            f"(probe={best.probe_acc:.3f}, "
            f"control={best.control_acc:.3f}, "
            f"selectivity={best.selectivity:+.3f})\n"
        )


if __name__ == "__main__":
    main()
