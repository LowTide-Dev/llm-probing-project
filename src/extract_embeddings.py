"""
extract_embeddings.py
---------------------
Extracts layer-wise embeddings from MatSciBERT (encoder) and Llama-3.2-3B (decoder)
for each example in the dataset.

For MatSciBERT: probes layers 1–12, hidden size 768
For Llama-3.2-3B: probes layers 1–28, hidden size 3072

Usage:
    python extract_embeddings.py --model matscibert --task convergence
    python extract_embeddings.py --model llama      --task convergence
    python extract_embeddings.py --model matscibert --task stability
    python extract_embeddings.py --model llama      --task stability

On HPC, pass --device cuda to use GPU (strongly recommended for llama).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from load_models import load_model, MODEL_REGISTRY
from extract_cls import get_all_layer_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="Which model to extract embeddings from.",
    )
    parser.add_argument(
        "--task",
        choices=["convergence", "stability"],
        required=True,
        help="Which labeled dataset to run on.",
    )
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean", "both"],
        default="both",
        help="Which pooling strategy to save. 'cls' = [CLS]/last-token, 'mean' = mean pool.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on. Use 'cuda' on HPC.",
    )
    args = parser.parse_args()

    data_path = f"data/processed/{args.task}_labels.csv"
    out_dir   = f"data/embeddings/{args.model}/{args.task}/"
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, model, arch = load_model(args.model, device=args.device)

    df = pd.read_csv(data_path)
    cls_all, mean_all, labels = [], [], []

    print(f"[extract_embeddings] Running {args.model} ({arch}) on {len(df)} examples...")

    for i, row in df.iterrows():
        embs = get_all_layer_embeddings(
            row["text"], tokenizer, model, device=args.device, arch=arch
        )
        cls_all.append(embs["cls"])
        mean_all.append(embs["mean"])
        labels.append(row["label"])

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(df)}")

    # Shape: (n_examples, n_layers, hidden_size)
    cls_all  = np.array(cls_all)
    mean_all = np.array(mean_all)
    labels   = np.array(labels)

    if args.pooling in ("cls", "both"):
        np.save(f"{out_dir}/cls_embeddings.npy", cls_all)
    if args.pooling in ("mean", "both"):
        np.save(f"{out_dir}/mean_embeddings.npy", mean_all)

    np.save(f"{out_dir}/labels.npy", labels)

    print(f"\nSaved embeddings to {out_dir}")
    print(f"  CLS shape:  {cls_all.shape}   (n_examples, n_layers, hidden_size)")
    print(f"  Mean shape: {mean_all.shape}")
    print(f"  Labels:     {labels.shape}  unique={set(labels)}")


if __name__ == "__main__":
    main()