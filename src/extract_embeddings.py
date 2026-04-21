"""
extract_embeddings.py
---------------------
Extracts layer-wise [CLS] and mean-pooled embeddings from SciBERT or MatSciBERT
for each example in the dataset.

Usage:
    python extract_embeddings.py --model scibert    --task convergence
    python extract_embeddings.py --model matscibert  --task stability
    python extract_embeddings.py --model materialsbert  --task stability
"""

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAMES = {
    "scibert":    "allenai/scibert_scivocab_uncased",
    "matscibert": "m3rg-iitd/matscibert",
    "materialsbert": "matscibert/MaterialsBERT",
}


def get_layer_embeddings(text: str, model, tokenizer, device: str = "cpu"):
    """
    Returns a dict with:
      - 'cls':  np.array of shape (n_layers, hidden_size)
      - 'mean': np.array of shape (n_layers, hidden_size)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors, each (1, seq_len, hidden)
    cls_embeddings = np.array([hs[0, 0, :].cpu().numpy() for hs in hidden_states[1:]])  # skip embedding layer
    mean_embeddings = np.array([hs[0, :, :].mean(dim=0).cpu().numpy() for hs in hidden_states[1:]])

    return {"cls": cls_embeddings, "mean": mean_embeddings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scibert", "matscibert", "materialsbert"], required=True)
    parser.add_argument("--task", choices=["convergence", "stability"], required=True)
    parser.add_argument("--pooling", choices=["cls", "mean", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_path = f"data/processed/{args.task}_labels.csv"
    out_dir = f"data/embeddings/{args.model}/{args.task}/"
    import os
    os.makedirs(out_dir, exist_ok=True)

    print(f"[extract_embeddings.py] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[args.model])
    model = AutoModel.from_pretrained(MODEL_NAMES[args.model]).to(args.device)
    model.eval()

    df = pd.read_csv(data_path)
    cls_all, mean_all, labels = [], [], []

    for _, row in df.iterrows():
        embs = get_layer_embeddings(row["text"], model, tokenizer, args.device)
        cls_all.append(embs["cls"])
        mean_all.append(embs["mean"])
        labels.append(row["label"])

    # Shape: (n_examples, n_layers, hidden_size)
    cls_all = np.array(cls_all)
    mean_all = np.array(mean_all)
    labels = np.array(labels)

    np.save(f"{out_dir}/cls_embeddings.npy",  cls_all)
    np.save(f"{out_dir}/mean_embeddings.npy", mean_all)
    np.save(f"{out_dir}/labels.npy",          labels)

    print(f"Saved embeddings to {out_dir}")
    print(f"  CLS shape:  {cls_all.shape}")
    print(f"  Mean shape: {mean_all.shape}")


if __name__ == "__main__":
    main()