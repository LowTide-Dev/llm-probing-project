# Interpreting Materials Science Knowledge in LLMs Using Linear Probes

**Course:** MATH498, Spring 2026  
**Instructor:** Michael Ivanitskiy  
**Author:** Talia Kumar

---

## Project Overview

This project applies linear probing to evaluate whether large language models encode
physical concepts in their internal representations, or merely recognize surface-level
patterns correlated with physical outcomes. I probe two pretrained models, 
MatSciBERT (domain-specific) and Llama-3.2-3B (general-purpose), for two binary
physical properties: convergence (Converged vs. Unconverged) and structural stability
(Stable vs. Unstable), using text descriptions of atomic configurations derived from
LAMMPS atomistic simulations.

---

## Repository Map
llm-probing-project-fresh/
│
├── README.md                          ← You are here. Navigation guide.
│
├── writeup/
│   ├── writeup.md                     ← Main paper writeup (Markdown, conference-style)
│   └── docs/                          ← Figures embedded in writeup
│       ├── comparison_convergence.png
│       ├── comparison_stability.png
│       ├── heatmap_convergence_llama.png
│       └── selectivity_convergence_matscibert.png
│
├── src/
│   ├── main.py                        ← Entry point; runs full pipeline
│   ├── load_models.py                 ← Loads MatSciBERT and Llama from HuggingFace
│   ├── extract_embeddings.py          ← Extracts layer-wise embeddings (CLS + mean pooling)
│   ├── extract_cls.py                 ← CLS/last-token extraction specifically
│   ├── train_probes.py                ← Trains L2 logistic regression probes per layer (LOSO CV)
│   ├── visualize_results.py           ← Generates all figures in results/figures/
│   └── convert_to_csv.py              ← Converts raw outputs to CSV format
│
├── data/
│   ├── embeddings/
│   │   ├── llama/
│   │   │   ├── convergence/           ← cls_embeddings.npy, mean_embeddings.npy, labels.npy
│   │   │   └── stability/             ← cls_embeddings.npy, mean_embeddings.npy, labels.npy
│   │   └── matscibert/
│   │       ├── convergence/           ← cls_embeddings.npy, mean_embeddings.npy, labels.npy
│   │       └── stability/             ← cls_embeddings.npy, mean_embeddings.npy, labels.npy
│   └── processed/
│       ├── convergence_labels.csv     ← Text descriptions + binary convergence label
│       ├── stability_labels.csv       ← Text descriptions + binary stability label
│       ├── descriptions.jsonl         ← Full dataset in JSONL format
│       └── descriptions_preview.txt   ← Human-readable sample of dataset
│
├── results/
│   ├── figures/                       ← All generated plots (15 total)
│   │   ├── comparison_convergence.png
│   │   ├── comparison_stability.png
│   │   ├── heatmap_convergence_llama.png
│   │   ├── heatmap_convergence_matscibert.png
│   │   ├── heatmap_stability_llama.png
│   │   ├── heatmap_stability_matscibert.png
│   │   ├── probe_accuracy_convergence_llama.png
│   │   ├── probe_accuracy_convergence_matscibert.png
│   │   ├── probe_accuracy_stability_llama.png
│   │   ├── probe_accuracy_stability_matscibert.png
│   │   ├── selectivity_convergence_llama.png
│   │   ├── selectivity_convergence_matscibert.png
│   │   ├── selectivity_stability_llama.png
│   │   ├── selectivity_stability_matscibert.png
│   │   └── summary_heatmap.png
│   ├── probe_results_convergence_cls_llama.csv
│   ├── probe_results_convergence_cls_matscibert.csv
│   ├── probe_results_convergence_mean_llama.csv
│   ├── probe_results_convergence_mean_matscibert.csv
│   ├── probe_results_stability_cls_llama.csv
│   ├── probe_results_stability_cls_matscibert.csv
│   ├── probe_results_stability_mean_llama.csv
│   ├── probe_results_stability_mean_matscibert.csv
│
├── .gitignore
└── requirements.txt                   ← Python dependencies


---

## Where to Find Everything

| Item | Location |
|------|----------|
| Main writeup / paper | `writeup/writeup.md` |
| Figures used in writeup | `writeup/docs/` |
| All generated figures | `results/figures/` |
| Probe results (CSV) | `results/` |
| Embedding extraction | `src/extract_embeddings.py`, `src/extract_cls.py` |
| Probe training code | `src/train_probes.py` |
| Model loading | `src/load_models.py` |
| Visualization code | `src/visualize_results.py` |
| Processed dataset | `data/processed/` |
| Raw embeddings | `data/embeddings/` |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/main.py

# Or run steps individually:
python src/extract_embeddings.py   # Extract layer-wise embeddings
python src/train_probes.py         # Train probes and compute selectivity
python src/visualize_results.py    # Generate figures
```

Embedding extraction for Llama was run on an NVIDIA L40S GPU (Wendian HPC cluster).
MatSciBERT embeddings were extracted on CPU.

---

## Models

| Model | Type | Layers | Hidden Size | Pretraining |
|-------|------|--------|-------------|-------------|
| MatSciBERT | BERT-style encoder | 12 | 768 | Materials science literature (~2.4M abstracts), initialized from SciBERT |
| Llama-3.2-3B | Decoder-only transformer | 28 | 3072 | Large general-purpose corpus (Meta, 2024) |

---

## Dataset

125 text descriptions derived from 25 unique LAMMPS atomistic simulations,
with 5 paraphrases per simulation. Labels assigned programmatically from
simulation output.

| Task | Examples | Unique Simulations | Class Balance |
|------|----------|--------------------|---------------|
| Convergence | 70 | 14 | 45 Converged / 25 Unconverged |
| Stability | 55 | 11 | 35 Stable / 20 Unstable |

---

## Current Status

- [x] Research question finalized
- [x] Literature review complete
- [x] Dataset generated (125 examples across 25 simulations)
- [x] Embeddings extracted for both models, both tasks
- [x] Probe training and LOSO cross-validation complete
- [x] Selectivity analysis complete
- [x] All figures generated
- [x] Writeup drafted (conference-paper style)

---

## Dependencies

See `requirements.txt`. Key packages: `transformers`, `torch`, `scikit-learn`,
`numpy`, `pandas`, `matplotlib`