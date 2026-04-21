# Interpreting Materials Science Knowledge in LLMs Using Linear Probes

**Course:** MATH498, Spring 2026  
**Instructor:** Michael Ivanitskiy 
**Author:** Talia Kumar

---

## Project Overview

This project applies linear probing to evaluate whether domain-specific materials science language models (MatSciBERT and MaterialsBERT) encode physical concepts more explicitly in their internal representations than general-purpose models (SciBERT). We probe for two binary properties: convergence (Converged vs. Unconverged) and structural stability (Stable vs. Unstable), using text descriptions of atomic configurations derived from LAMMPS and Quantum ESPRESSO simulations.

---

## Repository Map
*Currently in a purely hypothetical state, could drastically change*

```
llm-probing-project/
│
├── README.md                     ← You are here. Navigation guide.
│
├── writeup/
│   └── writeup.md                ← Main paper writeup (Markdown, conference-style)
│
├── src/
│   ├── generate_dataset.py       ← Script to generate labeled text descriptions
│   ├── extract_embeddings.py     ← Extracts [CLS] and mean-pooled embeddings from model layers
│   ├── train_probes.py           ← Trains logistic regression probes per layer
│   ├── compute_selectivity.py    ← Computes selectivity scores (probe acc - control acc)
│   └── visualize_results.py      ← Plots layer-wise probe accuracy curves
│
├── data/
│   ├── raw/                      ← Raw LAMMPS/QE output files (not committed if large)
│   ├── processed/
│   │   ├── convergence_labels.csv    ← Text + binary convergence label
│   │   └── stability_labels.csv      ← Text + binary stability label
│   └── README.md                 ← Data provenance and format description
│
├── results/
│   ├── probe_accuracy_convergence.csv   ← Layer-wise probe accuracy (SciBERT vs MatSciBERT vs MaterialsBERT)
│   ├── probe_accuracy_stability.csv
│   ├── selectivity_scores.csv           ← Selectivity = probe acc - control acc
│   └── figures/
│       ├── convergence_layerwise.png
│       └── stability_layerwise.png
│
└── requirements.txt              ← Python dependencies
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate / prepare the dataset
python src/generate_dataset.py

# 3. Extract embeddings from all layers of SciBERT and MatSciBERT
python src/extract_embeddings.py --model scibert --task convergence
python src/extract_embeddings.py --model matscibert --task convergence
python src/extract_embeddings.py --model materialsbert --task convergence

# 4. Train linear probes on each layer
python src/train_probes.py --task convergence

# 5. Compute selectivity scores
python src/compute_selectivity.py --task convergence

# 6. Generate figures
python src/visualize_results.py
```

---

## Group Members & Contributions

| Name | Contributions |
|------|--------------|
| Talia Kumar | Research question, dataset generation (LAMMPS/QE), embedding extraction, probe training, writeup |

---

## Where to Find Everything

| Item | Location |
|------|----------|
| Main writeup / paper | `writeup/writeup.md` |
| Data pipeline code | `src/generate_dataset.py`, `src/extract_embeddings.py` |
| Probe training code | `src/train_probes.py`, `src/compute_selectivity.py` |
| Visualizations | `results/figures/` |
| Raw & processed data | `data/` |
| Experimental results | `results/` |

---

## Current Status (Midpoint Update)

- [x] Research question finalized
- [x] Literature review complete
- [x] Dataset generation pipeline drafted
- [ ] Full dataset (300 examples) collected — *in progress*
- [ ] Embedding extraction run on both models
- [ ] Probe training and evaluation
- [ ] Selectivity analysis
- [ ] Final figures and writeup

---

## Dependencies

See `requirements.txt`. Key packages: `transformers`, `torch`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`.