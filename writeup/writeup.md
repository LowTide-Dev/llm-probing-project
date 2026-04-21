# Interpreting Materials Science Knowledge in LLMs Using Linear Probes

**Talia Kumar**  
MATH498 — Spring 2026

---

## Abstract

Large language models (LLMs) trained on scientific corpora are increasingly used in materials science work and research, yet it remains unclear whether improved task performance reflects genuinely richer internal representations of physical concepts or merely better surface-level pattern matching. This project applies linear probing to investigate if domain-specific models, MatSciBERT, and MaterialsBert, encode materials science concepts—specifically, simulation convergence and structural stability—more explicitly in its hidden layers than a general scientific model, SciBERT. I will construct a dataset of ~300 text descriptions of atomic configurations derived from LAMMPS and Quantum ESPRESSO simulations, each labeled with a binary physical property. Then I will train logistic regression probes on the layer-wise [CLS] embeddings of all models and report selectivity scores to distinguish encoded knowledge from task-learning artifacts. I expect MaterialsBERT to show higher selectivity in deeper layers; however, recent literature (Rubungo et al., 2023) suggests that general-purpose models may already capture significant domain knowledge, making the null result equally informative.

---

## 1. Introduction

The remarkable performance of large language models on scientific benchmarks has prompted substantial interest in their use as reasoning engines for materials discovery and property prediction. However, strong benchmark performance does not necessarily entail that a model has formed a coherent internal representation of the underlying physics. A model may achieve high accuracy on a materials question-answering task by associating lexical patterns with correct answers, without ever forming anything resembling a "world model" of atomic structure.

This distinction matters practically. If domain-specific pretraining (e.g., MatSciBERT, trained on materials science literature) improves task performance by encoding more explicitly physical concepts—making them linearly accessible from hidden representations—then we have grounds to trust such models as scientific reasoning tools. If, on the other hand, the improvement is superficial, domain-specific models carry a false sense of reliability.

Linear probing (Alain & Bengio, 2016) offers a principled way to test this question. A linear probe is a logistic regression classifier trained on the frozen hidden states of a pretrained model. High probe accuracy at a given layer indicates that the concept in question is *linearly decodable* from that layer's representation—a stronger claim than behavioral accuracy alone.

This project applies linear probing to two binary physical properties drawn from computational materials science:

1. **Convergence**: whether a self-consistent field (SCF) DFT calculation has converged.
2. **Structural Stability**: whether a given atomic configuration is energetically stable.

These properties are well-defined, simulation-grounded, and meaningful in materials research contexts. Our central question is:

> *Do materials science-focused LLMs (MatSciBERT and MaterialsBERT) encode domain-relevant concepts more explicitly than general scientific models (SciBERT), as measured by linear probe selectivity on hidden representations?*

The answer has implications not only for materials science AI, but for the broader question of what domain-specific pretraining actually buys us at the representational level.

---

## 2. Related Work (literature review)

### 2.1 Linear Probing as an Interpretability Method

Alain & Bengio (2016) introduced linear probes as a diagnostic tool for understanding the information geometry of intermediate neural network layers. Their key insight is that while deep layers may technically destroy Shannon information, they perform nonlinear transformations that can make information more linearly accessible. A classifier probe trained on layer $\ell$'s activations measures how "explicitly" that layer represents a target concept. We adopt their selectivity metric (probe accuracy minus control probe accuracy) to control for the possibility that the probe is learning the task rather than measuring representation.

### 2.2 Domain-Specific Language Models for Materials Science

MatSciBERT (Gupta et al., 2022) extends BERT-style pretraining to materials science literature and demonstrates strong gains on named entity recognition and relation extraction tasks in the domain. LLaMat similarly shows that domain-specific pretraining can outperform larger general-purpose models on materials tasks. These results motivate our hypothesis that domain-specific models encode physical concepts more explicitly—though behavioral gains do not, by themselves, establish this.

### 2.3 Surprising Strength of General-Purpose Models

Rubungo et al. (2023) present LLM-Prop, which finds that a general-purpose T5 encoder with a simple linear projection can outperform domain-specific models like MatBERT and state-of-the-art graph neural networks on materials property prediction tasks including band gap and unit cell volume. The authors argue this reflects the model's ability to access critical physical information—such as space group symmetry—from text descriptions. This finding sets up an important foil for our study: if a general-purpose model already captures the physical concepts we probe for, domain-specific pretraining may improve generation quality rather than conceptual encoding.

### 2.4 Token-Level Chemical Understanding

Zhang & Yang (PolyLLMem) visualize Llama 3 embeddings with UMAP and find that the model naturally clusters representations by physical properties (e.g., glass transition temperature, band gap) prior to task-specific training. Token-level analysis further reveals that the model internalizes structural chemistry, recognizing aromatic rings and the effects of functional groups. This supports the possibility that general pretraining on sufficiently scientific text may be enough to form meaningful physical representations.

### 2.5 Robustness Limitations

Tenney, Das, & Pavlick (2019) evaluate LLMs on materials science QA and property prediction, revealing sensitivity to prompt phrasing, degraded performance under distribution shift, and limited generalization. These results suggest behavioral performance is fragile, and motivate a probe-based approach that bypasses surface-level task performance in favor of internal representation quality.

### 2.6 Template: Domain-Specific vs. General Models via Probing

Hummel et al. (2026) provide a direct methodological template for our work. Using linear probing to compare domain-specific bioacoustic models against general-purpose audio models, they show that even embeddings dominated by irrelevant features (e.g., recording-specific IDs) can be filtered via linear probing to isolate domain-relevant features (e.g., ship acoustic signatures). Their systematic comparison framework directly informs our experimental design.

---

## 3. Methodology

### 3.1 Dataset Construction

We construct a dataset of approximately 300 short text descriptions of atomic configurations. Each description is derived from simulation output produced by LAMMPS (molecular dynamics) or Quantum ESPRESSO (DFT), and is labeled with one of two binary properties:

- **Convergence label**: Converged / Unconverged (based on SCF convergence thresholds in QE output)
- **Stability label**: Stable / Unstable (based on total energy relative to a reference configuration)

Descriptions are written in the style of materials science literature. For example:

> *"The simulation cell contains 64 atoms of FCC copper at 300 K. Total energy oscillated for 50 steps and failed to reach the SCF threshold of 1e-6 Ry."*

Labels are derived directly from simulation outputs rather than manually assigned, giving us ground-truth annotations grounded in physical definitions. We aim for approximately 150 examples per class per task.

### 3.2 Models

We compare two BERT-style encoder models:

| Model | Description |
|-------|-------------|
| **SciBERT** (Beltagy et al., 2019) | BERT pretrained on scientific text (biomedical + CS literature) |
| **MatSciBERT** (Gupta et al., 2022) | SciBERT further pretrained on materials science literature |
| **MaterialsBERT** (Shetty, P., . et al, 2023) | BERT trained from scratch on 750,000 inorganic, organic, and composite materials articles (2005–2019, ~3B words) | 

All models use 12 transformer layers with a [CLS] token. We extract the [CLS] embedding from each of the 12 layers of each model, yielding 24 embedding sets per example. We will also test mean-pooling across all tokens as an alternative to [CLS] extraction, given that physical knowledge in scientific text may be distributed across tokens rather than concentrated at [CLS].

### 3.3 Probe Training

For each layer $\ell \in \{1, ..., 12\}$ and each model, we train a logistic regression probe on the extracted embeddings using an 80/20 train/test split. The probe is intentionally simple—L2-regularized logistic regression—to avoid the probe itself overfitting and confounding our interpretation.

We also train a control probe on randomized labels to establish a baseline accuracy that captures only the information in the embedding geometry (not the labels). Selectivity is then defined as:

$$\text{Selectivity}(\ell) = \text{Acc}_{\text{probe}}(\ell) - \text{Acc}_{\text{control}}(\ell)$$

Positive selectivity at a layer indicates that layer genuinely encodes the target concept.

### 3.4 Evaluation

We report:

1. **Layer-wise probe accuracy** for SciBERT and MatSciBERT on both tasks (convergence, stability)
2. **Selectivity curves** across layers for both models
3. **Peak selectivity layer** — which layer most explicitly encodes each concept
4. **Model comparison**: Does MatSciBERT achieve higher peak selectivity than SciBERT?

---

## 4. Experiments and Preliminary Results

To add later

### 4.1 Dataset Status

To add later

### 4.2 Preliminary Embedding Extraction

To add later


---

## 5. Expected Results and Preliminary Conclusions

We anticipate the following outcomes, in rough order of probability:

**Most likely**: MaterialsBERT outpreforms MatSciBERT which outperforms SciBERT on all tasks. 

**Also plausible** (following Rubungo et al., 2023): SciBERT already achieves high selectivity on convergence or stability, suggesting that general scientific pretraining is sufficient to encode these concepts. If confirmed, this would indicate that domain-specific pretraining improves *generative* rather than representational quality.

Our conclusion will follow from which of these outcomes the experiments support. The key comparison is not raw accuracy but selectivity, which isolates representational encoding from task-learning.

---

## 6. Roadblocks and Open Challenges

1. **Data volume**: 300 examples is small for reliable probe training. We may face high variance in accuracy estimates. We are considering extending to 500 examples if simulation time permits.

2. **Distribution of descriptions**: The text descriptions are generated from our own simulations, which may have limited stylistic diversity. If descriptions are too similar (e.g., formulaic output from the same simulation code), the probe may learn surface n-gram patterns rather than physical semantics.

3. **Defining "stability"**: Structural stability is inherently relative and threshold-dependent. We must be precise about what constitutes a stable vs. unstable configuration in our labeling pipeline to avoid noisy labels.

4. **Compute**: Extracting embeddings across all 12 layers for both models on 300 examples is manageable; running a full grid of probes (12 layers × 2 models × 2 tasks × 5-fold CV) is feasible on CPU but will take time.

---

## 7. Questions for the Instructor

1. **Statistical testing for selectivity**: Should we use paired t-tests or bootstrap confidence intervals to assess whether MatSciBERT's selectivity is *significantly* higher than SciBERT's? Or is visual comparison of curves the standard in the probing literature?

2. **Control probe design**: Alain & Bengio use a permuted-label control. Is this the right baseline here, or would a majority-class baseline be more interpretable for a reader unfamiliar with the probing literature?

3. **Dataset validity**: Does generating descriptions from my own simulation data (rather than existing published materials science text) strengthen or weaken the claim that we're testing "domain knowledge" that could have been in the training corpus? I worry the models haven't seen descriptions in exactly my format.

4. **Mean-pool vs. [CLS]**: Is there a community norm in BERT-probing literature for which representation to prefer, or should we treat this as an experimental variable to report?

5. **Scope check**: Am I right to focus only on binary classification probes (2 classes)? Or should I attempt a regression probe on a continuous property (e.g., total energy value)?

---

## 8. Next Steps

- [ ] Complete dataset collection to 300 examples
- [ ] Run embedding extraction on full dataset (SciBERT + MatSciBERT, all 12 layers, both pooling strategies)
- [ ] Train and evaluate probes; compute selectivity curves
- [ ] Statistical comparison of peak selectivity between models
- [ ] Generate final figures (layer-wise accuracy plots, selectivity bar charts)
- [ ] Expand Sections 4 and 5 of this writeup with actual results
- [ ] Finalize bibliography and check all citations

---

## Contributions

| Contributor | Role |
|-------------|------|
| Talia Kumar | All aspects: research question, dataset generation, modeling, analysis, writing |

---

## References

Alain, Guillaume, and Yoshua Bengio. "Understanding intermediate layers using linear classifier probes." *arXiv preprint arXiv:1610.01644* (2016).

Beltagy, Iz, Kyle Lo, and Arman Cohan. "SciBERT: A pretrained language model for scientific text." *arXiv preprint arXiv:1903.10676* (2019).

Gupta, Tanishq, et al. "MatSciBERT: A materials domain language model for text mining and information extraction." *npj Computational Materials* 8 (2022): 1–11.

Hummel, [first name], et al. "Linear probing for domain-specific feature isolation in pretrained audio models." [venue] (2026). *(full citation to be completed)*

Rubungo, Andre Niyongabo, et al. "LLM-Prop: Predicting physical and electronic properties of crystalline solids from their text descriptions." *arXiv preprint arXiv:2310.05512* (2023).

Tang, Yingheng, et al. "MatterChat: A multi-modal LLM for material science." *arXiv preprint arXiv:2502.13107* (2025).

Tenney, Ian, Dipanjan Das, and Ellie Pavlick. "BERT Rediscovers the Classical NLP Pipeline." *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 4593–4601 (2019).

Zhang, [first name], and Yang, [first name]. "PolyLLMem: Exploring whether LLMs internalize chemical understanding for polymer property prediction." *(full citation to be completed)*

---
