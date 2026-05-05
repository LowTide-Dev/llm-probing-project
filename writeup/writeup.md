# Interpreting Materials Science Knowledge in LLMs Using Linear Probes

**Talia Kumar**  
MATH498 — Spring 2026

---

## Abstract
Large Language Models (LLMs) that are trained and/or pretrained on science materials and text have been used 
more and more for specialized work as they become smarter and more useful in such fields. One major example of this is in the world of materials science as being able to preform large scale atom simulations can be practially impossible on most computers and hard to compute without the help of an HPC, or perhaps now, an LLM. 
Although the use of LLMs for this kind of work is becoming more and more frequent in today's world (I would recommend looking up Periodic Labs if you are not familar), it remains unclear whether the tasks performed by these LLMs are truly reflecting the physical concepts and aspects of the materials and their atoms or just preforming the pattern matching that neural networks are so good at. 
This project hopes to apply linear probing to investiagte if science trained models can understad and wrap their metaphorical heads around the physics of quantum material level simulations. Specifically, by looking at the models' hidden layers and comparing them to a more general model. I plan on constructing a dataset of at least 50-100 (with hopes of up to 300 given time) text descriptions of atomic configurations derived from LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) simulations. Each will be labeled with physical properties that seperate them from others. Then I will train logistic regression probes on the layer-wise [CLS] embeddings of all models and report selectivity scores to distinguish encoded knowledge from task-learning artifacts. I expect the most materials science specific models to show the highest selectivity in deeper layers; however, the most recent  literature (Rubungo et al., 2023) suggests that general-purpose models may already capture significant domain knowledge, making the null result equally informative.

---

## 1. Introduction
As we've seen the growth and perfomance of LLMs sky-rocket on what kind of feels like a daily basis, scientists have wanted to test these models on specific knowledge. Notably, in the world of materials science- having a reliable LLM for things like predicting a materials properties or usefulness can be a huge assest, especially as the world that does test these things can be slower than the progress of AI. Currently, strong benchmark performance of an LLM does not necessarily mean that the given model have formed an actual, replicable understading/representation of the physics behind certain material structures and experiments/simulations. A model can achieve high accuracy on a materials question, answering the given task by using the known pattern recognition tools from texts and provided corpa, rather than truly knowing the atomic structure of anything it is give. 

Being able to understanding the underlying physics does matter in a practical sense. If domain-specific pretraining, like science specific models, shows to improve task performance using true physical concepts and properties, then we should be able to use such models for greater purposes to help a research timeline(or the like) jump drastically. If the models really do understand these concepts, we should be able to see them via hidden representation in the hidden layers of the model. If, on the other hand, the improvement is superficial, domain-specific models carry a false sense of reliability, which can be detrimental, especially long-term.

To truly examine the substantibility of these models understandings, I will be using Linear Probes. A linear probe (Alain & Bengio, 2016) is a logistic regression classifier that is trained on frozen hidden states of a pretrained model. If a probe trained on a particular layer achieves high accuracy, that layer's representations make the given concept linearly separable, meaning the model has already done the work of separating the two classes internally, a simple classifier on top is enough to read it out.

This project applies linear probing to two binary physical properties drawn from computational materials science:

1. **Convergence**: whether a self-consistent field (SCF) DFT calculation has converged.
2. **Structural Stability**: whether a given atomic configuration is energetically stable.

These properties are well-defined, simulation-grounded, and meaningful in materials research contexts. My central question is:

> *Do materials science-focused LLMs encode domain-relevant concepts more explicitly than general scientific models, as measured by linear probe selectivity on hidden representations?*

The answer has implications not only for materials science AI, but for the broader question of what domain-specific pretraining actually buys us at the representational level.

---

## 2. Related Work (literature review)

### 2.1 Linear Probing as an Interpretability Method

Alain & Bengio (2016) introduced linear probes as a diagnostic tool for understanding what information is encoded at each layer of a neural network. Their key insight is that deeper layers may disregard low-level input details while simultaneously reorganizing the rest of the information into a form easier to read out. This means that probe accuracy can increase with depth even if the representation of information drifts further from the raw input. Training a logistic regression on layer \ell's activations measures how "explicitly" that layer represents a target concept. This project will adopt their selectivity metric, probe accuracy minus control probe accuracy on shuffled labels, to ensure what is being measured is what the model has encoded, not just what a probe can learn to predict.

### 2.2 Domain-Specific Language Models for Materials Science

MatSciBERT (Gupta et al., 2022) extends BERT-style pretraining to materials science literature and demonstrates strong gains on named entity recognition and relation extraction tasks in the domain. LLaMat similarly shows that domain-specific pretraining can outperform larger general-purpose models on materials tasks. These results motivate the hypothesis that domain-specific models encode physical concepts more explicitly. Is is good to note that behavioral gains do not, by themselves, establish this.

### 2.3 Surprising Strength of General-Purpose Models

Rubungo et al. (2023) presents LLM-Prop, which finds that a general-purpose T5 encoder with a simple linear projection can outperform domain-specific models and state-of-the-art graph neural networks on materials property prediction tasks including band gap and unit cell volume, which are specific and important measurements for atomic simulations. The authors argue this reflects the model's ability to access critical physical information, such as space group symmetry, from text descriptions. This finding sets up an important foil for this project: if a general-purpose model already captures the physical concepts we probe for, domain-specific pretraining may improve generation quality rather than conceptual encoding.

### 2.4 Token-Level Chemical Understanding

Zhang & Yang (PolyLLMem) visualize Llama 3 embeddings with UMAP and find that the model naturally clusters representations by physical properties prior to task-specific training. Token-level analysis further reveals that the model internalizes structural chemistry, recognizing aromatic rings and the effects of functional groups. This supports the possibility that general pretraining on scientific text may be enough to form meaningful physical representations.

### 2.5 Robustness Limitations

Tenney, Das, & Pavlick (2019) evaluate LLMs on materials science QA and property prediction, revealing sensitivity to prompt phrasing, degraded performance under distribution shift, and limited generalization. These results suggest behavioral performance is fragile, and motivate a probe-based approach that bypasses surface-level task performance in favor of internal representation quality.

### 2.6 Template: Domain-Specific vs. General Models via Probing

Hummel et al. (2026) provide a direct methodological template for this project's work. Using linear probing to compare domain-specific bioacoustic models against general-purpose audio models, they show that even embeddings dominated by irrelevant features (e.g., recording-specific IDs) can be filtered via linear probing to isolate domain-relevant features (e.g., ship acoustic signatures). Their systematic comparison framework directly informs our experimental design.

---

## 3. Methodology

### 3.1 Dataset Construction
I constructed a dataset of 50–100 text descriptions of atomic simulations, each labeled with one of two binary properties. Descriptions are generated from LAMMPS molecular dynamics and geometry optimization outputs, with each unique simulation paraphrased into five stylistically distinct descriptions to increase surface-level diversity while preserving the underlying physical content.

Convergence label: 
* Converged / Unconverged: whether a geometry optimization satisfied both energy and force tolerance criteria, or instead terminated early (e.g., hitting the maximum iteration limit or encountering line search failures)
* Stability label - Stable / Unstable: whether an NVT or NPT molecular dynamics trajectory remained thermodynamically well-behaved, based on temperature stabilization and absence of pathological behaviors such as dangerous neighbor list builds

Labels are assigned programmatically from simulation output rather than manually, grounding them in physical definitions. An example description:
"A geometry optimization using LJ/cut was run on 4 atoms under periodic boundary conditions. The relaxation satisfied both the energy and force tolerance criteria. Final energetics: potential energy = −70.4177 eV, total energy = −70.1850 eV."

## 3.1.1 Helpful Background
Atomic level simulations model the behavior of large collections of atoms, specifically manipulating and/or recording how they interact and how they evolve over time or toward the materials equilibrium point. These interactions are known as the interatomic potential. A good optimization seeks a configuration where atomic forces are minimized; a standard molecular dynamic simulation involves relaxing a group of atoms under under a chosen thermodynamic ensemble, or given it certain volumetric parameters to see what happens under certain temperature conditions. Whether a given simulation has converged or produced a stable configuration are well-defined physical outcomes that I will use as probe targets.

* LJ/cut (Lennard-Jones potential) is a classical pairwise potential suitable for simple metals
* NequIP is a machine-learned interatomic potential based on equivariant neural networks, capable of capturing more complex bonding environments.
* NVT (constant **N**umber of atoms, **V**olume, and **T**emperature) and NPT (constant **N**umber of atoms, **P**ressure, and **T**emperature) are molecular dynamics runs




### 3.2 Models
*This section will likely change as I look to include a more recent model to compare.*

We compare three BERT-style encoder models:
We compare three BERT-style encoder models:

| Model | Description |
|-------|-------------|
| **SciBERT** (Beltagy et al., 2019) | BERT pretrained on scientific text (biomedical + CS literature) |
| **MatSciBERT** (Gupta et al., 2022) | SciBERT further pretrained on materials science literature |
| **MaterialsBERT** (Shetty, P., . et al, 2023) | BERT trained from scratch on 750,000 inorganic, organic, and composite materials articles (2005–2019, ~3B words) | 

All models use 12 transformer layers with a [CLS] token. I will extract the [CLS] embedding from each of the 12 layers of each model, yielding 24 embedding sets per example. I will also test mean-pooling across all tokens as an alternative to [CLS] extraction, given that physical knowledge in scientific text may be distributed across tokens rather than concentrated at [CLS].

### 3.3 Probe Training

For each layer \ell \in \{1, ..., 12\} and each model, I plan to train a logistic regression probe on the extracted embeddings. The data will be split by unique simulation (80/20 train/test), ensuring that paraphrases of the same simulation are never split across sets. The probe is intentionally simple, L2-regularized logistic regression, because a more powerful classifier could learn to predict the label from any weak statistical regularity in the activations, which would say more about the probe's capacity rather than the model's representations.
The plan is to also train a control probe on randomly shuffled labels. This establishes how much accuracy a probe can achieve, independent of the target concept. Selectivity is then defined as:
\text{Selectivity}(\ell) = \text{Acc}_{\text{probe}}(\ell) - \text{Acc}_{\text{control}}(\ell)
A layer with positive selectivity encodes the target concept in a way that goes beyond what a probe could pick up by chance; a layer near zero selectivity does not, regardless of its raw accuracy.

### 3.4 Evaluation

To report:
To report:

1. **Layer-wise probe accuracy** for all models on both tasks (convergence, stability)
2. **Selectivity curves** across layers for both models
3. **Peak selectivity layer** — which layer most explicitly encodes each concept
4. **Model comparison**: Which models achieves highest peak selectivity?

---

## 4. Experiments and Preliminary Results

To add later

### 4.1 Dataset Status

The current dataset contains 45 text descriptions derived from 9 unique simulations, 5 per simulation. Of these, 25 are labeled for convergence (10 Converged, 15 Unconverged) and 20 for stability (10 Stable, 10 Unstable). Simulations use either a LJ/cut or NequIP potential, covering relax, NVT, and NPT run types.
This is substantially smaller than the 100-300 examples originally proposed, and too small to draw reliable conclusions from probe training. The paraphrase structure means there are effectively only 9 independent data points, making an 80/20 train/test split unreliable. Time permitting, I hope to expand the dataset significantly before running probe training; results reported here should be treated as preliminary.


### 4.2 Preliminary Embedding Extraction

To add later


---

## 5. Expected Results and Preliminary Conclusions

I anticipate the following outcomes, in rough order of probability:
I anticipate the following outcomes, in rough order of probability:

**Most likely**: The materials science specific models will outpreform the less materials science specific models on all tasks. 

**Also plausible** (following Rubungo et al., 2023): SciBERT already achieves high selectivity on convergence or stability, suggesting that general scientific pretraining is sufficient to encode these concepts. If confirmed, this would indicate that domain-specific pretraining improves *generative* rather than representational quality.

Our conclusion will follow from which of these outcomes the experiments support. The key comparison is not raw accuracy but selectivity, which isolates representational encoding from task-learning.

---

## 6. Roadblocks and Open Challenges

1. **Data volume**: 50-100 examples is small for reliable probe training. I may face high variance in accuracy estimates. I are considering extending to 300 examples if simulation time permits.

2. **Distribution of descriptions**: The text descriptions are generated from our own simulations, which may have limited stylistic diversity. If descriptions are too similar (e.g., formulaic output from the same simulation code), the probe may learn surface patterns rather than physical semantics.

3. **Defining "stability"**: Structural stability is inherently relative and threshold-dependent. I must be precise about what constitutes a stable vs. unstable configuration in our labeling pipeline to avoid noisy labels.

4. **Compute**: Extracting embeddings across all 12 layers for both models on 50-100 examples is manageable; running a full grid of probes (12 layers × 2 models × 2 tasks × 5-fold CV) is feasible on CPU but will take time.

---

## 7. Next Steps
## 7. Next Steps

- [ ] Complete dataset collection to 100 examples
- [ ] Run embedding extraction on full dataset 
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
