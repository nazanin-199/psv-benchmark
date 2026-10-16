# When Does Protein Structure Add Value?
### A Sequence-Identity-Aware Benchmark for Protein Function Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![ESM-2](https://img.shields.io/badge/Model-ESM--2-blue.svg)](https://github.com/facebookresearch/esm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Research Motivation
Pretrained protein language models (pLMs) such as ESM-2 learn rich representations of evolutionary co-variation that correlate strongly with tertiary structure and function. Consequently, simple linear probes on frozen sequence embeddings frequently match or outperform graph neural networks (GNNs) trained on explicit 3D atomic coordinates.

This project investigates a hypothesis-driven empirical research question:
> **Under what specific conditions does explicitly incorporating 3D atomic coordinates provide measurable, orthogonal predictive value over pretrained sequence embeddings alone?**

Specifically, we examine two regimes:
1. **The Sequence Twilight Zone ($<30\%$ Identity):** When test proteins share low sequence identity with the training set, does explicit 3D message passing generalize better across divergent enzyme folds?
2. **Label Scarcity ($N \le 20$ samples/class):** Does a fixed 3D contact graph provide an inductive regularizer that mitigates overfitting when functional annotations are limited?

---

## Controlled Experimental Design

To isolate the contribution of structural inductive bias, we evaluate three controlled architectures sharing identical classification heads and loss objectives across primary Enzyme Commission (EC 1–6) classes:

* **Model A (Sequence Only):** Frozen ESM-2 (`esm2_t12_35M_UR50D`) mean-pooled representations ($480\text{d}$) $\rightarrow$ 2-layer MLP classifier.
* **Model B (Structure Only):** Residue-level $C_\alpha$ contact graph ($r \le 8.0\text{ \AA}$) with backbone dihedral angles ($\phi, \psi$) $\rightarrow$ Edge-conditioned message-passing GNN $\rightarrow$ Global pooling $\rightarrow$ MLP classifier.
* **Model C (Sequence–Structure Hybrid):** Per-residue ESM-2 token embeddings ($480\text{d}$) mapped as initial node features in the $C_\alpha$ contact graph $\rightarrow$ Edge-conditioned message-passing GNN $\rightarrow$ Global pooling $\rightarrow$ MLP classifier.

## Benchmark Results (5-Fold Cluster-Disjoint Cross-Validation, <30% Sequence Identity)

*Evaluated across 5 independent cluster-disjoint folds on non-redundant experimental structures (EC 1–6):*

| Architecture | Input Representations | Trainable Parameters | Mean Macro-F1 | Std Dev |
| :--- | :--- | :---: | :---: | :---: |
| **Model A: Sequence Only** | ESM-2 Global Mean Pool ($480\text{d}$) | 62,598 | **0.178** | $\pm 0.057$ |
| **Model B: Structure Only** | Residue One-Hot + Dihedrals ($25\text{d}$) + 3D Edges | 153,222 | **0.113** | $\pm 0.044$ |
| **Model C: Hybrid** | ESM-2 Residue Tokens ($480\text{d}$) + 3D Edges | 211,974 | **0.172** | $\pm 0.052$ |

### Scientific Conclusions
1. **Evolutionary Sequence Embeddings Remain the Dominant Signal:** Pretrained ESM-2 sequence embeddings retain sufficient functional co-variation that a lightweight MLP baseline matches or exceeds an explicit 3D message-passing network even in the sequence twilight zone ($<30\%$ pairwise identity).
2. **Structural Signal is Non-Trivial but High-Variance:** While explicit backbone geometry achieves above-collapse performance ($0.113 \pm 0.044$) and occasionally outperforms sequence alone on specific folds (e.g., Folds 1 and 5), training geometric message passing from scratch adds parameter capacity that does not yield a net statistical advantage when fused naively with frozen pLM representations.
3. **Parity Under Strict Leakage Control:** Under rigorous cluster-disjoint evaluation, hybrid sequence-structure models do not automatically yield a performance dividend over frozen pLM embeddings, highlighting that architectural inductive biases must be weighed against optimization complexity and data scale.
