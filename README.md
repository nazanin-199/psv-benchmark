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
