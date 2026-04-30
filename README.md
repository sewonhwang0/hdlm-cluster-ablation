# hdlm-cluster-ablation
Toy-scale ablation of cluster semantics in Hierarchical Diffusion Language Models
# HDLM Cluster Ablation

A toy-scale ablation study of cluster semantics in Hierarchical Diffusion Language Models (HDLM).

**Reference paper**: Zhou et al., "Next Semantic Scale Prediction via Hierarchical Diffusion Language Models" (NeurIPS 2025)

## Question

Which aspect of HDLM's cluster construction carries the contribution — the semantic content of the clusters, or the size distribution of the partition?

## Setup

- BERT-style MLM proxy (22M params, 6 layers, hidden 384)
- WikiText-103, 30K BPE vocab, sequence length 256
- 27 runs total: 3 cluster types × 3 K values × 3 seeds
- Trained for 20K steps, batch size 32, AdamW with cosine schedule

**Three cluster conditions:**
- `semantic` — K-means on GPT-2 word embeddings (freq-weighted)
- `rand_matched` — random partition, same size distribution as semantic
- `rand_uniform` — random partition with uniform cluster sizes

**K values swept**: 16, 64, 256

## Findings

- **Semantic vs size-matched-random**: small but consistent gap on word NLL, growing monotonically with K (Δ = −0.026 → −0.034 → −0.056 nats; all p < 0.05).
- **Size distribution channel**: null at every K (p ≥ 0.37).
- **Cluster vs word asymmetry**: the cluster-level NLL gap is large (~50% relative) while the word-level gap is small (3–6% PPL).

## Caveats

- Toy scale (22M params, ~360M tokens) vs HDLM production (170–425M params, 131B tokens).
- n = 3 seeds — variance estimates are noisy.
- Single-noise BERT-style MLM proxy, not the full HDLM diffusion ELBO.
- Auxiliary classifier head rather than the Bayesian-posterior route from the paper's Table 5.

## Author

Se Won — independent learner, transitioning into ML from a mathematics background.
