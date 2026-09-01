# Interpretable-VAE-for-Antimicrobial-Peptide-Design
This teaching project explores the generation of antimicrobial peptides (AMPs) using a two-dimensional variational autoencoder (VAE) that was trained on peptide sequences with known minimum inhibitory concentration (MIC) values against Escherichia coli. The goal is to generate peptide-like sequences and obtain an interpretable 2D latent space that can be visualized and related to properties such as activity, charge, and hydrophobicity.

## Project overview

The pipeline contains four main parts:
1. Sequence VAE
2. Latent space analysis
3. MIC-aware training
4. Candidate generation and ranking

## Main results

1) The VAE learns a latent space that captures meaningful biological structure:
- peptides with different charge and hydrophobicity occupy different regions
- lower MIC values are enriched in specific latent regions

2) Although a purely unsupervised VAE provides a useful generative latent space, MIC information is only weakly preserved. Adding an auxiliary MIC loss improves:
- the organization of the latent space with respect to activity
- the predictive performance of a small MIC regressor trained on two-dimensional latent coordinates

3.  Without additional filtering, focused generation tends to produce extremely cationic motifs. Applying simple physicochemical filters improves biological plausibility, diversity, and the balance between charge and hydrophobicity.
