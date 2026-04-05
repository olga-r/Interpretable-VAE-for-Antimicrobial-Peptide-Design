# Interpretable-VAE-for-Antimicrobial-Peptide-Design
 This project explores antimicrobial peptide (AMP) generation using a **2D variational autoencoder (VAE)** trained on peptide sequences with experimental MIC values against *Escherichia coli*.
The main goal is not only to generate peptide-like sequences, but also to obtain an **interpretable 2D latent space** that can be visualized and related to biologically meaningful properties such as activity, charge, and hydrophobicity.

---

## Project overview

The pipeline contains four main parts:

1. **Sequence VAE**
   - learns a 2D latent representation of AMP sequences
   - supports variable sequence length through a padding-aware reconstruction loss

2. **Latent space analysis**
   - visualizes latent space colored by:
     - `log10(MIC)`
     - normalized charge
     - hydrophobicity
   - identifies activity-enriched regions in latent space

3. **MIC-aware training**
   - adds an auxiliary MIC prediction loss to the VAE
   - makes the 2D latent space more informative for peptide activity
   - improves downstream MIC prediction from latent coordinates

4. **Candidate generation and ranking**
   - prior sampling from the full latent space
   - focused sampling from the most active latent region
   - post hoc filtering by physicochemical heuristics
   - diversity filtering
   - annotation of similarity to training peptides

---

## Main results

### 1. Interpretable 2D latent space
The VAE learns a latent space that captures meaningful biological structure:
- peptides with different charge and hydrophobicity occupy different regions
- lower MIC values are enriched in specific latent regions

### 2. MIC-aware VAE improves activity relevance
A purely unsupervised VAE gives a useful generative latent space, but MIC information is only weakly preserved.

Adding an auxiliary MIC loss improves:
- organization of the latent space with respect to activity
- predictive performance of a small MIC regressor trained on 2D latent coordinates

### 3. ESM (not included in the repository) is better for prediction, VAE is better for interpretation
Frozen ESM embeddings predict MIC substantially better than the 2D VAE latent space.


### 4. Post hoc filtering is important
Without additional filtering, focused generation tends to drift toward overly extreme cationic motifs.

Applying simple physicochemical filters improves:
- biological plausibility
- diversity
- balance between charge and hydrophobicity

---

## Repository structure

```text
.
├── config.py
├── process_dbaasp.py
├── dataset.py
├── utils_general.py
├── utils_model.py
├── utils_sample_latent_space.py
├── utils_filter_candidates.py
├── utils_plots.py
├── train_vae.py
├── generate_sequences.py
├── models/
├   ├── __init__.py
├   ├── vae_model.py
├   ├── mic_model.py
├   └── mic_train.py
└── example/
      ├── dbaasp_full_ecoli.csv
      ├── ecoli_peptide_vae_latent_charge.png
      ├── ecoli_peptide_vae_latent_hydrophobicity.png
      ├── ecoli_peptide_vae_latent_mic.png
      ├── ecoli_peptide_vae_active_region.png
      ├── ecoli_peptide_vae_MIC_true_predict.png
      ├── ecoli_peptide_vae_top_focus_candidates.csv
      └── ecoli_peptide_vae_top_gen_candidates.csv
