from typing import List
import numpy as np
import pandas as pd

from config import CFG, AA,  KD


def peptide_charge(seq: str) -> float:
    positive = sum(seq.count(x) for x in ["K", "R"])
    negative = sum(seq.count(x) for x in ["D", "E"])
    return positive - negative

def normalized_charge(seq: str) -> float:
    return peptide_charge(seq) / max(len(seq), 1)


def hydrophobicity_kd(seq: str) -> float:

    vals = [KD[a] for a in seq if a in KD]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))

def sequence_quality_filter(seq: str, min_len: int = 6, max_len: int = 30) -> bool:
    if len(seq) < min_len or len(seq) > max_len:
        return False
    if any(a not in AA for a in seq):
        return False
    return True

def build_generated_table(seqs: List[str]) -> pd.DataFrame:
    rows = []
    for s in seqs:
        rows.append({ "sequence": s, "length": len(s), "normalized_charge": normalized_charge(s), "hydrophobicity_kd": hydrophobicity_kd(s), "passes_basic_filter": sequence_quality_filter(s), })
    df = pd.DataFrame(rows)
    df = df[df["passes_basic_filter"]].copy()
    df = df.drop_duplicates(subset=["sequence"])
    return df


def no_long_homopolymer(seq: str, max_run: int = 3) -> bool:
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            if run > max_run:
                return False
        else:
            run = 1
    return True

def aa_fraction(seq: str, aa_set: set) -> float:
    if len(seq) == 0:
        return 0.0
    return sum(1 for a in seq if a in aa_set) / len(seq)

def basic_candidate_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out[out["length"].between(8, 30)].copy()

    out = out[out["normalized_charge"] >= 0.10].copy()

    out["frac_basic"] = out["sequence"].apply(lambda s: aa_fraction(s, {"K", "R", "H"}))
    out["frac_hydrophobic"] = out["sequence"].apply(
        lambda s: aa_fraction(s, {"A", "V", "I", "L", "M", "F", "W", "Y"})
    )

    # less extreme composition
    out = out[out["frac_basic"] <= 0.50].copy()
    out = out[out["frac_hydrophobic"].between(0.20, 0.60)].copy()

    # moderate KD hydrophobicity
    out = out[out["hydrophobicity_kd"].between(-1.0, 1.5)].copy()

    out = out[out["sequence"].apply(no_long_homopolymer)].copy()

    return out




def minmax_scale(series: pd.Series) -> pd.Series:
    smin = series.min()
    smax = series.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - smin) / (smax - smin)



def simple_sequence_identity(seq1: str, seq2: str) -> float:
    n = min(len(seq1), len(seq2))
    if n == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1[:n], seq2[:n]))
    return matches / n



def select_diverse_candidates(
    df: pd.DataFrame,
    top_n: int = 20,
    max_identity: float = 0.80,
) -> pd.DataFrame:
    ranked = df.sort_values("pred_log10_mic_uM", ascending=True).copy()

    selected_rows = []
    selected_seqs = []

    for _, row in ranked.iterrows():
        seq = row["sequence"]

        too_similar = False
        for prev_seq in selected_seqs:
            if simple_sequence_identity(seq, prev_seq) >= max_identity:
                too_similar = True
                break

        if not too_similar:
            selected_rows.append(row)
            selected_seqs.append(seq)

        if len(selected_rows) >= top_n:
            break

    if len(selected_rows) == 0:
        return ranked.head(0).copy()

    return pd.DataFrame(selected_rows).reset_index(drop=True)

def max_identity_to_training(seq: str, training_sequences: List[str]) -> float:
    if len(training_sequences) == 0:
        return 0.0
    return max(simple_sequence_identity(seq, t) for t in training_sequences)


def rank_generated_candidates(
    df: pd.DataFrame,
    training_sequences=None,
    top_n: int = 20) -> pd.DataFrame:
    """
    Simple final ranking pipeline:

    1. physicochemical filtering
    2. sort by predicted MIC
    3. enforce diversity
    4. optionally annotate similarity to training set
    """
    out = basic_candidate_filter(df).copy()

    if len(out) == 0:
        return out

    out = out.sort_values("pred_log10_mic_uM", ascending=True).copy()
    out = select_diverse_candidates(out, top_n=top_n, max_identity=CFG.max_identity)

    if training_sequences is not None and len(out) > 0:
        out["max_identity_to_train"] = out["sequence"].apply(
            lambda s: max_identity_to_training(s, training_sequences)
        )

    return out

