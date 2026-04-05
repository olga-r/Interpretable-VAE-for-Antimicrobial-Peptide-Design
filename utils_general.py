import math
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from config import CFG, AA, AA_TO_IDX, PAD_IDX, KD


#=========================
#1. Encode sequences
#=========================

def valid_sequence(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    if len(seq) < CFG.min_len or len(seq) > CFG.max_len:
        return False
    return all(aa in AA_TO_IDX for aa in seq)

def encode_sequence(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns
    -------
    tokens : shape (max_len,)
    mask   : shape (max_len,) with 1 for real positions and 0 for padding
    """
    seq = seq.strip().upper()
    tokens = np.full(CFG.max_len, PAD_IDX, dtype=np.int64)
    mask = np.zeros(CFG.max_len, dtype=np.float32)
    for i, aa in enumerate(seq[:CFG.max_len]):
        tokens[i] = AA_TO_IDX[aa]
        mask[i] = 1.0
    return tokens, mask

def decode_tokens(tokens: np.ndarray) -> str:
    chars = []
    for tok in tokens:
        if tok == PAD_IDX:
            break
        chars.append(AA[tok])
    return "".join(chars)


#=========================
# 2.Aggregate database
#=========================
def simplify_bond_class(x: str) -> str:
    if pd.isna(x):
        return "none"
    s = str(x).strip()
    if s == "" or s == "[]":
        return "none"
    # already simplified in the user's CSV most of the time
    if not s.startswith("["):
        return s
    try:
        items = json.loads(s)
    except Exception:
        return s
    items = sorted(set(str(i).strip() for i in items if str(i).strip()))
    if len(items) == 0:
        return "none"
    if len(items) == 1:
        return items[0]
    return "mixed"

def build_sequence_level_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df = df[df["sequence"].apply(lambda s: valid_sequence(s))].copy()
    if "bond_class" in df.columns:
        df["bond_class_simple"] = df["bond_class"].apply(simplify_bond_class)
    else:
        df["bond_class_simple"] = "none"

    # For the first pass keep only biologically simpler bond classes.
    allowed_bonds = {"none", "disulfide", "amide_cycle"}
    df = df[df["bond_class_simple"].isin(allowed_bonds)].copy()

    physchem_cols = [
        "Normalized Hydrophobic Moment",
        "Normalized Hydrophobicity",
        "Net Charge",
        "Normalized charge",
        "Isoelectric Point",
        "Penetration Depth",
        "Tilt Angle",
        "Disordered Conformation Propensity",
        "Linear Moment",
        "Propensity to in vitro Aggregation",
        "Angle Subtended by the Hydrophobic Residues",
        "Amphiphilicity Index",
        "Propensity to PPII coil",
        "helix_propensity_normalized",
    ]
    physchem_cols = [c for c in physchem_cols if c in df.columns]

    def agg_binary(series: pd.Series):
        vals = sorted(set(v for v in series.dropna().tolist()))
        if len(vals) == 0:
            return np.nan
        if len(vals) == 1:
            return vals[0]
        return np.nan

    agg = {
        "sequenceLength": "first",
        "log10_mic_uM": "median",
        "mic_mean_uM": "median",
        "n_generic_records": "max",
        "n_total_records": "max",
        "has_strain_specific": "max",
        "strong_active_25": agg_binary,
        "active_100": agg_binary,
        "bond_class_simple": lambda s: s.mode().iloc[0] if not s.mode().empty else "none",
        "pdb": lambda s: int(s.notna().any()),
    }
    for c in physchem_cols:
        agg[c] = "median"

    seq_df = df.groupby("sequence", as_index=False).agg(agg)
    seq_df = seq_df.rename(columns={
        "mic_mean_uM": "mic_generic_median_uM",
        "pdb": "has_pdb_structure",
    })
    seq_df["length"] = seq_df["sequence"].str.len()
    return seq_df



#=========================
# 3.Miscellaneous
#=========================

def make_regression_arrays(df_latent: pd.DataFrame):
    use = df_latent[df_latent["log10_mic_uM"].notna()].copy()
    X = use[["z1", "z2"]].values.astype(np.float32)
    y = use["log10_mic_uM"].values.astype(np.float32)
    return X, y, use



def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
