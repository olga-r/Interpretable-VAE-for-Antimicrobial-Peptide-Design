import math
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataset import PeptideSequenceDataset
from config import CFG, AA, AA_TO_IDX, PAD_IDX
from torch.utils.data import DataLoader

from utils_general import  decode_tokens


@torch.no_grad()
def sample_sequences(model, n_samples: int = 10, device: str = "cpu") -> List[str]:
    model.eval()
    z = torch.randn(n_samples, CFG.latent_dim, device=device)
    logits = model.decode(z)
    probs = torch.softmax(logits, dim=-1)
    tokens = torch.argmax(probs, dim=-1).cpu().numpy()
    seqs = [decode_tokens(tok) for tok in tokens]
    return seqs

@torch.no_grad()
def decode_from_latent_points(model, z_points: np.ndarray, device: str = "cpu") -> List[str]:
    model.eval()
    z = torch.tensor(z_points, dtype=torch.float32, device=device)
    logits = model.decode(z)
    probs = torch.softmax(logits, dim=-1)
    tokens = torch.argmax(probs, dim=-1).cpu().numpy()
    return [decode_tokens(tok) for tok in tokens]

@torch.no_grad()
def encode_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    ds = PeptideSequenceDataset(df)
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False)
    model.eval()

    mus = []
    for batch in loader:
        tokens = batch["tokens"].to(CFG.device)
        mu, logvar = model.encode(tokens)
        mus.append(mu.cpu().numpy())

    mus = np.vstack(mus)
    out = df.copy().reset_index(drop=True)
    out["z1"] = mus[:, 0]
    if CFG.latent_dim > 1:
        out["z2"] = mus[:, 1]
    return out

