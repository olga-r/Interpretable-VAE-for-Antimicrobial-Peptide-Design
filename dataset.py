from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from config import CFG
from utils_general import encode_sequence
import numpy as np

class PeptideSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.max_len = CFG.max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        tokens, mask = encode_sequence(row["sequence"])
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "length": torch.tensor(len(row["sequence"]), dtype=torch.long),
             "log10_mic_uM": torch.tensor(row.get("log10_mic_uM", np.nan), dtype=torch.float32)
        }
