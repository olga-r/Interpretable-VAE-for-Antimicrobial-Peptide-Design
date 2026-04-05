from dataclasses import dataclass
import torch

@dataclass
class Config:
    csv_path: str = "./dbaasp_full_ecoli.csv"
    random_seed: int = 42
    batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    latent_dim: int = 2
    hidden_dim: int = 128
    dropout: float = 0.2
    beta: float = 0.1
    max_len: int = 30
    min_len: int = 4
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path: str = "./ecoli_peptide_vae.pt"
    plot_prefix: str = "./ecoli_peptide_vae"
    frac: float = 0.1
    sigma: float = 0.2
    max_identity: float = 0.80
    pad_weight: float = 0.3
    mic_weight: float = 0.2

CFG = Config()
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
PAD_IDX = len(AA)
VOCAB_SIZE = len(AA) + 1

KD = { "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3, }
