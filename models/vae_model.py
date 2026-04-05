import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


class SequenceVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        latent_dim: int = 2,
        hidden_dim: int = 128,
        embed_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.input_dim = max_len * embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_len * vocab_size),
        )

        # MIC prediction head from latent mean
        self.mic_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode(self, tokens: torch.Tensor):
        x = self.embed(tokens)                # (B, L, E)
        x = x.reshape(x.size(0), -1)         # (B, L*E)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        logits = self.decoder(z)
        logits = logits.view(-1, self.max_len, self.vocab_size)
        return logits

    def forward(self, tokens: torch.Tensor):
        mu, logvar = self.encode(tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        # use mu, not sampled z, for more stable MIC supervision
        mic_pred = self.mic_head(mu).squeeze(-1)

        return logits, mu, logvar, z, mic_pred



def vae_loss(
    logits,
    tokens,
    mask,
    mu,
    logvar,
    mic_pred=None,
    mic_true=None,
):
    """
    Reconstruction + KL + optional MIC supervision.

    - real positions have full weight
    - padding positions have smaller weight
    - MIC loss is used only where mic_true is available
    """
    ce = F.cross_entropy(
        logits.transpose(1, 2),
        tokens,
        reduction="none",
    )

    weights = mask + CFG.pad_weight * (1.0 - mask)
    rec = (ce * weights).sum() / (weights.sum() + 1e-8)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = rec + CFG.beta * kl

    mic_loss = torch.tensor(0.0, device=logits.device)

    if mic_pred is not None and mic_true is not None:
        valid = ~torch.isnan(mic_true)
        if valid.any():
            mic_loss = F.mse_loss(mic_pred[valid], mic_true[valid])
            loss = loss + CFG.mic_weight * mic_loss

    return loss, rec, kl, mic_loss
