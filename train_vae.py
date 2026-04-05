import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from config import CFG, VOCAB_SIZE
from utils_general import set_seed, build_sequence_level_table
from utils_plots import plot_latent, plot_structure_overlay, plot_strong_active
from utils_model import encode_dataframe
from dataset import PeptideSequenceDataset
from models.vae_model import SequenceVAE, vae_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0
    total_mic = 0.0
    n_batches = 0

    for batch in loader:
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        mic_true = batch["log10_mic_uM"].to(device)

        logits, mu, logvar, z, mic_pred = model(tokens)

        loss, rec, kl, mic_loss = vae_loss(
            logits,
            tokens,
            mask,
            mu,
            logvar,
            mic_pred=mic_pred,
            mic_true=mic_true,
        )

        total_loss += loss.item()
        total_rec += rec.item()
        total_kl += kl.item()
        total_mic += mic_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "rec": total_rec / max(n_batches, 1),
        "kl": total_kl / max(n_batches, 1),
        "mic": total_mic / max(n_batches, 1),
    }


def train_model(model, train_loader, val_loader, patience: int = 10):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )

    best_val = float("inf")
    best_epoch = 0
    counter = 0
    history = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        total_mic = 0.0
        n_batches = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(CFG.device)
            mask = batch["mask"].to(CFG.device)
            mic_true = batch["log10_mic_uM"].to(CFG.device)

            optimizer.zero_grad()

            logits, mu, logvar, z, mic_pred = model(tokens)

            loss, rec, kl, mic_loss = vae_loss(
                logits,
                tokens,
                mask,
                mu,
                logvar,
                mic_pred=mic_pred,
                mic_true=mic_true
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()
            total_mic += mic_loss.item()
            n_batches += 1

        train_metrics = {
            "loss": total_loss / max(n_batches, 1),
            "rec": total_rec / max(n_batches, 1),
            "kl": total_kl / max(n_batches, 1),
            "mic": total_mic / max(n_batches, 1),
        }

        val_metrics = evaluate(model, val_loader, CFG.device)
        history.append((epoch, train_metrics, val_metrics))

        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), CFG.model_path)
        else:
            counter += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} rec={train_metrics['rec']:.4f} "
            f"kl={train_metrics['kl']:.4f} mic={train_metrics['mic']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} rec={val_metrics['rec']:.4f} "
            f"kl={val_metrics['kl']:.4f} mic={val_metrics['mic']:.4f} | "
            f"best_epoch={best_epoch} patience={counter}/{patience}"
        )

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    return history





def main():
    set_seed(CFG.random_seed)
    df = pd.read_csv(CFG.csv_path)
    seq_df = build_sequence_level_table(df)
    print("Sequence-level dataset shape:", seq_df.shape)
    print(seq_df[["sequence", "sequenceLength", "log10_mic_uM", "bond_class_simple", "has_pdb_structure"]].head())

    train_df, val_df = train_test_split(seq_df, test_size=0.2, random_state=CFG.random_seed)
    train_ds = PeptideSequenceDataset(train_df)
    val_ds = PeptideSequenceDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    model = SequenceVAE(
        vocab_size=VOCAB_SIZE,
        max_len=CFG.max_len,
        latent_dim=CFG.latent_dim,
        hidden_dim=CFG.hidden_dim,
        embed_dim=16,
        dropout = CFG.dropout
    ).to(CFG.device)

    _ = train_model(model, train_loader, val_loader, patience=10)
    model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))

    latent_df = encode_dataframe(model, seq_df)
    latent_csv = CFG.plot_prefix + "_latent.csv"
    latent_df.to_csv(latent_csv, index=False)
    print(f"Saved latent table to: {latent_csv}")

    common_mask = (
        latent_df["log10_mic_uM"].notna() &
        latent_df["Normalized charge"].notna() &
        latent_df["Normalized Hydrophobicity"].notna()
    )

    plot_df = latent_df[common_mask].copy()

    plot_latent(plot_df, "log10_mic_uM", "Latent space colored by log10(MIC)", CFG.plot_prefix + "_latent_mic.png")
    plot_latent(plot_df, "Normalized charge", "Latent space colored by normalized charge", CFG.plot_prefix + "_latent_charge.png")
    plot_latent(plot_df, "Normalized Hydrophobicity", "Latent space colored by hydrophobicity", CFG.plot_prefix + "_latent_hydrophobicity.png")
    plot_structure_overlay(plot_df, CFG.plot_prefix + "_latent_structure.png")
    plot_strong_active(plot_df, CFG.plot_prefix + "_strong_active.png")




if __name__ == "__main__":
    main()
