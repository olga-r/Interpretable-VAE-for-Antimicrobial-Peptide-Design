import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_latent(df_latent: pd.DataFrame, color_col: str, title: str, out_path: str):
    plt.figure(figsize=(6, 5))
    valid = df_latent[color_col].notna()
    plt.scatter(
        df_latent.loc[valid, "z1"],
        df_latent.loc[valid, "z2"],
        c=df_latent.loc[valid, color_col],
        s=2,
        alpha=0.8,
    )
    plt.colorbar(label=color_col)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_structure_overlay(df_latent: pd.DataFrame, out_path: str):
    plt.figure(figsize=(6, 5))
    no_pdb = df_latent["has_pdb_structure"] == 0
    has_pdb = df_latent["has_pdb_structure"] == 1
    plt.scatter(df_latent.loc[no_pdb, "z1"], df_latent.loc[no_pdb, "z2"], s=2, alpha=0.35, label="no PDB")
    plt.scatter(df_latent.loc[has_pdb, "z1"], df_latent.loc[has_pdb, "z2"], s=2, alpha=0.9, label="has PDB")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space with structure subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_strong_active(df_latent: pd.DataFrame, out_path: str):
    plt.figure(figsize=(6, 5))
    inactive = df_latent["strong_active_25"] == 0
    active = df_latent["strong_active_25"] == 1
    missed = df_latent["strong_active_25"].isna()

    plt.scatter(df_latent.loc[missed, "z1"], df_latent.loc[missed, "z2"], s=2, alpha=0.35, label="missed")
    plt.scatter(df_latent.loc[inactive, "z1"], df_latent.loc[inactive, "z2"], s=2, alpha=0.9, label="MIC>25")
    plt.scatter(df_latent.loc[active, "z1"], df_latent.loc[active, "z2"], s=2, alpha=0.9, label="MIC<25")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space with strong active peptides")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_true_vs_predict(y_true, y_predict, title, out_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(
        y_true,
        y_predict,
        s=2,
        alpha=0.8,
    )
    plt.xlabel("true")
    plt.ylabel("predict")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_active_region(
    df_latent: pd.DataFrame,
    top_active_df: pd.DataFrame,
    center: np.ndarray,
    out_path: str,):

    plt.figure(figsize=(6, 5))
    plt.scatter(df_latent["z1"], df_latent["z2"], s=2, alpha=0.25, color="lightgray", label="all")
    plt.scatter(
        top_active_df["z1"],
        top_active_df["z2"],
        s=5,
        alpha=0.9,
        label="top active",
    )
    plt.scatter(
        [center[0]],
        [center[1]],
        s=20,
        marker="X",
        label="active center",
    )

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Most active latent region")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

