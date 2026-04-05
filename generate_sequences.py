import numpy as np
import pandas as pd
import torch
from config import CFG, VOCAB_SIZE
from utils_general import set_seed, build_sequence_level_table
from utils_sample_latent_space import find_active_region_center, sample_around_center, get_top_active_points
from utils_plots import plot_active_region
from utils_model import sample_sequences, decode_from_latent_points, encode_dataframe
from utils_filter_candidates import build_generated_table, rank_generated_candidates

from models.vae_model import SequenceVAE
from models.mic_model import MICPredictor
from models.mic_train import train_mic_predictor, score_generated_sequences


set_seed(CFG.random_seed)
latent_df = pd.read_csv(CFG.plot_prefix + "_latent.csv")
df = pd.read_csv(CFG.csv_path)
seq_df = build_sequence_level_table(df)
training_sequences = seq_df["sequence"].tolist()

vae_model = SequenceVAE(
    vocab_size=VOCAB_SIZE,
    max_len=CFG.max_len,
    latent_dim=CFG.latent_dim,
    hidden_dim=CFG.hidden_dim,
    embed_dim=16,
    dropout = CFG.dropout
).to(CFG.device)

vae_model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))

mic_model, best_val_mse = train_mic_predictor(latent_df, epochs=400, lr=1e-3)
print(f"Best MIC predictor val MSE: {best_val_mse:.4f}")



#####################################################
# Sample from the most active region (Focus region)
#####################################################
active_center = find_active_region_center(latent_df, activity_col="log10_mic_uM")
print(f"Active-region center: z1={active_center[0]:.3f}, z2={active_center[1]:.3f}")

top_active_df = get_top_active_points(latent_df, activity_col="log10_mic_uM")
top_active_df.to_csv(CFG.plot_prefix + "_top_active_latent_points.csv", index=False)
plot_active_region(latent_df, top_active_df, active_center,CFG.plot_prefix + "_active_region.png")

focus_points = sample_around_center(active_center, n_samples=150)
focus_seqs = decode_from_latent_points(vae_model, focus_points, device=CFG.device)
df_focus = build_generated_table(focus_seqs)

if len(df_focus) > 0:
    tmp_focus = df_focus.copy()
    tmp_focus["log10_mic_uM"] = np.nan
    tmp_focus["Normalized charge"] = tmp_focus["normalized_charge"]
    tmp_focus["Normalized Hydrophobicity"] = tmp_focus["hydrophobicity_kd"]
    tmp_focus["has_pdb_structure"] = 0

    tmp_focus_lat = encode_dataframe(vae_model, tmp_focus)
    df_focus = pd.concat([df_focus.reset_index(drop=True), tmp_focus_lat[["z1", "z2"]].reset_index(drop=True)], axis=1)

    df_focus = score_generated_sequences(mic_model, df_focus)
    df_focus.to_csv(CFG.plot_prefix + "_generated_focus_sequences.csv", index=False)
    top_focus = rank_generated_candidates( df_focus, training_sequences=training_sequences,
    top_n=20)

    top_focus.to_csv(CFG.plot_prefix + "_top_focus_candidates.csv", index=False)

    print("\nTop diverse AMP-like candidates from active region:")
    print(
    top_focus[
        [
            "sequence",
            "length",
            "normalized_charge",
            "hydrophobicity_kd",
            "pred_log10_mic_uM",
            "max_identity_to_train",
        ]
    ].to_string(index=False)
    )
####################################
# Sample from the whole latent prior
####################################
samples = sample_sequences(vae_model, n_samples=200, device=CFG.device)
df_gen = build_generated_table(samples)
if len(df_gen) > 0:
    tmp = df_gen.copy()
    tmp["log10_mic_uM"] = np.nan
    tmp["Normalized charge"] = tmp["normalized_charge"]
    tmp["Normalized Hydrophobicity"] = tmp["hydrophobicity_kd"]
    tmp["has_pdb_structure"] = 0
    tmp_lat = encode_dataframe(vae_model, tmp)
    df_gen = pd.concat([df_gen.reset_index(drop=True), tmp_lat[["z1", "z2"]].reset_index(drop=True)], axis=1)
    df_gen = score_generated_sequences(mic_model, df_gen)
    df_gen.to_csv(CFG.plot_prefix + "_generated_prior_sequences.csv", index=False)

    top_gen = rank_generated_candidates(
        df_gen,
        training_sequences=training_sequences,
        top_n=20)

    top_gen.to_csv(CFG.plot_prefix + "_top_gen_candidates.csv", index=False)

    print("\nTop diverse AMP-like candidates from prior sampling:")
    print(
        top_gen[
            [
            "sequence",
            "length",
            "normalized_charge",
            "hydrophobicity_kd",
            "pred_log10_mic_uM",
            "max_identity_to_train",
            ]
        ].to_string(index=False)
    )




