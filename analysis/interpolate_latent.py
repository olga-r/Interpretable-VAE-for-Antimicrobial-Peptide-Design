import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("../")
from config import CFG,AA_TO_IDX, PAD_IDX, AA, VOCAB_SIZE
from models.vae_model import SequenceVAE
from utils_general import decode_tokens, encode_sequence
from utils_filter_candidates import  build_generated_table, hydrophobicity_kd, normalized_charge
from utils_model import encode_dataframe, decode_from_latent_points
import matplotlib.pyplot as plt


@torch.no_grad()
def encode_single_sequence(model, seq: str, device: str):
    tokens, mask = encode_sequence(seq)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    tokens = tokens.to(device)
    mu, logvar = model.encode(tokens)
    return mu.squeeze(0).cpu().numpy()



def interpolate(z1, z2, n_steps=9):
    ts = np.linspace(0.0, 1.0, n_steps)
    zs = np.array([(1 - t) * z1 + t * z2 for t in ts], dtype=np.float32)
    return ts, zs

def plot_interpolation_on_latent(
    z_all,
    z_a,
    z_b,
    z_interp,
    seqs_interp,
    ts,
    save_path="latent_interpolation.png"
):

    plt.figure(figsize=(6, 6))

    plt.scatter(
        z_all[:, 0],
        z_all[:, 1],
        s=10,
        alpha=0.3,
        label="dataset"
    )

    plt.plot(
        z_interp[:, 0],
        z_interp[:, 1],
        '-o',
        color='black',
        label="interpolation path"
    )

    plt.scatter(z_a[0], z_a[1], color='red', s=80)
    #plt.text(z_a[0], z_a[1], "A", fontsize=12)

    plt.scatter(z_b[0], z_b[1], color='blue', s=80)
    #plt.text(z_b[0], z_b[1], "B", fontsize=12)

    for i, (z, t, seq) in enumerate(zip(z_interp, ts, seqs_interp)):
        if i in [0, len(ts)//2, len(ts)-1]:
            short = seq[:30] + "..." if len(seq) > 30 else seq
            plt.text(z[0], z[1], short, fontsize=8)



    #for i, (z, t) in enumerate(zip(z_interp, ts)):
    #    if i in [0, len(ts)//2, len(ts)-1]:
    #        plt.text(z[0], z[1], f"{t:.2f}", fontsize=9)

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space interpolation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.show()



def main():
    # load model
    model = SequenceVAE(
        vocab_size=VOCAB_SIZE,
        max_len=CFG.max_len,
        latent_dim=CFG.latent_dim,
        hidden_dim=CFG.hidden_dim,
        dropout=CFG.dropout,
    ).to(CFG.device)

    model.load_state_dict(torch.load("../ecoli_peptide_vae.pt", map_location=CFG.device))
    model.eval()

    # choose two sequences manually
    seq_a = "GRPLILRRIRKLVKKLFRPILRPI"
    seq_b = "GIGKFLKKFTKKVKKIFGLILGVIS"

    z_a = encode_single_sequence(model, seq_a, CFG.device)
    #z_b = encode_single_sequence(model, seq_b, CFG.device)

    # choose z_b manually
    z_b = np.array([1.2, -1.0])

    ts, z_interp = interpolate(z_a, z_b, n_steps=11)
    seqs_interp = decode_from_latent_points(model, z_interp, CFG.device)
    seqs_b = decode_from_latent_points(model, z_b, CFG.device)
    print(seqs_b)
    rows = []

    for t, z, s in zip(ts, z_interp, seqs_interp):
        rows.append({
            "t": round(float(t), 2),
            "z1": float(z[0]),
            "z2": float(z[1]),
            "sequence": s,
            "length": len(s),
            "normalized_charge": normalized_charge(s),
            "hydrophobicity_kd": hydrophobicity_kd(s),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(CFG.plot_prefix + "_latent_interpolation.csv", index=False)

    df_z = pd.read_csv('../ecoli_peptide_vae_latent.csv')
    z_all = df_z[["z1", "z2"]].values
    plot_interpolation_on_latent(
        z_all=z_all,
        z_a=z_a,
        z_b=z_b,
        z_interp=z_interp,
        seqs_interp=seqs_interp,
        ts=ts,
        save_path="latent_interpolation.png"
        )

if __name__ == "__main__":
    main()



