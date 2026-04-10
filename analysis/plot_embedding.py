import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
sys.path.append("../")
from config import CFG
from models.vae_model import SequenceVAE


model = SequenceVAE(
    vocab_size=21,
    max_len=CFG.max_len,
    latent_dim=CFG.latent_dim,
    hidden_dim=CFG.hidden_dim,
    dropout=CFG.dropout,
).to(CFG.device)

model.load_state_dict(torch.load("../ecoli_peptide_vae.pt", map_location=CFG.device))
model.eval()


embedding_weights = model.embed.weight.detach().cpu().numpy()

# shape: (vocab_size, embed_dim)
print("Embedding shape:", embedding_weights.shape)

AA_LIST = [
    "A","C","D","E","F","G","H","I","K","L",
    "M","N","P","Q","R","S","T","V","W","Y","<PAD>"
]
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embedding_weights)

df = pd.DataFrame({
    "aa": AA_LIST,
    "x": emb_2d[:, 0],
    "y": emb_2d[:, 1],
})

charge_dict = {"K": 1, "R": 1, "H": 0.5, "D": -1, "E": -1}
df["charge"] = df["aa"].map(charge_dict).fillna(0)

plt.figure(figsize=(6, 6))
#plt.scatter(df["x"], df["y"],s=200, c=df["charge"], cmap="coolwarm")
for _, row in df.iterrows():
    aa = row["aa"]
    if aa == "<PAD>":
        continue

    plt.scatter(row["x"], row["y"],s=200)
    plt.text(row["x"], row["y"], aa, fontsize=12)

plt.title("Amino acid embedding (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()

plt.savefig("aa_embedding_pca.png", dpi=150)
plt.show()
