import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from config import CFG
from utils_plots import plot_true_vs_predict
from utils_general import  make_regression_arrays
from models.mic_model import MICPredictor


def train_mic_predictor(df_latent: pd.DataFrame,  epochs: int = 200, lr: float = 1e-3):
    X, y, used_df = make_regression_arrays(df_latent)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=CFG.random_seed)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=CFG.device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=CFG.device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=CFG.device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=CFG.device)

    model = MICPredictor(input_dim=2, hidden_dim=32, dropout=0.2).to(CFG.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 20
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"MIC predictor epoch {epoch:03d} | train_mse={loss.item():.4f} | val_mse={val_loss:.4f}")

        if counter >= patience:
            print(f"MIC predictor early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).cpu().numpy()
    plot_true_vs_predict(y_val.cpu().numpy(), val_pred, "MIC_true_vs_predict",  CFG.plot_prefix + "_MIC_true_predict.png")
    return model, best_val

@torch.no_grad()
def score_generated_sequences(mic_model: MICPredictor, df_gen: pd.DataFrame) -> pd.DataFrame:
    X = torch.tensor(df_gen[["z1", "z2"]].values.astype(np.float32), dtype=torch.float32, device=CFG.device)
    pred = mic_model(X).cpu().numpy()
    out = df_gen.copy()
    out["pred_log10_mic_uM"] = pred
    return out.sort_values("pred_log10_mic_uM", ascending=True)



