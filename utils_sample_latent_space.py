import numpy as np
import pandas as pd

from config import CFG

def find_active_region_center(
    df_latent: pd.DataFrame,
    activity_col: str = "log10_mic_uM") -> np.ndarray:
    """
    Find the center of the most active region in latent space.

    We define 'most active' as the lowest values of log10(MIC).
    frac=0.10 means top 10% most active peptides.
    """
    use = df_latent[df_latent[activity_col].notna()].copy()
    if len(use) == 0:
        raise ValueError(f"No non-NaN values found in {activity_col}")

    n_top = max(5, int(len(use) * CFG.frac))
    top = use.nsmallest(n_top, activity_col)

    center = top[["z1", "z2"]].mean().values.astype(np.float32)
    return center


def sample_around_center(
    center: np.ndarray,
    n_samples: int = 100,
    ) -> np.ndarray:
    """
    Sample latent points around a given center.
    """
    center = np.asarray(center, dtype=np.float32).reshape(1, 2)
    noise = np.random.normal(loc=0.0, scale=CFG.sigma, size=(n_samples, 2)).astype(np.float32)
    return center + noise


def get_top_active_points(
    df_latent: pd.DataFrame,
    activity_col: str = "log10_mic_uM",
    ) -> pd.DataFrame:
    use = df_latent[df_latent[activity_col].notna()].copy()
    n_top = max(5, int(len(use) * CFG.frac))
    return use.nsmallest(n_top, activity_col).copy()

