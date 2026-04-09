from __future__ import annotations

import numpy as np
import torch


def extract_case_embedding(model: torch.nn.Module, x0: np.ndarray, device: str) -> np.ndarray:
    """
    Embedding from first hidden projection layer global pooled.
    x0 shape [2,H,W]
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x0[None, ...]).to(device)
        # FNO/PINO both expose input_proj in this codebase.
        z = model.input_proj(x)  # [1,W,H,W]
        emb = z.mean(dim=(-2, -1)).cpu().numpy()[0]
    return emb.astype(np.float32)


def ood_distance_scores(train_embeddings: np.ndarray, emb: np.ndarray) -> dict[str, float]:
    """
    Returns simple OOD scores:
      - distance to train centroid
      - nearest-neighbor distance
    """
    centroid = np.mean(train_embeddings, axis=0)
    d_centroid = float(np.linalg.norm(emb - centroid))
    dists = np.linalg.norm(train_embeddings - emb[None, :], axis=1)
    d_nn = float(np.min(dists))
    return {"ood_centroid_distance": d_centroid, "ood_nn_distance": d_nn}
