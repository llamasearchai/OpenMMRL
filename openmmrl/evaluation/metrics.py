import numpy as np
import torch
from scipy.stats import spearmanr

def accuracy(preds, labels):
    """
    Computes accuracy for classification tasks.
    """
    if isinstance(preds, torch.Tensor):
        if preds.dim() > 1 and preds.shape[-1] > 1:
            preds = torch.argmax(preds, dim=-1)
        return (preds == labels).float().mean().item()
    else:
        if preds.ndim > 1 and preds.shape[-1] > 1:
            preds = np.argmax(preds, axis=-1)
        return (preds == labels).mean()

def correlation(preds, labels):
    """
    Computes Spearman correlation for similarity tasks.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    # Assuming predictions and labels are embeddings
    if preds.ndim > 1 and labels.ndim > 1:
        preds = np.linalg.norm(preds, axis=1)
        labels = np.linalg.norm(labels, axis=1)

    corr, _ = spearmanr(preds, labels)
    return corr 