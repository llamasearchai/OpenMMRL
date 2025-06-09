from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F


def contrastive_loss_jax(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 0.07,
    reduction: str = "mean",
) -> jnp.ndarray:
    """
    Compute InfoNCE/NT-Xent loss for contrastive learning.
    
    Args:
        embeddings: Normalized embeddings of shape [batch_size, embedding_dim]
        labels: Labels for positive pairs, shape [batch_size]
        temperature: Temperature parameter for softmax
        reduction: Reduction method ('none', 'mean', 'sum')
    
    Returns:
        Contrastive loss
    """
    # Compute similarity matrix
    similarity = jnp.matmul(embeddings, embeddings.T) / temperature
    
    # Mask out self-similarity
    batch_size = similarity.shape[0]
    mask = jnp.eye(batch_size)
    similarity = similarity - mask * 1e9
    
    # Create positive pair mask
    # positive_mask[i,j] = 1 if embeddings[i] and embeddings[j] have same label
    positive_mask = labels[:, None] == labels[None, :]
    positive_mask = positive_mask.astype(jnp.int32) - mask  # Remove self
    
    # Calculate loss for each element
    exp_similarity = jnp.exp(similarity)
    
    # For each row, compute numerator (sum of exp similarities for positive pairs)
    positive_similarities = jnp.where(positive_mask, exp_similarity, 0.0)
    numerator = jnp.sum(positive_similarities, axis=1)
    
    # Denominator is sum of all exp similarities
    denominator = jnp.sum(exp_similarity, axis=1)
    
    # Compute loss
    losses = -jnp.log(numerator / denominator + 1e-8)
    
    # Apply reduction
    if reduction == "none":
        return losses
    elif reduction == "mean":
        return jnp.mean(losses)
    elif reduction == "sum":
        return jnp.sum(losses)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def hard_negative_mining_jax(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    k: int = 10,
    margin: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Mine hard negative examples for improved contrastive learning.
    
    Hard negatives are embeddings with different labels that are close
    in the embedding space.
    
    Args:
        embeddings: Normalized embeddings of shape [batch_size, embedding_dim]
        labels: Labels for positive pairs, shape [batch_size]
        k: Number of hard negatives to mine per example
        margin: Margin for semi-hard negatives
    
    Returns:
        Hard negative mask and weights
    """
    # Compute similarity matrix
    similarity = jnp.matmul(embeddings, embeddings.T)
    
    # Create negative mask (embedding pairs with different labels)
    negative_mask = (labels[:, None] != labels[None, :]).astype(jnp.float32)
    
    # For each anchor, find the k hardest negatives
    # (highest similarity among negatives)
    similarities_with_negatives = similarity * negative_mask - (1 - negative_mask) * 1e9
    
    # Get top-k hard negatives
    hard_negative_similarities, hard_negative_indices = jax.lax.top_k(
        similarities_with_negatives, k
    )
    
    # Create hard negative mask
    batch_size = embeddings.shape[0]
    hard_negative_mask = jnp.zeros((batch_size, batch_size))
    
    # Expensive sequential version
    # JAX doesn't support scattered assignment directly
    def update_row(i, mask):
        indices = hard_negative_indices[i]
        updates = jnp.ones_like(indices, dtype=jnp.float32)
        row_update = jnp.zeros((batch_size,), dtype=jnp.float32).at[indices].set(updates)
        return mask.at[i].set(row_update)
    
    hard_negative_mask = jax.lax.fori_loop(
        0, batch_size, update_row, hard_negative_mask
    )
    
    # Create weights for hard negatives based on similarity
    # Applies higher weight to harder negatives (more similar)
    hard_negative_weights = hard_negative_mask * (
        similarity * hard_negative_mask + margin
    )
    
    return hard_negative_mask, hard_negative_weights


def contrastive_loss_with_hard_negatives_jax(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 0.07,
    hard_negative_k: int = 10,
    hard_negative_weight: float = 2.0,
    reduction: str = "mean",
) -> jnp.ndarray:
    """
    Compute contrastive loss with hard negative mining.
    
    Args:
        embeddings: Normalized embeddings of shape [batch_size, embedding_dim]
        labels: Labels for positive pairs, shape [batch_size]
        temperature: Temperature parameter
        hard_negative_k: Number of hard negatives to mine per example
        hard_negative_weight: Weight factor for hard negatives
        reduction: Reduction method ('none', 'mean', 'sum')
    
    Returns:
        Enhanced contrastive loss
    """
    # Get base contrastive loss
    base_loss = contrastive_loss_jax(
        embeddings, labels, temperature, reduction="none"
    )
    
    # Mine hard negatives
    hard_negative_mask, hard_negative_weights = hard_negative_mining_jax(
        embeddings, labels, k=hard_negative_k
    )
    
    # Compute similarity matrix
    similarity = jnp.matmul(embeddings, embeddings.T) / temperature
    
    # Create negative pair mask
    negative_mask = (labels[:, None] != labels[None, :]).astype(jnp.float32)
    
    # Compute loss from hard negatives
    hard_negative_exp = jnp.exp(similarity) * hard_negative_mask * hard_negative_weight
    
    # Get positive mask (same as in regular contrastive loss)
    batch_size = similarity.shape[0]
    mask = jnp.eye(batch_size)
    positive_mask = (labels[:, None] == labels[None, :]).astype(jnp.int32) - mask
    positive_similarities = jnp.where(positive_mask, jnp.exp(similarity), 0.0)
    numerator = jnp.sum(positive_similarities, axis=1)
    
    # Include hard negatives in denominator with increased weight
    exp_similarity = jnp.exp(similarity)
    denominator = jnp.sum(exp_similarity - hard_negative_exp, axis=1) + jnp.sum(hard_negative_exp, axis=1)
    
    # Compute enhanced loss
    enhanced_losses = -jnp.log(numerator / denominator + 1e-8)
    
    # Combine base loss with enhanced loss
    losses = base_loss + enhanced_losses
    
    # Apply reduction
    if reduction == "none":
        return losses
    elif reduction == "mean":
        return jnp.mean(losses)
    elif reduction == "sum":
        return jnp.sum(losses)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


# PyTorch versions for training with DeepSpeed/FSDP

def contrastive_loss_torch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    reduction: str = "mean",
) -> torch.Tensor:
    """PyTorch implementation of contrastive loss."""
    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Mask out self-similarity
    batch_size = similarity.shape[0]
    mask = torch.eye(batch_size, device=embeddings.device)
    similarity = similarity - mask * 1e9
    
    # Create positive pair mask
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() - mask
    
    # Calculate loss for each element
    exp_similarity = torch.exp(similarity)
    
    # For each row, compute numerator (sum of exp similarities for positive pairs)
    positive_similarities = torch.where(positive_mask > 0, exp_similarity, torch.zeros_like(exp_similarity))
    numerator = torch.sum(positive_similarities, dim=1)
    
    # Denominator is sum of all exp similarities
    denominator = torch.sum(exp_similarity, dim=1)
    
    # Compute loss
    losses = -torch.log(numerator / denominator + 1e-8)
    
    # Apply reduction
    if reduction == "none":
        return losses
    elif reduction == "mean":
        return torch.mean(losses)
    elif reduction == "sum":
        return torch.sum(losses)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def hard_negative_mining_torch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
    margin: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of hard negative mining."""
    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T)
    
    # Create negative mask (embedding pairs with different labels)
    negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    
    # For each anchor, find the k hardest negatives
    similarities_with_negatives = similarity * negative_mask - (1 - negative_mask) * 1e9
    
    # Get top-k hard negatives
    hard_negative_similarities, hard_negative_indices = torch.topk(
        similarities_with_negatives, k
    )
    
    # Create hard negative mask
    batch_size = embeddings.shape[0]
    hard_negative_mask = torch.zeros((batch_size, batch_size), device=embeddings.device)
    
    # Set hard negative mask
    for i in range(batch_size):
        hard_negative_mask[i, hard_negative_indices[i]] = 1.0
    
    # Create weights for hard negatives based on similarity
    hard_negative_weights = hard_negative_mask * (
        similarity * hard_negative_mask + margin
    )
    
    return hard_negative_mask, hard_negative_weights


def contrastive_loss_with_hard_negatives_torch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    hard_negative_k: int = 10,
    hard_negative_weight: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """PyTorch implementation of contrastive loss with hard negative mining."""
    # Get base contrastive loss
    base_loss = contrastive_loss_torch(
        embeddings, labels, temperature, reduction="none"
    )
    
    # Mine hard negatives
    hard_negative_mask, hard_negative_weights = hard_negative_mining_torch(
        embeddings, labels, k=hard_negative_k
    )
    
    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create negative pair mask
    negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    
    # Compute loss from hard negatives
    hard_negative_exp = torch.exp(similarity) * hard_negative_mask * hard_negative_weight
    
    # Get positive mask (same as in regular contrastive loss)
    batch_size = similarity.shape[0]
    mask = torch.eye(batch_size, device=embeddings.device)
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() - mask
    positive_similarities = torch.where(positive_mask > 0, torch.exp(similarity), torch.zeros_like(similarity))
    numerator = torch.sum(positive_similarities, dim=1)
    
    # Include hard negatives in denominator with increased weight
    exp_similarity = torch.exp(similarity)
    denominator = torch.sum(exp_similarity - hard_negative_exp, dim=1) + torch.sum(hard_negative_exp, dim=1)
    
    # Compute enhanced loss
    enhanced_losses = -torch.log(numerator / denominator + 1e-8)
    
    # Combine base loss with enhanced loss
    losses = base_loss + enhanced_losses
    
    # Apply reduction
    if reduction == "none":
        return losses
    elif reduction == "mean":
        return torch.mean(losses)
    elif reduction == "sum":
        return torch.sum(losses)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}") 