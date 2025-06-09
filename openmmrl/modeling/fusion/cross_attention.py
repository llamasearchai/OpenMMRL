from typing import Dict, List, Optional, Tuple, Union

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = object

if FLAX_AVAILABLE:
    class CrossAttentionBlock(nn.Module):
        """
        Cross-attention block for fusing information between modalities.
        
        Based on advanced cross-attention mechanisms from Flamingo and PaLI-X.
        """
        hidden_size: int
        num_heads: int
        mlp_dim: int
        dropout_rate: float = 0.1
        attention_dropout_rate: float = 0.1
        dtype: jnp.dtype = jnp.float32
        
        @nn.compact
        def __call__(
            self,
            q_inputs: jnp.ndarray,
            kv_inputs: jnp.ndarray,
            q_mask: Optional[jnp.ndarray] = None,
            kv_mask: Optional[jnp.ndarray] = None,
            deterministic: bool = True,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Layer normalization
            q = nn.LayerNorm(dtype=self.dtype)(q_inputs)
            kv = nn.LayerNorm(dtype=self.dtype)(kv_inputs)
            
            # Cross-attention
            x, attention_weights = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                head_dim=self.hidden_size // self.num_heads,
                dropout_rate=self.attention_dropout_rate,
                dtype=self.dtype,
            )(
                q,
                kv,
                mask=kv_mask,
                deterministic=deterministic,
                decode=False,
            )
            
            # Apply attention mask if provided
            if q_mask is not None:
                x = x * q_mask[:, :, None]
            
            # Residual connection
            x = x + q_inputs
            
            # MLP block
            y = nn.LayerNorm(dtype=self.dtype)(x)
            y = nn.Dense(
                self.mlp_dim,
                dtype=self.dtype,
            )(y)
            y = nn.gelu(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
            y = nn.Dense(
                self.hidden_size,
                dtype=self.dtype,
            )(y)
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
            
            # Second residual connection
            return x + y, attention_weights


    class CrossAttentionFusion(nn.Module):
        """
        Cross-attention-based fusion of multiple modalities.
        
        Implements hierarchical cross-attention where each modality attends to
        every other modality in a configurable order.
        """
        hidden_size: int
        num_layers: int
        num_heads: int
        mlp_dim: int
        dropout_rate: float = 0.1
        dtype: jnp.dtype = jnp.float32
        
        @nn.compact
        def __call__(
            self,
            modality_encodings: Dict[str, jnp.ndarray],
            attention_masks: Dict[str, jnp.ndarray],
            fusion_order: Optional[List[Tuple[str, str]]] = None,
            deterministic: bool = True,
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            # Default fusion order if not specified
            if fusion_order is None:
                # Default hierarchical fusion: text→image→audio→video
                available_modalities = list(modality_encodings.keys())
                if len(available_modalities) <= 1:
                    # Only one modality, no fusion needed
                    modality = available_modalities[0]
                    return modality_encodings[modality], {}
                    
                # Create pairs for cross-attention: each modality attends to all previous ones
                fusion_order = []
                for i in range(1, len(available_modalities)):
                    target = available_modalities[i]
                    for j in range(i):
                        source = available_modalities[j]
                        fusion_order.append((target, source))
            
            # Store attention weights for visualization/analysis
            all_attention_weights = {}
            
            # Apply cross-attention layers according to fusion order
            for layer in range(self.num_layers):
                layer_attention_weights = {}
                
                # Project all modalities to the same hidden dimension if needed
                projected_encodings = {}
                for modality, encodings in modality_encodings.items():
                    if encodings.shape[-1] != self.hidden_size:
                        projected_encodings[modality] = nn.Dense(
                            self.hidden_size,
                            dtype=self.dtype,
                            name=f"{modality}_projection",
                        )(encodings)
                    else:
                        projected_encodings[modality] = encodings
                
                # Perform cross-attention between modalities
                for target, source in fusion_order:
                    # Skip if either modality is missing
                    if target not in projected_encodings or source not in projected_encodings:
                        continue
                    
                    # Get query from target modality
                    q = projected_encodings[target]
                    q_mask = attention_masks.get(target)
                    
                    # Get key/value from source modality
                    kv = projected_encodings[source]
                    kv_mask = attention_masks.get(source)
                    
                    # Apply cross-attention
                    fusion_block = CrossAttentionBlock(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_dim=self.mlp_dim,
                        dropout_rate=self.dropout_rate,
                        dtype=self.dtype,
                        name=f"layer_{layer}_{target}_to_{source}",
                    )
                    
                    new_q, attn_weights = fusion_block(
                        q,
                        kv,
                        q_mask=q_mask,
                        kv_mask=kv_mask,
                        deterministic=deterministic,
                    )
                    
                    # Update the target modality with fused information
                    projected_encodings[target] = new_q
                    
                    # Store attention weights
                    layer_attention_weights[f"{target}_to_{source}"] = attn_weights
                
                # Update all modality encodings
                modality_encodings = projected_encodings
                
                # Store attention weights for this layer
                all_attention_weights[f"layer_{layer}"] = layer_attention_weights
            
            # Combine all modalities into a unified representation
            # Use weighted average of modality encodings
            combined_encodings = []
            combined_mask = []
            
            for modality, encodings in modality_encodings.items():
                combined_encodings.append(encodings)
                mask = attention_masks.get(modality, jnp.ones((encodings.shape[0], encodings.shape[1])))
                combined_mask.append(mask)
            
            # Stack all encodings and masks
            stacked_encodings = jnp.concatenate(combined_encodings, axis=1)
            stacked_mask = jnp.concatenate(combined_mask, axis=1)
            
            # Final self-attention to fuse all modalities
            final_attn = nn.SelfAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name="final_fusion",
            )(
                stacked_encodings,
                mask=stacked_mask,
                deterministic=deterministic,
            )
            
            # Get CLS embedding as the final representation
            # Alternative: use attention pooling over all tokens
            cls_embedding = final_attn[:, 0]
            
            return cls_embedding, all_attention_weights
else:
    CrossAttentionBlock = None
    CrossAttentionFusion = None 