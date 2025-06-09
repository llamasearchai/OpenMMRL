from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlockPyTorch(nn.Module):
    """
    PyTorch implementation of the Cross-attention block.
    """
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate=0.1, attention_dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.ln1_q = nn.LayerNorm(hidden_size)
        self.ln1_kv = nn.LayerNorm(hidden_size)
        
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=attention_dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, q_inputs, kv_inputs, q_mask=None, kv_mask=None):
        q = self.ln1_q(q_inputs)
        kv = self.ln1_kv(kv_inputs)

        # PyTorch's MultiheadAttention expects key_padding_mask for kv_mask
        # It should have shape (N, S) where N is batch size, S is sequence length.
        # It masks positions with True.
        
        attn_output, attn_weights = self.cross_attention(q, kv, kv, key_padding_mask=kv_mask)
        
        x = q_inputs + self.dropout(attn_output)
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights


class CrossAttentionFusionPyTorch(nn.Module):
    """
    PyTorch implementation of the cross-attention-based fusion of multiple modalities.
    """
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.fusion_layers = nn.ModuleList()
        for _ in range(num_layers):
            # This is a simplified implementation. A real one would have more complex connections.
            # Here we just create a list of blocks. The logic in forward will determine how they are used.
            self.fusion_layers.append(CrossAttentionBlockPyTorch(hidden_size, num_heads, mlp_dim, dropout_rate))

        self.final_fusion = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.projections = nn.ModuleDict()

    def forward(self, modality_encodings, attention_masks, fusion_order=None):
        if fusion_order is None:
            available_modalities = list(modality_encodings.keys())
            if len(available_modalities) <= 1:
                return next(iter(modality_encodings.values())), {}
            
            fusion_order = []
            for i in range(1, len(available_modalities)):
                target = available_modalities[i]
                for j in range(i):
                    source = available_modalities[j]
                    fusion_order.append((target, source))

        all_attention_weights = {}

        # Project all modalities to the same hidden dimension
        projected_encodings = {}
        for modality, encodings in modality_encodings.items():
            if encodings.shape[-1] != self.hidden_size:
                if modality not in self.projections:
                    self.projections[modality] = nn.Linear(encodings.shape[-1], self.hidden_size)
                projected_encodings[modality] = self.projections[modality](encodings)
            else:
                projected_encodings[modality] = encodings

        for i in range(self.num_layers):
            layer_attention_weights = {}
            fusion_block = self.fusion_layers[i] # Simplified: using one block per layer for all pairs
            
            for target, source in fusion_order:
                if target not in projected_encodings or source not in projected_encodings:
                    continue
                
                q = projected_encodings[target]
                kv = projected_encodings[source]
                # Assuming attention_masks contains key_padding_masks
                kv_mask = attention_masks.get(source)

                fused_q, attn_weights = fusion_block(q, kv, kv_mask=kv_mask)
                projected_encodings[target] = fused_q
                layer_attention_weights[f"{target}_to_{source}"] = attn_weights
            
            all_attention_weights[f"layer_{i}"] = layer_attention_weights

        # Final fusion by concatenating and self-attending
        combined_encodings = torch.cat(list(projected_encodings.values()), dim=1)
        
        # Create a combined attention mask
        combined_mask = None
        if attention_masks:
            masks = [attention_masks.get(m) for m in projected_encodings.keys() if attention_masks.get(m) is not None]
            if masks:
                 # The masks need to be reshaped/padded to be concatenated
                max_len = max(m.shape[1] for m in projected_encodings.values())
                padded_masks = []
                for m_name, m_enc in projected_encodings.items():
                    mask = attention_masks.get(m_name)
                    if mask is not None:
                        pad_len = combined_encodings.shape[1] - mask.shape[1]
                        padded_mask = F.pad(mask, (0, pad_len), 'constant', True) # Pad with True for masking
                        padded_masks.append(padded_mask)

                if padded_masks:
                    combined_mask = torch.cat(padded_masks, dim=1)


        final_fused, _ = self.final_fusion(combined_encodings, combined_encodings, combined_encodings, key_padding_mask=combined_mask)
        
        cls_embedding = final_fused[:, 0]
        
        return cls_embedding, all_attention_weights 