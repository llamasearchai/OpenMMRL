import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import flax.linen as flax_nn
    import jax.numpy as jnp
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

class VideoEncoderPyTorch(nn.Module):
    """
    PyTorch-based video encoder using 3D convolutions and transformer layers.
    """
    
    def __init__(
        self,
        patch_size=16,
        frame_stride=4,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout_rate=0.1,
        num_frames=32,
        image_size=224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.frame_stride = frame_stride
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.image_size = image_size
        
        # 3D patch embedding
        self.patch_embed = nn.Conv3d(
            3, hidden_size,
            kernel_size=(frame_stride, patch_size, patch_size),
            stride=(frame_stride, patch_size, patch_size)
        )
        
        # Positional embeddings
        num_patches_per_frame = (image_size // patch_size) ** 2
        num_temporal_patches = num_frames // frame_stride
        total_patches = num_patches_per_frame * num_temporal_patches
        
        self.pos_embed = nn.Parameter(torch.randn(1, total_patches + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            VideoTransformerBlock(hidden_size, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, training=False):
        # x shape: (batch_size, num_frames, channels, height, width)
        B, T, C, H, W = x.shape
        
        # Rearrange for 3D conv: (batch, channels, time, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, hidden_size, T', H', W')
        
        # Flatten spatial and temporal dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, training=training)
        
        x = self.norm(x)
        
        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones(B, x.shape[1], device=x.device)
        
        return x, attention_mask

class VideoTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, x, training=False):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

if FLAX_AVAILABLE:
    class VideoEncoder(flax_nn.Module):
        """
        Flax-based video encoder using 3D convolutions and transformer layers.
        """
        patch_size: int = 16
        frame_stride: int = 4
        hidden_size: int = 768
        num_layers: int = 12
        num_heads: int = 12
        mlp_dim: int = 3072
        dropout_rate: float = 0.1
        num_frames: int = 32
        image_size: int = 224
        dtype: jnp.dtype = jnp.float32
        
        @flax_nn.compact
        def __call__(self, x, deterministic=True):
            # x shape: (batch_size, num_frames, height, width, channels)
            B, T, H, W, C = x.shape
            
            # Rearrange for 3D conv: (batch, time, height, width, channels)
            # Flax expects (batch, spatial_dims..., features)
            
            # 3D patch embedding
            x = flax_nn.Conv(
                features=self.hidden_size,
                kernel_size=(self.frame_stride, self.patch_size, self.patch_size),
                strides=(self.frame_stride, self.patch_size, self.patch_size),
                dtype=self.dtype,
                name="patch_embed",
            )(x)
            
            # Flatten spatial and temporal dimensions
            x = x.reshape(B, -1, self.hidden_size)
            
            # Add CLS token
            cls_token = self.param(
                "cls_token", 
                flax_nn.initializers.normal(stddev=0.02), 
                (1, 1, self.hidden_size)
            )
            cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.hidden_size))
            x = jnp.concatenate([cls_tokens, x], axis=1)
            
            # Add positional embeddings
            pos_embed = self.param(
                "pos_embed",
                flax_nn.initializers.normal(stddev=0.02),
                (1, x.shape[1], self.hidden_size),
            )
            x = x + pos_embed
            x = flax_nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
            
            # Apply transformer layers
            for i in range(self.num_layers):
                x = VideoTransformerBlockFlax(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    name=f"layer_{i}",
                )(x, deterministic=deterministic)
            
            x = flax_nn.LayerNorm(dtype=self.dtype, name="norm")(x)
            
            # Create attention mask (all tokens are valid)
            attention_mask = jnp.ones((B, x.shape[1]))
            
            return x, attention_mask

    class VideoTransformerBlockFlax(flax_nn.Module):
        hidden_size: int
        num_heads: int
        mlp_dim: int
        dropout_rate: float
        dtype: jnp.dtype = jnp.float32
        
        @flax_nn.compact
        def __call__(self, x, deterministic=True):
            # Self-attention
            norm_x = flax_nn.LayerNorm(dtype=self.dtype, name="norm1")(x)
            attn_out = flax_nn.SelfAttention(
                num_heads=self.num_heads,
                head_dim=self.hidden_size // self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name="attention",
            )(norm_x, deterministic=deterministic)
            x = x + attn_out
            
            # MLP
            norm_x = flax_nn.LayerNorm(dtype=self.dtype, name="norm2")(x)
            mlp_out = flax_nn.Dense(self.mlp_dim, dtype=self.dtype, name="mlp_dense1")(norm_x)
            mlp_out = flax_nn.gelu(mlp_out)
            mlp_out = flax_nn.Dropout(rate=self.dropout_rate)(mlp_out, deterministic=deterministic)
            mlp_out = flax_nn.Dense(self.hidden_size, dtype=self.dtype, name="mlp_dense2")(mlp_out)
            mlp_out = flax_nn.Dropout(rate=self.dropout_rate)(mlp_out, deterministic=deterministic)
            
            x = x + mlp_out
            
            return x
else:
    # Placeholder for when Flax is not available
    class VideoEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flax is not available. Please install jax and flax to use the Flax implementation.") 