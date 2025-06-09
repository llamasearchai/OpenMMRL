import functools
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as torch_nn

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

from openmmrl.modeling.encoders import (
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    VideoEncoder,
)
from openmmrl.modeling.fusion import CrossAttentionFusion, CrossAttentionFusionPyTorch
from openmmrl.modeling.heads import ProjectionHead

if FLAX_AVAILABLE:
    class OpenMMRLModel(nn.Module):
        """
        Core multimodal transformer model implementing cross-modal fusion.
        """
        config: Dict[str, Any]
        dtype: jnp.dtype = jnp.float32

        def setup(self):
            # Initialize modality-specific encoders
            self.text_encoder = TextEncoder(
                model_name=self.config["text"]["model_name"],
                pretrained=self.config["text"].get("pretrained", True),
            )

            self.image_encoder = ImageEncoder(
                model_name=self.config["image"]["model_name"],
                pretrained=self.config["image"].get("pretrained", True),
            )

            self.audio_encoder = AudioEncoder(
                pretrained=self.config["audio"].get("pretrained", True),
            )
            
            self.video_encoder = VideoEncoder() # Placeholder

            # Cross-attention fusion module
            self.fusion = CrossAttentionFusion(
                hidden_size=self.config["fusion"]["hidden_size"],
                num_layers=self.config["fusion"]["num_layers"],
                num_heads=self.config["fusion"]["num_heads"],
                mlp_dim=self.config["fusion"]["mlp_dim"],
                dropout_rate=self.config["fusion"]["dropout_rate"],
                dtype=self.dtype,
            )

            # Task-specific heads
            self.projection_head = ProjectionHead(
                input_dim=self.config["heads"]["input_dim"],
                hidden_dim=self.config["heads"]["hidden_dim"],
                output_dim=self.config["heads"]["output_dim"],
            )

        def get_modality_encodings(
            self,
            text=None,
            images=None,
            audio=None,
            video=None,
            training: bool = False,
        ) -> Dict[str, jnp.ndarray]:
            """Encode each provided modality."""
            encodings = {}
            
            if text is not None:
                encodings["text"] = self.text_encoder(
                    input_ids=text["input_ids"],
                    attention_mask=text["attention_mask"],
                )

            if images is not None:
                encodings["image"] = self.image_encoder(images)

            if audio is not None:
                encodings["audio"] = self.audio_encoder(audio)
                
            if video is not None:
                encodings["video"] = self.video_encoder(video)

            return encodings

        def __call__(
            self,
            text=None,
            images=None,
            audio=None,
            video=None,
            training: bool = False,
        ) -> Dict[str, jnp.ndarray]:
            encodings = self.get_modality_encodings(
                text=text,
                images=images,
                audio=audio,
                video=video,
                training=training,
            )

            attention_masks = {k: jnp.ones(v.shape[:-1]) for k, v in encodings.items() if hasattr(v, 'shape')}

            fused_embeddings, attention_weights = self.fusion(
                encodings,
                attention_masks,
                deterministic=not training,
            )

            projected_embeddings = self.projection_head(fused_embeddings)

            return {
                "fused_embeddings": fused_embeddings,
                "projected_embeddings": projected_embeddings,
                "attention_weights": attention_weights,
                "modality_encodings": encodings,
            }
else:
    OpenMMRLModel = None

class OpenMMRLModelPyTorch(torch_nn.Module):
    """PyTorch implementation of the multimodal model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        from openmmrl.modeling.encoders import (
            TextEncoderPyTorch,
            ImageEncoderPyTorch,
            AudioEncoderPyTorch,
            VideoEncoderPyTorch,
        )
        from openmmrl.modeling.heads import ProjectionHeadPyTorch

        self.text_encoder = TextEncoderPyTorch(
            model_name=self.config["text"]["model_name"],
            pretrained=self.config["text"].get("pretrained", True),
        )

        self.image_encoder = ImageEncoderPyTorch(
            model_name=self.config["image"]["model_name"],
            pretrained=self.config["image"].get("pretrained", True),
        )

        self.audio_encoder = AudioEncoderPyTorch(
            pretrained=self.config["audio"].get("pretrained", True),
        )
        
        self.video_encoder = VideoEncoderPyTorch()

        self.fusion = CrossAttentionFusionPyTorch(
            hidden_size=self.config["fusion"]["hidden_size"],
            num_layers=self.config["fusion"]["num_layers"],
            num_heads=self.config["fusion"]["num_heads"],
            mlp_dim=self.config["fusion"]["mlp_dim"],
            dropout_rate=self.config["fusion"]["dropout_rate"],
        )

        self.projection_head = ProjectionHeadPyTorch(
            input_dim=self.config["heads"]["input_dim"],
            hidden_dim=self.config["heads"]["hidden_dim"],
            output_dim=self.config["heads"]["output_dim"],
        )

    def get_modality_encodings(
        self,
        text=None,
        images=None,
        audio=None,
        video=None,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encodings = {}

        if text is not None:
            encodings["text"] = self.text_encoder(
                input_ids=text["input_ids"],
                attention_mask=text["attention_mask"],
            )

        if images is not None:
            encodings["image"] = self.image_encoder(images)

        if audio is not None:
            encodings["audio"] = self.audio_encoder(audio)
            
        if video is not None:
            encodings["video"] = self.video_encoder(video)

        return encodings

    def forward(
        self,
        text=None,
        images=None,
        audio=None,
        video=None,
        training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encodings = self.get_modality_encodings(
            text=text,
            images=images,
            audio=audio,
            video=video,
            training=training,
        )

        attention_masks = {}
        if text and 'attention_mask' in text:
            attention_masks['text'] = text['attention_mask']

        fused_embeddings, attention_weights = self.fusion(encodings, attention_masks)
        
        projected_embeddings = self.projection_head(fused_embeddings)

        return {
            "fused_embeddings": fused_embeddings,
            "projected_embeddings": projected_embeddings,
            "attention_weights": attention_weights,
            "modality_encodings": encodings,
        } 