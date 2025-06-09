try:
    import flax.linen as nn
    import jax.numpy as jnp
    from transformers import FlaxAutoModel
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False
    nn = object # type: ignore

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

if FLAX_AVAILABLE:
    class TextEncoder(nn.Module):
        """
        A Flax-based text encoder using a pre-trained transformer model.
        """
        model_name: str
        pretrained: bool = True
        output_hidden_states: bool = False

        def setup(self):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.transformer = FlaxAutoModel.from_pretrained(self.model_name)

        def __call__(self, input_ids, attention_mask):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.output_hidden_states,
            )
            return outputs.last_hidden_state
else:
    TextEncoder = None

class TextEncoderPyTorch(torch.nn.Module):
    """
    A PyTorch-based text encoder using a pre-trained transformer model.
    """
    def __init__(self, model_name: str, pretrained: bool = True, output_hidden_states: bool = False):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.output_hidden_states = output_hidden_states

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.pretrained:
            self.transformer = AutoModel.from_pretrained(self.model_name)
        else:
            config = AutoConfig.from_pretrained(self.model_name)
            self.transformer = AutoModel.from_config(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
        )
        return outputs.last_hidden_state 