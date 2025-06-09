import torch
import torchaudio
from torchaudio.models import wav2vec2_base

class AudioEncoderPyTorch(torch.nn.Module):
    """
    A PyTorch-based audio encoder using a pre-trained Wav2Vec2 model.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = wav2vec2_base(pretrained=pretrained)
        # We can treat the output of the feature extractor as the embeddings
        # Or we can further process the output of the transformer layers

    def forward(self, waveform, length=None):
        """
        Args:
            waveform (torch.Tensor): Audio waveform of shape (batch, num_frames)
            length (torch.Tensor, optional): Specifies the valid length of each audio in the batch. Shape (batch,).
        
        Returns:
            torch.Tensor: The audio embeddings.
        """
        return self.model.feature_extractor(waveform)

# TODO: Add a Flax-based AudioEncoder.
class AudioEncoder:
    def __init__(self):
        raise NotImplementedError("Flax AudioEncoder is not yet implemented.") 