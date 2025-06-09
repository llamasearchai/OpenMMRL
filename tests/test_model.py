import unittest
import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openmmrl.modeling.model import OpenMMRLModelPyTorch

class TestOpenMMRLModelPyTorch(unittest.TestCase):
    def get_mock_config(self):
        return {
            "text": {"model_name": "bert-base-uncased", "pretrained": False},
            "image": {"model_name": "vit_base_patch16_224", "pretrained": False},
            "audio": {"pretrained": False},
            "video": {},
            "fusion": {
                "hidden_size": 768,
                "num_layers": 1,
                "num_heads": 8,
                "mlp_dim": 1024,
                "dropout_rate": 0.1,
            },
            "heads": {
                "input_dim": 768,
                "hidden_dim": 512,
                "output_dim": 256
            },
        }

    def test_model_initialization(self):
        config = self.get_mock_config()
        model = OpenMMRLModelPyTorch(config)
        self.assertIsInstance(model, OpenMMRLModelPyTorch)

    def test_forward_pass(self):
        config = self.get_mock_config()
        model = OpenMMRLModelPyTorch(config)

        mock_text = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones((2, 10)),
        }
        mock_images = torch.rand(2, 3, 224, 224)
        mock_audio = torch.rand(2, 16000)
        mock_video = torch.rand(2, 16, 3, 224, 224)

        output = model(
            text=mock_text,
            images=mock_images,
            audio=mock_audio,
            video=mock_video
        )
        
        self.assertIn("projected_embeddings", output)
        self.assertEqual(output["projected_embeddings"].shape, (2, config["heads"]["output_dim"]))

if __name__ == "__main__":
    unittest.main() 