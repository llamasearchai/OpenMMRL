import argparse
import json
import os
import torch
import torch.distributed as dist

from openmmrl.data.tfdata.dataset import OpenMMRLDatasetTF
from openmmrl.data.tfdata.torch_wrapper import TFDatasetToTorch
from openmmrl.modeling.model import OpenMMRLModelPyTorch
from openmmrl.training.distributed.deepspeed_trainer import DeepSpeedTrainer
from openmmrl.utils.tracking.experiment_tracker import ExperimentTracker
from openmmrl.utils.logging import get_logger
from openmmrl.evaluation.metrics import accuracy, correlation

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train OpenMMRL model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()

def load_config(config_path: str):
    """Load config from file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)

    # Initialize tracker, model, datasets, and trainer
    tracker = ExperimentTracker(
        experiment_name="OpenMMRL",
        config=config,
        use_wandb=config.get("use_wandb", False),
    )
    
    model = OpenMMRLModelPyTorch(config["model"])

    train_dataset = TFDatasetToTorch(
        OpenMMRLDatasetTF(
            dataset_name=config["dataset"]["name"],
            split="train",
            data_dir=config["dataset"]["data_dir"],
            modalities=config["dataset"]["modalities"],
            batch_size=config["training"]["batch_size"],
        ).get_dataset()
    )
    
    eval_dataset = TFDatasetToTorch(
        OpenMMRLDatasetTF(
            dataset_name=config["dataset"]["name"],
            split=config["dataset"]["eval_split"],
            data_dir=config["dataset"]["data_dir"],
            modalities=config["dataset"]["modalities"],
            batch_size=config["training"]["eval_batch_size"],
        ).get_dataset()
    )

    def compute_metrics(predictions, labels):
        # This is a placeholder. You may need to adjust this based on your task.
        return {
            "accuracy": accuracy(predictions, labels),
            "correlation": correlation(predictions, labels),
        }

    trainer = DeepSpeedTrainer(
        model=model,
        config=config["training"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="outputs",
    )

    trainer.train(
        num_epochs=config["training"].get("num_epochs", 10),
        compute_metrics=compute_metrics,
    )

    tracker.finish()

if __name__ == "__main__":
    main() 