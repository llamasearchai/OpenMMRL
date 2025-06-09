import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from openmmrl.utils.logging import get_logger

logger = get_logger(__name__)


class DeepSpeedTrainer:
    """
    Trainer for multimodal models using DeepSpeed.
    
    Handles distributed training with DeepSpeed's ZeRO optimizer stages
    for efficient large-scale training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        local_rank: int = -1,
        deepspeed_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs",
    ):
        """
        Initialize DeepSpeed trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            local_rank: Local rank for distributed training
            deepspeed_config: DeepSpeed configuration
            output_dir: Output directory for checkpoints
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.local_rank = local_rank if local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))
        self.output_dir = output_dir
        
        # Set up DeepSpeed
        self.deepspeed_config = deepspeed_config or self._get_default_deepspeed_config()
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=self.deepspeed_config,
        )
        
        # Set up data loaders
        self.train_dataloader = self._setup_dataloader(train_dataset, is_train=True)
        self.eval_dataloader = self._setup_dataloader(eval_dataset, is_train=False)
        
        # Track training progress
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
    
    def _get_default_deepspeed_config(self) -> Dict[str, Any]:
        """Get default DeepSpeed configuration."""
        return {
            "train_batch_size": self.config.get("train_batch_size", 32),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 1),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.get("learning_rate", 1e-4),
                    "weight_decay": self.config.get("weight_decay", 0.01),
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config.get("learning_rate", 1e-4),
                    "warmup_num_steps": self.config.get("warmup_steps", 1000),
                },
            },
            "fp16": {
                "enabled": self.config.get("fp16", True),
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
            "zero_optimization": {
                "stage": self.config.get("zero_stage", 2),
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                } if self.config.get("offload_optimizer", False) else False,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                } if self.config.get("offload_param", False) else False,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": self.config.get("reduce_bucket_size", 5e8),
                "stage3_prefetch_bucket_size": self.config.get("stage3_prefetch_bucket_size", 5e8),
                "stage3_param_persistence_threshold": self.config.get("stage3_param_persistence_threshold", 1e6),
            } if self.config.get("zero_stage", 2) > 0 else None,
            "gradient_clipping": self.config.get("gradient_clipping", 1.0),
            "steps_per_print": self.config.get("steps_per_print", 100),
            "wall_clock_breakdown": False,
        }
    
    def _setup_dataloader(
        self, 
        dataset: Optional[torch.utils.data.Dataset],
        is_train: bool = True,
    ) -> Optional[torch.utils.data.DataLoader]:
        """Set up data loader for training or evaluation."""
        if dataset is None:
            return None
        
        batch_size = (
            self.config.get("train_batch_size", 32) 
            if is_train 
            else self.config.get("eval_batch_size", 32)
        )
        
        # Adjust batch size for gradient accumulation
        if "gradient_accumulation_steps" in self.deepspeed_config:
            batch_size = batch_size // self.deepspeed_config.get("gradient_accumulation_steps", 1)
        
        # Create sampler for distributed training
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=is_train,
        )
        
        # Create data loader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=is_train,
        )
    
    def train(
        self,
        num_epochs: int,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        log_steps: int = 10,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            log_steps: Number of steps between logging
            compute_metrics: Function to compute evaluation metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.train_dataloader is None:
            raise ValueError("Training dataset is not provided")
        
        # Total training steps
        total_steps = len(self.train_dataloader) * num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs ({total_steps} steps)")
        
        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            
            self.model_engine.train()
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model_engine(**batch)
                loss = outputs["loss"] if "loss" in outputs else outputs[0]
                
                # Backward pass
                self.model_engine.backward(loss)
                
                # Update weights
                self.model_engine.step()
                
                # Update global step
                self.global_step += 1
                
                # Log progress
                if self.global_step % log_steps == 0:
                    logger.info(
                        f"Epoch: {epoch}/{num_epochs} | "
                        f"Step: {self.global_step}/{total_steps} | "
                        f"Loss: {loss.item():.4f}"
                    )
                
                # Evaluate model
                if eval_steps is not None and self.global_step % eval_steps == 0:
                    metrics = self.evaluate(compute_metrics)
                    
                    # Save best model
                    if metrics is not None and "eval_loss" in metrics:
                        if metrics["eval_loss"] < self.best_metric:
                            self.best_metric = metrics["eval_loss"]
                            self.save_checkpoint(f"best_model")
                            logger.info(f"New best model saved with eval_loss: {self.best_metric:.4f}")
                
                # Save checkpoint
                if save_steps is not None and self.global_step % save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
            
            # Save checkpoint at the end of each epoch
            self.save_checkpoint(f"checkpoint-epoch-{epoch}")
        
        # Final evaluation
        metrics = self.evaluate(compute_metrics)
        
        # Save final model
        self.save_checkpoint("final_model")
        
        return metrics
    
    def evaluate(
        self,
        compute_metrics: Optional[Callable] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate the model.
        
        Args:
            compute_metrics: Function to compute evaluation metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            logger.warning("Evaluation dataset is not provided")
            return None
        
        logger.info("Starting evaluation")
        
        self.model_engine.eval()
        
        all_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model_engine(**batch)
                
                # Get loss
                if "loss" in outputs:
                    loss = outputs["loss"]
                    all_losses.append(loss.item())
                
                # Get predictions and labels for metrics
                if compute_metrics is not None:
                    if "logits" in outputs:
                        preds = outputs["logits"]
                    elif "contrastive_embeddings" in outputs:
                        preds = outputs["contrastive_embeddings"]
                    else:
                        preds = outputs[0]
                    
                    all_preds.append(preds.cpu())
                    
                    if "labels" in batch:
                        all_labels.append(batch["labels"].cpu())
        
        # Compute average loss
        eval_loss = sum(all_losses) / len(all_losses) if all_losses else float("nan")
        
        logger.info(f"Evaluation loss: {eval_loss:.4f}")
        
        # Compute additional metrics
        metrics = {"eval_loss": eval_loss}
        
        if compute_metrics is not None and all_preds and all_labels:
            # Concatenate predictions and labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute metrics
            additional_metrics = compute_metrics(all_preds, all_labels)
            metrics.update(additional_metrics)
            
            # Log metrics
            for name, value in additional_metrics.items():
                logger.info(f"Evaluation {name}: {value:.4f}")
        
        return metrics
    
    def save_checkpoint(self, tag: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            tag: Tag for the checkpoint
        """
        output_dir = os.path.join(self.output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DeepSpeed checkpoint
        self.model_engine.save_checkpoint(output_dir)
        
        logger.info(f"Model checkpoint saved to {output_dir}")
        
        # If we're using ZeRO-3, also save a consolidated FP32 checkpoint
        if self.deepspeed_config.get("zero_optimization", {}).get("stage", 0) == 3:
            fp32_output_dir = os.path.join(output_dir, "fp32")
            os.makedirs(fp32_output_dir, exist_ok=True)
            
            # Get consolidated state dict
            state_dict = get_fp32_state_dict_from_zero_checkpoint(output_dir)
            
            # Save consolidated checkpoint
            torch.save(state_dict, os.path.join(fp32_output_dir, "pytorch_model.bin"))
            
            logger.info(f"Consolidated FP32 model saved to {fp32_output_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str, load_optimizer_states: bool = True) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
            load_optimizer_states: Whether to load optimizer states
        """
        # Load DeepSpeed checkpoint
        _, client_state = self.model_engine.load_checkpoint(
            checkpoint_dir,
            load_optimizer_states=load_optimizer_states,
        )
        
        # Update global step and epoch from checkpoint
        if client_state is not None:
            self.global_step = client_state.get("global_step", self.global_step)
            self.epoch = client_state.get("epoch", self.epoch)
            self.best_metric = client_state.get("best_metric", self.best_metric)
        
        logger.info(f"Model checkpoint loaded from {checkpoint_dir}")
        logger.info(f"Resuming from global step {self.global_step}, epoch {self.epoch}") 