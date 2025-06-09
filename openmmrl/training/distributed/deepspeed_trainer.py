import os
import time
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from openmmrl.utils.logging import get_logger

logger = get_logger(__name__)


class DeepSpeedTrainer:
    """
    Comprehensive trainer for multimodal models using DeepSpeed.
    
    Handles distributed training with DeepSpeed's ZeRO optimizer stages
    for efficient large-scale training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_dataset = None,
        eval_dataset = None,
        local_rank: int = -1,
        deepspeed_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs",
        resume_from_checkpoint: Optional[str] = None,
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
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.local_rank = local_rank if local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up DeepSpeed configuration
        self.deepspeed_config = deepspeed_config or self._get_default_deepspeed_config()
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, self.train_dataloader, self.lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            config=self.deepspeed_config,
        )
        
        # Set up evaluation dataloader
        self.eval_dataloader = self._setup_eval_dataloader()
        
        # Track training progress
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        self.start_time = time.time()
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
    
    def _get_default_deepspeed_config(self) -> Dict[str, Any]:
        """Get default DeepSpeed configuration."""
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        train_batch_size = self.config.get("batch_size", 32) * gradient_accumulation_steps
        
        if dist.is_initialized():
            train_batch_size *= dist.get_world_size()
        
        config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": self.config.get("batch_size", 32),
            "gradient_accumulation_steps": gradient_accumulation_steps,
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
            "gradient_clipping": self.config.get("gradient_clipping", 1.0),
            "steps_per_print": self.config.get("steps_per_print", 100),
            "wall_clock_breakdown": False,
        }
        
        # Add FP16 configuration
        if self.config.get("fp16", True):
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        
        # Add ZeRO configuration
        zero_stage = self.config.get("zero_stage", 2)
        if zero_stage > 0:
            config["zero_optimization"] = {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": self.config.get("reduce_bucket_size", 5e8),
                "stage3_prefetch_bucket_size": self.config.get("stage3_prefetch_bucket_size", 5e8),
                "stage3_param_persistence_threshold": self.config.get("stage3_param_persistence_threshold", 1e6),
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
            
            # Add CPU offloading if requested
            if self.config.get("offload_optimizer", False):
                config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
                
            if self.config.get("offload_param", False) and zero_stage == 3:
                config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
        
        return config
    
    def _setup_eval_dataloader(self):
        """Set up evaluation dataloader."""
        if self.eval_dataset is None:
            return None
        
        eval_batch_size = self.config.get("eval_batch_size", self.config.get("batch_size", 32))
        
        # Create sampler for distributed evaluation
        eval_sampler = None
        if dist.is_initialized():
            eval_sampler = DistributedSampler(
                self.eval_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
        
        # Create evaluation dataloader
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=eval_batch_size,
            sampler=eval_sampler,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )
        
        return eval_dataloader
    
    def train(
        self,
        num_epochs: int,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        log_steps: int = 10,
        compute_metrics: Optional[Callable] = None,
        early_stopping_patience: Optional[int] = None,
        metric_for_best_model: str = "eval_loss",
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            log_steps: Number of steps between logging
            compute_metrics: Function to compute evaluation metrics
            early_stopping_patience: Number of evaluations to wait before early stopping
            metric_for_best_model: Metric to use for determining best model
        
        Returns:
            Dictionary of final evaluation metrics
        """
        if self.train_dataloader is None:
            raise ValueError("Training dataset is not provided")
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training batch size: {self.deepspeed_config['train_batch_size']}")
        logger.info(f"Gradient accumulation steps: {self.deepspeed_config['gradient_accumulation_steps']}")
        
        # Initialize early stopping
        early_stopping_counter = 0
        
        # Training loop
        for epoch in range(self.epoch, self.epoch + num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            self.model_engine.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(self.train_dataloader):
                start_time = time.time()
                
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.model_engine(**batch)
                loss = self._extract_loss(outputs, batch)
                
                # Backward pass
                self.model_engine.backward(loss)
                
                # Update weights
                self.model_engine.step()
                
                # Update global step
                self.global_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if self.global_step % log_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    learning_rate = self.model_engine.get_lr()[0]
                    step_time = time.time() - start_time
                    
                    logger.info(
                        f"Epoch: {epoch}/{self.epoch + num_epochs - 1} | "
                        f"Step: {self.global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"LR: {learning_rate:.2e} | "
                        f"Step Time: {step_time:.2f}s"
                    )
                
                # Evaluate model
                if eval_steps is not None and self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(compute_metrics)
                    
                    if eval_metrics is not None:
                        # Check for best model
                        current_metric = eval_metrics.get(metric_for_best_model, float("inf"))
                        
                        if current_metric < self.best_metric:
                            self.best_metric = current_metric
                            self.save_checkpoint("best_model")
                            logger.info(f"New best model saved with {metric_for_best_model}: {self.best_metric:.4f}")
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                        
                        # Early stopping check
                        if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {early_stopping_counter} evaluations without improvement")
                            return eval_metrics
                
                # Save checkpoint
                if save_steps is not None and self.global_step % save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
            
            # Save checkpoint at the end of each epoch
            self.save_checkpoint(f"checkpoint-epoch-{epoch}")
            
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final evaluation
        final_metrics = self.evaluate(compute_metrics)
        
        # Save final model
        self.save_checkpoint("final_model")
        
        # Log training summary
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return final_metrics
    
    def evaluate(
        self,
        compute_metrics: Optional[Callable] = None,
        dataset: Optional[DataLoader] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate the model.
        
        Args:
            compute_metrics: Function to compute evaluation metrics
            dataset: Dataset to evaluate on (uses self.eval_dataloader if None)
        
        Returns:
            Dictionary of evaluation metrics
        """
        eval_dataloader = dataset or self.eval_dataloader
        
        if eval_dataloader is None:
            logger.warning("No evaluation dataset provided")
            return None
        
        logger.info("Starting evaluation")
        
        self.model_engine.eval()
        
        total_loss = 0.0
        num_samples = 0
        all_predictions = []
        all_labels = []
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.model_engine(**batch)
                
                # Compute loss
                loss = self._extract_loss(outputs, batch)
                total_loss += loss.item() * self._get_batch_size(batch)
                num_samples += self._get_batch_size(batch)
                
                # Collect predictions and labels for metrics
                if compute_metrics is not None:
                    predictions = self._extract_predictions(outputs)
                    labels = self._extract_labels(batch)
                    
                    all_predictions.append(predictions.cpu())
                    all_labels.append(labels.cpu())
        
        # Aggregate results across all processes
        if dist.is_initialized():
            total_loss = self._all_reduce_mean(total_loss)
            num_samples = self._all_reduce_sum(num_samples)
        
        # Compute average loss
        avg_loss = total_loss / num_samples
        
        eval_time = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        
        # Compute additional metrics
        metrics = {"eval_loss": avg_loss}
        
        if compute_metrics is not None and all_predictions and all_labels:
            # Concatenate all predictions and labels
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Gather predictions and labels from all processes
            if dist.is_initialized():
                all_predictions = self._gather_tensor(all_predictions)
                all_labels = self._gather_tensor(all_labels)
            
            # Compute metrics only on rank 0
            if not dist.is_initialized() or dist.get_rank() == 0:
                try:
                    additional_metrics = compute_metrics(all_predictions, all_labels)
                    metrics.update(additional_metrics)
                    
                    # Log additional metrics
                    for name, value in additional_metrics.items():
                        logger.info(f"Evaluation {name}: {value:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to compute additional metrics: {e}")
        
        return metrics
    
    def _move_to_device(self, batch):
        """Move batch to the appropriate device."""
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(item) for item in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.model_engine.device, non_blocking=True)
        else:
            return batch
    
    def _extract_loss(self, outputs, batch):
        """Extract loss from model outputs."""
        if isinstance(outputs, dict):
            if "loss" in outputs:
                return outputs["loss"]
            elif "projected_embeddings" in outputs and "labels" in batch:
                # Compute contrastive loss
                embeddings = outputs["projected_embeddings"]
                labels = batch["labels"]
                return self._compute_contrastive_loss(embeddings, labels)
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        else:
            raise ValueError("Could not extract loss from model outputs")
    
    def _compute_contrastive_loss(self, embeddings, labels, temperature=0.07):
        """Compute contrastive loss."""
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
        
        # Create positive pairs mask
        batch_size = embeddings.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(embeddings.device)
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size).to(embeddings.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)
        
        loss = -mean_log_prob_pos.mean()
        return loss
    
    def _extract_predictions(self, outputs):
        """Extract predictions from model outputs."""
        if isinstance(outputs, dict):
            if "projected_embeddings" in outputs:
                return outputs["projected_embeddings"]
            elif "logits" in outputs:
                return outputs["logits"]
            else:
                # Return first tensor value
                for v in outputs.values():
                    if isinstance(v, torch.Tensor):
                        return v
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        else:
            return outputs
    
    def _extract_labels(self, batch):
        """Extract labels from batch."""
        if isinstance(batch, dict):
            if "labels" in batch:
                return batch["labels"]
            elif "label" in batch:
                return batch["label"]
        
        # If no labels found, create dummy labels
        batch_size = self._get_batch_size(batch)
        return torch.zeros(batch_size, dtype=torch.long)
    
    def _get_batch_size(self, batch):
        """Get batch size from batch."""
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
        elif isinstance(batch, torch.Tensor):
            return batch.shape[0]
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            return self._get_batch_size(batch[0])
        
        return 1
    
    def _all_reduce_mean(self, tensor):
        """All-reduce mean across processes."""
        if not dist.is_initialized():
            return tensor
        
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor).to(self.model_engine.device)
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        return tensor.item()
    
    def _all_reduce_sum(self, tensor):
        """All-reduce sum across processes."""
        if not dist.is_initialized():
            return tensor
        
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor).to(self.model_engine.device)
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()
    
    def _gather_tensor(self, tensor):
        """Gather tensor from all processes."""
        if not dist.is_initialized():
            return tensor
        
        # Get the size of tensor from all processes
        world_size = dist.get_world_size()
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        # Gather tensors
        dist.all_gather(tensor_list, tensor)
        
        # Concatenate tensors
        return torch.cat(tensor_list, dim=0)
    
    def save_checkpoint(self, tag: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            tag: Tag for the checkpoint
        """
        checkpoint_dir = os.path.join(self.output_dir, tag)
        
        # Save DeepSpeed checkpoint
        self.model_engine.save_checkpoint(checkpoint_dir)
        
        # Save additional training state
        training_state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Save consolidated FP32 model for ZeRO-3
        if self.deepspeed_config.get("zero_optimization", {}).get("stage") == 3:
            try:
                fp32_dir = os.path.join(checkpoint_dir, "fp32_model")
                os.makedirs(fp32_dir, exist_ok=True)
                
                # Get consolidated state dict
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
                
                # Save consolidated model
                torch.save(state_dict, os.path.join(fp32_dir, "pytorch_model.bin"))
                logger.info(f"Consolidated FP32 model saved to {fp32_dir}")
                
            except Exception as e:
                logger.warning(f"Failed to save consolidated FP32 model: {e}")
    
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
        
        # Load additional training state
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                training_state = json.load(f)
            
            self.epoch = training_state.get("epoch", 0)
            self.global_step = training_state.get("global_step", 0)
            self.best_metric = training_state.get("best_metric", float("inf"))
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
        logger.info(f"Resuming from epoch {self.epoch}, global step {self.global_step}") 