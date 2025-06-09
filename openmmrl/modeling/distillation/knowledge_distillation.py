from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openmmrl.utils.logging import get_logger

logger = get_logger(__name__)


class DistillationTrainer:
    """
    Trainer for knowledge distillation from a teacher model to a student model.
    
    Implements various distillation techniques to transfer knowledge from a 
    large multimodal model to a smaller, more efficient one.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Dict[str, Any],
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs",
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            optimizer: Optimizer for student model
            scheduler: Learning rate scheduler
            device: Device to use for training
            output_dir: Output directory for checkpoints
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.output_dir = output_dir
        
        # Set models to appropriate modes
        self.teacher_model.to(device)
        self.student_model.to(device)
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Create optimizer if not provided
        self.optimizer = optimizer or torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Create scheduler if not provided
        self.scheduler = scheduler
        if self.scheduler is None and config.get("use_scheduler", True):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get("num_epochs", 10) * len(self.train_dataloader),
                eta_min=config.get("min_learning_rate", 1e-6),
            )
        
        # Set up loss weights
        self.kd_loss_weight = config.get("kd_loss_weight", 0.5)
        self.task_loss_weight = config.get("task_loss_weight", 0.5)
        self.feature_loss_weight = config.get("feature_loss_weight", 0.1)
        self.attention_loss_weight = config.get("attention_loss_weight", 0.1)
        
        # Track training progress
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        
        # Set temperature for distillation
        self.temperature = config.get("temperature", 2.0)
    
    def train(
        self,
        num_epochs: int,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        log_steps: int = 10,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Train the student model via knowledge distillation.
        
        Args:
            num_epochs: Number of epochs to train
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            log_steps: Number of steps between logging
            compute_metrics: Function to compute evaluation metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Total training steps
        total_steps = len(self.train_dataloader) * num_epochs
        
        logger.info(f"Starting distillation for {num_epochs} epochs ({total_steps} steps)")
        
        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            self.student_model.train()
            self.teacher_model.eval()
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass - teacher model
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)
                
                # Forward pass - student model
                student_outputs = self.student_model(**batch)
                
                # Compute distillation loss
                loss = self.compute_distillation_loss(
                    teacher_outputs=teacher_outputs,
                    student_outputs=student_outputs,
                    batch=batch,
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.get("gradient_clipping", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.get("gradient_clipping"),
                    )
                
                # Update weights
                self.optimizer.step()
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
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
    
    def compute_distillation_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the distillation loss.
        
        Combines several loss components:
        - Embedding distillation loss
        - Task-specific loss
        - Feature distillation loss
        - Attention distillation loss
        
        Args:
            teacher_outputs: Outputs from teacher model
            student_outputs: Outputs from student model
            batch: Input batch
        
        Returns:
            Total distillation loss
        """
        # Initialize total loss
        total_loss = 0.0
        
        # 1. Embedding distillation loss (KL divergence)
        if "contrastive_embeddings" in teacher_outputs and "contrastive_embeddings" in student_outputs:
            teacher_embeddings = teacher_outputs["contrastive_embeddings"]
            student_embeddings = student_outputs["contrastive_embeddings"]
            
            # Normalize embeddings
            teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)
            student_embeddings = F.normalize(student_embeddings, p=2, dim=1)
            
            # Compute similarity matrices
            teacher_sim = torch.matmul(teacher_embeddings, teacher_embeddings.t()) / self.temperature
            student_sim = torch.matmul(student_embeddings, student_embeddings.t()) / self.temperature
            
            # Apply softmax
            teacher_prob = F.softmax(teacher_sim, dim=1)
            student_prob = F.log_softmax(student_sim, dim=1)
            
            # Compute KL divergence
            kd_loss = F.kl_div(student_prob, teacher_prob, reduction="batchmean") * (self.temperature ** 2)
            total_loss += self.kd_loss_weight * kd_loss
        
        # 2. Task-specific loss
        if "loss" in student_outputs:
            task_loss = student_outputs["loss"]
            total_loss += self.task_loss_weight * task_loss
        elif "generative_logits" in student_outputs and "labels" in batch:
            # Compute cross-entropy loss for generative tasks
            logits = student_outputs["generative_logits"].view(-1, student_outputs["generative_logits"].size(-1))
            labels = batch["labels"].view(-1)
            task_loss = F.cross_entropy(logits, labels)
            total_loss += self.task_loss_weight * task_loss
        
        # 3. Feature distillation loss
        if "modality_encodings" in teacher_outputs and "modality_encodings" in student_outputs:
            teacher_encodings = teacher_outputs["modality_encodings"]
            student_encodings = student_outputs["modality_encodings"]
            
            feature_loss = 0.0
            num_features = 0
            
            # Compute MSE loss for each modality
            for modality in teacher_encodings:
                if modality in student_encodings:
                    teacher_feat = teacher_encodings[modality]
                    student_feat = student_encodings[modality]
                    
                    # Handle dimension mismatch with projection
                    if teacher_feat.shape[-1] != student_feat.shape[-1]:
                        # Project to smaller dimension
                        if teacher_feat.shape[-1] > student_feat.shape[-1]:
                            projection = getattr(self, f"{modality}_projection", None)
                            if projection is None:
                                projection = nn.Linear(teacher_feat.shape[-1], student_feat.shape[-1]).to(self.device)
                                setattr(self, f"{modality}_projection", projection)
                            teacher_feat = projection(teacher_feat)
                        else:
                            projection = getattr(self, f"{modality}_projection", None)
                            if projection is None:
                                projection = nn.Linear(student_feat.shape[-1], teacher_feat.shape[-1]).to(self.device)
                                setattr(self, f"{modality}_projection", projection)
                            student_feat = projection(student_feat)
                    
                    # Compute MSE loss
                    feat_loss = F.mse_loss(student_feat, teacher_feat)
                    feature_loss += feat_loss
                    num_features += 1
            
            if num_features > 0:
                feature_loss /= num_features
                total_loss += self.feature_loss_weight * feature_loss
        
        # 4. Attention distillation loss
        if "attention_weights" in teacher_outputs and "attention_weights" in student_outputs:
            teacher_attn = teacher_outputs["attention_weights"]
            student_attn = student_outputs["attention_weights"]
            
            attention_loss = 0.0
            num_attention_layers = 0
            
            # Compute MSE loss for attention weights
            for layer in teacher_attn:
                if layer in student_attn:
                    teacher_layer_attn = teacher_attn[layer]
                    student_layer_attn = student_attn[layer]
                    
                    for attn_key in teacher_layer_attn:
                        if attn_key in student_layer_attn:
                            teacher_attn_weights = teacher_layer_attn[attn_key]
                            student_attn_weights = student_layer_attn[attn_key]
                            
                            # Handle dimension mismatch by interpolation
                            if teacher_attn_weights.shape != student_attn_weights.shape:
                                # Interpolate attention maps to match shapes
                                b, h, q_t, k_t = teacher_attn_weights.shape
                                b, h, q_s, k_s = student_attn_weights.shape
                                
                                if q_t > q_s or k_t > k_s:
                                    # Downsample teacher attention
                                    teacher_attn_weights = F.interpolate(
                                        teacher_attn_weights.view(b * h, 1, q_t, k_t),
                                        size=(q_s, k_s),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).view(b, h, q_s, k_s)
                                else:
                                    # Upsample student attention
                                    student_attn_weights = F.interpolate(
                                        student_attn_weights.view(b * h, 1, q_s, k_s),
                                        size=(q_t, k_t),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).view(b, h, q_t, k_t)
                            
                            # Compute MSE loss
                            attn_loss = F.mse_loss(student_attn_weights, teacher_attn_weights)
                            attention_loss += attn_loss
                            num_attention_layers += 1
            
            if num_attention_layers > 0:
                attention_loss /= num_attention_layers
                total_loss += self.attention_loss_weight * attention_loss
        
        return total_loss
    
    def evaluate(
        self,
        compute_metrics: Optional[Callable] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate the student model.
        
        Args:
            compute_metrics: Function to compute evaluation metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            logger.warning("Evaluation dataset is not provided")
            return None
        
        logger.info("Starting evaluation")
        
        self.student_model.eval()
        
        all_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.student_model(**batch)
                
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
        import os
        
        output_dir = os.path.join(self.output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.student_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        # Save training state
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
            },
            os.path.join(output_dir, "training_state.bin"),
        )
        
        logger.info(f"Model checkpoint saved to {output_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
        """
        import os
        
        # Load model weights
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        self.student_model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load scheduler
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path) and self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        
        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.bin")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            self.best_metric = state.get("best_metric", float("inf"))
        
        logger.info(f"Model checkpoint loaded from {checkpoint_dir}")
        logger.info(f"Resuming from epoch {self.epoch}, global step {self.global_step}") 