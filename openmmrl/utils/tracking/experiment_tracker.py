import os
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import wandb
from git import Repo, InvalidGitRepositoryError

from openmmrl.utils.logging import get_logger
from openmmrl.utils.visualization.embedding_visualizer import EmbeddingVisualizer

logger = get_logger(__name__)


class ExperimentTracker:
    """
    Track experiments using MLflow and Weights & Biases.
    
    Provides unified interface for logging metrics, parameters, and artifacts
    to both platforms for comprehensive experiment tracking.
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        use_mlflow: bool = True,
        use_wandb: bool = True,
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking server
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity/username
            config: Configuration parameters
            run_name: Name for the run
            tags: Tags for the run
            use_mlflow: Whether to use MLflow
            use_wandb: Whether to use Weights & Biases
        """
        self.experiment_name = experiment_name
        self.config = config or {}
        self.run_name = run_name or f"run_{int(time.time())}"
        self.tags = tags or {}
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        
        # Add system info to tags
        self.tags.update(self._get_system_info())
        
        # Add git info to tags
        self.tags.update(self._get_git_info())
        
        # Initialize MLflow
        if self.use_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            
            # Log tags
            mlflow.set_tags(self.tags)
            
            # Log config parameters
            for key, value in self.config.items():
                mlflow.log_param(key, value)
        
        # Initialize Weights & Biases
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=wandb_project or experiment_name,
                entity=wandb_entity,
                name=self.run_name,
                config=self.config,
                tags=list(self.tags.values()),
                reinit=True,
            )
    
    def log_metric(
        self,
        key: str,
        value: Union[float, int],
        step: Optional[int] = None,
    ) -> None:
        """
        Log a metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        if self.use_mlflow:
            mlflow.log_metric(key, value, step=step)
        
        if self.use_wandb:
            metrics = {key: value}
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number
        """
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.use_mlflow:
            mlflow.log_param(key, value)
        
        if self.use_wandb:
            wandb.config.update({key: value})
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if self.use_mlflow:
            mlflow.log_params(params)
        
        if self.use_wandb:
            wandb.config.update(params)
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log an artifact.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Path within the artifact store
        """
        if self.use_mlflow:
            mlflow.log_artifact(local_path, artifact_path)
        
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=os.path.basename(local_path),
                type=os.path.splitext(local_path)[1][1:],  # Use extension as type
            )
            artifact.add_file(local_path)
            wandb.log_artifact(artifact)
    
    def log_figure(
        self,
        figure,
        artifact_path: str,
    ) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            artifact_path: Path for the figure
        """
        if self.use_mlflow:
            mlflow.log_figure(figure, artifact_path)
        
        if self.use_wandb:
            wandb.log({artifact_path: wandb.Image(figure)})
    
    def log_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        artifact_path: str,
    ) -> None:
        """
        Log an image.
        
        Args:
            image: Image as numpy array or torch tensor
            artifact_path: Path for the image
        """
        # Convert torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        if self.use_mlflow:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.axis("off")
            mlflow.log_figure(plt.gcf(), artifact_path)
            plt.close()
        
        if self.use_wandb:
            wandb.log({artifact_path: wandb.Image(image)})
    
    def log_table(
        self,
        data: Dict[str, List],
        artifact_path: str,
    ) -> None:
        """
        Log a table.
        
        Args:
            data: Dictionary of column names and values
            artifact_path: Path for the table
        """
        if self.use_mlflow:
            import pandas as pd
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV and log
            csv_path = f"{artifact_path}.csv"
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            os.remove(csv_path)
        
        if self.use_wandb:
            wandb.log({artifact_path: wandb.Table(data=data)})
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str,
    ) -> None:
        """
        Log a PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path for the model
        """
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, artifact_path)
        
        if self.use_wandb:
            # Save model to a temporary file
            tmp_path = f"{artifact_path}.pt"
            torch.save(model.state_dict(), tmp_path)
            
            # Log model
            artifact = wandb.Artifact(
                name=artifact_path,
                type="model",
            )
            artifact.add_file(tmp_path)
            wandb.log_artifact(artifact)
            
            # Clean up
            os.remove(tmp_path)
    
    def log_embedding(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        metadata: Optional[List[str]] = None,
        artifact_path: str = "embeddings",
    ) -> None:
        """
        Log embeddings for visualization.
        
        Args:
            embeddings: Embeddings as numpy array or torch tensor
            metadata: Metadata for the embeddings
            artifact_path: Path for the embeddings
        """
        # Convert torch tensor to numpy array
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        if self.use_mlflow:
            # Create visualizer
            visualizer = EmbeddingVisualizer(output_dir=".")
            
            # Create visualization
            visualizer.visualize_embeddings(
                embeddings,
                labels=metadata,
                method="umap",
                title="Embeddings",
                filename=artifact_path,
            )
            
            # Log visualization
            mlflow.log_artifact(f"{artifact_path}_umap.png")
            
            # Clean up
            os.remove(f"{artifact_path}_umap.png")
            os.remove(f"{artifact_path}_umap.pdf")
        
        if self.use_wandb:
            wandb.log({
                f"{artifact_path}_projector": wandb.Table(
                    columns=["embedding"] + (["metadata"] if metadata else []),
                    data=[[e] + ([m] if metadata else []) for e, m in zip(embeddings, metadata or [None] * len(embeddings))],
                )
            })
    
    def log_confusion_matrix(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        labels: Optional[List[str]] = None,
        artifact_path: str = "confusion_matrix",
    ) -> None:
        """
        Log a confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            artifact_path: Path for the confusion matrix
        """
        # Convert to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        
        # Log figure
        self.log_figure(plt.gcf(), artifact_path)
        plt.close()
    
    def log_pr_curve(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_score: Union[np.ndarray, torch.Tensor, List],
        artifact_path: str = "pr_curve",
    ) -> None:
        """
        Log a precision-recall curve.
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            artifact_path: Path for the PR curve
        """
        # Convert to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_score, torch.Tensor):
            y_score = y_score.detach().cpu().numpy()
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_score, list):
            y_score = np.array(y_score)
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
        # Plot PR curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, marker=".", label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Log figure
        self.log_figure(plt.gcf(), artifact_path)
        plt.close()
    
    def finish(self) -> None:
        """End the tracking session."""
        if self.use_mlflow:
            mlflow.end_run()
        
        if self.use_wandb:
            wandb.finish()
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        import platform
        import psutil
        
        system_info = {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "cpu_count": str(psutil.cpu_count()),
            "memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        }
        
        # Add GPU info if available
        try:
            if torch.cuda.is_available():
                system_info["gpu"] = torch.cuda.get_device_name(0)
                system_info["gpu_count"] = str(torch.cuda.device_count())
                system_info["cuda"] = torch.version.cuda
        except:
            pass
        
        return {f"system.{k}": v for k, v in system_info.items()}
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information."""
        git_info = {}
        
        try:
            repo = Repo(search_parent_directories=True)
            git_info["git.commit"] = repo.head.commit.hexsha
            git_info["git.branch"] = repo.active_branch.name
            
            # Check if repo is dirty
            git_info["git.dirty"] = str(repo.is_dirty())
            
            # Get remote URL
            if len(repo.remotes) > 0:
                git_info["git.remote"] = repo.remotes.origin.url
        except (InvalidGitRepositoryError, Exception):
            pass
        
        return git_info 