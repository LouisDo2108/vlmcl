"""
Trainer module for Qwen3 VL Embeddings V2.

This module provides a custom trainer for multimodal retrieval training,
extending Tevatron's trainer with improved logging and checkpoint handling.
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.trainer import Trainer as HFTrainer, TRAINING_ARGS_NAME
from transformers.trainer_utils import SaveStrategy

from tevatron.retriever.trainer import TevatronTrainer

logger = logging.getLogger(__name__)


class Trainer(TevatronTrainer):
    """
    Custom trainer for Qwen3 VL Embeddings V2.
    
    This trainer extends TevatronTrainer with improved loss tracking,
    checkpoint saving, and distributed training support.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the trainer."""
        super().__init__(*args, **kwargs)
        
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.other_losses: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0, device=self.args.device)
        )
    
    def _save(
        self, 
        output_dir: Optional[str] = None, 
        state_dict: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            output_dir: Output directory path.
            state_dict: Optional state dictionary to save.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        if state_dict is None:
            state_dict = self.model.state_dict()
            
            # Filter state dict to only include relevant parameters
            model_state_dict = {
                k: v for k, v in state_dict.items() if k.startswith("encoder.")
            }
            
            # Remove 'encoder.' prefix
            prefix = "encoder."
            model_state_dict = {
                k[len(prefix):]: v 
                for k, v in model_state_dict.items() 
                if k.startswith(prefix)
            }
            
            self.model.encoder.save_pretrained(
                output_dir,
                state_dict=model_state_dict,
                safe_serialization=self.args.save_safetensors,
            )
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def compute_loss(
        self, 
        model: torch.nn.Module, 
        inputs: Any, 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute training loss.
        
        Args:
            model: The model being trained.
            inputs: Batch of input data.
            return_outputs: Whether to return model outputs.
            num_items_in_batch: Number of items in batch (for gradient accumulation).
            
        Returns:
            Loss tensor or tuple of (loss, outputs).
        """
        query, passage = inputs
        outputs = model(query=query, passage=passage)
        
        # Extract loss from outputs
        if isinstance(outputs, dict):
            loss = outputs.pop("loss")
            
            # Track additional losses
            for loss_name, some_loss in outputs.items():
                self.other_losses[loss_name] += (
                    some_loss / self.args.gradient_accumulation_steps
                )
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None
        
        return (loss, outputs) if return_outputs else loss
    
    @staticmethod
    def recall_at_k(
        scores: torch.Tensor, 
        target: torch.Tensor, 
        k: int = 5
    ) -> torch.Tensor:
        """
        Compute recall@k metric.
        
        Args:
            scores: Similarity scores [num_queries, num_passages].
            target: Indices of positive passages [num_queries].
            k: Number of top results to consider.
            
        Returns:
            Binary tensor indicating correct predictions.
        """
        topk = scores.topk(k, dim=1).indices
        correct = (topk == target.unsqueeze(1)).any(dim=1)
        return correct.float()
    
    @staticmethod
    def cosine_diagnostics(
        scores: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine similarity diagnostics.
        
        Args:
            scores: Similarity scores.
            target: Indices of positive passages.
            
        Returns:
            Tuple of (positive_scores, negative_scores).
        """
        pos_scores = scores[torch.arange(scores.size(0)), target]
        
        neg_mask = torch.ones_like(scores, dtype=torch.bool)
        neg_mask[torch.arange(scores.size(0)), target] = False
        neg_scores = scores[neg_mask].view(scores.size(0), -1).mean(dim=-1)
        
        return pos_scores, neg_scores
    
    def training_step(
        self, 
        model: torch.nn.Module, 
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            model: The model being trained.
            inputs: Batch of input data.
            
        Returns:
            Scaled loss tensor.
        """
        return (
            super().training_step(model, inputs) 
            / self._dist_loss_scale_factor
        )
    
    def _load_from_checkpoint(
        self, 
        resume_from_checkpoint: str, 
        model: Optional[torch.nn.Module] = None
    ):
        """
        Load model from checkpoint.
        
        Note: This is overridden to disable automatic checkpoint loading.
        """
        pass
    
    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: Optional[torch.Tensor],
        model: torch.nn.Module,
        trial: Any,
        epoch: float,
        ignore_keys_for_eval: Optional[List[str]],
        start_time: float,
        learning_rate: Optional[float] = None,
    ):
        """
        Log metrics, save checkpoints, and evaluate.
        
        Args:
            tr_loss: Training loss.
            grad_norm: Gradient norm.
            model: The model being trained.
            trial: Trial object for hyperparameter optimization.
            epoch: Current epoch.
            ignore_keys_for_eval: Keys to ignore during evaluation.
            start_time: Start time of training.
            learning_rate: Current learning rate.
        """
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            logs: Dict[str, float] = {}
            
            # Gather and average loss across processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            
            logs["loss"] = round(
                tr_loss_scalar 
                / (self.state.global_step - self._globalstep_last_logged),
                2,
            )
            
            # Log additional losses
            for loss_name, some_loss in self.other_losses.items():
                some_loss_scalar = self._nested_gather(some_loss).mean().item()
                self.other_losses[loss_name] -= some_loss
                logs[loss_name] = round(
                    some_loss_scalar 
                    / (self.state.global_step - self._globalstep_last_logged),
                    2,
                )
            
            # Log gradient norm
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item() 
                    if isinstance(grad_norm, torch.Tensor) 
                    else grad_norm
                )
                logs["grad_norm"] = round(logs["grad_norm"], 2)
            
            # Log learning rate
            if learning_rate is not None:
                logs["lr"] = f"{learning_rate:.2e}"
            else:
                logs["lr"] = f"{self._get_learning_rate():.2e}"
            
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            
            self.log(logs, start_time)
        
        # Evaluation
        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )
            
            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric
        
        # Checkpoint saving
        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )
