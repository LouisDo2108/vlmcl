import logging
import os
from collections import defaultdict
from typing import Optional

import torch
import torch.distributed as dist
from grad_cache.grad_cache import GradCache
from tevatron.hyperbolic.collator import get_dense_rep, split_dense_inputs
from tevatron.hyperbolic.loss import build_contrastive_loss
from tevatron.retriever.trainer import TevatronTrainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import SaveStrategy

logger = logging.getLogger(__name__)

_LOSS_LOG_DECIMALS = 4


class CLIPTrainer(TevatronTrainer):
    """Hugging Face Trainer hook for CLIP contrastive batches (qry, tgt)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signature_columns = []
        self.loss_fn = build_contrastive_loss(
            is_ddp=self.is_ddp,
            temperature=self.model.temperature,
            bidirectional=self.args.bidirectional_loss,
        )
        self._loss_logs = defaultdict(
            lambda: torch.tensor(0.0, device=self.args.device)
        )

    def _accumulate_component_logs(self, losses: Optional[dict[str, float]] = None) -> None:
        losses = losses or getattr(self.loss_fn, "losses", None)
        if not losses:
            return
        scale = 1.0 / self.args.gradient_accumulation_steps
        for name, value in losses.items():
            if name == "loss":
                continue
            self._loss_logs[name] = self._loss_logs[name] + value * scale

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        qry, tgt = inputs
        qry_reps, tgt_reps = model(qry=qry, tgt=tgt)
        loss = self.loss_fn(qry_reps, tgt_reps)
        self._accumulate_component_logs()
        return (loss, qry_reps, tgt_reps) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            logs = {}
            steps = self.state.global_step - self._globalstep_last_logged

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / steps, _LOSS_LOG_DECIMALS)

            for loss_name, accumulated in self._loss_logs.items():
                scalar = self._nested_gather(accumulated).mean().item()
                accumulated -= accumulated
                logs[loss_name] = round(scalar / steps, _LOSS_LOG_DECIMALS)

            if grad_norm is not None:
                logs["grad_norm"] = round(
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm,
                    _LOSS_LOG_DECIMALS,
                )

            if learning_rate is not None:
                lr = (
                    learning_rate.item()
                    if isinstance(learning_rate, torch.Tensor)
                    else learning_rate
                )
            else:
                lr = self._get_learning_rate()
            logs["lr"] = f"{float(lr):.2e}"

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )
            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = "encoder."
        state_dict = {
            k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        self.model.encoder.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class CLIPGradCacheTrainer(CLIPTrainer):
    """GradCache trainer for CLIP (one pass on CLIP-space qry/tgt reps)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=self.loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        queries, targets = self._prepare_inputs(inputs)
        self.gc.models = [model, model]
        loss = self.gc(
            {"qry": queries},
            {"tgt": targets},
            no_sync_except_last=self._use_no_sync(model),
        )
        self._accumulate_component_logs()
        return loss / self._dist_loss_scale_factor

    @staticmethod
    def _use_no_sync(model) -> bool:
        return (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and isinstance(model, torch.nn.parallel.DistributedDataParallel)
        )
