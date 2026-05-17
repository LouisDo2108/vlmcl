import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import nullcontext
from pdb import set_trace as st
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import SaveStrategy, has_length

from tevatron.retriever.trainer import TevatronTrainer

logger = logging.getLogger(__name__)


class Trainer(TevatronTrainer):

    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.other_losses = defaultdict(lambda: torch.tensor(0.0).to(self.args.device))


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()
            # Remove the base_model which is only used for KL loss
            model_state_dict = {
                k: v for k, v in state_dict.items() if k.startswith("base_model.")
            }

            # Remove the encoder of Tevatron's DenseModel wrapper.
            prefix = "encoder."
            model_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
                
            self.model.encoder.save_pretrained(
                output_dir,
                state_dict=model_state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        query, doc = inputs  # query is a dummy list, doc contains (docid, doc)
        outputs = model(query=query, passage=doc)

        loss = outputs.pop("loss")

        for loss_name, some_loss in outputs.items():
            self.other_losses[loss_name] += some_loss / self.args.gradient_accumulation_steps

        return (loss, outputs) if return_outputs else loss

    def recall_at_k(self, scores, target, k=5):
        # scores: [num_queries, num_passages]
        # target: [num_queries] → index of positive passage per query
        topk = scores.topk(k, dim=1).indices  # [num_queries, k]
        correct = (topk == target.unsqueeze(1)).any(dim=1)
        return correct.float()

    def cosine_diagnostics(self, scores, target):
        pos_scores = scores[torch.arange(scores.size(0)), target]
        neg_mask = torch.ones_like(scores, dtype=torch.bool)
        neg_mask[torch.arange(scores.size(0)), target] = False
        neg_scores = scores[neg_mask].view(scores.size(0), -1).mean(dim=-1)

        return pos_scores, neg_scores

    # def eval_step(self, model, inputs):
    #     query, passage = inputs

    #     if isinstance(model.adaptor, torch.nn.Module):
    #         _, q_reps = model.encode_query(query) if query else None
    #         _, p_reps = model.encode_passage(passage) if passage else None
    #     else:
    #         q_reps = model.encode_query(query) if query else None
    #         p_reps = model.encode_passage(passage) if passage else None
        
        
    #     metrics = {}
    #     for m in self.model.matryoshka_dim_list:

    #         scores_semantic = self.model.compute_similarity(q_reps[:, :m], p_reps[:, :m])

    #         num_neg = p_reps.size(0) // q_reps.size(0)
    #         target = torch.arange(
    #             scores_semantic.size(0),
    #             device=scores_semantic.device,
    #             dtype=torch.long,
    #         )
    #         target = target * num_neg

    #         # ---- Add diagnostics ----
    #         with torch.no_grad():
    #             recall1 = self.recall_at_k(scores_semantic, target, k=1)
    #             recall5 = self.recall_at_k(scores_semantic, target, k=5)
    #             pos_sim, neg_sim = self.cosine_diagnostics(scores_semantic, target)

    #             metrics.update({
    #                 f"loss_{m}": F.cross_entropy(scores_semantic / self.model.temperature, target, reduction="none"),
    #                 f"recall@1_{m}": recall1,
    #                 f"recall@5_{m}": recall5,
    #                 f"pos_sim_{m}": pos_sim,
    #                 f"neg_sim_{m}": neg_sim,
    #             })
    #     return metrics

    def training_step(self, *args):
        return (
            super(TevatronTrainer, self).training_step(*args)
            / self._dist_loss_scale_factor
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        pass

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
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                2,
            )
            for loss_name, some_loss in self.other_losses.items():
                some_loss_scalar = self._nested_gather(some_loss).mean().item()
                self.other_losses[loss_name] -= some_loss
                logs[loss_name] = round(
                    some_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    2,
                )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
                logs["grad_norm"] = round(logs["grad_norm"], 2)

            if learning_rate is not None:
                logs["lr"] = learning_rate
            else:
                logs["lr"] = self._get_learning_rate()
            logs["lr"] = f"{learning_rate:.2e}"

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