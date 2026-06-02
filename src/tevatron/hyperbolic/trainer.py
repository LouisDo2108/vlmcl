import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from grad_cache.grad_cache import GradCache
from tevatron.hyperbolic.collator import get_dense_rep, split_dense_inputs
from tevatron.hyperbolic.loss import DistributedContrastiveLoss, SimpleContrastiveLoss
from tevatron.hyperbolic.model import CLIPContrastiveModel
from tevatron.retriever.trainer import TevatronTrainer
from transformers.trainer import TRAINING_ARGS_NAME

logger = logging.getLogger(__name__)


class CLIPTrainer(TevatronTrainer):
    """Hugging Face Trainer hook for CLIP contrastive batches (qry, tgt)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signature_columns = []
        loss_fn_cls = (
            DistributedContrastiveLoss if self.is_ddp else SimpleContrastiveLoss
        )
        self.loss_fn = loss_fn_cls(temperature=self.model.temperature)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        qry, tgt = inputs
        qry_reps, tgt_reps = model(qry=qry, tgt=tgt)
        loss = self.loss_fn(qry_reps, tgt_reps)
        return (loss, qry_reps, tgt_reps) if return_outputs else loss

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
    """GradCache trainer for large CLIP batches (pre-tokenized collator output)."""

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
        queries, targets = inputs
        self.gc.models = [model, model]
        use_no_sync = (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and all(
                isinstance(m, torch.nn.parallel.DistributedDataParallel)
                for m in self.gc.models
            )
        )
        # Pass inner CLIP batch dicts directly so each GradCache stream chunk is:
        # {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
        loss = self.gc({"qry": queries}, {"tgt": targets}, no_sync_except_last=use_no_sync)
        # loss = self.gc(queries, targets, no_sync_except_last=use_no_sync)
        return loss / self._dist_loss_scale_factor