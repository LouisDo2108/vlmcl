"""
CCLIP training: two GradCache steps, both using SimpleContrastiveLoss.

  1. encode_clip_input      -> loss on (x, y)
  2. encode_projected_input -> loss on (projected_x, projected_y)

Use with train_cclip.py (separate from train.py).
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from grad_cache.grad_cache import GradCache
from tevatron.hyperbolic.collator import get_dense_rep, split_dense_inputs
from tevatron.hyperbolic.loss import build_contrastive_loss
from tevatron.hyperbolic.trainer import CLIPGradCacheTrainer, CLIPTrainer


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _set_gc_encode_method(model: nn.Module, method_name: str) -> None:
    _unwrap_model(model)._gc_encode_method = method_name


def _prefix_losses(losses: dict[str, float], prefix: str) -> dict[str, float]:
    out = {}
    for key, value in losses.items():
        if key == "loss":
            continue
        out[f"{prefix}_{key}"] = value
    return out


class CCLIPTrainer(CLIPTrainer):
    """Non–GradCache: two forward passes, same contrastive loss class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_ckc_fn = build_contrastive_loss(
            is_ddp=self.is_ddp,
            temperature=self.model.temperature,
            bidirectional=self.args.bidirectional_loss,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        qry, tgt = inputs
        base = _unwrap_model(model)
        qry_clip = base.encode_clip_input(qry)
        tgt_clip = base.encode_clip_input(tgt)
        qry_proj = base.encode_projected_input(qry)
        tgt_proj = base.encode_projected_input(tgt)

        loss_clip = self.loss_fn(qry_clip, tgt_clip)
        loss_ckc = self.loss_ckc_fn(qry_proj, tgt_proj)
        loss = loss_clip + loss_ckc

        merged = {}
        merged.update(_prefix_losses(self.loss_fn.losses, "clip"))
        merged.update(_prefix_losses(self.loss_ckc_fn.losses, "ckc"))
        self._accumulate_component_logs(merged)

        outputs = (qry_clip, tgt_clip)
        return (loss, outputs) if return_outputs else loss


class CCLIPGradCacheTrainer(CCLIPTrainer):
    """Two GradCache steps; reuses SimpleContrastiveLoss for CLIP and projected reps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gc_kwargs = dict(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )
        self.gc_clip = GradCache(loss_fn=self.loss_fn, **gc_kwargs)
        self.gc_ckc = GradCache(loss_fn=self.loss_ckc_fn, **gc_kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        queries, targets = self._prepare_inputs(inputs)
        use_no_sync = CLIPGradCacheTrainer._use_no_sync(model)
        self.gc_clip.models = [model, model]
        self.gc_ckc.models = [model, model]

        _set_gc_encode_method(model, "encode_clip_input")
        loss_clip = self.gc_clip(
            {"qry": queries},
            {"tgt": targets},
            no_sync_except_last=use_no_sync,
        )

        _set_gc_encode_method(model, "encode_projected_input")
        loss_ckc = self.gc_ckc(
            {"qry": queries},
            {"tgt": targets},
            no_sync_except_last=use_no_sync,
        )

        merged = {}
        merged.update(_prefix_losses(self.loss_fn.losses, "clip"))
        merged.update(_prefix_losses(self.loss_ckc_fn.losses, "ckc"))
        self._accumulate_component_logs(merged)

        return (loss_clip + loss_ckc) / self._dist_loss_scale_factor
