import os

"""
CCLIP training: two GradCache steps, both using SimpleContrastiveLoss.

  1. CLIP (self.gc): qry-vs-tgt contrastive on CLIP embeddings; reps captured.
  2. CKC  (self.gc_ckc): reuses step-1 CLIP reps, applies projector only:
       x = interleaved cat(proj(qry_clip), proj(tgt_clip))  [slot 1]
       y = interleaved cat(cached old_qry, cached old_tgt)    [slot 2, frozen]

Use with train_cclip.py (separate from train.py).
"""

import torch
import torch.nn as nn
from grad_cache.grad_cache import GradCache
from tevatron.hyperbolic.collator import get_dense_rep
from tevatron.hyperbolic.loss import build_ckc_loss
from tevatron.hyperbolic.model import CCLIP, HyperbolicCCLIP
from tevatron.hyperbolic.old_embedding_cache import (
    OldEmbeddingCache,
    cat_qry_tgt_by_chunk,
)
from tevatron.hyperbolic.trainer import CLIPGradCacheTrainer, CLIPTrainer


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class _CapturingGradCache(GradCache):
    """GradCache that stores no-grad encoder reps before the loss backward pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_reps: tuple[torch.Tensor, ...] | None = None

    def cache_step(self, *model_inputs, no_sync_except_last=False, **loss_kwargs):
        all_reps = []
        all_rnd_states = []

        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'

        model_inputs = [self.split_inputs(x, chunk_size) for x, chunk_size in zip(model_inputs, self.chunk_sizes)]

        for model, x in zip(self.models, model_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        self.last_reps = tuple(all_reps)

        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for model, x, model_cache, rnd_states in zip(
                self.models, model_inputs, cache, all_rnd_states):
            self.forward_backward(model, x, model_cache, rnd_states, no_sync_except_last=no_sync_except_last)

        return loss


class _ProjectCKCEncoder(nn.Module):
    """GradCache slot-1: project cached CLIP reps (no ViT re-encode)."""

    def __init__(self, cclip: CCLIP):
        super().__init__()
        self.cclip = cclip

    def forward(
        self,
        qry_reps: torch.Tensor | None = None,
        tgt_reps: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if qry_reps is None or tgt_reps is None:
            raise ValueError("CKC projector slot requires qry_reps and tgt_reps.")
        return self.cclip.project_ckc_from_clip_reps(qry_reps, tgt_reps)


class _OldRepEncoder(nn.Module):
    """GradCache slot-2 stand-in for frozen old reps (no trainable parameters).

    Values stay detached from the old checkpoint. A fresh leaf with
    requires_grad=True is returned on the re-forward pass so GradCache's
    surrogate backward can run without updating any old weights.
    """

    def forward(self, old_reps: torch.Tensor, **kwargs) -> torch.Tensor:
        return old_reps.detach().requires_grad_(True)


def split_ckc_inputs(model_input: dict, chunk_size: int):
    """Split inputs for gc_ckc (both slots use rep row count as chunk_size).

    Slot 1: cached CLIP rep tensors split with chunk_size // 2; projector output
      has chunk_size rows (interleaved proj_q, proj_t).

    Slot 2: frozen interleaved old reps split directly by chunk_size.
    """
    if "old_reps" in model_input:
        return [
            {"old_reps": chunk}
            for chunk in model_input["old_reps"].split(chunk_size, dim=0)
        ]

    q_chunk = chunk_size // 2
    if chunk_size != 2 * q_chunk:
        raise ValueError(f"CKC rep chunk_size must be even, got {chunk_size}.")
    q_chunks = model_input["qry_reps"].split(q_chunk, dim=0)
    t_chunks = model_input["tgt_reps"].split(q_chunk, dim=0)
    return [
        {"qry_reps": qc, "tgt_reps": tc}
        for qc, tc in zip(q_chunks, t_chunks)
    ]


def get_ckc_rep(x):
    return x if isinstance(x, torch.Tensor) else get_dense_rep(x)


def _prefix_losses(losses: dict[str, float], prefix: str) -> dict[str, float]:
    out = {}
    for key, value in losses.items():
        if key == "loss":
            continue
        out[f"{prefix}_{key}"] = value
    return out


def _unpack_inputs(inputs):
    if len(inputs) == 3:
        return inputs[0], inputs[1], inputs[2]
    return inputs[0], inputs[1], None


def _build_ckc_new_reps_from_clip(
    cclip: CCLIP, qry_clip: torch.Tensor, tgt_clip: torch.Tensor, q_chunk: int
) -> torch.Tensor:
    return cat_qry_tgt_by_chunk(
        cclip._project(qry_clip),
        cclip._project(tgt_clip),
        q_chunk,
    )


class CCLIPTrainer(CLIPTrainer):
    """Non-GradCache path: CLIP loss + optional CKC loss when old embeddings are cached."""

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        output_dir = output_dir or self.args.output_dir
        torch.save(
            _unwrap_model(self.model).cclip_projector.state_dict(),
            os.path.join(output_dir, "cclip_projector.pt"),
        )

    def __init__(self, *args, **kwargs):
        self.old_embedding_cache = kwargs.pop("old_embedding_cache", None)
        super().__init__(*args, **kwargs)
        self.loss_ckc_fn = build_contrastive_loss(
            is_ddp=self.is_ddp,
            temperature=self.model.temperature,
            bidirectional=self.args.bidirectional_loss,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        qry, tgt, indices = _unpack_inputs(inputs)
        base = _unwrap_model(model)
        qry_clip = base.encode_clip_input(qry)
        tgt_clip = base.encode_clip_input(tgt)
        loss = self.loss_fn(qry_clip, tgt_clip)

        merged = {}
        merged.update(_prefix_losses(self.loss_fn.losses, "clip"))
        if self.old_embedding_cache is not None:
            if indices is None:
                raise ValueError(
                    "CKC training requires dataset indices; enable return_index on "
                    "CLIPTrainDataset and return_indices on CLIPCollator."
                )
            q_chunk = qry_clip.size(0)
            old_reps = self.old_embedding_cache.get_ckc_old_reps(
                indices,
                q_chunk,
                qry_clip.device,
            )
            new_reps = _build_ckc_new_reps_from_clip(
                base, qry_clip, tgt_clip, q_chunk
            )
            loss_ckc = self.loss_ckc_fn(new_reps, old_reps)
            loss = loss + loss_ckc
            merged.update(_prefix_losses(self.loss_ckc_fn.losses, "ckc"))

        self._accumulate_component_logs(merged)
        outputs = (qry_clip, tgt_clip)
        return (loss, outputs) if return_outputs else loss


class CCLIPGradCacheTrainer(CLIPGradCacheTrainer):
    """CLIP GradCache step + optional CKC GradCache step against cached old embeddings."""

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        output_dir = output_dir or self.args.output_dir
        torch.save(
            _unwrap_model(self.model).cclip_projector.state_dict(),
            os.path.join(output_dir, "cclip_projector.pt"),
        )

    def __init__(self, *args, **kwargs):
        self.old_embedding_cache: OldEmbeddingCache | None = kwargs.pop(
            "old_embedding_cache", None
        )
        super().__init__(*args, **kwargs)
        self.loss_ckc_fn = build_contrastive_loss(
            is_ddp=self.is_ddp,
            temperature=self.model.temperature,
            bidirectional=self.args.bidirectional_loss,
        )
        clip_gc = self.gc
        self.gc = _CapturingGradCache(
            models=clip_gc.models,
            chunk_sizes=clip_gc.chunk_sizes,
            loss_fn=clip_gc.loss_fn,
            split_input_fn=clip_gc.split_input_fn,
            get_rep_fn=clip_gc.get_rep_fn,
            fp16=clip_gc.fp16,
            scaler=clip_gc.scaler,
        )
        self._old_rep_encoder = _OldRepEncoder()
        self._project_ckc_encoder = _ProjectCKCEncoder(_unwrap_model(self.model))
        q_chunk = self.args.gc_q_chunk_size
        rep_chunk = 2 * q_chunk
        self.gc_ckc = GradCache(
            models=[self._project_ckc_encoder, self._old_rep_encoder],
            chunk_sizes=[rep_chunk, rep_chunk],
            loss_fn=self.loss_ckc_fn,
            split_input_fn=split_ckc_inputs,
            get_rep_fn=get_ckc_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        queries, targets, indices = _unpack_inputs(self._prepare_inputs(inputs))
        no_sync = self._use_no_sync(model)

        # Step 1: CLIP GradCache; last_reps = (qry_clip, tgt_clip) for CKC reuse.
        self.gc.models = [model, model]
        loss_clip = self.gc(
            {"qry": queries},
            {"tgt": targets},
            no_sync_except_last=no_sync,
        )

        loss_ckc = torch.zeros((), device=loss_clip.device, dtype=loss_clip.dtype)
        if self.old_embedding_cache is not None:
            if indices is None:
                raise ValueError(
                    "CKC training requires dataset indices; enable return_index on "
                    "CLIPTrainDataset and return_indices on CLIPCollator."
                )
            if self.gc.last_reps is None or len(self.gc.last_reps) != 2:
                raise RuntimeError("CLIP GradCache did not capture qry/tgt reps for CKC.")
            qry_clip_reps, tgt_clip_reps = self.gc.last_reps
            qry_clip_reps = qry_clip_reps.to(dtype=torch.bfloat16)
            tgt_clip_reps = tgt_clip_reps.to(dtype=torch.bfloat16)

            cclip = _unwrap_model(model)
            self._project_ckc_encoder.cclip = cclip
            self.gc_ckc.models = [self._project_ckc_encoder, self._old_rep_encoder]
            old_reps = self.old_embedding_cache.get_ckc_old_reps(
                indices,
                self.args.gc_q_chunk_size,
                qry_clip_reps.device,
            )

            loss_ckc = self.gc_ckc(
                {"qry_reps": qry_clip_reps, "tgt_reps": tgt_clip_reps},
                {"old_reps": old_reps},
                no_sync_except_last=False,
            )

        merged = {}
        merged.update(_prefix_losses(self.loss_fn.losses, "clip"))
        if self.old_embedding_cache is not None:
            merged.update(_prefix_losses(self.loss_ckc_fn.losses, "ckc"))
        self._accumulate_component_logs(merged)

        return (loss_clip + loss_ckc) / self._dist_loss_scale_factor
