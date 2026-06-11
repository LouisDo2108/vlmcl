"""Precompute and index old-checkpoint CLIP embeddings for CKC training."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from tevatron.hyperbolic.collator import CLIPCollator
from tevatron.hyperbolic.dataset import CLIPTrainDataset
from tevatron.hyperbolic.utils import batch_to_device, print_master, print_rank
from torch.utils.data import DataLoader
from tqdm import tqdm


def cat_qry_tgt_by_chunk(
    q_reps: torch.Tensor, t_reps: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Interleave qry/tgt in chunk-sized blocks: [q0..q_{c-1}, t0..t_{c-1}, ...]."""
    if q_reps.size(0) != t_reps.size(0):
        raise ValueError(
            f"qry/tgt rep counts must match, got {q_reps.size(0)} and {t_reps.size(0)}."
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    chunks = []
    for i in range(0, q_reps.size(0), chunk_size):
        chunks.append(
            torch.cat([q_reps[i : i + chunk_size], t_reps[i : i + chunk_size]], dim=0)
        )
    return torch.cat(chunks, dim=0)


@dataclass(frozen=True)
class OldEmbeddingCacheMeta:
    num_samples: int
    rep_dim: int
    old_checkpoint_path: str
    subset_name: list[str]
    image_dir: str
    num_sample_per_subset: int | None

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "num_samples": self.num_samples,
                "old_checkpoint_path": self.old_checkpoint_path,
                "subset_name": self.subset_name,
                "image_dir": self.image_dir,
                "num_sample_per_subset": self.num_sample_per_subset,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_dataset(
        cls,
        dataset: CLIPTrainDataset,
        old_checkpoint_path: str,
        *,
        rep_dim: int = 0,
    ) -> OldEmbeddingCacheMeta:
        return cls(
            num_samples=len(dataset),
            rep_dim=rep_dim,
            old_checkpoint_path=old_checkpoint_path,
            subset_name=list(dataset.data_args.subset_name),
            image_dir=dataset.data_args.image_dir,
            num_sample_per_subset=dataset.data_args.num_sample_per_subset,
        )


class OldEmbeddingCache:
    """Row-aligned qry/tgt CLIP reps from a frozen old checkpoint."""

    def __init__(
        self,
        qry_reps: torch.Tensor,
        tgt_reps: torch.Tensor,
        meta: OldEmbeddingCacheMeta,
    ):
        if qry_reps.shape != tgt_reps.shape:
            raise ValueError(
                f"qry/tgt cache shapes must match, got {qry_reps.shape} and {tgt_reps.shape}."
            )
        self.qry_reps = qry_reps
        self.tgt_reps = tgt_reps
        self.meta = meta

    def __len__(self) -> int:
        return self.qry_reps.size(0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "qry_reps": self.qry_reps,
                "tgt_reps": self.tgt_reps,
                "meta": self.meta,
            },
            path,
        )
        print_master(f"Saved old embedding cache to {path} ({len(self)} rows)")

    @classmethod
    def load(cls, path: str, *, expected_meta: OldEmbeddingCacheMeta | None = None) -> OldEmbeddingCache:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cache = cls(
            qry_reps=payload["qry_reps"],
            tgt_reps=payload["tgt_reps"],
            meta=payload["meta"],
        )
        if expected_meta is not None and cache.meta.fingerprint() != expected_meta.fingerprint():
            raise ValueError(
                f"Old embedding cache at {path} does not match current dataset/checkpoint "
                f"(expected {expected_meta.fingerprint()}, got {cache.meta.fingerprint()})."
            )
        print_rank(f"Loaded old embedding cache from {path} ({len(cache)} rows)")
        return cache

    def get_ckc_old_reps(
        self,
        indices: torch.Tensor,
        q_chunk: int,
        device: torch.device,
    ) -> torch.Tensor:
        if indices.dtype != torch.long:
            indices = indices.to(dtype=torch.long)
        indices = indices.cpu()
        old_q = self.qry_reps.index_select(0, indices).to(device=device, non_blocking=True)
        old_t = self.tgt_reps.index_select(0, indices).to(device=device, non_blocking=True)
        return cat_qry_tgt_by_chunk(old_q, old_t, q_chunk)


def default_cache_path(output_dir: str, meta: OldEmbeddingCacheMeta) -> str:
    return os.path.join(output_dir, f"old_embeddings_{meta.fingerprint()}.pt")


def _encode_batches(
    old_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
    rep_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qry_rows = torch.empty((num_samples, rep_dim), dtype=torch.bfloat16)
    tgt_rows = torch.empty((num_samples, rep_dim), dtype=torch.bfloat16)
    device_type = device.type if isinstance(device, torch.device) else "cuda"

    old_model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            for qry_batch, tgt_batch, batch_indices in tqdm(
                loader, desc="Precomputing old checkpoint embeddings"
            ):
                qry_batch = batch_to_device(qry_batch, device)
                tgt_batch = batch_to_device(tgt_batch, device)
                old_q = old_model.encode_clip_input(qry_batch).detach().cpu() # type: ignore
                old_t = old_model.encode_clip_input(tgt_batch).detach().cpu() # type: ignore
                qry_rows[batch_indices] = old_q.to(dtype=torch.bfloat16)
                tgt_rows[batch_indices] = old_t.to(dtype=torch.bfloat16)

    return qry_rows, tgt_rows


def build_old_embedding_cache(
    old_model: nn.Module,
    dataset: CLIPTrainDataset,
    collator: CLIPCollator,
    *,
    device: torch.device,
    batch_size: int,
    old_checkpoint_path: str,
    output_dir: str,
    cache_path: str | None = None,
    num_workers: int = 0,
) -> OldEmbeddingCache:
    meta = OldEmbeddingCacheMeta.from_dataset(
        dataset,
        old_checkpoint_path,
        rep_dim=old_model.rep_dim, # type: ignore
    )
    resolved_path = cache_path or default_cache_path(output_dir, meta)

    if os.path.isfile(resolved_path):
        return OldEmbeddingCache.load(resolved_path, expected_meta=meta)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        indexed_dataset = CLIPTrainDataset(dataset.data_args, return_index=True)
        indexed_collator = CLIPCollator(
            data_args=collator.data_args,
            processor=collator.processor,
            return_indices=True,
        )
        loader = DataLoader(
            indexed_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=indexed_collator,
        )
        qry_rows, tgt_rows = _encode_batches(
            old_model,
            loader,
            device,
            num_samples=len(indexed_dataset),
            rep_dim=meta.rep_dim,
        )
        cache = OldEmbeddingCache(qry_reps=qry_rows, tgt_reps=tgt_rows, meta=meta)
        cache.save(resolved_path)

    if dist.is_initialized():
        dist.barrier()

    return OldEmbeddingCache.load(resolved_path, expected_meta=meta)
