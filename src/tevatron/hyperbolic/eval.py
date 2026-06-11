import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
from datasets import load_dataset
from tevatron.hyperbolic.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from tevatron.hyperbolic.collator import CLIPEvalCollator
from tevatron.hyperbolic.dataset import EvalDataset
from tevatron.hyperbolic.metrics import RankingMetrics
from tevatron.hyperbolic.model import CLIPContrastiveModel
from tevatron.hyperbolic.train import load_clip_processor
from tevatron.hyperbolic.utils import batch_to_device, init, print_master, print_rank
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def encode_eval_dataset(
    model,
    loader,
    device,
    output_path: str,
    paired_data,
    *,
    is_query: bool,
    desc: str,
) -> torch.Tensor:
    """Encode query or target loader and save (embeddings, paired_data) pickle."""
    encoded = []
    device_type = device.type if isinstance(device, torch.device) else "cuda"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch = batch_to_device(batch, device)
                output = model(qry=batch) if is_query else model(tgt=batch)
                key = "qry_reps" if is_query else "tgt_reps"
                encoded.append(output[key].detach().cpu())
    encoded = torch.cat(encoded, dim=0).to(torch.bfloat16)

    with open(output_path, "wb") as f:
        pickle.dump((encoded, paired_data), f)
    
    return encoded, paired_data


def _pair_key(text, img_path):
    return (text, img_path)


def _embeddings_to_dict(tensor, index):
    return {_pair_key(m["text"], m["img_path"]): t for t, m in zip(tensor, index)}


def _to_float32(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.float().numpy()
    return np.asarray(x, dtype=np.float32)


def build_ranking_test_cases(eval_data, qry_dict, tgt_dict, *, normalize: bool):
    """
    One test case per MMEB row: rank row-specific candidates; GT is index 0.

    RankingMetrics expects hashable doc IDs in prediction/label lists.
    """
    test_cases = []
    for row in eval_data:
        qry_key = _pair_key(row["qry_text"], row["qry_img_path"])
        qry_t = _to_float32(qry_dict[qry_key])
        cand_keys = [_pair_key(t, p) for t, p in zip(row["tgt_text"], row["tgt_img_path"])]
        tgt_t = np.stack([_to_float32(tgt_dict[k]) for k in cand_keys], axis=0)
        if normalize:
            qry_norm = np.linalg.norm(qry_t)
            tgt_norms = np.linalg.norm(tgt_t, axis=1)
            scores = np.dot(tgt_t, qry_t) / (tgt_norms * qry_norm + 1e-8)
        else:
            scores = np.dot(tgt_t, qry_t)
        ranked = np.argsort(-scores)
        test_cases.append(
            {
                "prediction": [cand_keys[i] for i in ranked],
                "label": [cand_keys[0]],
                "rel_scores": None,
            }
        )
    return test_cases


def main():
    for arg in list(sys.argv):
        if arg.startswith("--local-rank="):
            rank = arg.split("=", 1)[1]
            sys.argv.remove(arg)
            sys.argv.extend(["--local_rank", rank])
    model_args, data_args, training_args = init(
        ModelArguments, DataArguments, TrainingArguments, verbose=False
    )
    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU eval is not supported.")
    
    output_dir = training_args.output_dir
    device = torch.device(training_args.device)

    processor = load_clip_processor(model_args)
    if model_args.lora_name_or_path:
        model = CLIPContrastiveModel.load_merged_adapters(model_args)
    else:
        model = CLIPContrastiveModel.load(model_args)
    model.eval()
    model = model.to(device, dtype=torch.bfloat16)
    model.encoder = torch.compile(model.encoder) # type: ignore

    eval_collator = CLIPEvalCollator(data_args=data_args, processor=processor)
    metrics = RankingMetrics(metric_list=["hit", "ndcg"], k_list=(1, 5, 10))

    for idx, subset in enumerate(data_args.subset_name):
        print_rank(f"{idx + 1}/{len(data_args.subset_name)}: {subset}")
        encode_qry_path = os.path.join(output_dir, f"{subset}_qry.pkl")
        encode_tgt_path = os.path.join(output_dir, f"{subset}_tgt.pkl")
        score_path = os.path.join(output_dir, f"{subset}_score.json")

        eval_data = load_dataset("TIGER-Lab/MMEB-eval", subset, split="test")

        # Disable caching
        # pickles_exist = os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path)
        # if pickles_exist:
        #     print_master(f"Loading cached embeddings for {subset}")
        #     with open(encode_qry_path, "rb") as f:
        #         qry_tensor, qry_index = pickle.load(f)
        #     with open(encode_tgt_path, "rb") as f:
        #         tgt_tensor, tgt_index = pickle.load(f)
        # else:
        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
            eval_data=eval_data,
        )
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
            eval_data=eval_data,
        )
        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        qry_tensor, qry_index = encode_eval_dataset(
            model,
            eval_qry_loader,
            device,
            encode_qry_path,
            eval_qry_dataset.paired_data,
            is_query=True,
            desc=f"{subset} encode query",
        )
        tgt_tensor, tgt_index = encode_eval_dataset(
            model,
            eval_tgt_loader,
            device,
            encode_tgt_path,
            eval_tgt_dataset.paired_data,
            is_query=False,
            desc=f"{subset} encode target",
        )

        qry_dict = _embeddings_to_dict(qry_tensor, qry_index)
        tgt_dict = _embeddings_to_dict(tgt_tensor, tgt_index)

        test_cases = build_ranking_test_cases(
            eval_data, qry_dict, tgt_dict, normalize=model_args.normalize
        )

        score_dict = metrics.evaluate(test_cases)
        for k, v in score_dict.items():
            score_dict[k] = round(v, 4)
        
        # Print nDCG@1, nDCG@5, nDCG@10
        print_master(f"nDCG@1: {score_dict['ndcg_linear@1']}")
        print_master(f"nDCG@5: {score_dict['ndcg_linear@5']}")
        print_master(f"nDCG@10: {score_dict['ndcg_linear@10']}")
        
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(score_dict, f, indent=4)


if __name__ == "__main__":
    main()
