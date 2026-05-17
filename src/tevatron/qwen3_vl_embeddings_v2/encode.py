#!/usr/bin/env python
# coding=utf-8
"""
Encoding script for Qwen3-VL embeddings.

This module provides functionality to encode text and image inputs into embeddings
using the Qwen3-VL model. It follows the Tevatron framework conventions for
distributed inference and output formatting.

Usage:
    python -m src.tevatron.qwen3_vl_embeddings_v2.encode \
        --output_dir <output_dir> \
        --model_name_or_path <model_path> \
        --data <data_path> \
        --batch_size <batch_size> \
        --fp16
"""

import logging
import os
import sys
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser

from .arguments import Qwen3VLDataArguments, Qwen3VLModelArguments, Qwen3VLTrainingArguments
from .dataset import Qwen3VLEvalDataset
from .modeling_qwen3_vl import Qwen3VLForEmbedding
from .collator import Qwen3VLEvalCollator

logger = logging.getLogger(__name__)


def run_encoding(
    model_args: Qwen3VLModelArguments,
    data_args: Qwen3VLDataArguments,
    training_args: Qwen3VLTrainingArguments,
) -> None:
    """
    Run the encoding process to generate embeddings.

    Args:
        model_args: Model configuration arguments.
        data_args: Data configuration arguments.
        training_args: Training/configuration arguments.
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.should_log else logging.WARN,
    )
    logger.warning(f"Process rank: {training_args.local_rank}, device: {training_args.device}")

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = Qwen3VLForEmbedding.from_pretrained(
        model_args.model_name_or_path,
        config_name=model_args.config_name,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    model.to(training_args.device)
    model.eval()

    # Load dataset
    logger.info(f"Loading dataset from {data_args.data}")
    dataset = Qwen3VLEvalDataset(
        data_path=data_args.data,
        processor_name_or_path=model_args.model_name_or_path,
        max_len=data_args.max_len,
    )

    # Collator
    collator = Qwen3VLEvalCollator(
        processor=dataset.processor,
        max_length=data_args.max_len,
    )

    # DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    # Encoding loop
    all_embeddings = []
    all_doc_ids = []

    logger.info("Starting encoding...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Encoding", disable=not training_args.is_local_process_zero()):
            # Move batch to device
            batch_inputs = {k: v.to(training_args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch_inputs)
            embeddings = outputs.embeddings  # type: ignore[attr-defined]

            # Normalize embeddings if required
            if model_args.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Collect results
            all_embeddings.append(embeddings.cpu())
            
            # Extract doc IDs if available
            if "doc_id" in batch:
                all_doc_ids.extend(batch["doc_id"])
            elif "_id" in batch:
                all_doc_ids.extend(batch["_id"])

    # Concatenate results
    if all_embeddings:
        all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    else:
        logger.warning("No embeddings generated.")
        return

    # Save results
    if training_args.is_local_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # Save embeddings
        embedding_path = os.path.join(training_args.output_dir, "embeddings.pt")
        torch.save(all_embeddings_tensor, embedding_path)
        logger.info(f"Saved embeddings to {embedding_path}")

        # Save doc IDs if available
        if all_doc_ids:
            id_path = os.path.join(training_args.output_dir, "doc_ids.txt")
            with open(id_path, "w", encoding="utf-8") as f:
                for doc_id in all_doc_ids:
                    f.write(f"{doc_id}\n")
            logger.info(f"Saved doc IDs to {id_path}")

    logger.info("Encoding completed successfully.")


def main():
    """Main entry point for the encoding script."""
    parser = HfArgumentParser((Qwen3VLModelArguments, Qwen3VLDataArguments, Qwen3VLTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    run_encoding(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
