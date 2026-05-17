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
        --dataset_name <dataset_name> \
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

from .arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from .dataset import EncodeDataset
from .models import Qwen3VLForEmbedding, Qwen3VLProcessor
from .collator import EncodeCollator

logger = logging.getLogger(__name__)


def run_encoding(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TevatronTrainingArguments,
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

    # Load model and processor
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = Qwen3VLForEmbedding.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    processor = Qwen3VLProcessor.from_pretrained(
        model_args.model_name_or_path,
        padding_side='right',
    )
    
    # Configure image processor
    patch_size = getattr(processor.image_processor, "patch_size", None)
    merge_size = getattr(processor.image_processor, "merge_size", None)
    
    if patch_size is None or merge_size is None:
        raise ValueError(
            "Qwen3VL image processor is missing `patch_size` or `merge_size`."
        )
    
    tile = patch_size * merge_size
    processor.image_processor.max_pixels = 768 * tile * tile
    processor.image_processor.size["longest_edge"] = processor.image_processor.max_pixels
    
    model.to(training_args.device)
    model.eval()

    # Load dataset
    logger.info(f"Loading dataset from {data_args.dataset_name}")
    dataset = EncodeDataset(data_args=data_args)

    # Collator
    collator = EncodeCollator(data_args=data_args)

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
            if isinstance(batch, dict):
                batch_inputs = {
                    k: v.to(training_args.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
            else:
                # Handle tuple output from collator (content_ids, messages)
                content_ids, messages = batch
                batch_inputs = messages

            # Forward pass through embedder
            outputs = model.encode(messages=batch_inputs)
            embeddings = outputs

            # Normalize embeddings if required
            if model_args.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Collect results
            all_embeddings.append(embeddings.cpu())
            
            # Collect doc IDs
            if isinstance(batch, tuple) and len(batch) == 2:
                all_doc_ids.extend(batch[0])

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
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    run_encoding(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
