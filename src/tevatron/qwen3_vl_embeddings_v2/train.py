"""
Training script for Qwen3 VL Embeddings V2.

This script demonstrates how to train multimodal embedding models using the
improved Qwen3 VL Embeddings V2 module following Tevatron framework conventions.

Usage:
    python train.py \
        --model_name_or_path <path_to_model> \
        --dataset_name <dataset_name> \
        --output_dir <output_directory> \
        [other arguments]
"""

import logging
import os
import sys
from dataclasses import asdict
from copy import deepcopy

import torch
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model
from transformers.utils.import_utils import is_flash_attn_2_available

from tevatron.colpali.utils import get_params_info, init, write_json

from .arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from .models import Qwen3VLForEmbedding, DenseModel, Qwen3VLProcessor
from .dataset import TrainDataset
from .collator import TrainCollator
from .trainer import Trainer

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Parse arguments
    model_args, data_args, training_args = init(
        ModelArguments, DataArguments, TevatronTrainingArguments
    )
    
    # Set random seed for reproducibility
    set_seed(training_args.seed)
    
    # Initialize logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}"
    )
    
    # Load model and processor
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    model = Qwen3VLForEmbedding.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        ),
        trust_remote_code=True,
    )
    
    processor = Qwen3VLProcessor.from_pretrained(
        model_args.model_name_or_path, 
        padding_side='right'
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
    
    # Configure LoRA
    logger.info("Configuring LoRA adapter")
    
    lora_config = LoraConfig(
        r=model_args.lora_r if hasattr(model_args, 'lora_r') else 8,
        lora_alpha=model_args.lora_alpha if hasattr(model_args, 'lora_alpha') else 8,
        lora_dropout=model_args.lora_dropout if hasattr(model_args, 'lora_dropout') else 0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=model_args.lora_target_modules.split(','),
    )
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    model = get_peft_model(model, lora_config)
    
    # Wrap with DenseModel for retrieval training
    model = DenseModel(
        encoder=model,
        processor=processor,
        max_pixels=786432,  # Mimic ColQwen3
        pooling=model_args.pooling if hasattr(model_args, 'pooling') else "last",
        normalize=model_args.normalize if hasattr(model_args, 'normalize') else True,
        filter_false_negatives=training_args.filter_false_negatives,
        min_pixels=data_args.min_pixels if hasattr(data_args, 'min_pixels') else None,
        max_pixels=data_args.max_pixels if hasattr(data_args, 'max_pixels') else None,
        total_pixels=data_args.total_pixels if hasattr(data_args, 'total_pixels') else None,
        fps=data_args.fps if hasattr(data_args, 'fps') else None,
        max_frames=data_args.max_frames if hasattr(data_args, 'max_frames') else None,
        default_instruction=data_args.default_instruction if hasattr(data_args, 'default_instruction') else None,
    )
    
    # Log model parameters
    get_params_info(model)
    
    # Load dataset
    logger.info("Loading training dataset")
    train_dataset = TrainDataset(data_args)
    
    # Initialize collator
    collator = TrainCollator(data_args)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    
    train_dataset.set_trainer(trainer)
    
    # Check for existing checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    # Start training
    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    
    # Save final model and configurations
    logger.info("Saving final model and configurations")
    
    training_args_to_save = asdict(deepcopy(training_args))
    training_args_to_save.update(asdict(model_args))
    training_args_to_save.update(asdict(data_args))
    
    write_json(
        os.path.join(training_args.output_dir, "full_config.json"),
        training_args_to_save,
    )
    
    processor.save_pretrained(training_args.output_dir)
    
    logger.info(f"Training completed. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
