"""
Arguments module for Qwen3 VL Embeddings V2.

This module defines configuration classes for model, data, and training arguments,
following Tevatron framework conventions with improved type hints and documentation.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tevatron.retriever.arguments import DataArguments as TevatronDataArguments
from tevatron.retriever.arguments import ModelArguments as TevatronModelArguments
from transformers import TrainingArguments


def default_field(obj):
    """Create a dataclass field with a deep-copied default factory."""
    return field(default_factory=lambda: copy.deepcopy(obj))


@dataclass
class ModelArguments(TevatronModelArguments):
    """
    Model arguments for Qwen3 VL Embeddings V2.
    
    Extends Tevatron's ModelArguments with LoRA-specific configurations.
    """
    
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA adaptation."}
    )
    
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension (rank)."}
    )
    
    lora_alpha: int = field(
        default=8,
        metadata={"help": "LoRA alpha scaling parameter."}
    )
    
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA layers."}
    )
    
    pooling: str = field(
        default="last",
        metadata={
            "help": "Pooling strategy for embeddings. Options: 'cls', 'mean', 'last'."
        }
    )
    
    normalize: bool = field(
        default=True,
        metadata={"help": "Whether to L2-normalize embeddings."}
    )
    
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature scaling for contrastive loss."}
    )


@dataclass
class DataArguments(TevatronDataArguments):
    """
    Data arguments for Qwen3 VL Embeddings V2.
    
    Extends Tevatron's DataArguments with multimodal-specific configurations.
    """
    
    query_max_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum sequence length for query after tokenization."
        },
    )
    
    passage_max_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length for passage after tokenization."
        },
    )
    
    num_proc: int = field(
        default=1,
        metadata={"help": "Number of processes for dataset loading."}
    )
    
    padding_side: str = field(
        default='left',
        metadata={"help": "Padding side for tokenizer ('left' or 'right')."}
    )
    
    colpali_source: str = field(
        default="all",
        metadata={
            "help": "Filter ColPali dataset by source. Use 'all' for no filtering."
        }
    )
    
    # Vision-specific arguments
    min_pixels: int = field(
        default=4 * 16 * 16 * 2 * 2,
        metadata={"help": "Minimum number of pixels for image processing."}
    )
    
    max_pixels: int = field(
        default=1800 * 16 * 16 * 2 * 2,
        metadata={"help": "Maximum number of pixels for image processing."}
    )
    
    total_pixels: int = field(
        default=10 * 768 * 16 * 16 * 2 * 2,
        metadata={"help": "Maximum total pixels for video processing."}
    )
    
    fps: float = field(
        default=1.0,
        metadata={"help": "Frames per second for video sampling."}
    )
    
    max_frames: int = field(
        default=64,
        metadata={"help": "Maximum number of frames to sample from videos."}
    )
    
    default_instruction: str = field(
        default="Represent the user's input.",
        metadata={"help": "Default instruction prompt for embedding generation."}
    )


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    """
    Training arguments for Qwen3 VL Embeddings V2.
    
    Extends HuggingFace's TrainingArguments with Tevatron and multimodal-specific configurations.
    """
    
    output_dir: str = field(
        default="/tmp/vlmcl_output",
        metadata={"help": "Output directory for checkpoints and results."}
    )
    
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per device for training."}
    )
    
    warmup_ratio: float = field(
        default=0.025,
        metadata={"help": "Warmup ratio for learning rate scheduler."}
    )
    
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X steps."}
    )
    
    # Gradient cache settings
    grad_cache: bool = field(
        default=False,
        metadata={"help": "Use gradient cache update for memory efficiency."}
    )
    
    gc_q_chunk_size: int = field(
        default=4,
        metadata={"help": "Query chunk size for gradient cache."}
    )
    
    gc_p_chunk_size: int = field(
        default=32,
        metadata={"help": "Passage chunk size for gradient cache."}
    )
    
    # Gradient checkpointing
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing for memory savings."}
    )
    
    gradient_checkpointing_kwargs: Dict = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={"help": "Keyword arguments for gradient checkpointing."}
    )
    
    dataloader_num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses for data loading."}
    )
    
    # Precision settings
    fp16: bool = field(
        default=False,
        metadata={"help": "Use FP16 mixed precision training."}
    )
    
    bf16: bool = field(
        default=True,
        metadata={"help": "Use BF16 mixed precision training."}
    )
    
    tf32: bool = field(
        default=True,
        metadata={"help": "Use TF32 precision for matrix operations."}
    )
    
    report_to: str = field(
        default="none",
        metadata={"help": "Reporting backend (e.g., 'wandb', 'tensorboard')."}
    )
    
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Maximum number of checkpoints to keep."}
    )
    
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    
    data_seed: int = field(
        default=42,
        metadata={"help": "Random seed for data shuffling."}
    )
    
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": "Overwrite output directory content if it exists."
        },
    )
    
    continual_learning: bool = field(
        default=False,
        metadata={
            "help": "Enable continual learning from previous LoRA checkpoints."
        },
    )
    
    # ColPali-specific settings
    max_num_visual_tokens: int = field(
        default=768,
        metadata={"help": "Maximum number of visual tokens for ColPali-style models."}
    )
    
    embedding_projection: bool = field(
        default=True,
        metadata={"help": "Apply embedding projection layer."}
    )
    
    # Loss function settings
    pairwise_ce_loss: bool = field(
        default=False,
        metadata={"help": "Use pairwise cross-entropy loss."}
    )
    
    pairwise_inbatch_neg_loss: bool = field(
        default=False,
        metadata={"help": "Include in-batch negatives in pairwise loss."}
    )
    
    pairwise_inbatch_neg_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for in-batch negative loss in pairwise training."}
    )
    
    filter_false_negatives: bool = field(
        default=False,
        metadata={"help": "Filter potential false negatives during training."}
    )
    
    # Evaluation settings
    eval_on_start: bool = field(
        default=False,
        metadata={"help": "Run evaluation before training starts."}
    )
    
    eval_strategy: str = field(
        default="no",
        metadata={"help": "Evaluation strategy ('no', 'steps', 'epoch')."}
    )
    
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Checkpoint saving strategy ('no', 'steps', 'epoch')."}
    )
    
    per_device_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per device for evaluation."}
    )
