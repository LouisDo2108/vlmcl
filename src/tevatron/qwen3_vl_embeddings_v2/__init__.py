"""
Qwen3 VL Embeddings V2 - Improved multimodal embedding module for Tevatron framework.

This module provides enhanced code structure, better type hints, and improved 
code quality compared to the original qwen3vl_embedding implementation.
"""

from .models import (
    Qwen3VLForEmbedding,
    DenseModel,
    Qwen3VLEmbedder,
)
from .dataset import TrainDataset, EncodeDataset
from .collator import TrainCollator, EncodeCollator
from .trainer import Trainer
from .arguments import ModelArguments, DataArguments, TevatronTrainingArguments

__all__ = [
    "Qwen3VLForEmbedding",
    "DenseModel",
    "Qwen3VLEmbedder",
    "TrainDataset",
    "EncodeDataset",
    "TrainCollator",
    "EncodeCollator",
    "Trainer",
    "ModelArguments",
    "DataArguments",
    "TevatronTrainingArguments",
]
