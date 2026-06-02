"""
Qwen3 VL Embeddings V2 - Improved multimodal embedding module for Tevatron framework.

This module provides enhanced code structure, better type hints, and improved 
code quality compared to the original qwen3vl_embedding implementation.
"""

from .arguments import DataArguments, ModelArguments, TevatronTrainingArguments
from .collator import EncodeCollator, TrainCollator
from .dataset import EncodeDataset, TrainDataset
from .models import DenseModel, Qwen3VLForEmbedding
from .trainer import Trainer

__all__ = [
    "Qwen3VLForEmbedding",
    "DenseModel",
    "TrainDataset",
    "EncodeDataset",
    "TrainCollator",
    "EncodeCollator",
    "Trainer",
    "ModelArguments",
    "DataArguments",
    "TevatronTrainingArguments",
]
