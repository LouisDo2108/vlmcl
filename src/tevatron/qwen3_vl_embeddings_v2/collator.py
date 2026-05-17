"""
Collator module for Qwen3 VL Embeddings V2.

This module provides data collation utilities for training and encoding,
with improved type hints and documentation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from .arguments import DataArguments

logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    """
    Collator for training data.
    
    This class formats training samples into the expected input format
    for the multimodal model.
    """
    
    data_args: DataArguments
    
    def __call__(
        self, 
        features: List[Tuple[str, List[Any]]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Collate training features into model inputs.
        
        Args:
            features: List of (query_text, passage_images) tuples.
            
        Returns:
            Tuple of (query_messages, passage_messages).
        """
        query_messages = [
            {"text": f[0]} for f in features
        ]
        
        passage_messages = [
            {"image": f[1][0]} if len(f[1]) == 1 else {"image": f[1]} 
            for f in features 
        ]
        
        return query_messages, passage_messages


@dataclass
class EncodeCollator:
    """
    Collator for encoding data.
    
    This class formats data samples for inference/encoding tasks.
    """
    
    data_args: DataArguments
    
    def __call__(
        self, 
        features: List[Tuple[str, str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Collate encoding features into model inputs.
        
        Args:
            features: List of (id, text, content) tuples.
            
        Returns:
            Tuple of (content_ids, input_messages).
        """
        content_ids = [x[0] for x in features]
        
        if self.data_args.encode_is_query:
            # Query encoding
            query_inputs = [{"text": f[1]} for f in features]
            return content_ids, query_inputs
        else:
            # Passage encoding
            passage_inputs = [
                {"image": f[2][0]} if len(f[1]) == 1 else {"image": f[2]}
                for f in features
            ]
            return content_ids, passage_inputs
