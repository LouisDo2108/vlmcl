import logging
from dataclasses import dataclass
from typing import List, Tuple
from pdb import set_trace as st

import torch
from qwen_vl_utils import process_vision_info
from tevatron.colpali.arguments import DataArguments
from transformers import ProcessorMixin

logger = logging.getLogger(__name__)

N_AUGMENTATION_TOKENS = 10


@dataclass
class TrainCollator:
    data_args: DataArguments

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        query_messages = [
            {"text": f[0]} for f in features
        ]
        passage_messages = [
            {"image": f[1][0]} if len(f[1]) == 1 else {"image": f[1]} for f in features 
        ]   

        return query_messages, passage_messages

@dataclass
class EncodeCollator:
    data_args: DataArguments

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        content_ids = [x[0] for x in features]

        if self.data_args.encode_is_query:
            query_inputs = [{"text": f[1]} for f in features]
            return content_ids, query_inputs
        else:
            passage_inputs = [
                {"image": f[2][0]} if len(f[1]) == 1 else {"image": f[2]}
                for f in features
            ]
            return content_ids, passage_inputs
