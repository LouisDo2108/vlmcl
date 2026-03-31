import logging
import os
import random
from dataclasses import dataclass
from pdb import set_trace as st
from typing import List, Tuple

import torch
from arguments import DataArguments
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, ProcessorMixin

from tevatron.retriever.arguments import DataArguments

import msgspec
from pathlib import Path
from msgspec.json import format
from pprint import pformat, pprint

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

temp = load_dataset("vidore/arxivqa_test_subsampled_beir", "qrels", split="test")

# with open(f"/home/thuy0050/code/colpali/tevatron/examples/colpali/arxiv_test_subsampled_beir_qrel.txt", "w") as f:
#     for row in temp:
#         query_id, docid, score = row["query-id"], row["corpus-id"], row["score"]
#         f.write(f"{int(query_id)} 0 {int(docid)} 1\n")

with open(f"/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/archive_after_submission/test/qrel.txt", "r") as f:       
    for line in f.readlines():
        q, _, d, _ = line.split(" ")
        if q == d:
            print(q, d)