import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from pdb import set_trace as st
from pprint import pformat, pprint
from typing import List, Tuple

import msgspec
from datasets import load_dataset
from msgspec.json import format
from tqdm import tqdm

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

temp = load_dataset("vidore/tatdqa_test_beir", "qrels", split="test")

# with open(f"/home/thuy0050/mg61_scratch2/thuy0050/data/vlmcl/vidore_v1_qrels/tatdqa_test_beir_qrel.txt", "w") as f:
#     for row in temp:
#         query_id, docid, score = row["query-id"], row["corpus-id"], row["score"]
#         f.write(f"{int(query_id)} 0 {int(docid)} 1\n")