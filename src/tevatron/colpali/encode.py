import logging
import os
import pickle
import sys
from contextlib import nullcontext
from pdb import set_trace as st

import numpy as np
import torch
from collator import EncodeCollator, MultiModalEncodeCollator
from colpali_engine.models import (
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColQwen3,
    ColQwen3Processor,
)
from dataset import EncodeDataset, EncodeICLRDataset
from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

model_cls_dict = {
    "vidore/colqwen2.5-v0.2": (ColQwen2_5, ColQwen2_5_Processor),
    "colqwen3": (ColQwen3, ColQwen3Processor),
}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    for k, v in model_cls_dict.items():
        # Sequentially select the appropriate processor 
        if k.lower() in model_args.model_name_or_path.lower():
            model_cls, processor_cls = v
            break

    processor = processor_cls.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        trust_remote_code=True,
        query_max_len=data_args.query_max_len,
        trust_remote_code=True, #type:ignore
        # cache_dir=model_args.cache_dir,
    )

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        trust_remote_code=True,
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = MultiModalEncodeCollator(
        data_args=data_args,
        processor=processor,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=0, # training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    
    for ix, (batch_ids, batch) in enumerate(tqdm(encode_loader)):
        lookup_indices.extend(batch_ids)
        with (
            torch.autocast(
                "cuda", dtype=torch.float16 if training_args.fp16 else torch.bfloat16
            )
            if training_args.fp16 or training_args.bf16
            else nullcontext()
        ):
            with torch.no_grad():
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.to(training_args.device)

                if data_args.encode_is_query:
                    texts, _ = batch['texts'], batch['images']
                    model_output = model(**texts)
                else:
                    _, images = batch['texts'], batch['images']
                    model_output = model(**images)
                
                encoded.append(model_output.cpu().detach().half()) 
                print(encoded[-1].shape)
            # if ix > 1:
            #     break

    encoded = torch.cat(encoded, dim=0)
    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
