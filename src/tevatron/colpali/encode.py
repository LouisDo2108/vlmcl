import logging
import os
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from pdb import set_trace as st

import numpy as np
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from peft import LoraConfig, PeftModel, TaskType, get_model_status, get_peft_model
from tevatron.colpali.arguments import DataArguments, ModelArguments
from tevatron.colpali.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.colpali.collator import EncodeCollator
from tevatron.colpali.dataset import EncodeDataset, EncodeICLRDataset
from tevatron.colpali.models import ColQwen3, ColQwen3Processor, DenseModel
from tevatron.retriever.modeling.encoder import EncoderOutput
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers.utils.import_utils import is_flash_attn_2_available

# from tevatron.colpali.index import save_colbert_index

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
        fix_mistral_regex=True,
        max_num_visual_tokens=getattr(training_args, "max_num_visual_tokens", None),
    )

    base_model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        trust_remote_code=True,
    )
    
    def load_lora_model(lora_name_or_path, base_model):
        print(f"Loading LoRA from {lora_name_or_path}")
        lora_config = LoraConfig.from_pretrained(
            lora_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
            trust_remote_code=True
        )
        lora_model = PeftModel.from_pretrained(
            base_model, lora_name_or_path, 
            config=lora_config, 
            adapter_name=lora_name_or_path
        )
        return lora_model

    if model_args.lora_name_or_path == "":
        lora_model = base_model
    elif Path(model_args.lora_name_or_path).suffix == ".txt":
        # Read the list of lora paths
        with open(model_args.lora_name_or_path) as f:
            lora_paths = f.readlines()
        
        lora_paths = [x.strip() for x in lora_paths]
        
        # The most naive way to load
        # lora_model1 = load_lora_model(lora_paths[0].strip(), base_model)
        # lora_model1 = lora_model1.merge_and_unload(progressbar=True)
        # lora_model1 = load_lora_model(lora_paths[1].strip(), lora_model1)
        # lora_model1 = lora_model1.merge_and_unload(progressbar=True)
        
        # Load the base model along with the first LoRA
        lora_model = load_lora_model(lora_paths[0], base_model)
    
        # Load the remaining LoRAs in the list
        for lora_path in lora_paths[1:]:
            print(f"Loading LoRA from {lora_path}")
            lora_model.load_adapter(
                lora_path, 
                adapter_name=lora_path
            )
            # lora_model.set_adapter(lora_path.strip())
            # print(lora_model.active_adapter)
            # print(get_model_status(lora_model).available_adapters)
        
        lora_model = lora_model.merge_and_unload(
            progressbar=True, 
            adapter_names=lora_paths if lora_paths is not None else [model_args.lora_name_or_path.strip()]
        ) # type: ignore
        
        # Another way but with worse performance
        # lora_model.add_weighted_adapter([x.strip() for x in lora_paths], [1.0]*len(lora_paths), "merge", combination_type="linear")
        # lora_model = lora_model.merge_and_unload(adapter_names=["merge"])
    else:
        lora_model = load_lora_model(model_args.lora_name_or_path, base_model)
    
    model = DenseModel(
        encoder=lora_model,
    )
    # model.encoder = torch.compile(model.encoder)

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = EncodeCollator(
        data_args=data_args,
        processor=processor,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    
    dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    context = torch.autocast("cuda", dtype=dtype) if training_args.fp16 or training_args.bf16 else nullcontext()
    
    print("Encoding with dtype:", dtype)

    for ix, (batch_ids, batch) in enumerate(tqdm(encode_loader)):
        lookup_indices.extend(batch_ids)
        with context:
            with torch.no_grad():
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.to(training_args.device)
                    
                if data_args.encode_is_query:
                    model_output: EncoderOutput = model(
                        query=batch, 
                        embedding_projection=training_args.embedding_projection
                    )
                    encoded.append(model_output.q_reps.cpu().half())
                else:
                    model_output: EncoderOutput = model(
                        passage=batch, 
                        embedding_projection=training_args.embedding_projection
                    )
                    encoded.append(model_output.p_reps.cpu().half())
                
                print(encoded[-1].shape)
            # if ix > 1:
            #     break

    # save_colbert_index(
    #     embeddings_list=encoded,
    #     doc_ids=lookup_indices,
    #     output_dir="/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl/colqwen3_tevatron/1epoch_colpali_inbatchneg/checkpoint-3694/out_no_embedding_projection/vidore/arxivqa_test_subsampled_beir/colbert_index" # data_args.encode_output_path,
    # )
    # Old logic that save a fixed size tensor index
    encoded = torch.cat(encoded, dim=0)
    
    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

    print(f"Writting embeddings to {data_args.encode_output_path}")

if __name__ == "__main__":
    main()
