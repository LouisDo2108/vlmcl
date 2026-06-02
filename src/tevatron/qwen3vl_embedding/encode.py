import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tevatron.colpali.utils import get_params_info, init, write_json

# from tevatron.retriever.arguments import ModelArguments, DataArguments, \
#     TevatronTrainingArguments as TrainingArguments
# from tevatron.retriever.dataset import TrainDataset
# from tevatron.retriever.collator import TrainCollator
from tevatron.qwen3vl_embedding.arguments import DataArguments, ModelArguments
from tevatron.qwen3vl_embedding.arguments import (
    TevatronTrainingArguments as TrainingArguments,
)
from tevatron.qwen3vl_embedding.collator import EncodeCollator, TrainCollator
from tevatron.qwen3vl_embedding.dataset import (  # TrainRankedDataset,
    EncodeDataset,
    TrainDataset,
)
from tevatron.qwen3vl_embedding.models import *
from tevatron.qwen3vl_embedding.models import DenseModel
from tevatron.qwen3vl_embedding.trainer import Trainer

# from tevatron.retriever.modeling import DenseModel
# from tevatron.retriever.trainer import TevatronTrainer as Trainer
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = init(
        ModelArguments, DataArguments, TrainingArguments
    )

    base_model = Qwen3VLForEmbedding.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        ),
        trust_remote_code=True,
    )
    processor = Qwen3VLProcessor.from_pretrained(
        model_args.model_name_or_path, padding_side="right"
    )
    patch_size = getattr(processor.image_processor, "patch_size", None)
    merge_size = getattr(processor.image_processor, "merge_size", None)
    if patch_size is None or merge_size is None:
        raise ValueError(
            "Qwen3VL image processor is missing `patch_size` or `merge_size`."
        )

    tile = patch_size * merge_size
    processor.image_processor.max_pixels = 768 * tile * tile
    processor.image_processor.size["longest_edge"] = (
        processor.image_processor.max_pixels
    )

    def load_lora_model(lora_name_or_path, base_model):
        print(f"Loading LoRA from {lora_name_or_path}")
        lora_config = LoraConfig.from_pretrained(
            lora_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            ),
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_name_or_path,
            config=lora_config,
            adapter_name=lora_name_or_path,
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
            lora_model.load_adapter(lora_path, adapter_name=lora_path)
            # lora_model.set_adapter(lora_path.strip())
            # print(lora_model.active_adapter)
            # print(get_model_status(lora_model).available_adapters)

        lora_model = lora_model.merge_and_unload(
            progressbar=True,
            adapter_names=(
                lora_paths
                if lora_paths is not None
                else [model_args.lora_name_or_path.strip()]
            ),
        )  # type: ignore

        # Another way but with worse performance
        # lora_model.add_weighted_adapter([x.strip() for x in lora_paths], [1.0]*len(lora_paths), "merge", combination_type="linear")
        # lora_model = lora_model.merge_and_unload(adapter_names=["merge"])
    else:
        lora_model = load_lora_model(model_args.lora_name_or_path, base_model)

    model = DenseModel(
        encoder=lora_model,
        processor=processor,
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = EncodeCollator(
        data_args=data_args,
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
    context = (
        torch.autocast("cuda", dtype=dtype)
        if training_args.fp16 or training_args.bf16
        else nullcontext()
    )

    print("Encoding with dtype:", dtype)

    for ix, (batch_ids, batch) in enumerate(tqdm(encode_loader)):
        lookup_indices.extend(batch_ids)
        with context:
            with torch.no_grad():
                if data_args.encode_is_query:
                    model_output: EncoderOutput = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().half())
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().half())

                print(encoded[-1].shape)
            # if ix > 1:
            #     break

    encoded = torch.cat(encoded, dim=0)

    with open(data_args.encode_output_path, "wb") as f:
        pickle.dump((encoded, lookup_indices), f)

    print(f"Writting embeddings to {data_args.encode_output_path}")


if __name__ == "__main__":
    main()
