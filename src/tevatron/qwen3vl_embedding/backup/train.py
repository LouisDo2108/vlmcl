import logging
import os
import sys
import torch
from dataclasses import asdict
from copy import deepcopy
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

# from tevatron.retriever.arguments import ModelArguments, DataArguments, \
#     TevatronTrainingArguments as TrainingArguments
# from tevatron.retriever.dataset import TrainDataset
# from tevatron.retriever.collator import TrainCollator
from tevatron.colpali.arguments import DataArguments, ModelArguments
from tevatron.colpali.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.colpali.utils import get_params_info, init, write_json
from transformers.utils.import_utils import is_flash_attn_2_available
from tevatron.qwen3vl_embedding.collator import TrainCollator
from tevatron.qwen3vl_embedding.dataset import TrainDataset, TrainRankedDataset
from tevatron.qwen3vl_embedding.models import DenseModel
from tevatron.qwen3vl_embedding.trainer import Trainer
# from tevatron.retriever.modeling import DenseModel
# from tevatron.retriever.trainer import TevatronTrainer as Trainer
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer
from tevatron.qwen3vl_embedding.models import *

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = init(ModelArguments, DataArguments, TrainingArguments)

    model = Qwen3VLForEmbedding.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        trust_remote_code=True,
    )
    processor = Qwen3VLProcessor.from_pretrained(
        model_args.model_name_or_path, padding_side='right'
    )
    patch_size = getattr(processor.image_processor, "patch_size", None)
    merge_size = getattr(processor.image_processor, "merge_size", None)
    if patch_size is None or merge_size is None:
        raise ValueError("Qwen3VL image processor is missing `patch_size` or `merge_size`.")

    tile = patch_size * merge_size
    processor.image_processor.max_pixels = 768 * tile * tile
    processor.image_processor.size["longest_edge"] = processor.image_processor.max_pixels

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj).*$|.*(custom_text_proj).*$)",
        # target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
    )
    # model.add_adapter(lora_config)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model = DenseModel(
        encoder=model,
        processor=processor,
        max_pixels=786432, # Mimic colqwen3
        pooling="last",
        normalize=True,
        filter_false_negatives=training_args.filter_false_negatives,
    )

    get_params_info(model)

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args)

    trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=collator,
    )
    train_dataset.set_trainer(trainer)
    last_checkpoint = None
    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))

    training_args_to_save = asdict(deepcopy(training_args))
    training_args_to_save.update(asdict(model_args))
    training_args_to_save.update(asdict(data_args))
    write_json(
        os.path.join(training_args.output_dir, "full_config.json"),
        training_args_to_save,
    )
    processor.save_pretrained(training_args.output_dir)

    # model = DenseModel.build(
    #     model_args,
    #     training_args,
    #     cache_dir=model_args.cache_dir,
    #     torch_dtype=torch_dtype,
    #     attn_implementation=model_args.attn_implementation,
    # )

    # train_dataset = TrainDataset(data_args)
    # collator = TrainCollator(data_args, tokenizer)

    # trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    # trainer = trainer_cls(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     data_collator=collator
    # )
    # train_dataset.set_trainer(trainer)

    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    # trainer.save_model()
    # if trainer.is_world_process_zero():
    #     tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
