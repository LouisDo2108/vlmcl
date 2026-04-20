import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import yaml
from pdb import set_trace as st
from pprint import pprint
from typing import Dict

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

# from tevatron.retriever.arguments import DataArguments, ModelArguments
# from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.colpali.arguments import DataArguments, ModelArguments
from tevatron.colpali.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.colpali.collator import TrainCollator
from tevatron.colpali.dataset import TrainDataset, TrainRankedDataset
# from tevatron.colpali.losses import ColbertLoss
from tevatron.colpali.models import ColQwen3, ColQwen3Processor, DenseModel
from tevatron.colpali.trainer import Trainer
from tevatron.colpali.utils import get_params_info, init, write_json
from tevatron.retriever.modeling.encoder import EncoderModel, EncoderOutput
from transformers.utils.import_utils import is_flash_attn_2_available
from torchinfo import summary
from transformers.trainer_pt_utils import AcceleratorConfig

logger = logging.getLogger(__name__)


model_cls_dict = {
    # "vidore/colqwen2.5-v0.2": (ColQwen2_5, ColQwen2_5_Processor),
    "colqwen3": (ColQwen3, ColQwen3Processor),
}


def main():
    model_args, data_args, training_args = init(ModelArguments, DataArguments, TrainingArguments)

    for k, v in model_cls_dict.items():
        # Sequentially select the appropriate processor 
        if k.lower() in model_args.model_name_or_path.lower():
            model_cls, processor_cls = v
            break

    processor = processor_cls.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        trust_remote_code=True,
        query_max_len=data_args.query_max_len,
        fix_mistral_regex=True,
        max_num_visual_tokens=getattr(training_args, "max_num_visual_tokens", None),
    )
    
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        trust_remote_code=True,
    )
    
    if training_args.continual_learning:
        # Load the base model, merge with the trained adapter
        lora_config = LoraConfig.from_pretrained(
            model_args.lora_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model, model_args.lora_name_or_path, config=lora_config
        )
        model = model.merge_and_unload()


    lora_config=LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
    )
    model.add_adapter(lora_config)
    model = DenseModel(
        encoder=model,
        pooling="last",
        normalize=True,
        pairwise_ce_loss=training_args.pairwise_ce_loss,
        filter_false_negatives=training_args.filter_false_negatives,
        pairwise_inbatch_neg_loss=training_args.pairwise_inbatch_neg_loss,
        pairwise_inbatch_neg_loss_weight=training_args.pairwise_inbatch_neg_loss_weight,
    )
    
    get_params_info(model)
    
    # # Override 
    # with open("/home/thuy0050/mg61_scratch2/thuy0050/.cache/huggingface/accelerate/default_config.yaml") as f:
    #     accelerate_config = yaml.safe_load(f)

    # training_args.accelerator_config = training_args.accelerator_config.to_dict()
    # training_args.accelerator_config.update(accelerate_config)
    # training_args.accelerator_config = AcceleratorConfig(**training_args.accelerator_config)
    
    print("*"*50)
    pprint(model_args)
    print("*"*50)
    pprint(data_args)
    print("*"*50)
    pprint(training_args)

    train_dataset = TrainRankedDataset(data_args)
    
    # eval_args = deepcopy(data_args)
    # eval_args.dataset_name = "vidore/colpali_train_set"
    # eval_args.dataset_split = "test"
    # eval_args.corpus_split = "test"
    # eval_dataset = TrainDataset(eval_args)
    
    collator = TrainCollator(data_args, processor)

    # trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer_cls = Trainer
    
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=collator,
    )
    train_dataset.set_trainer(trainer)
    # eval_dataset.set_trainer(trainer)
    
    """
    # For resuming from a checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
    """
    
    last_checkpoint = None
    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    
    """
    # Wandb
    if wandb.run is not None:
        wandb.run.config.update(asdict(model_args), allow_val_change=True)
        wandb.run.config.update(asdict(data_args), allow_val_change=True)
    """
    training_args_to_save = asdict(deepcopy(training_args))
    training_args_to_save.update(asdict(model_args))
    training_args_to_save.update(asdict(data_args))
    write_json(
        os.path.join(training_args.output_dir, "full_config.json"),
        training_args_to_save,
    )
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
