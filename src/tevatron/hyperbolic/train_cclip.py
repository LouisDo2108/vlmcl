"""
Train CCLIP / HyperbolicCCLIP (separate entry point from train.py).

Dual objective = CLIP contrastive on fused embeddings + contrastive on projected
embeddings. With --grad_cache, runs two GradCache passes (see trainer_cclip.py).
"""

import logging
import os
import sys

from tevatron.hyperbolic.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from tevatron.hyperbolic.collator import CLIPCollator
from tevatron.hyperbolic.dataset import CLIPTrainDataset
from tevatron.hyperbolic.model import CCLIP
from tevatron.hyperbolic.train import _maybe_enable_wandb, load_clip_processor
from tevatron.hyperbolic.trainer_cclip import CCLIPGradCacheTrainer, CCLIPTrainer
from tevatron.hyperbolic.utils import get_params_info, init, print_master, print_rank

logger = logging.getLogger(__name__)


def main():
    for arg in list(sys.argv):
        if arg.startswith("--local-rank="):
            rank = arg.split("=", 1)[1]
            sys.argv.remove(arg)
            sys.argv.extend(["--local_rank", rank])

    model_args, data_args, training_args = init(
        ModelArguments, DataArguments, TrainingArguments
    )
    _maybe_enable_wandb(training_args)

    processor = load_clip_processor(model_args)
    if model_args.lora and model_args.lora_name_or_path:
        raise NotImplementedError("Continual LoRA for CCLIP is not wired yet.")
    model = CCLIP.build(model_args)

    if training_args.local_rank in [-1, 0]:
        get_params_info(model)

    train_dataset = CLIPTrainDataset(data_args)
    collator = CLIPCollator(data_args=data_args, processor=processor)

    trainer_cls = (
        CCLIPGradCacheTrainer if training_args.grad_cache else CCLIPTrainer
    )
    print_rank(f"Trainer: {trainer_cls.__name__}, grad_cache={training_args.grad_cache}")

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
