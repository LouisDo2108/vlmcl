import logging
import sys

from tevatron.hyperbolic.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from tevatron.hyperbolic.collator import CLIPCollator
from tevatron.hyperbolic.dataset import CLIPTrainDataset
from tevatron.hyperbolic.model import CLIPContrastiveModel
from tevatron.hyperbolic.trainer import CLIPGradCacheTrainer, CLIPTrainer
from tevatron.hyperbolic.utils import get_params_info, init, print_master, print_rank
from transformers import CLIPProcessor

logger = logging.getLogger(__name__)


def load_clip_processor(model_args):
    model_name = model_args.processor_name or model_args.model_name_or_path
    print_master(f"Loading CLIPProcessor from {model_name}")
    return CLIPProcessor.from_pretrained(model_name)


def main():
    for arg in list(sys.argv):
        if arg.startswith("--local-rank="):
            rank = arg.split("=", 1)[1]
            sys.argv.remove(arg)
            sys.argv.extend(["--local_rank", rank])

    # Sets up logging, prints training/model/data args, and seeds RNGs.
    model_args, data_args, training_args = init(
        ModelArguments, DataArguments, TrainingArguments
    )

    processor = load_clip_processor(model_args)
    model = CLIPContrastiveModel.build(model_args)

    # Print trainable params once before training begins (rank 0 only).
    if training_args.local_rank in [-1, 0]:
        get_params_info(model)

    train_dataset = CLIPTrainDataset(data_args)
    collator = CLIPCollator(data_args=data_args, processor=processor)

    trainer_cls = CLIPGradCacheTrainer if training_args.grad_cache else CLIPTrainer
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
