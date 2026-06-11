import copy
import logging
import os
import sys

import torch
from tevatron.hyperbolic.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from tevatron.hyperbolic.collator import CLIPCollator
from tevatron.hyperbolic.dataset import CLIPTrainDataset
from tevatron.hyperbolic.model import CCLIP, CLIPContrastiveModel, HyperbolicCCLIP
from tevatron.hyperbolic.old_embedding_cache import (
    OldEmbeddingCache,
    OldEmbeddingCacheMeta,
    build_old_embedding_cache,
    default_cache_path,
)
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
    model_cls = HyperbolicCCLIP if model_args.hyperbolic else CCLIP
    print_master(f"Building model: {model_cls.__name__}")
    model = model_cls.build(model_args)

    old_embedding_cache = None
    if training_args.old_checkpoint_path:
        """
        Build old embedding cache from old checkpoint.
        """
        train_dataset_for_cache = CLIPTrainDataset(data_args)
        cache_meta = OldEmbeddingCacheMeta.from_dataset(
            train_dataset_for_cache,
            training_args.old_checkpoint_path,
        )
        resolved_cache_path = (
            training_args.old_embedding_cache_path
            or default_cache_path(training_args.output_dir, cache_meta)
        )
        if os.path.isfile(resolved_cache_path):
            print_master(f"Loading cached old embeddings from {resolved_cache_path}")
            old_embedding_cache = OldEmbeddingCache.load(
                resolved_cache_path,
                expected_meta=cache_meta,
            )
        else:
        """
        If old cache is not found, build it from old checkpoint, which is the pre-trained CLIP model or a LoRA-finetuned CLIP model.
        """
            old_model_args = copy.deepcopy(model_args)
            if old_model_args.model_name_or_path == "OpenAI/clip-vit-large-patch14":
                old_model_args.lora_name_or_path = []
            else:
                old_model_args.lora_name_or_path = [training_args.old_checkpoint_path]
            print_master(
                f"Loading old checkpoint from {old_model_args.lora_name_or_path}"
            )
            old_checkpoint = CLIPContrastiveModel.load(old_model_args)
            old_checkpoint.eval()
            old_checkpoint.to(training_args.device)

            cache_collator = CLIPCollator(data_args=data_args, processor=processor)
            old_embedding_cache = build_old_embedding_cache(
                old_checkpoint,
                train_dataset_for_cache,
                cache_collator,
                device=training_args.device,
                batch_size=training_args.per_device_train_batch_size,
                old_checkpoint_path=training_args.old_checkpoint_path,
                output_dir=training_args.output_dir,
                cache_path=training_args.old_embedding_cache_path,
                num_workers=training_args.dataloader_num_workers,
            )
            del old_checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if training_args.local_rank in [-1, 0]:
        get_params_info(model)

    use_ckc = old_embedding_cache is not None
    train_dataset = CLIPTrainDataset(data_args, return_index=use_ckc)
    collator = CLIPCollator(
        data_args=data_args,
        processor=processor,
        return_indices=use_ckc,
    )

    trainer_cls = CCLIPGradCacheTrainer if training_args.grad_cache else CCLIPTrainer
    print_rank(f"Trainer: {trainer_cls.__name__}, grad_cache={training_args.grad_cache}")

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
        old_embedding_cache=old_embedding_cache,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
