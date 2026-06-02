import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from transformers import TrainingArguments
from tevatron.retriever.arguments import DataArguments as TevatronDataArguments
from tevatron.retriever.arguments import ModelArguments as TevatronModelArguments
import copy


def default_field(obj):
    return field(default_factory=lambda: copy.deepcopy(obj))


@dataclass
class ModelArguments(TevatronModelArguments):
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj",
        metadata={"help": "lora target modules"}
    )
    # lora_name_or_path: List[str] = field(default_factory=lambda:[""])


@dataclass
class DataArguments(TevatronDataArguments):

    query_max_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_proc: int = field(
        default=1, metadata={"help": "number of processes to use for loading the dataset"}
    )
    padding_side: str = field(default='left')
    
    colpali_source: str = field(default="all")


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    output_dir: str = field(default="/home/thuy0050/mg61_scratch2/thuy0050/exp/vlmcl")
    per_device_train_batch_size: int = field(default=32) # Following the ColPali paper
    warmup_ratio: float = field(default=0.025) # Following the ColPali paper
    logging_steps: int = field(default=10)

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    
    gradient_checkpointing: bool = field(default=False)
    gradient_checkpointing_kwargs: Dict = default_field({"use_reentrant": False})
    dataloader_num_workers: int = field(default=8)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    report_to: str = field(default="none") # wandb
    save_total_limit: int = field(default=1)
    max_length: int = field(default=512)
    seed: int = field(default=42)
    data_seed: int = field(default=42)
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    continual_learning: bool = field(
        default=False,
        metadata={
            "help": (
                "A flag for continual learning from previous LoRA checkpoints. This is not applied for the first fine-tuning session from the base model."
            )
        },
    )
    
    # Colpali specific
    max_num_visual_tokens: int = field(default=768)
    embedding_projection: bool = field(default=True)
    
    pairwise_ce_loss: bool = field(default=False)
    pairwise_inbatch_neg_loss: bool = field(default=False)
    pairwise_inbatch_neg_loss_weight: float = field(default=0.5)
    
    filter_false_negatives: bool = field(default=False)
    
    # For evaluation
    eval_on_start: bool = field(default=False)
    eval_strategy: str = field(default="no") # epoch
    save_strategy: str = field(default="steps")
    per_device_eval_batch_size: int = field(default=64)
    
    # load_best_model_at_end: bool = field(default=True)
    # prediction_loss_only: bool = field(default=True)
    # metric_for_best_model: str = field(default="eval_recall@1")
    # greater_is_better: bool = field(default=True)