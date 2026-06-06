from dataclasses import dataclass, field
from typing import List, Optional, Union

from tevatron.retriever.arguments import DataArguments as TevatronDataArguments
from tevatron.retriever.arguments import ModelArguments as TevatronModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments


@dataclass
class ModelArguments(TevatronModelArguments):
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "CLIPProcessor name (defaults to model_name)"}
    )
    normalize: bool = field(
        default=True, metadata={"help": "L2-normalize embeddings before similarity"}
    )
    temperature: float = field(
        default=0.02, metadata={"help": "Softmax temperature for contrastive loss"}
    )
    lora: bool = field(default=False, metadata={"help": "Train with LoRA adapters"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj,gate_proj,up_proj",# "all-linear"
    )
    lora_merge_coeff: float = field(default=1.0, metadata={"help": "LoRA merge coefficient used in C-CLIP"})


@dataclass
class DataArguments(TevatronDataArguments):
    subset_name: List[str] = field(
        default=None, metadata={"help": "MMEB subset folder names under image_dir"}
    )
    num_sample_per_subset: Optional[int] = field(
        default=100000, metadata={"help": "Cap training rows per subset (optional)"}
    )
    image_dir: str = field(
        default="/home/thuy0050/mg61_scratch2/thuy0050/data/MMEB/MMEB-train",
        metadata={"help": "Root of MMEB-train parquet + image files"},
    )
    max_len: Optional[int] = field(
        default=77,
        metadata={"help": "Max text tokens (CLIP uses 77; capped to model_max_length)"},
    )
    add_instructions: bool = field(
        default=False,
        metadata={"help": "Add instructions to the text, should not be used for CLIP models"},
    )


@dataclass
class TrainingArguments(TevatronTrainingArguments):
    bidirectional_loss: bool = field(
        default=True,
        metadata={
            "help": "Average query->target and target->query contrastive loss (CLIP-style)"
        },
    )
    grad_cache: bool = field(
        default=True, metadata={"help": "Use GradCache for memory-efficient training"}
    )
    gc_q_chunk_size: int = field(default=4, metadata={"help": "GradCache query chunk size"})
    gc_p_chunk_size: int = field(default=4, metadata={"help": "GradCache target chunk size"})
    report_to: Union[None, str, list[str]] = field(
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    save_strategy: str = field(default="epoch", metadata={"help": "Save strategy"})
    num_train_epochs: int = field(default=10, metadata={"help": "Number of training epochs"})
    learning_rate: float = field(default=3e-5, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=1e-2, metadata={"help": "Weight decay"})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of workers for dataloader"})
    save_safetensors: bool = field(default=True, metadata={"help": "Save safetensors"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove unused columns"})
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})
    bf16: bool = field(default=True, metadata={"help": "Use BF16 mixed precision training"})