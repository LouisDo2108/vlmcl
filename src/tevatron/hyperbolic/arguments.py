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


@dataclass
class DataArguments(TevatronDataArguments):
    subset_name: List[str] = field(
        default=None, metadata={"help": "MMEB subset folder names under image_dir"}
    )
    num_sample_per_subset: Optional[int] = field(
        default=None, metadata={"help": "Cap training rows per subset (optional)"}
    )
    image_dir: str = field(
        default=None,
        metadata={"help": "Root of MMEB-train parquet + image files"},
    )
    max_len: Optional[int] = field(
        default=77,
        metadata={"help": "Max text tokens (CLIP uses 77; capped to model_max_length)"},
    )


@dataclass
class TrainingArguments(TevatronTrainingArguments):
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
