import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pprint
from typing import Dict

import torch
from peft import LoraConfig

# from tevatron.retriever.arguments import DataArguments, ModelArguments
# from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.colpali.arguments import DataArguments, ModelArguments
from tevatron.colpali.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.colpali.collator import TrainCollator
from tevatron.colpali.dataset import TrainDataset
from tevatron.colpali.losses import ColbertLoss
from tevatron.colpali.models import ColQwen3, ColQwen3Processor
from tevatron.colpali.trainer import Trainer
from tevatron.colpali.utils import get_params_info, init, write_json
from tevatron.retriever.modeling.encoder import EncoderModel, EncoderOutput
from torch import Tensor
from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = ColbertLoss(
            temperature=0.02,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        return query_hidden_states
        # query_hidden_states = query_hidden_states.last_hidden_state
        # return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
        
    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                
            scores, loss = self.loss_func(
                query_embeddings=q_reps,
                doc_embeddings=p_reps,
                offset=0,
            )
            losses = {
                "loss": loss,
                "infonce_loss": loss.clone().detach(),
            }

            # scores = self.compute_similarity(q_reps, p_reps)
            # scores = scores.view(q_reps.size(0), -1)

            # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            # target = target * (p_reps.size(0) // q_reps.size(0))

            # loss = self.compute_loss(scores / self.temperature, target)
            # if self.is_ddp:
            #     loss = loss * self.world_size  # counter average weight reduction
            return losses
        # for eval
        else:
            scores = self.loss_func.compute_similarity(q_reps, p_reps, offset=0)
            loss = None
            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )


model_cls_dict = {
    # "vidore/colqwen2.5-v0.2": (ColQwen2_5, ColQwen2_5_Processor),
    "colqwen3": (ColQwen3, ColQwen3Processor),
}


def main():
    model_args, data_args, training_args = init(ModelArguments, DataArguments, TrainingArguments)
    
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

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
    peft_config=LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
    )
    model.add_adapter(peft_config)
    model = DenseModel(
        encoder=model,
        pooling="last",
        normalize=True,
    )
    get_params_info(model)
    print("*"*50)
    pprint(model_args)
    print("*"*50)
    pprint(data_args)
    print("*"*50)
    pprint(training_args)

    train_dataset = TrainDataset(data_args)
    
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
