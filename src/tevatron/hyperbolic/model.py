from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tevatron.hyperbolic.arguments import ModelArguments, TrainingArguments
from tevatron.hyperbolic.utils import print_master
from torch import Tensor, nn
from transformers import CLIPModel


class CLIPContrastiveModel(nn.Module):
    """Contrastive wrapper around Hugging Face CLIPModel for Tevatron / GradCache training."""

    def __init__(
        self,
        encoder: CLIPModel,
        normalize: bool = True,
        temperature: float = 0.02,
    ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_input(self, inputs: Dict[str, Tensor]) -> Tensor:
        image_features, text_features = None, None

        pixel_values = inputs.get("pixel_values")
        image_features = self.encoder.get_image_features(pixel_values=pixel_values)
        # if pixel_values is not None and pixel_values.abs().sum() > 0:
        #     has_image = pixel_values.flatten(1).abs().sum(dim=1) > 0
        #     if has_image.any():
        #         image_features = self.encoder.get_image_features(pixel_values=pixel_values)
        #         if not has_image.all():
        #             image_features = image_features.clone()
        #             image_features[~has_image] = 0

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if input_ids is not None:
            text_features = self.encoder.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if image_features is not None and text_features is not None:
            reps = image_features + text_features
        elif image_features is not None:
            reps = image_features
        elif text_features is not None:
            reps = text_features
        else:
            raise ValueError("Batch must contain text and/or image inputs.")

        if self.normalize:
            reps = F.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments = None, **kwargs):
        del training_args
        print_master(f"Loading CLIP from {model_args.model_name_or_path}")
        dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        base_model = CLIPModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=dtype,
            **kwargs,
        )

        if model_args.lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules if model_args.lora_target_modules == "all-linear" else model_args.lora_target_modules.split(","),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                # use_dora=True, # VLM2Vec uses DORA
                use_dora=False,
                inference_mode=False,
            )
            base_model = get_peft_model(base_model, lora_config)

        return cls(
            encoder=base_model,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
        )

    def forward(
        self,
        qry: Optional[Dict[str, Tensor]] = None,
        tgt: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ):
        # # GradCache per-stream call path: model(**batch_dict)
        # # where kwargs contains input_ids/attention_mask/pixel_values directly.
        # if qry is None and tgt is None and kwargs:
        #     return self.encode_input(kwargs)

        qry_reps = self.encode_input(qry) if qry is not None else None
        tgt_reps = self.encode_input(tgt) if tgt is not None else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps, all_tgt_reps = qry_reps, tgt_reps
            
        return all_qry_reps, all_tgt_reps

    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        t = t.contiguous()
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.process_rank] = t
        return torch.cat(gathered, dim=0)
