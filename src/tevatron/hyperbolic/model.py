import os
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tevatron.hyperbolic.arguments import ModelArguments, TrainingArguments
from tevatron.hyperbolic.poincare import embed_sphere_to_ball, expmap0, project_to_ball
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

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    @property
    def rep_dim(self) -> int:
        """Output dim of get_image/get_text_features (CLIP uses projection_dim)."""
        config = self.encoder.config
        if getattr(config, "projection_dim", None) is not None:
            return config.projection_dim
        if getattr(config, "hidden_size", None) is not None:
            return config.hidden_size
        return config.text_config.hidden_size

    def encode_clip_input(self, inputs: Dict[str, Tensor]) -> Tensor:
        return self.encode_input(inputs)

    def _fuse_multimodal_features(
        self,
        image_features: Tensor,
        text_features: Tensor,
        has_image: Tensor,
        has_text: Tensor,
    ) -> Tensor:
        """
        Per-sample fusion for mixed-modality batches.

        Each row independently uses text-only, image-only, or image+text:
          text only  -> text_features
          image only -> image_features
          both       -> image_features + text_features (VLM2Vec approach)
        Placeholder inputs are masked out before fusion.
        """
        has_image = has_image.to(device=image_features.device, dtype=torch.bool)
        has_text = has_text.to(device=text_features.device, dtype=torch.bool)

        if (~has_image & ~has_text).any():
            raise ValueError(
                "Batch contains samples with neither text nor image; "
                "check dataset filtering."
            )

        img_mask = has_image.unsqueeze(-1).to(dtype=image_features.dtype)
        txt_mask = has_text.unsqueeze(-1).to(dtype=text_features.dtype)
        return image_features * img_mask + text_features * txt_mask

    def encode_input(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Expects collator batch dicts already on the encoder device (Trainer._prepare_inputs
        or eval batch_to_device). GradCache splits those GPU tensors into chunks.
        """
        has_image = inputs.get("has_image")
        has_text = inputs.get("has_text")
        if has_image is None or has_text is None:
            raise ValueError("Batch must include has_image and has_text from the collator.")

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]

        image_features = self.encoder.get_image_features(
            pixel_values=pixel_values
        )
        text_features = self.encoder.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        reps = self._fuse_multimodal_features(
            image_features, text_features, has_image, has_text
        )

        if self.normalize:
            reps = F.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def load_clip_encoder(cls, model_name_or_path: str, **hf_kwargs) -> CLIPModel:
        load_kwargs = dict(hf_kwargs)
        print_master(f"Loading CLIP from {model_name_or_path} with {load_kwargs}")
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            model = CLIPModel.from_pretrained(model_name_or_path, **load_kwargs)
            print_master("Using flash attention 2")
            return model
        except Exception as e:
            try:
                load_kwargs["attn_implementation"] = "sdpa"
                print_master("Using sdpa")
                return CLIPModel.from_pretrained(model_name_or_path, **load_kwargs)
            except Exception as e2:
                load_kwargs["attn_implementation"] = "eager"
                print_master("Using eager")
                return CLIPModel.from_pretrained(model_name_or_path, **load_kwargs)

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        hf_kwargs = {"torch_dtype": dtype, **kwargs}
        base_model = cls.load_clip_encoder(model_args.model_name_or_path, **hf_kwargs)

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

    @staticmethod
    def normalize_lora_paths(
        lora_name_or_path: Union[str, List[str], None],
    ) -> List[str]:
        if not lora_name_or_path:
            return []
        if isinstance(lora_name_or_path, str):
            return [lora_name_or_path]
        return list(lora_name_or_path)

    @classmethod
    def _merge_lora_into_encoder(
        cls,
        encoder: CLIPModel,
        adapter_path: str,
        merge_coeff: float,
    ) -> CLIPModel:
        lora_config = LoraConfig.from_pretrained(adapter_path)
        print_master(f"Merging LoRA from {adapter_path} (coeff={merge_coeff})")
        setattr(
            lora_config,
            "lora_alpha",
            int(lora_config.lora_alpha * merge_coeff),
        )
        lora_model = PeftModel.from_pretrained(
            encoder,
            adapter_path,
            config=lora_config,
        )
        return lora_model.merge_and_unload()

    @classmethod
    def load_merged_adapters(cls, model_args: ModelArguments, **kwargs):
        """
        Load base CLIP and sequentially merge each LoRA checkpoint in
        model_args.lora_name_or_path (in order).
        """
        print_master("-----START LOADING MERGED ADAPTERS-----")
        adapter_paths = cls.normalize_lora_paths(model_args.lora_name_or_path)
        if not adapter_paths:
            raise ValueError("load_merged_adapters requires --lora_name_or_path.")

        dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        hf_kwargs = {"torch_dtype": dtype, **kwargs}
        merge_coeff = model_args.lora_merge_coeff

        first_config = LoraConfig.from_pretrained(adapter_paths[0])
        encoder = cls.load_clip_encoder(
            first_config.base_model_name_or_path,
            **hf_kwargs,
        )
        for ix, adapter_path in enumerate(adapter_paths):
            if ix == len(adapter_paths) - 1:
                merge_coeff = 1.0
            encoder = cls._merge_lora_into_encoder(
                encoder, adapter_path, merge_coeff
            )

        return cls(
            encoder=encoder,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
        )
        print_master("-----END LOADING MERGED ADAPTERS-----")

    @classmethod
    def load(cls, model_args: ModelArguments, **kwargs):
        """
        Load a checkpoint produced by training.

        Symmetry with build():
        - build(lora=True): base from model_name_or_path + new LoRA adapters
        - load (PEFT checkpoint): base from adapter_config.base_model_name_or_path + adapter weights from checkpoint dir

        Pass multiple dirs via lora_name_or_path to merge them sequentially.
        """
        print_master("-----START LOADING-----")
        adapter_paths = cls.normalize_lora_paths(model_args.lora_name_or_path)
        if len(adapter_paths) > 1:
            return cls.load_merged_adapters(model_args, **kwargs)

        dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        hf_kwargs = {"torch_dtype": dtype, **kwargs}

        # Adapter checkpoint: same dir as trainer.save_model (adapter_config + weights)
        adapter_path = (
            adapter_paths[0]
            if adapter_paths
            else model_args.model_name_or_path
        )

        lora_config = None
        if model_args.lora or adapter_paths:
            try:
                lora_config = LoraConfig.from_pretrained(adapter_path)
            except Exception as e:
                print_master(str(e))
                print_master(
                    "Could not load LoRA config from adapter_path; "
                    "falling back to full CLIP weights."
                )

        if lora_config is not None:
            base_model = cls.load_clip_encoder(
                lora_config.base_model_name_or_path,
                **hf_kwargs,
            )
            print_master(f"Loading LoRA checkpoint from {adapter_path}")

            print_master(f"Merging LoRA with merge coefficient {model_args.lora_merge_coeff}")
            setattr(lora_config, "lora_alpha", int(lora_config.lora_alpha * model_args.lora_merge_coeff))
            print_master(f"lora_config.lora_alpha: {lora_config.lora_alpha}")

            lora_model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                config=lora_config,
            )

            encoder = lora_model.merge_and_unload()
        else:
            print_master(f"This is not a PEFT model!!! ")
            encoder = cls.load_clip_encoder(
                model_args.model_name_or_path,
                **hf_kwargs,
            )

        print_master("-----END LOADING-----")

        return cls(
            encoder=encoder,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
        )

    @classmethod
    def load_merge_build(
        cls, model_args: ModelArguments, training_args: TrainingArguments = None, **kwargs
    ):
        """
        Continual LoRA: sequentially merge prior adapter(s), then attach new LoRA.
        """
        print_master("-----START LOADING MERGE BUILD-----")
        
        adapter_paths = cls.normalize_lora_paths(model_args.lora_name_or_path)
        if not adapter_paths:
            raise ValueError("load_merge_build requires --lora_name_or_path.")
        if not model_args.lora:
            raise ValueError("load_merge_build requires --lora to initialize new adapters.")

        merged_model = cls.load_merged_adapters(model_args, **kwargs)

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=(
                model_args.lora_target_modules
                if model_args.lora_target_modules == "all-linear"
                else model_args.lora_target_modules.split(",")
            ),
            lora_dropout=model_args.lora_dropout,
            init_lora_weights="gaussian",
            use_dora=False,
            inference_mode=False,
        )
        encoder = get_peft_model(merged_model.encoder, lora_config)
        print_master("Initialized new LoRA on merged encoder")
        print_master(f"Active adapters: {encoder.active_adapters}")
        print_master("-----END LOADING MERGE BUILD-----")
        return cls(
            encoder=encoder,
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


class CCLIP(CLIPContrastiveModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.rep_dim
        enc_dtype = next(self.encoder.parameters()).dtype
        """
        Introduce a projector $h_\psi\colon \mathcal{Z}\rightarrow\mathcal{Z}$ 
        after the vision and text encoders, optimizing the model in the 
        projected space to keep the new and old feature spaces connected but 
        not identical.
        """
        self.cclip_projector = nn.Linear(
            hidden, hidden, bias=False, dtype=enc_dtype
        )
        
    def forward(
        self,
        qry: Optional[Dict[str, Tensor]] = None,
        tgt: Optional[Dict[str, Tensor]] = None,
        ckc=False,
        **kwargs,
    ):
        if ckc:
            if qry is None or tgt is None:
                raise ValueError("ckc=True requires both qry and tgt.")
            # q_bs = qry["input_ids"].size(0)
            # t_bs = tgt["input_ids"].size(0)
            # if q_bs != t_bs:
            #     raise ValueError(
            #         f"ckc qry/tgt batch sizes must match, got {q_bs} and {t_bs}."
            #     )
            return self.encode_ckc_reps(qry, tgt)

        return super().forward(qry=qry, tgt=tgt, **kwargs)


    def _project(self, reps: Tensor) -> Tensor:
        reps = self.cclip_projector(reps)
        if self.normalize:
            reps = F.normalize(reps, p=2, dim=-1)
        return reps

    def encode_projected_input(self, inputs: Dict[str, Tensor]) -> Tensor:
        return self._project(self.encode_input(inputs))

    def project_ckc_from_clip_reps(
        self, qry_reps: Tensor, tgt_reps: Tensor
    ) -> Tensor:
        """Project cached CLIP reps and interleave: [proj_q..., proj_t...]."""
        return torch.cat(
            [self._project(qry_reps), self._project(tgt_reps)], dim=0
        )

    def encode_ckc_reps(
        self, qry: Dict[str, Tensor], tgt: Dict[str, Tensor]
    ) -> Tensor:
        """Full encode + project path (non-GradCache fallback)."""
        qry_reps = self.encode_input(qry)
        tgt_reps = self.encode_input(tgt)
        return self.project_ckc_from_clip_reps(qry_reps, tgt_reps)


class HyperbolicCCLIP(CCLIP):
    """
    C-CLIP with CKC regularization in hyperbolic (Poincaré) space.

    New features: linear projector -> expmap0.
    Old cached sphere features: expmap0(α · y) at CKC time.
    CKC similarity should use hyperbolic distance (see build_ckc_loss).
    """

    def __init__(
        self,
        *args,
        curvature: float = 1.0,
        old_tangent_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.curvature = curvature
        self.old_tangent_scale = old_tangent_scale

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        hf_kwargs = {"torch_dtype": dtype, **kwargs}
        base_model = cls.load_clip_encoder(model_args.model_name_or_path, **hf_kwargs)

        if model_args.lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=(
                    model_args.lora_target_modules
                    if model_args.lora_target_modules == "all-linear"
                    else model_args.lora_target_modules.split(",")
                ),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=False,
                inference_mode=False,
            )
            base_model = get_peft_model(base_model, lora_config)

        return cls(
            encoder=base_model,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
            curvature=model_args.curvature,
            old_tangent_scale=model_args.old_tangent_scale,
        )

    def to_hyperbolic(self, reps: Tensor) -> Tensor:
        reps = project_to_ball(
            expmap0(reps, c=self.curvature),
            c=self.curvature,
        )
        return reps

    def _project(self, reps: Tensor) -> Tensor:
        reps = self.cclip_projector(reps)
        if self.normalize:
            reps = self.to_hyperbolic(reps)
        return reps

    def embed_old_ckc_reps(self, reps: Tensor) -> Tensor:
        return embed_sphere_to_ball(
            reps,
            scale=self.old_tangent_scale,
            c=self.curvature,
        )