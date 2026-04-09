import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pdb import set_trace as st
from pprint import pprint
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from peft import LoraConfig
from PIL import Image
from tevatron.colpali.losses import ColbertLoss
from tevatron.colpali.utils import get_params_info, init, set_seed, write_json
from tevatron.retriever.modeling.encoder import EncoderModel, EncoderOutput
from torch import Tensor, nn
from transformers import BatchEncoding, BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor
from transformers.utils.import_utils import is_flash_attn_2_available


class DenseModel(EncoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.use_smooth_max = False
        self.normalize_scores = True
        self.tau = 0.1 # Default from ColbertModule
        self.norm_tol = 1e-3
        self.pairwise_ce_loss = kwargs.get("pairwise_ce_loss", None)
        self.filter_false_negatives = kwargs.get("filter_false_negatives", None)
        self.pairwise_inbatch_neg_loss = kwargs.get("pairwise_inbatch_neg_loss", None)
        self.pairwise_inbatch_neg_loss_weight = kwargs.get("pairwise_inbatch_neg_loss_weight", None)
        # self.loss_func = ColbertLoss(
        #     temperature=0.02,
        #     normalize_scores=True,
        #     use_smooth_max=False,
        #     pos_aware_negative_filtering=False,
        # )

    def encode_query(self, qry, embedding_projection=True):
        query_hidden_states = self.encoder(**qry, embedding_projection=embedding_projection, return_dict=True)
        return query_hidden_states
        # query_hidden_states = query_hidden_states.last_hidden_state
        # return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg, embedding_projection=True):
        # encode passage is the same as encode query
        return self.encode_query(psg, embedding_projection=embedding_projection)
        
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
    
    def _smooth_max(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute smooth max via log-sum-exp along a given dimension.
        """
        return self.tau * torch.logsumexp(scores / self.tau, dim=dim)
    
    def _aggregate(
        self,
        scores_raw: torch.Tensor,
        use_smooth_max: bool,
        dim_max: int,
        dim_sum: int,
    ) -> torch.Tensor:
        """
        Aggregate token-level scores into document-level.

        Args:
            scores_raw (Tensor): Raw scores tensor.
            use_smooth_max (bool): Use smooth-max if True.
            dim_max (int): Dimension to perform max/logsumexp.
            dim_sum (int): Dimension to sum over after max.
        """
        # if use_smooth_max:
        #     return self._smooth_max(scores_raw, dim=dim_max).sum(dim=dim_sum)
        return scores_raw.amax(dim=dim_max).sum(dim=dim_sum)
    
    def _apply_normalization(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Normalize scores by query lengths and enforce bounds.

        Args:
            scores (Tensor): Unnormalized score matrix [B, C].
            lengths (Tensor): Query lengths [B].

        Returns:
            Tensor: Normalized scores.

        Raises:
            ValueError: If normalized scores exceed tolerance.
        """
        if scores.ndim == 2:
            normalized = scores / lengths.unsqueeze(1)
        else:
            normalized = scores / lengths

        # mn, mx = torch.aminmax(normalized)
        # if mn < -self.norm_tol or mx > 1 + self.norm_tol:
        #     print(
        #         f"Scores out of bounds after normalization: "
        #         f"min={mn.item():.4f}, max={mx.item():.4f}, tol={self.norm_tol}"
        #     )
        return normalized
    
    def compute_similarity(self, q_reps, p_reps):
        lengths = (q_reps[:, :, 0] != 0).sum(dim=1)
        raw = torch.einsum("bnd,csd->bcns", q_reps, p_reps)
        scores = self._aggregate(raw, use_smooth_max=False, dim_max=3, dim_sum=2)
        if self.normalize_scores:
            scores = self._apply_normalization(scores, lengths)
        return scores
    
    def get_index_and_masks(self, num_neg, scores):
        B = scores.size(0)
        device = scores.device
        
        # The positive passages
        target = torch.arange(
            B,
            device=device,
            dtype=torch.long,
        ) * num_neg

        # # A mask to filter out the positive and annotated negatives
        # no_filter_mask = torch.stack(
        #     [
        #         torch.arange(
        #             x,
        #             x + num_neg,
        #             device=scores_semantic.device,
        #             dtype=torch.long,
        #         )
        #         for x in target
        #     ]
        # )
        no_filter_mask = target.unsqueeze(1) + torch.arange(num_neg, device=device, dtype=torch.long)
        
        row_idx = torch.arange(
            B, device=device
        )

        return target, row_idx, no_filter_mask

    def mask_inbatch_negative_percentile(self, scores, target, row_idx, no_filter_mask=None, percentile=0.95):
        # semantic_thresholds = scores_semantic[row_idx, target] * percentile  # shape: [B]
        pos_scores = scores.gather(1, target.unsqueeze(1))
        threshold = pos_scores * percentile

        # mask = scores_semantic > semantic_thresholds.unsqueeze(1)  # Mask out those greater than the threshold
        mask = scores > threshold
        
        if no_filter_mask is not None:
            B = scores.size(0)
            device = scores.device
            
            rows = torch.arange(B, device=device).unsqueeze(1).expand_as(no_filter_mask)
            mask[rows, no_filter_mask] = False  # type: ignore # don't mask the positive and annotated negatives, only consider the other in-batch negatives

        # Apply the mask
        scores.masked_fill_(mask, float('-inf'))
        return scores
    
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, embedding_projection=True): # type: ignore
        q_reps = self.encode_query(query, embedding_projection) if query else None
        p_reps = self.encode_passage(passage, embedding_projection) if passage else None

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
                
            # scores, loss = self.loss_func(
            #     query_embeddings=q_reps,
            #     doc_embeddings=p_reps,
            #     offset=0,
            # )
            target = None
            row_idx = None
            no_filter_mask = None
            num_neg = p_reps.size(0) // q_reps.size(0)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            # target = target * (p_reps.size(0) // q_reps.size(0))
            
            # Create useful masks for calculating different losses
            # Which will be reused later on
            if target is None:
                target, row_idx, no_filter_mask = self.get_index_and_masks(num_neg, scores)

            # (Optional) Filter potential false negatives by filtering samples with simialrty greater than 95% compared to the ground truth
            if self.filter_false_negatives:
                scores = self.mask_inbatch_negative_percentile(scores, target, row_idx, no_filter_mask, percentile=0.95)
            
            if self.pairwise_ce_loss:
                top2 = scores.topk(2, dim=1).values
                pos_scores = scores[row_idx, target]
                neg_scores = torch.where(top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0])
                
                loss_pairwise_ce = F.softplus((neg_scores - pos_scores) / self.temperature).mean()
                
                if self.pairwise_inbatch_neg_loss:
                    loss_infonce_w_inbatch_neg = self.compute_loss(scores / self.temperature, target)
                    loss = (1 - self.pairwise_inbatch_neg_loss_weight) * loss_pairwise_ce + self.pairwise_inbatch_neg_loss_weight * loss_infonce_w_inbatch_neg
                    losses = {
                        "loss": loss,
                        "total_loss": loss.clone().detach(),
                        "loss_infonce_w_inbatch_neg": self.pairwise_inbatch_neg_loss_weight * loss_infonce_w_inbatch_neg.clone().detach(),
                        "loss_pairwise_ce": (1 - self.pairwise_inbatch_neg_loss_weight) * loss_pairwise_ce.clone().detach(),
                    }
                else:
                    loss = loss_pairwise_ce
                    losses = {
                        "loss": loss,
                        "loss_pairwise_ce": loss.clone().detach(),
                    }
            else:
                # Normal in-batch negatives with optional annotated hard negatives InfoNCE loss
                loss = self.compute_loss(scores / self.temperature, target)
                losses = {
                    "loss": loss,
                    "infonce_loss": loss.clone().detach(),
                }
            
            # if self.is_ddp:
            #     loss = loss * self.world_size  # counter average weight reduction
            return losses
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps) # self.loss_func.compute_similarity(q_reps, p_reps, offset=0)
            loss = None
            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )


class ColQwen3Processor(BaseVisualRetrieverProcessor, Qwen3VLProcessor):
    """
    Processor for ColQwen3.

    Args:
        *args: Variable length argument list to be passed to the parent `Qwen3VLProcessor` class.
        max_num_visual_tokens: The maximum number of visual tokens that can be processed by the model.
        **kwargs: Arbitrary keyword arguments to be passed to the parent `Qwen3VLProcessor` class.
    """

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )
        if "max_num_visual_tokens" in kwargs and kwargs["max_num_visual_tokens"]:
            patch_size = getattr(instance.image_processor, "patch_size", None)
            merge_size = getattr(instance.image_processor, "merge_size", None)
            if patch_size is None or merge_size is None:
                raise ValueError("Qwen3VL image processor is missing `patch_size` or `merge_size`.")
            tile = patch_size * merge_size
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * tile * tile
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels
            
        return instance

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColQwen3.

        Args:
            images: List of PIL images.
        """

        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        # Need to have the following code to batch images with different size together (i.e., fix "pixel_values" that has been flatten into 2d tensor by split them and pad to max_length of the batch)
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)
        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]
        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)

        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColQwen3.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self(
            text=texts,
            return_tensors="pt",
            # padding="longest",
            padding="max_length",
        )

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen3VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        patch_size = self.image_processor.patch_size

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask that identifies the image tokens in the batch.
        """
        return batch_images.input_ids == self.image_token_id


class ColQwen3(Qwen3VLModel):
    """
    ColQwen3 model implementation, following the architecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen3-VL backbone.

    Args:
        config (Qwen3VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    _checkpoint_conversion_mapping = {
        r"^base_model\.model\.custom_text_proj": "custom_text_proj",
        r"^model\.visual": "visual",
        r"^model\.language_model": "language_model",
        r"^model\.": "",
    }

    def __init__(
        self,
        config: Qwen3VLConfig,
        mask_non_image_embeddings: bool = False,
        **kwargs,
    ):
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        attn_impl = kwargs.pop("attn_implementation", None)
        use_cache = kwargs.pop("use_cache", None)

        super().__init__(config=config)

        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = self.config.text_config.hidden_size
        if hidden_size is None:
            raise ValueError("Unable to determine text hidden size for Qwen3VLConfig.")

        self.dim = 320
        self.custom_text_proj = nn.Linear(hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

        if dtype is not None:
            self.to(dtype=dtype)
        if use_cache is not None:
            self.config.use_cache = use_cache
        if attn_impl is not None and hasattr(self, "set_attn_implementation"):
            self.set_attn_implementation(attn_impl)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = dict(getattr(super(), "_checkpoint_conversion_mapping", {}))
            key_mapping.update(getattr(cls, "_checkpoint_conversion_mapping", {}))
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen3Processor` through unpadding
        # print("pixel_values before", kwargs["pixel_values"].shape)
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )
        # print("pixel_values after", kwargs["pixel_values"].shape)
        

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)
        
        # print("last_hidden_states before projection", last_hidden_states.shape)
        
        if isinstance(self.custom_text_proj, nn.Module) and kwargs["embedding_projection"]:

            proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

            # L2 normalization
            proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
            proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
            # print("last_hidden_states after projection", proj.shape)
        else:
            last_hidden_states = last_hidden_states / last_hidden_states.norm(dim=-1, keepdim=True)
            last_hidden_states = last_hidden_states * kwargs["attention_mask"].unsqueeze(-1)
            return last_hidden_states

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
