import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pdb import set_trace as st
from pprint import pprint
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from peft import LoraConfig
from PIL import Image

from tevatron.colpali.utils import get_params_info, init, set_seed, write_json
from torch import nn
from transformers import BatchEncoding, BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor
from transformers.utils.import_utils import is_flash_attn_2_available


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
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )
        

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

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
