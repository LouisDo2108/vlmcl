"""
Models module for Qwen3 VL Embeddings V2.

This module provides improved model implementations for multimodal embeddings
using Qwen3-VL, following Tevatron framework conventions with enhanced code quality,
better type hints, and modular design.
"""

import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from tevatron.retriever.modeling.encoder import EncoderModel as TevatronEncoderModel
from tevatron.retriever.modeling.encoder import EncoderOutput
from transformers import AutoModel, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLProcessor,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .arguments import DataArguments, ModelArguments

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1.0
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
PAD_TOKEN = ""


# =============================================================================
# Helper Functions
# =============================================================================

def sample_frames(
    frames: List[Union[str, Image.Image]], 
    max_segments: int
) -> List[Union[str, Image.Image]]:
    """
    Sample frames from a video sequence using uniform sampling.
    
    Args:
        frames: List of frame paths or PIL Images.
        max_segments: Maximum number of frames to sample.
        
    Returns:
        List of sampled frames.
    """
    duration = len(frames)
    if duration <= max_segments:
        return frames
    
    frame_indices = np.linspace(0, duration - 1, max_segments, dtype=int).tolist()
    return [frames[idx] for idx in frame_indices]


def is_image_path(path: str) -> bool:
    """
    Check if a string path points to an image file.
    
    Args:
        path: File path or URL to check.
        
    Returns:
        True if the path has an image extension, False otherwise.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    if path.startswith(('http://', 'https://')):
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def is_video_input(video: Any) -> bool:
    """
    Determine if input represents a video.
    
    Args:
        video: Input to check (string path, list of frames, etc.).
        
    Returns:
        True if the input appears to be a video, False otherwise.
    """
    if isinstance(video, str):
        return True
    
    if isinstance(video, list) and len(video) > 0:
        first_elem = video[0]
        if isinstance(first_elem, Image.Image):
            return True
        if isinstance(first_elem, str) and is_image_path(first_elem):
            return True
    
    return False


# =============================================================================
# Base Encoder Model
# =============================================================================

class EncoderModel(nn.Module):
    """
    Base encoder model for retrieval tasks.
    
    This class provides common functionality for encoding queries and passages,
    including distributed training support and contrastive loss computation.
    """
    
    TRANSFORMER_CLS = AutoModel
    
    def __init__(
        self,
        encoder: PreTrainedModel,
        processor: Any,
        pooling: str = 'cls',
        normalize: bool = False,
        temperature: float = 1.0,
        max_length: int = MAX_LENGTH,
    ):
        super().__init__()
        
        self.config = encoder.model.config
        self.encoder = encoder
        self.processor = processor
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        
        # Distributed training setup
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        self.max_length = max_length
    
    def forward(
        self, 
        query: Optional[Dict[str, torch.Tensor]] = None, 
        passage: Optional[Dict[str, torch.Tensor]] = None
    ) -> EncoderOutput:
        """
        Forward pass for training and inference.
        
        Args:
            query: Tokenized query inputs.
            passage: Tokenized passage inputs.
            
        Returns:
            EncoderOutput containing embeddings and optionally loss/scores.
        """
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None
        
        # Inference mode
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)
        
        # Training mode
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            
            loss = self.compute_loss(scores / self.temperature, target)
            if self.is_ddp:
                loss = loss * self.world_size
            
            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )
        
        # Evaluation mode
        scores = self.compute_similarity(q_reps, p_reps)
        return EncoderOutput(
            loss=None,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    
    def encode_query(self, qry: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode query inputs to embeddings."""
        raise NotImplementedError("Subclasses must implement encode_query")
    
    def encode_passage(self, psg: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode passage inputs to embeddings."""
        raise NotImplementedError("Subclasses must implement encode_passage")
    
    def compute_similarity(self, q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix between query and passage embeddings."""
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    
    def compute_loss(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss from similarity scores."""
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing for memory efficiency."""
        self.encoder.model.gradient_checkpointing_enable()
    
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Gather tensors across distributed processes."""
        if t is None:
            return None
        
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)
    
    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: Any,
        **hf_kwargs,
    ) -> "EncoderModel":
        """
        Build encoder model from pretrained checkpoint.
        
        Args:
            model_args: Model configuration arguments.
            train_args: Training configuration arguments.
            **hf_kwargs: Additional keyword arguments for HuggingFace model loading.
            
        Returns:
            Initialized encoder model.
        """
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.model_name_or_path, **hf_kwargs
        )
        
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        # LoRA configuration
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(
                    model_args.lora_name_or_path, **hf_kwargs
                )
                lora_model = PeftModel.from_pretrained(
                    base_model, model_args.lora_name_or_path, is_trainable=True
                )
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            
            return cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        
        return cls(
            encoder=base_model,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
        )
    
    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        pooling: str = 'cls',
        normalize: bool = False,
        lora_name_or_path: Optional[str] = None,
        **hf_kwargs
    ) -> "EncoderModel":
        """
        Load encoder model from checkpoint.
        
        Args:
            model_name_or_path: Path to model checkpoint.
            pooling: Pooling strategy.
            normalize: Whether to normalize embeddings.
            lora_name_or_path: Optional LoRA adapter path.
            **hf_kwargs: Additional keyword arguments.
            
        Returns:
            Loaded encoder model.
        """
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_name_or_path, **hf_kwargs
        )
        
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(
                base_model, lora_name_or_path, config=lora_config
            )
            lora_model = lora_model.merge_and_unload()
            return cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
            )
        
        return cls(
            encoder=base_model,
            pooling=pooling,
            normalize=normalize,
        )
    
    def save(self, output_dir: str):
        """Save model checkpoint."""
        self.encoder.model.save_pretrained(output_dir)


# =============================================================================
# Dense Retrieval Model
# =============================================================================

@dataclass
class Qwen3VLForEmbeddingOutput:
    """Output dataclass for Qwen3VL embedding model."""
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """
    Qwen3-VL model adapted for embedding generation.
    
    This class wraps the Qwen3-VL model to produce embeddings suitable
    for retrieval tasks.
    """
    
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config_class = Qwen3VLConfig
    
    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embedding layer."""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Module):
        """Set input embedding layer."""
        self.model.set_input_embeddings(value)
    
    def set_decoder(self, decoder: nn.Module):
        """Set decoder module."""
        self.model.set_decoder(decoder)
    
    def get_decoder(self) -> nn.Module:
        """Get decoder module."""
        return self.model.get_decoder()
    
    def get_video_features(
        self, 
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """Extract video features from the model."""
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)
    
    def get_image_features(
        self, 
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """Extract image features from the model."""
        return self.model.get_image_features(pixel_values, image_grid_thw)
    
    @property
    def language_model(self) -> nn.Module:
        """Get language model component."""
        return self.model.language_model
    
    @property
    def visual(self) -> nn.Module:
        """Get visual encoder component."""
        return self.model.visual
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3VLForEmbeddingOutput:
        """
        Forward pass through the embedding model.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            past_key_values: Cached key-value states.
            inputs_embeds: Input embeddings.
            pixel_values: Image pixel values.
            pixel_values_videos: Video pixel values.
            image_grid_thw: Image grid dimensions.
            video_grid_thw: Video grid dimensions.
            cache_position: Cache position indices.
            logits_to_keep: Number of logits to keep.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Qwen3VLForEmbeddingOutput with last hidden states and attention mask.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class DenseModel(EncoderModel):
    """
    Dense retrieval model for Qwen3-VL.
    
    This class implements dense retrieval with multimodal support,
    including text, image, and video encoding capabilities.
    """
    
    def __init__(
        self,
        *args,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        filter_false_negatives: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        self.filter_false_negatives = filter_false_negatives
    
    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        """
        Truncate token sequence while preserving special tokens.
        
        Args:
            token_ids: List of token IDs.
            max_length: Maximum sequence length.
            
        Returns:
            Truncated token ID list.
        """
        if len(token_ids) <= max_length:
            return token_ids
        
        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for tid in token_ids if tid in special_token_ids)
        num_non_special_to_keep = max_length - num_special
        
        final_token_ids = []
        non_special_kept_count = 0
        
        for tid in token_ids:
            if tid in special_token_ids:
                final_token_ids.append(tid)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(tid)
                non_special_kept_count += 1
        
        return final_token_ids
    
    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[
            List[Union[str, List[Union[str, Image.Image]]]], 
            str, 
            List[Union[str, Image.Image]]
        ]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format input data into conversation format for the model.
        
        Args:
            text: Text content(s).
            image: Image content(s) as paths or PIL Images.
            video: Video content(s) as paths or frame sequences.
            instruction: Instruction prompt.
            fps: Frames per second for video.
            max_frames: Maximum frames to sample.
            
        Returns:
            Formatted conversation list.
        """
        # Normalize instruction
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'
        
        # Initialize conversation structure
        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]
        
        # Normalize inputs to lists
        texts = [] if text is None else ([text] if isinstance(text, str) else text)
        images = [] if image is None else ([image] if not isinstance(image, list) else image)
        
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            videos = video
        
        # Handle empty input
        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation
        
        # Process videos
        for vid in videos:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}
            
            if isinstance(vid, list):
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")
            
            if video_content:
                content.append({
                    'type': 'video', 
                    'video': video_content,
                    **video_kwargs
                })
        
        # Process images
        for img in images:
            image_content = None
            
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")
            
            if image_content:
                content.append({
                    'type': 'image', 
                    'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })
        
        # Process text
        for txt in texts:
            content.append({'type': 'text', 'text': txt})
        
        return conversation
    
    def _preprocess_inputs(
        self, 
        conversations: List[List[Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess conversations for model consumption.
        
        Args:
            conversations: List of formatted conversations.
            
        Returns:
            Dictionary of preprocessed tensors.
        """
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        
        try:
            images, _ = process_vision_info(
                conversations, image_patch_size=16,
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )
        
        inputs = self.processor(
            text=text, 
            images=images, 
            max_length=self.max_length, 
            padding=True, 
            do_resize=False, 
            return_tensors='pt',
        )
        
        return inputs
    
    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool embeddings using the last valid token position.
        
        Args:
            hidden_state: Hidden states from the model.
            attention_mask: Attention mask indicating valid positions.
            
        Returns:
            Pooled embeddings.
        """
        flipped_mask = attention_mask.flip(dims=[1])
        last_one_positions = flipped_mask.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]
    
    def process(
        self, 
        inputs: List[Dict[str, Any]], 
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Process inputs to generate normalized embeddings.
        
        Args:
            inputs: List of input dictionaries.
            normalize: Whether to L2-normalize embeddings.
            
        Returns:
            Normalized embedding tensor.
        """
        conversations = [
            self.format_model_input(
                text=ele.get('text'),
                image=ele.get('image'),
            ) for ele in inputs
        ]
        
        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.encoder.device) for k, v in processed_inputs.items()}
        
        outputs = self.encoder(
            **processed_inputs,
            use_cache=False,
            output_hidden_states=True,
        )
        
        embeddings = self._pooling_last(outputs.last_hidden_state, outputs.attention_mask)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def encode_query(
        self, 
        qry: Union[Dict[str, torch.Tensor], List[Dict[str, Any]]],
        embedding_projection: bool = True
    ) -> torch.Tensor:
        """
        Encode query inputs to embeddings.
        
        Args:
            qry: Query inputs (tokenized or raw).
            embedding_projection: Whether to apply embedding projection.
            
        Returns:
            Query embeddings.
        """
        if isinstance(qry, dict):
            # Already tokenized
            query_hidden_states = self.process([qry])
        else:
            query_hidden_states = self.process(qry)
        return query_hidden_states
    
    def encode_passage(
        self, 
        psg: Union[Dict[str, torch.Tensor], List[Dict[str, Any]]],
        embedding_projection: bool = True
    ) -> torch.Tensor:
        """
        Encode passage inputs to embeddings.
        
        Args:
            psg: Passage inputs (tokenized or raw).
            embedding_projection: Whether to apply embedding projection.
            
        Returns:
            Passage embeddings.
        """
        return self.encode_query(psg, embedding_projection)
    
    def forward(
        self, 
        query: Optional[Union[Dict[str, torch.Tensor], List[Dict[str, Any]]]] = None, 
        passage: Optional[Union[Dict[str, torch.Tensor], List[Dict[str, Any]]]] = None,
        embedding_projection: bool = True
    ) -> Union[EncoderOutput, Dict[str, torch.Tensor]]:
        """
        Forward pass for training and inference.
        
        Args:
            query: Query inputs.
            passage: Passage inputs.
            embedding_projection: Whether to apply embedding projection.
            
        Returns:
            EncoderOutput for inference, Dict of losses for training.
        """
        q_reps = self.encode_query(query, embedding_projection) if query else None
        p_reps = self.encode_passage(passage, embedding_projection) if passage else None
        
        # Inference mode
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)
        
        # Training mode
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            
            bs = q_reps.size(0)
            num_neg = p_reps.size(0) // bs
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(bs, -1)
            
            # Get indices and masks
            target, row_idx, no_filter_mask = self._get_index_and_masks(num_neg, scores)
            
            # Optional false negative filtering
            if self.filter_false_negatives:
                scores = self._mask_inbatch_negative_percentile(
                    scores, target, row_idx, no_filter_mask, percentile=0.95
                )
            
            loss = self.compute_loss(scores / self.temperature, target)
            
            return {
                "loss": loss,
                "infonce_loss": loss.clone().detach(),
            }
        
        # Evaluation mode
        scores = self.compute_similarity(q_reps, p_reps)
        return EncoderOutput(
            loss=None,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    
    def _get_index_and_masks(
        self, 
        num_neg: int, 
        scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get target indices and masks for loss computation.
        
        Args:
            num_neg: Number of negatives per query.
            scores: Similarity scores.
            
        Returns:
            Tuple of (target, row_idx, no_filter_mask).
        """
        B = scores.size(0)
        device = scores.device
        
        target = torch.arange(B, device=device, dtype=torch.long) * num_neg
        no_filter_mask = target.unsqueeze(1) + torch.arange(num_neg, device=device, dtype=torch.long)
        row_idx = torch.arange(B, device=device)
        
        return target, row_idx, no_filter_mask
    
    def _mask_inbatch_negative_percentile(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        row_idx: torch.Tensor,
        no_filter_mask: Optional[torch.Tensor] = None,
        percentile: float = 0.95
    ) -> torch.Tensor:
        """
        Mask in-batch negatives based on percentile threshold.
        
        Args:
            scores: Similarity scores.
            target: Positive passage indices.
            row_idx: Row indices.
            no_filter_mask: Mask for passages that shouldn't be filtered.
            percentile: Percentile threshold for masking.
            
        Returns:
            Scores with masked negatives.
        """
        pos_scores = scores.gather(1, target.unsqueeze(1))
        threshold = pos_scores * percentile
        
        mask = scores > threshold
        
        if no_filter_mask is not None:
            B = scores.size(0)
            device = scores.device
            rows = torch.arange(B, device=device).unsqueeze(1).expand_as(no_filter_mask)
            mask[rows, no_filter_mask] = False
        
        scores.masked_fill_(mask, float('-inf'))
        return scores