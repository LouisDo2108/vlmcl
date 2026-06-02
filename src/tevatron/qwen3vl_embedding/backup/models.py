import os
import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging

from PIL import Image
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from qwen_vl_utils.vision_process import process_vision_info
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from peft import LoraConfig
from PIL import Image
# from tevatron.colpali.losses import ColbertLoss
from tevatron.retriever.modeling.encoder import EncoderModel, EncoderOutput
from torch import Tensor, nn
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from pdb import set_trace as st


# Constants for configuration
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
PAD_TOKEN = "<|endoftext|>"


logger = logging.getLogger(__name__)

class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder: PreTrainedModel,
                 processor,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 max_length: int = MAX_LENGTH,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS,
                 total_pixels: int = MAX_TOTAL_PIXELS,
                 fps: float = FPS,
                 max_frames: int = MAX_FRAMES,
                 default_instruction: str = "Represent the user's input.",
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.config = encoder.model.config
        self.encoder = encoder
        self.processor = processor
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

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

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)
            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args,
            train_args,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
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
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.model.save_pretrained(output_dir)


class DenseModel(EncoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_false_negatives = kwargs.get("filter_false_negatives", None)
        self.default_instruction = "Represent the user's input.",
        
    # Truncate token sequence to a specified max length
    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids

        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for token_idx in token_ids if token_idx in special_token_ids)
        num_non_special_to_keep = max_length - num_special

        final_token_ids = []
        non_special_kept_count = 0
        # Ensure retention of special tokens while truncating the rest
        for token_idx in token_ids:
            if token_idx in special_token_ids:
                final_token_ids.append(token_idx)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(token_idx)
                non_special_kept_count += 1
        return final_token_ids

    def format_model_input(
        self, 
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:

        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        # Initialize conversation with system prompts
        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]

        # Normalize text input to list
        if text is None:
            texts = []
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image
        
        # Normalize video input to list
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            # Assume it's a list of videos
            videos = video

        # Add text, image, or video content to conversation
        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation

        # Process each video
        for vid in videos:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}
            
            if isinstance(vid, list):
                # Video as frame sequence
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                # Video as file path
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")

            # Add video input to content
            if video_content:
                content.append({
                    'type': 'video', 
                    'video': video_content,
                    **video_kwargs
                })

        # Process each image
        for img in images:
            image_content = None
            
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append({
                    'type': 'image', 
                    'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        # Process each text
        for txt in texts:
            content.append({'type': 'text', 'text': txt})

        return conversation

    # Preprocess input conversations for model consumption
    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            # images, video_inputs, video_kwargs = process_vision_info(
            #     conversations, image_patch_size=16,
            #     return_video_metadata=True, return_video_kwargs=True
            # )
            images, _ = process_vision_info(
                conversations, image_patch_size=16,
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )

        # if video_inputs is not None:
        #     videos, video_metadata = zip(*video_inputs)
        #     videos = list(videos)
        #     video_metadata = list(video_metadata)
        # else:
        #     videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, 
            # videos=videos, video_metadata=video_metadata, 
            # truncation=True, 
            max_length=self.max_length, padding=True, do_resize=False, 
            # **video_kwargs
            # padding="longest",
            return_tensors='pt',
        )
        # offsets = inputs["image_grid_thw"][:, 1] * inputs["image_grid_thw"][:, 2]
        # pixel_values = list(torch.split(inputs["pixel_values"], offsets.tolist()))
        # inputs["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True)
        return inputs

    # Pool the last hidden state by attention mask for embeddings
    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    # Process inputs to generate normalized embeddings
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> tuple:
        conversations = [self.format_model_input(
            text=ele.get('text'),
            image=ele.get('image'),
            # video=ele.get('video'),
            # instruction=ele.get('instruction'),
            # fps=ele.get('fps'),
            # max_frames=ele.get('max_frames')
        ) for ele in inputs]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.encoder.device) for k, v in processed_inputs.items()}

        # for k, v in processed_inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        
        outputs = self.encoder(
            **processed_inputs,
            use_cache=False, # added
            output_hidden_states=True, # added
        )
        embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

        # Normalize the embeddings if specified
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


    def encode_query(self, qry, embedding_projection=True):
        query_hidden_states = self.process(qry)
        return query_hidden_states
    
    def encode_passage(self, psg, embedding_projection=True):
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
    
    def _aggregate(
        self,
        scores_raw: torch.Tensor,
        dim_max: int,
        dim_sum: int,
    ) -> torch.Tensor:
        """
        Aggregate token-level scores into document-level.

        Args:
            scores_raw (Tensor): Raw scores tensor.
            dim_max (int): Dimension to perform max/logsumexp.
            dim_sum (int): Dimension to sum over after max.
        """
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
        return normalized

    def get_index_and_masks(self, num_neg, scores):
        B = scores.size(0)
        device = scores.device
        
        # The positive passages
        target = torch.arange(
            B,
            device=device,
            dtype=torch.long,
        ) * num_neg

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
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            bs = q_reps.size(0)
            device = q_reps.device
            num_neg = p_reps.size(0) // bs
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(bs, -1)

            # 3. Indexing & Masking
            target, row_idx, no_filter_mask = self.get_index_and_masks(num_neg, scores)

            # 4. Optional False Negative Filtering
            if self.filter_false_negatives:
                scores = self.mask_inbatch_negative_percentile(
                    scores, target, row_idx, no_filter_mask, percentile=0.95
                )
                
            # Normal in-batch negatives InfoNCE loss with optional annotated hard negatives
            loss = self.compute_loss(scores / self.temperature, target)
            # ndcgloss = neuralNDCG_transposed(
            #     y_pred=scores.gather(1, no_filter_mask),
            #     y_true=torch.arange(num_neg, device=device, dtype=torch.float32).expand(bs, -1), 
            #     device=device
            # )
            losses = {
                "loss": loss,# + ndcgloss,
                "infonce_loss": loss.clone().detach(),
                # "ndcg_loss": ndcgloss.clone().detach(),
            }
            
            # if self.is_ddp:
            #     loss = loss * self.world_size  # counter average weight reduction
            return losses
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps,
            )
            
    def _calc_pairwise_ce_loss(self, scores: Tensor, target: Tensor, row_idx: Tensor) -> dict:
        """Calculates Pairwise CE loss, optionally mixed with InfoNCE."""
        top2 = scores.topk(2, dim=1).values
        pos_scores = scores[row_idx, target]
        
        # Determine neg scores: if top1 is positive, take top2; else take top1
        neg_scores = torch.where(top2[:, 0] == pos_scores, top2[:, 1], top2[:, 0])
        
        loss_pairwise_ce = F.softplus((neg_scores - pos_scores) / self.temperature).mean()
        
        if self.pairwise_inbatch_neg_loss:
            loss_infonce = self.compute_loss(scores / self.temperature, target)

            weight = self.pairwise_inbatch_neg_loss_weight
            loss = (1 - weight) * loss_pairwise_ce + weight * loss_infonce
            
            return {
                "loss": loss,
                "total_loss": loss.clone().detach(),
                "loss_infonce_w_inbatch_neg": weight * loss_infonce.clone().detach(),
                "loss_pairwise_ce": (1 - weight) * loss_pairwise_ce.clone().detach(),
            }
        else:
            return {
                "loss": loss_pairwise_ce,
                "loss_pairwise_ce": loss_pairwise_ce.clone().detach(),
            }

# Define output structure for embeddings
@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None

# Define model class to compute embeddings
class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Extract video features from model
    def get_video_features(self, pixel_values_videos: torch.FloatTensor,
                           video_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    # Extract image features from model
    def get_image_features(self, pixel_values: torch.FloatTensor,
                           image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules accessible through properties
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # Forward pass through model with input parameters
    # @check_model_inputs
    def forward(self,
                input_ids: torch.LongTensor = None,
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
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        # Pass inputs through the model
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
        # Return the model output
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

def sample_frames(frames: List[Union[str, Image.Image]], max_segments: int) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [ frames[frame_idx] for frame_idx in frame_id_list ]
    return sampled_frames

def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    if path.startswith(('http://', 'https://')):
        # Parse URL to remove query parameters
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    
    # Check file extension
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions

def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    
    if isinstance(video, list) and len(video) > 0:
        # Check first element to determine the type
        first_elem = video[0]
        
        if isinstance(first_elem, Image.Image):
            return True
        
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    
    return False

# Define embedder class for processing inputs and generating embeddings
class Qwen3VLEmbedder():
    def __init__(
        self, 
        model_name_or_path: str, 
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        **kwargs
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames

        self.default_instruction = default_instruction

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right'
        )
        
         # Mimic colqwen3
        patch_size = getattr(self.processor.image_processor, "patch_size", None)
        merge_size = getattr(self.processor.image_processor, "merge_size", None)
        if patch_size is None or merge_size is None:
                raise ValueError("Qwen3VL image processor is missing `patch_size` or `merge_size`.")
        
        tile = patch_size * merge_size
        self.processor.image_processor.max_pixels = 768 * tile * tile
        self.processor.image_processor.size["longest_edge"] = self.processor.image_processor.max_pixels
        # self.model.eval()

    # @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            **inputs,
            use_cache=False, # added
            output_hidden_states=True, # added
        )
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'attention_mask': inputs.get('attention_mask')
        }

    # Truncate token sequence to a specified max length
    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids

        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for token_idx in token_ids if token_idx in special_token_ids)
        num_non_special_to_keep = max_length - num_special

        final_token_ids = []
        non_special_kept_count = 0
        # Ensure retention of special tokens while truncating the rest
        for token_idx in token_ids:
            if token_idx in special_token_ids:
                final_token_ids.append(token_idx)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(token_idx)
                non_special_kept_count += 1
        return final_token_ids

    def format_model_input(
        self, 
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:

        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        # Initialize conversation with system prompts
        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]

        # Normalize text input to list
        if text is None:
            texts = []
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image
        
        # Normalize video input to list
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            # Assume it's a list of videos
            videos = video

        # Add text, image, or video content to conversation
        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation

        # Process each video
        for vid in videos:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}
            
            if isinstance(vid, list):
                # Video as frame sequence
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                # Video as file path
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")

            # Add video input to content
            if video_content:
                content.append({
                    'type': 'video', 
                    'video': video_content,
                    **video_kwargs
                })

        # Process each image
        for img in images:
            image_content = None
            
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append({
                    'type': 'image', 
                    'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        # Process each text
        for txt in texts:
            content.append({'type': 'text', 'text': txt})

        return conversation

    # Preprocess input conversations for model consumption
    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            # images, video_inputs, video_kwargs = process_vision_info(
            #     conversations, image_patch_size=16,
            #     return_video_metadata=True, return_video_kwargs=True
            # )
            images, _ = process_vision_info(
                conversations, image_patch_size=16,
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )

        # if video_inputs is not None:
        #     videos, video_metadata = zip(*video_inputs)
        #     videos = list(videos)
        #     video_metadata = list(video_metadata)
        # else:
        #     videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, 
            # videos=videos, video_metadata=video_metadata, 
            # truncation=True, 
            max_length=self.max_length, padding=True, do_resize=False, 
            # **video_kwargs
            # padding="longest",
            return_tensors='pt',
        )
        # offsets = inputs["image_grid_thw"][:, 1] * inputs["image_grid_thw"][:, 2]
        # pixel_values = list(torch.split(inputs["pixel_values"], offsets.tolist()))
        # inputs["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True)
        return inputs

    # Pool the last hidden state by attention mask for embeddings
    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    # Process inputs to generate normalized embeddings
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> tuple:
        conversations = [self.format_model_input(
            text=ele.get('text'),
            image=ele.get('image'),
            # video=ele.get('video'),
            # instruction=ele.get('instruction'),
            # fps=ele.get('fps'),
            # max_frames=ele.get('max_frames')
        ) for ele in inputs]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.model.device) for k, v in processed_inputs.items()}

        for k, v in processed_inputs.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        
        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

        # Normalize the embeddings if specified
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
