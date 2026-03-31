import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin
from qwen_vl_utils import process_vision_info
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from PIL import Image
from pdb import set_trace as st
import torch

from arguments import DataArguments


logger = logging.getLogger(__name__)

N_AUGMENTATION_TOKENS = 10


@dataclass
class TrainCollator:
    data_args: DataArguments
    processor: ProcessorMixin


    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        
        query_messages = []
        for query in all_queries:
            message = [
                {
                    'role': 'user',
                    'content': [
                        # {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height': 1, 'resized_width': 1},
                        # {'type': 'image', 'image': image},
                        {'type': 'text', 'text': f'Query: {query}'}
                    ]
                }
            ]
            query_messages.append(message)

        passage_messages = []
        for idx in range(len(all_passages)):
            image = all_passages[idx]
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image},#, 'resized_height': 748, 'resized_width': 748},
                        {'type': 'text', 'text': f'Describe the image.'}
                    ]
                }
            ]
            passage_messages.append(message)
        
        query_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) + '<|endoftext|>' * N_AUGMENTATION_TOKENS
            for msg in query_messages
        ]

        passage_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) + '<|endoftext|>'
            for msg in passage_messages
        ]

        query_image_inputs, _ = process_vision_info(query_messages, image_patch_size=16)
        passage_image_inputs, _ = process_vision_info(passage_messages, image_patch_size=16)

        query_inputs = self.processor(
            text=query_texts,
            images=query_image_inputs,
            # videos=query_video_inputs,
            padding="longest",
            return_tensors="pt",
        )
        # print("Query")
        # for k, v in query_inputs.items():
        #     print(k, v.shape)

        passage_inputs = self.processor(
            text=passage_texts,
            images=passage_image_inputs,
            # videos=passage_video_inputs,
            padding="longest",
            return_tensors="pt",
        )
        # print("before pixel_value fix passage")
        # for k, v in passage_inputs.items():
        #     print(k, v.shape)
            
        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        offsets = passage_inputs["image_grid_thw"][:, 1] * passage_inputs["image_grid_thw"][:, 2]  # (batch_size,)

        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(passage_inputs["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        passage_inputs["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)
        # print("after pixel_value fix passage")
        # for k, v in passage_inputs.items():
        #     print(k, v.shape)

        return query_inputs, passage_inputs
    

@dataclass
class EncodeCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        passages = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        
        if self.data_args.encode_is_query:
            query_texts = []
            for text in passages:
                query_texts.append(text)
            inputs = self.processor(
                text=query_texts,
                padding="max_length",
                max_length=max_length,
            )
        else:
            images = []
            for idx in range(len(passages)):
                image = passages[idx]
                images.append(image)
            # inputs = self.processor(
            #     images=images,
            # )
            # if isinstance(self.processor, ColQwen2_5_Processor):
            inputs = self.processor.process_images(images)
            # else:
            #     raise ValueError("Not ColQwen2_5_Processor")
        
        return text_ids, inputs


@dataclass
class MultiModalEncodeCollator:
    """
    collator for text-visual data.
    """
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        content_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[2] for x in features]

        collated_inputs = {
            "texts": self.processor.process_texts(texts) if None not in texts else None,
            "images": self.processor.process_images(images) if None not in images else None
        }
        
        return content_ids, collated_inputs