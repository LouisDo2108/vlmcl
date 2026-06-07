from dataclasses import dataclass, field
from itertools import repeat
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from tevatron.hyperbolic.arguments import DataArguments
from tevatron.hyperbolic.utils import clip_text_max_length
from transformers import CLIPProcessor

Example = Tuple[str, Image.Image | None]


def split_dense_inputs(model_input: dict, chunk_size: int):
    """Split GradCache inputs when the collator already produced tensors."""
    arg_key = next(iter(model_input))
    arg_val = model_input[arg_key]
    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [
        dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))
    ]
    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if isinstance(x, dict):
        query_rep, target_rep = x["qry_reps"], x["tgt_reps"]
    elif isinstance(x, tuple):
        query_rep, target_rep = x
    else:
        raise ValueError(f"Unsupported type: {type(x)}")
    if query_rep is None:
        return target_rep
    if target_rep is None:
        return query_rep
    return query_rep, target_rep


def _parse_examples(examples: Sequence) -> Tuple[List[str], List[Image.Image | None], List[bool], List[bool]]:
    texts: List[str] = []
    images: List[Image.Image | None] = []
    has_text_flags: List[bool] = []
    has_image_flags: List[bool] = []

    for example in examples:
        if example is None:
            text, image = "", None
        else:
            text, image = example

        has_text = bool(text and str(text).strip())
        has_image = image is not None
        texts.append(text if has_text else "")
        images.append(image if has_image else None)
        has_text_flags.append(has_text)
        has_image_flags.append(has_image)

    return texts, images, has_text_flags, has_image_flags


@dataclass
class CLIPCollator:
    """
    Collate MMEB (qry, pos) pairs into tokenized batches for CLIP.

    Each dataset item: ((qry_text, qry_image), (pos_text, pos_image)).
    Returns (qry_batch, pos_batch).
    """

    data_args: DataArguments
    processor: CLIPProcessor
    return_indices: bool = False
    _max_length: int = field(init=False, repr=False)
    _pixel_shape: Tuple[int, int, int] = field(init=False, repr=False)

    def __post_init__(self):
        self._max_length = clip_text_max_length(
            self.processor.tokenizer, self.data_args.max_len
        )
        crop_size = self.processor.image_processor.crop_size
        channels = getattr(self.processor.image_processor, "num_channels", 3)
        self._pixel_shape = (
            channels,
            crop_size["height"],
            crop_size["width"],
        )

    def __call__(self, examples):
        indices = None
        if self.return_indices:
            indices = [ex[1] for ex in examples]
            examples = [ex[0] for ex in examples]
        qry_examples = [ex[0] for ex in examples]
        pos_examples = [ex[1] for ex in examples]
        batches = self._batch(qry_examples), self._batch(pos_examples)
        if indices is None:
            return batches
        return (*batches, torch.tensor(indices, dtype=torch.long))

    def _batch(self, examples: Sequence) -> dict:
        """
        Build a batch that may mix text-only, image-only, and multimodal rows.

        Zero pixel_values / empty tokenization are batching placeholders only;
        has_image / has_text tell the model which tower(s) to use per row.
        """
        texts, images, has_text_flags, has_image_flags = _parse_examples(examples)
        batch_size = len(examples)

        enc = self.processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if input_ids.size(1) > self._max_length:
            input_ids = input_ids[:, : self._max_length]
            attention_mask = attention_mask[:, : self._max_length]

        pixel_values = torch.zeros(batch_size, *self._pixel_shape)
        image_indices = [i for i, img in enumerate(images) if img is not None]
        if image_indices:
            batch_images = [images[i] for i in image_indices]
            processed = self.processor.image_processor(
                images=batch_images,
                return_tensors="pt",
            )["pixel_values"]
            pixel_values[image_indices] = processed

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "has_image": torch.tensor(has_image_flags, dtype=torch.bool),
            "has_text": torch.tensor(has_text_flags, dtype=torch.bool),
        }


@dataclass
class CLIPEvalCollator:
    """
    Eval collator for MMEB (mirrors VLM2Vec EvalCollator).

    Each example is (text, image). Returns one batch dict for model(qry=...) or model(tgt=...).
    """

    data_args: DataArguments
    processor: CLIPProcessor
    _batch_collator: CLIPCollator = field(init=False, repr=False)

    def __post_init__(self):
        self._batch_collator = CLIPCollator(self.data_args, self.processor)

    def __call__(self, examples):
        return self._batch_collator._batch(examples)
