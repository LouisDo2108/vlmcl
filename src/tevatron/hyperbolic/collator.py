from dataclasses import dataclass
from itertools import repeat

import torch
from tevatron.hyperbolic.arguments import DataArguments
from tevatron.hyperbolic.utils import clip_text_max_length
from transformers import CLIPProcessor


def split_dense_inputs(model_input: dict, chunk_size: int):
    """Split GradCache inputs when the collator already produced tensors."""
    # # Preferred CLIP path: model_input is already the inner batch dict
    # # {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
    # if all(isinstance(v, torch.Tensor) for v in model_input.values()):
    #     keys = list(model_input.keys())
    #     chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
    #     return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    # Backward compatible path: wrapped dict {"qry": inner_batch} / {"tgt": inner_batch}
    # assert len(model_input) == 1
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


@dataclass
class CLIPCollator:
    """
    Collate MMEB (qry, pos) pairs into tokenized batches for CLIP.

    Each dataset item: ((qry_text, qry_image), (pos_text, pos_image)).
    Returns (qry_batch, pos_batch).
    """

    data_args: DataArguments
    processor: CLIPProcessor

    def __call__(self, examples):
        qry_examples = [ex[0] for ex in examples]
        pos_examples = [ex[1] for ex in examples]
        return self._batch(qry_examples), self._batch(pos_examples)

    def _batch(self, examples):
        input_ids = []
        pixel_values = []
        max_length = clip_text_max_length(self.processor.tokenizer, self.data_args.max_len)

        crop_size = self.processor.image_processor.crop_size
        h, w = crop_size["height"], crop_size["width"]
        channels = getattr(self.processor.image_processor, "num_channels", 3)
        pixel_shape = (channels, h, w)

        for example in examples:
            if example is None:
                text, image = "  ", None
            else:
                text, image = example

            if image is not None:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                pv = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            else:
                pv = torch.zeros(pixel_shape)

            pixel_values.append(pv)
            text_inputs = self.processor.tokenizer(
                text or "  ",
                padding=False,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids.append(text_inputs["input_ids"].squeeze(0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, :max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.processor.tokenizer.pad_token_id),
            "pixel_values": torch.stack(pixel_values, dim=0),
        }


@dataclass
class CLIPEvalCollator:
    """
    Eval collator for MMEB (mirrors VLM2Vec EvalCollator).

    Each example is (text, image). Returns one batch dict for model(qry=...) or model(tgt=...).
    """

    data_args: DataArguments
    processor: CLIPProcessor

    def __call__(self, examples):
        return CLIPCollator(self.data_args, self.processor)._batch(examples)
