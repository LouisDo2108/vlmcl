import os
from typing import List, Tuple

import datasets
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tevatron.hyperbolic.arguments import DataArguments
from tevatron.hyperbolic.utils import print_rank
from torch.utils.data import Dataset

MMEB_IMAGE_TOKEN = "<|image_1|>"

MMEB_retrieval_instruction_dict = {
    "CIRR": {
        "query": "Given an image, find a similar everyday image with the described changes:",
        "target": "Represent the given image.",
    },
    
    "NIGHTS": {
        "query": "Find a day-to-day image that looks similar to the provided image.",
        "target": "Represent the given image.",
    },
    "VisDial": {
        "query": "Represent the given dialogue about an image, which is used for image retrieval:",
        "target": "Represent the given image",
    },
    "VisualNews_i2t": {
        "query": "Find a caption for the news in the given photo.",
        "target": "",
    },
    "VisualNews_t2i": {
        "query": "Retrieve an image of this news caption.",
        "target": "Represent the given image.",
    },
    "MSCOCO_i2t": {
        "query": "Find an image caption describing the given everyday image.",
        "target": "",
    },
    "MSCOCO_t2i": {
        "query": "Find me an everyday image that matches the given caption:",
        "target": "Represent the given image.",
    },
    "WebQA": {
        "query": "Find a Wikipedia image that answers this question:",
        "target": "Represent the given Wikipedia image with related text information:",
    },
    ### OOD tasks
    "EDIS": {
        "query": "Find a news image that matches the provided caption:",
        "target": "Represent the given image with related text information:",
    },
    "FashionIQ": {
        "query": "Find an image to match the fashion image and style note:",
        "target": "Represent the given image.",
    },
    "OVEN": {
        "query": "Retrieve a Wikipedia image-description pair that provides evidence for the question of this image:",
        "target": "Represent the given Wikipedia image with related text information:",
    },
    "Wiki-SS-NQ": {
        "query": "Find the document image that can answer the given query:",
        "target": "Represent the given image",
    },
}


def _first_scalar(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value
    
def _clean_mmeb_text(text: str, instruction: str) -> str:
    """Strip MMEB image token, task instruction, and surrounding whitespace."""
    if not text:
        return ""
    text = text.replace(MMEB_IMAGE_TOKEN, "", 1)
    text = text.replace(instruction, "", 1)
    return text.strip()


def remove_mmeb_instructions(batch, query_instruction, target_instruction):
    batch["qry"] = [_clean_mmeb_text(t, query_instruction) for t in batch["qry"]]
    batch["pos_text"] = [_clean_mmeb_text(t, target_instruction) for t in batch["pos_text"]]
    return batch

def remove_mmeb_instructions_eval(batch, instruction):
    batch["text"] = [_clean_mmeb_text(t, instruction) for t in batch["text"]]
    return batch


def get_unique_pairs(eval_data, text_field: str, img_path_field: str):
    """Unique (text, img_path) pairs from MMEB eval rows for qry or tgt encoding."""
    unique_pair = set()
    for row in eval_data:
        text_val = row[text_field]
        img_val = row[img_path_field]
        if isinstance(text_val, str):
            if text_val:
                unique_pair.add((text_val, img_val))
            else:
                if isinstance(img_val, list):
                    for img_path in img_val:
                        unique_pair.add((text_val, img_path))
                else:
                    unique_pair.add((text_val, img_val))
        elif isinstance(text_val, list):
            assert isinstance(img_val, list) and len(img_val) == len(text_val)
            for text, img_path in zip(text_val, img_val):
                unique_pair.add((text, img_path))
    return [{"text": t, "img_path": p} for t, p in unique_pair]


def _load_mmeb_train_subset(data_args: DataArguments, subset: str):
    if subset not in MMEB_retrieval_instruction_dict:
        raise ValueError(
            f"Unknown subset {subset!r}; add it to MMEB_retrieval_instruction_dict"
        )
    ds = load_dataset(
        "parquet",
        data_dir=os.path.join(data_args.image_dir, subset),
        data_files={"original": "original-00000-of-00001.parquet"},
    )["original"]
    if (
        data_args.num_sample_per_subset is not None
        and data_args.num_sample_per_subset < ds.num_rows
    ):
        ds = ds.select(range(int(data_args.num_sample_per_subset)))
    if not data_args.add_instructions:
        instr = MMEB_retrieval_instruction_dict[subset]
        ds = ds.map(
            lambda x: remove_mmeb_instructions(
                x, instr["query"], instr["target"]
            ),
            batched=True,
            batch_size=2048,
            drop_last_batch=False,
            desc=f"strip MMEB instructions ({subset})",
        )
    return ds


class CLIPTrainDataset(Dataset):
    """MMEB parquet rows as CLIP (text, image) pairs for query and positive."""

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self._image_root = data_args.image_dir
        subsets = []
        print_rank(f"Loading {len(data_args.subset_name)} subsets: {data_args.subset_name}")
        for subset in data_args.subset_name:
            ds = _load_mmeb_train_subset(data_args, subset)
            subsets.append(ds)
            print_rank(f"{subset}: {len(ds)} rows")
        self.train_data = concatenate_datasets(subsets)
        # Columnar refs avoid per-row dict materialization in __getitem__.
        self._qry = self.train_data["qry"]
        self._pos_text = self.train_data["pos_text"]
        self._qry_image_path = self.train_data["qry_image_path"]
        self._pos_image_path = self.train_data["pos_image_path"]
        
        print(self.train_data[0])
        from pdb import set_trace; set_trace()

    def __len__(self):
        return len(self.train_data)

    def _load_image(self, img_path) -> Image.Image | None:
        img_path = _first_scalar(img_path)
        if not img_path:
            return None
        path = os.path.join(self._image_root, img_path)
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _pair_text_image(text, image: Image.Image | None):
        text = _first_scalar(text)
        if text is None:
            text = ""
        text = str(text)
        if text.strip() or image is not None:
            return text, image
        return "", image

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[str, Image.Image | None], Tuple[str, Image.Image | None]]:
        qry_text, qry_image = self._pair_text_image(
            self._qry[idx], self._load_image(self._qry_image_path[idx])
        )
        pos_text, pos_image = self._pair_text_image(
            self._pos_text[idx], self._load_image(self._pos_image_path[idx])
        )
        return (qry_text, qry_image), (pos_text, pos_image)


class EvalDataset(Dataset):
    """
    MMEB eval: unique (text, image_path) pairs for query or target encoding.

    Mirrors VLM2Vec EvalDataset; each item is (text, PIL image) for CLIPEvalCollator.
    """

    def __init__(
        self,
        data_args: DataArguments,
        subset: str,
        text_field: str,
        img_path_field: str,
        eval_data=None,
    ):
        self.data_args = data_args
        self.eval_data = eval_data or load_dataset(
            "TIGER-Lab/MMEB-eval",
            subset,
            split="test",
        )
        self.paired_data = get_unique_pairs(self.eval_data, text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [p["text"] for p in self.paired_data],
            "img_path": [p["img_path"] for p in self.paired_data],
        })
        if text_field == "qry_text":
            instruction = MMEB_retrieval_instruction_dict[subset]["query"]
        elif text_field == "tgt_text":
            instruction = MMEB_retrieval_instruction_dict[subset]["target"]
        else:
            raise ValueError(f"Invalid text field: {text_field}")

        if not data_args.add_instructions:
            self.paired_dataset = self.paired_dataset.map(lambda x: remove_mmeb_instructions_eval(x, instruction),batched=True,batch_size=2048,drop_last_batch=False)
        print_rank(f"{subset} {text_field}: {len(self.paired_data)} unique pairs")

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item: int) -> Tuple[str, Image.Image | None]:
        text = self.paired_dataset[item]["text"]
        img_path = self.paired_dataset[item]["img_path"]
        return text, self._load_image(img_path)

    def _load_image(self, img_path: str) -> Image.Image | None:
        if not img_path:
            return None
        path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(path)
        return image.convert("RGB") if image.mode != "RGB" else image