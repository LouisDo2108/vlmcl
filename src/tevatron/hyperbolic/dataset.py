import os
from typing import List, Tuple

import datasets
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tevatron.hyperbolic.arguments import DataArguments
from tevatron.hyperbolic.utils import print_rank
from torch.utils.data import Dataset


def _first_scalar(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value


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


class CLIPTrainDataset(Dataset):
    """MMEB parquet rows as CLIP (text, image) pairs for query and positive."""

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        subsets = []
        print_rank(f"Loading {len(data_args.subset_name)} subsets: {data_args.subset_name}")
        for subset in data_args.subset_name:
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
            subsets.append(ds)
            print_rank(f"{subset}: {len(ds)} rows")
        self.train_data = concatenate_datasets(subsets)

    def __len__(self):
        return len(self.train_data)

    def _load_image(self, img_path) -> Image.Image | None:
        img_path = _first_scalar(img_path)
        if not img_path:
            return None
        path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(path)
        return image.convert("RGB") if image.mode != "RGB" else image

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[str, Image.Image | None], Tuple[str, Image.Image | None]]:
        row = self.train_data[idx]

        qry_text = _first_scalar(row["qry"])
        pos_text = _first_scalar(row["pos_text"])
        qry_image = self._load_image(row["qry_image_path"])
        pos_image = self._load_image(row["pos_image_path"])

        if not qry_text and qry_image is None:
            qry_text = "  "
        if not pos_text and pos_image is None:
            pos_text = "  "

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