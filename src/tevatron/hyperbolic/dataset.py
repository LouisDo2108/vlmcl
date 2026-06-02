import os
from typing import Tuple

from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tevatron.hyperbolic.arguments import DataArguments
from tevatron.hyperbolic.utils import print_rank
from torch.utils.data import Dataset


def _first_scalar(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value


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
