from pathlib import Path

import msgspec
import pandas as pd
from datasets import (
    Image,
    Value,
    Features,
    load_dataset,
)
from pdb import set_trace as st

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

dataset_root = Path("/home/thuy0050/mg61_scratch2/thuy0050/data/marqo-gs-dataset")
image_folder = dataset_root / "images_wfash"
metadata_folder = dataset_root / "marqo_gs_wfash_1m"

mapping_dict = {
    "in_domain": "query_0_product_id_0.csv",
    "novel_query": "query_1_product_id_0.csv",
    "novel_doc": "query_0_product_id_1.csv",
    "zeroshot_doc": "query_1_product_id_1.csv",
    "corpus1": "corpus_1.json",
    "corpus2": "corpus_2.json"
}

in_domain = pd.read_csv(metadata_folder / mapping_dict["in_domain"])[["product_id", "image_local", "title"]]
novel_query = pd.read_csv(metadata_folder / mapping_dict["novel_query"])[["product_id", "image_local", "title"]]
corpus = pd.concat([in_domain, novel_query], axis=0).drop_duplicates().reset_index(drop=True)
corpus.rename(columns={
    "image_local": "file_name",
    "product_id": "docid",
}, inplace=True)
corpus["file_name"] = corpus["file_name"].apply(lambda x: Path(x).name)
corpus['docid'] = corpus['docid'].astype(str)
corpus['title'] = corpus['title'].astype(str)

corpus.to_csv("/home/thuy0050/mg61_scratch2/thuy0050/data/marqo-gs-dataset/images_wfash/metadata.csv", index=False)

base_path = "/home/thuy0050/mg61_scratch2/thuy0050/data/marqo-gs-dataset/images_wfash/"

dataset = load_dataset(
    "imagefolder", 
    data_dir=base_path, 
    split="train",
    features=Features({'docid': Value('string'), 'image': Image(), 'title': Value('string'), 'file_name': Value('string')}),
)
dataset.push_to_hub("LouisDo2108/marqo_gs_wfash_1m_corpus_tevatron")