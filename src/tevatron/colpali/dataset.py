import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from pdb import set_trace as st
from pprint import pformat, pprint
from typing import List, Tuple

import msgspec
from arguments import DataArguments
from datasets import load_dataset
from msgspec.json import format
from tevatron.retriever.arguments import DataArguments
from torch.utils.data import Dataset

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

def write_json(file_path, data, jsonl=False):
    with open(file_path, "wb") as file:
        if jsonl:
            file.write(encoder.encode_lines(data))
        else:
            file.write(format(encoder.encode(data)))
    print(f"The file contains {len(data)} items.")
    print("Saved to", file_path)


def read_json(file_path, jsonl=False):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError("filepath is not a file")
    # if not file_path.suffix == ".jsonl" and jsonl:
    #     raise ValueError("the file is not jsonl")
    # if not file_path.suffix == ".json" and not jsonl:
    #     raise ValueError("the file is not json")
    file_path = file_path.__str__()

    output = []

    with open(file_path, "rb") as file:
        data = file.read()
        if jsonl:
            output = decoder.decode_lines(data)
        else:
            output = decoder.decode(data)

    print(f"The file is of type: {type(output)}")
    print(f"The file contains {len(output)} items.")
    return output


logger = logging.getLogger(__name__)


def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()


def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    return f'{prefix} {title.strip()} {text.strip()}'.strip()


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
        
        self.corpus = load_dataset(
            self.data_args.corpus_name,
            self.data_args.corpus_config,
            data_files=self.data_args.corpus_path,
            split=self.data_args.corpus_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        try:
            self.train_data = self.train_data.rename_column("positive_document_ids", "positive_passages")
        except Exception as e:
            print(e)
        
        try:
            self.train_data = self.train_data.rename_column("negative_document_ids", "negative_passages")
        except Exception as e:
            print(e)
            

        self.docid2idx = {}
        if 'docid' in self.corpus.features:
            for idx, docid in enumerate(self.corpus['docid']):
                self.docid2idx[str(docid)] = idx
        else:
            # handle docmatix
            for idx in range(len(self.corpus)):
                self.docid2idx[str(idx)] = idx
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
    
    def set_trainer(self, trainer):
        """Sets the trainer for the dataset."""
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def _get_image(self, docid):
        if 'image' in self.corpus.features:
            image = self.corpus[self.docid2idx[docid]]['image']
        elif 'images' in self.corpus.features:
            # handle docmatrix
            example_id, image_id = docid.split('_')
            image = self.corpus[self.docid2idx[example_id]]['images'][int(image_id)]
        return image
        

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        # query = group.get('query', "")
        query = group.get("query_text", "")
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        formated_passages.append(self._get_image(pos_psg))
        # formated_passages.append(self._get_image(pos_psg['docid']))

        num_negatives = len(group_negatives)
        negative_size = self.data_args.train_group_size - 1
        
        if num_negatives > 0 and num_negatives < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif num_negatives == 0 or self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(self._get_image(neg_psg))
            # formated_passages.append(self._get_image(neg_psg['docid']))

        return formated_query, formated_passages


class EncodeICLRDataset(Dataset):
    """
    Dataset for encoding.
    Loads data and optionally shards it for distributed processing.
    """

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        
        # This was used for testing ICLR paper retrieval
        if self.data_args.encode_is_query:
            self.encode_data = read_json("/home/thuy0050/code/colpali/louis/2017_images_resized/test/query.jsonl", jsonl=True)
        else:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        content = self.encode_data[item]
        
        if self.data_args.encode_is_query:
            content_id = content['query_id']
            content_text = content.get('query_text', "")
            content_text = self.data_args.query_prefix + content_text
            content_image = content.get('image', None)
            # content_video = content.get('query_video', None)
            # content_audio = content.get('query_audio', None)
        else:
            content_id = content['docid']
            content_text = content.get('text', '')
            if content_text is None:
                content_text = ""

            content_text = self.data_args.passage_prefix + content_text.strip()
            content_image = content.get('image', None)
            # content_video = content.get('video', None)
            # content_audio = content.get('audio', None)


        # if content_video is not None and self.data_args.encode_video:
        #     content_video = os.path.join(self.data_args.assets_path, content_video)
        #     # check if the file exists
        #     if not os.path.exists(content_video):
        #         logger.warning(f"Video file {content_video} does not exist.")
        #         content_video = None

        # if content_audio is not None: # either an dict with 'array' key or a string .mp3 path
        #     if isinstance(content_audio, dict) and 'array' in content_audio:
        #         content_audio = content_audio['array']
        #     else:
        #         assert isinstance(content_audio, str) and content_audio.endswith('.mp3')
        #         content_audio = os.path.join(self.data_args.assets_path, content_audio)
        #         # check if the file exists
        #         if not os.path.exists(content_audio):
        #             logger.warning(f"Audio file {content_audio} does not exist.")
        #             content_audio = None

        if not self.data_args.encode_text:
            content_text = None
        if not self.data_args.encode_image:
            content_image = None
        # if not self.data_args.encode_video:
        #     content_video = None
        # if not self.data_args.encode_audio:
        #     content_audio = None

        return content_id, content_text, content_image # , content_video, content_audio


class EncodeDataset(Dataset):
    """
    Dataset for encoding.
    Loads data and optionally shards it for distributed processing.
    """

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        
        # This was used for testing ICLR paper retrieval
        # if self.data_args.encode_is_query:
        #     self.encode_data = read_json("/home/thuy0050/code/colpali/louis/2017_images_resized/test/query.jsonl", jsonl=True)
        # else:
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        
        try:
            self.encode_data = self.encode_data.rename_column("corpus-id", "docid")
        except Exception as e:
            print(e)
        try:
            self.encode_data = self.encode_data.rename_column("query-id", "query_id")
        except Exception as e:
            print(e)
        try:
            self.encode_data = self.encode_data.rename_column("query", "query_text")
        except Exception as e:
            print(e)
        print(self.encode_data)

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        content = self.encode_data[item]
        
        if self.data_args.encode_is_query:
            content_id = content['query_id']
            content_text = content.get('query_text', "")
            content_text = self.data_args.query_prefix + content_text
            content_image = content.get('image', None)
            # content_video = content.get('query_video', None)
            # content_audio = content.get('query_audio', None)
        else:
            content_id = content['docid']
            content_text = content.get('text', '')
            if content_text is None:
                content_text = ""
            if 'title' in content:
                content_text = content['title'] + ' ' + content_text
            content_text = self.data_args.passage_prefix + content_text.strip()
            content_image = content.get('image', None)
            # content_video = content.get('video', None)
            # content_audio = content.get('audio', None)


        # if content_video is not None and self.data_args.encode_video:
        #     content_video = os.path.join(self.data_args.assets_path, content_video)
        #     # check if the file exists
        #     if not os.path.exists(content_video):
        #         logger.warning(f"Video file {content_video} does not exist.")
        #         content_video = None

        # if content_audio is not None: # either an dict with 'array' key or a string .mp3 path
        #     if isinstance(content_audio, dict) and 'array' in content_audio:
        #         content_audio = content_audio['array']
        #     else:
        #         assert isinstance(content_audio, str) and content_audio.endswith('.mp3')
        #         content_audio = os.path.join(self.data_args.assets_path, content_audio)
        #         # check if the file exists
        #         if not os.path.exists(content_audio):
        #             logger.warning(f"Audio file {content_audio} does not exist.")
        #             content_audio = None

        if not self.data_args.encode_text:
            content_text = None
        if not self.data_args.encode_image:
            content_image = None
        # if not self.data_args.encode_video:
        #     content_video = None
        # if not self.data_args.encode_audio:
        #     content_audio = None

        return content_id, content_text, content_image # , content_video, content_audio