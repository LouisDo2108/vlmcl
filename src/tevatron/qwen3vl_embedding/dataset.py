import logging
import random
from pathlib import Path

import msgspec
from datasets import load_dataset
from msgspec.json import format
from tevatron.colpali.arguments import DataArguments
from torch.utils.data import Dataset
from pdb import set_trace as st

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
            self.train_data = self.train_data.rename_column("query", "query_text")
        except Exception as e:
            print(e)
        try:
            self.train_data = self.train_data.rename_column("positive_document_ids", "positive_passages")
        except Exception as e:
            print(e)
        
        try:
            self.train_data = self.train_data.rename_column("negative_document_ids", "negative_passages")
        except Exception as e:
            print(e)
            
        """
        https://discuss.huggingface.co/t/from-pandas-dataframe-to-huggingface-dataset/9322/3
        import pandas as pd
        import datasets
        from datasets import Dataset, DatasetDict


        tdf = pd.DataFrame({"a": [1, 2, 3], "b": ['hello', 'ola', 'thammi']})
        vdf = pd.DataFrame({"a": [4, 5, 6], "b": ['four', 'five', 'six']})
        tds = Dataset.from_pandas(tdf)
        vds = Dataset.from_pandas(vdf)


        ds = DatasetDict()

        ds['train'] = tds
        ds['validation'] = vds

        print(ds)
        """
            
        if data_args.colpali_source != "all":
            print(f"Train with samples from {data_args.colpali_source}")
            print(f"Length of dataset before filtering: {len(self.train_data)}")
            self.train_data = self.train_data.filter(
                lambda x: data_args.colpali_source in x['source'],
                num_proc=4
            )
            print(f"Length of dataset after filtering: {len(self.train_data)}")
            
            """
            https://github.com/huggingface/datasets/issues/4684
            One option is use map with a function that overwrites the labels (dset = dset.map(lamba _: {"label": 0}, features=dset.features)). Or you can use the remove_column + add_column combination (dset = dset.remove_columns("label").add_column("label", [0]*len(data)).cast(dset.features), but note that this approach creates an in-memory table for the added column instead of writing to disk, which could be problematic for large datasets.
            """
            
            temp = self.train_data.remove_columns(["query_image"])
            def extract_docids(batch):
                docids = []
                for pos, src in zip(batch["positive_passages"], batch["source"]):
                    if data_args.colpali_source in src:
                        docids.append(int(pos[0]))  # same logic as your lambda
                return {"_docid_candidate": docids}  # temporary column

            temp_with_docids = temp.map(extract_docids, batched=True, num_proc=4)
            docid_set = set(docid for docid in temp_with_docids["_docid_candidate"] if docid is not None)
            
            del temp_with_docids

            print(f"Length of corpus before filtering: {len(self.corpus)}")
            self.corpus = self.corpus.select(list(docid_set))
            print(f"Length of corpus after filtering: {len(self.corpus)}")

            def filter_negatives(batch):
                batch["negative_passages"] = [
                    [i for i in negs if int(i) in docid_set] 
                    for negs in batch["negative_passages"]
                ]
                return batch

            # Update the annotated hard negatives that must exist in the filtered corpus
            self.train_data = self.train_data.map(filter_negatives, batched=True, num_proc=4).cast(self.train_data.features)
            
        else:
            print(f"Train with all data")
            

        self.docid2idx = {}
        if 'docid' in self.corpus.features:
            for idx, docid in enumerate(self.corpus['docid']):
                self.docid2idx[str(docid)] = idx
        else:
            # handle docmatix
            for idx in range(len(self.corpus)):
                self.docid2idx[str(idx)] = idx
        if self.data_args.dataset_number_of_shards > 1:
            self.train_data = self.train_data.shard(
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
        image = self.corpus[self.docid2idx[docid]]['image'] 
        return image
        # if 'image' in self.corpus.features:
        #     image = self.corpus[self.docid2idx[docid]]['image']
        # elif 'images' in self.corpus.features:
        #     # handle docmatrix
        #     example_id, image_id = docid.split('_')
        #     image = self.corpus[self.docid2idx[example_id]]['images'][int(image_id)]
        # return image
        

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group.get("query_text", "")
        group_positives = group['positive_passages']

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        formated_passages.append(self._get_image(pos_psg))
        # formated_passages.append(self._get_image(pos_psg['docid']))

        if "negative_passages" in group:
            group_negatives = group['negative_passages']
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


class TrainRankedDataset(Dataset):
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
            self.train_data = self.train_data.rename_column("query", "query_text")
        except Exception as e:
            print(e)
        try:
            self.train_data = self.train_data.rename_column("positive_document_ids", "positive_passages")
        except Exception as e:
            print(e)
        
        try:
            self.train_data = self.train_data.rename_column("negative_document_ids", "negative_passages")
        except Exception as e:
            print(e)
            
        """
        https://discuss.huggingface.co/t/from-pandas-dataframe-to-huggingface-dataset/9322/3
        import pandas as pd
        import datasets
        from datasets import Dataset, DatasetDict


        tdf = pd.DataFrame({"a": [1, 2, 3], "b": ['hello', 'ola', 'thammi']})
        vdf = pd.DataFrame({"a": [4, 5, 6], "b": ['four', 'five', 'six']})
        tds = Dataset.from_pandas(tdf)
        vds = Dataset.from_pandas(vdf)


        ds = DatasetDict()

        ds['train'] = tds
        ds['validation'] = vds

        print(ds)
        """
            
        if data_args.colpali_source != "all":
            print(f"Train with samples from {data_args.colpali_source}")
            print(f"Length of dataset before filtering: {len(self.train_data)}")
            self.train_data = self.train_data.filter(
                lambda x: data_args.colpali_source in x['source'],
                num_proc=4
            )
            print(f"Length of dataset after filtering: {len(self.train_data)}")
            
            """
            https://github.com/huggingface/datasets/issues/4684
            One option is use map with a function that overwrites the labels (dset = dset.map(lamba _: {"label": 0}, features=dset.features)). Or you can use the remove_column + add_column combination (dset = dset.remove_columns("label").add_column("label", [0]*len(data)).cast(dset.features), but note that this approach creates an in-memory table for the added column instead of writing to disk, which could be problematic for large datasets.
            """
            
            temp = self.train_data.remove_columns(["query_image"])
            def extract_docids(batch):
                docids = []
                for pos, src in zip(batch["positive_passages"], batch["source"]):
                    if data_args.colpali_source in src:
                        docids.append(int(pos[0]))  # same logic as your lambda
                return {"_docid_candidate": docids}  # temporary column

            temp_with_docids = temp.map(extract_docids, batched=True, num_proc=4)
            docid_set = set(docid for docid in temp_with_docids["_docid_candidate"] if docid is not None)
            
            del temp_with_docids

            print(f"Length of corpus before filtering: {len(self.corpus)}")
            self.corpus = self.corpus.select(list(docid_set))
            print(f"Length of corpus after filtering: {len(self.corpus)}")

            def filter_negatives(batch):
                batch["negative_passages"] = [
                    [i for i in negs if int(i) in docid_set] 
                    for negs in batch["negative_passages"]
                ]
                return batch

            # Update the annotated hard negatives that must exist in the filtered corpus
            self.train_data = self.train_data.map(filter_negatives, batched=True, num_proc=4).cast(self.train_data.features)
            
        else:
            print(f"Train with all data")
            

        self.docid2idx = {}
        if 'docid' in self.corpus.features:
            for idx, docid in enumerate(self.corpus['docid']):
                self.docid2idx[str(docid)] = idx
        else:
            # handle docmatix
            for idx in range(len(self.corpus)):
                self.docid2idx[str(idx)] = idx
        if self.data_args.dataset_number_of_shards > 1:
            self.train_data = self.train_data.shard(
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
        image = self.corpus[self.docid2idx[docid]]['image'] 
        return image
        # if 'image' in self.corpus.features:
        #     image = self.corpus[self.docid2idx[docid]]['image']
        # elif 'images' in self.corpus.features:
        #     # handle docmatrix
        #     example_id, image_id = docid.split('_')
        #     image = self.corpus[self.docid2idx[example_id]]['images'][int(image_id)]
        # return image
        

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group.get("query_text", "")
        group_positives = group['positive_passages']

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        # if self.data_args.positive_passage_no_shuffle:
        #     pos_psg = group_positives[0]
        # else:
        #     pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        # formated_passages.append(self._get_image(pos_psg))

        # Handling ranked negatives for training
        rank = group.get("rank", None)
        rank_psg = group.get("positive_passages", None)
    
        if rank is None or rank_psg is None:
            return formated_query, formated_passages
        
        rank_size = self.data_args.train_group_size # no - 1, since we will treat the positives as ranked highest, and the rest as negatives
        paired_rank = list(zip(rank_psg, rank))
        num_rank = len(paired_rank)

        if num_rank > 0 and num_rank < rank_size:
            negs = random.choices(paired_rank, k=rank_size)
        elif self.data_args.negative_passage_no_shuffle:
            negs = paired_rank[:rank_size]
        else:
            _offset = epoch * rank_size % len(paired_rank)
            negs = [x for x in paired_rank]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + rank_size]

        negs = sorted(negs, key=lambda x: x[1]) # sort by rank, so that we know that the first element is the most relevant positive, and the rest are negatives with decreasing relevance
        for docid, rank in negs:
            formated_passages.append(self._get_image(docid))
            # formated_passages.append(self._get_image(neg_psg['docid']))

        return formated_query, formated_passages

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
            if content_text is None:
                content_text = ""
            content_text = self.data_args.query_prefix + content_text.strip()
            content_image = content.get('image', None)
        else:
            content_id = content['docid']
            content_text = content.get('text', "")
            if content_text is None:
                content_text = ""
            if 'title' in content:
                content_text = content['title'] + ' ' + content_text
            content_text = self.data_args.passage_prefix + content_text.strip()
            content_image = content.get('image', None)

        return content_id, content_text, content_image
    

class EncodeICLRDataset(EncodeDataset):
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

