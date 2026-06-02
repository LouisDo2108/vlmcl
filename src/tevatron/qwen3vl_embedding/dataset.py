"""
Dataset module for Qwen3 VL Embeddings V2.

This module provides dataset classes for training and encoding with improved
code structure and better error handling.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgspec
from datasets import load_dataset
from torch.utils.data import Dataset

from .arguments import DataArguments

logger = logging.getLogger(__name__)

# MsgSpec encoders/decoders
encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()


def write_json(file_path: Path, data: Any, jsonl: bool = False) -> None:
    """
    Write data to JSON file.
    
    Args:
        file_path: Output file path.
        data: Data to write.
        jsonl: Whether to write in JSONL format.
    """
    with open(file_path, "wb") as file:
        if jsonl:
            file.write(encoder.encode_lines(data))
        else:
            file.write(msgspec.json.format(encoder.encode(data)))
    
    logger.info(f"The file contains {len(data)} items.")
    logger.info(f"Saved to {file_path}")


def read_json(file_path: Path, jsonl: bool = False) -> Any:
    """
    Read data from JSON file.
    
    Args:
        file_path: Input file path.
        jsonl: Whether to read in JSONL format.
        
    Returns:
        Decoded data.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f"File not found: {file_path}")
    
    with open(file_path, "rb") as file:
        data = file.read()
        if jsonl:
            output = decoder.decode_lines(data)
        else:
            output = decoder.decode(data)
    
    logger.info(f"The file is of type: {type(output)}")
    logger.info(f"The file contains {len(output)} items.")
    return output


def format_query(query: str, prefix: str = '') -> str:
    """Format query text with optional prefix."""
    return f'{prefix} {query.strip()}'.strip()


def format_passage(text: str, title: str = '', prefix: str = '') -> str:
    """Format passage text with optional title and prefix."""
    return f'{prefix} {title.strip()} {text.strip()}'.strip()


class TrainDataset(Dataset):
    """
    Training dataset for multimodal retrieval.
    
    This class loads and processes training data consisting of queries and 
    associated positive/negative passages (images).
    """
    
    def __init__(self, data_args: DataArguments, trainer: Optional[Any] = None):
        """
        Initialize training dataset.
        
        Args:
            data_args: Data configuration arguments.
            trainer: Optional trainer reference for epoch tracking.
        """
        self.data_args = data_args
        self.trainer = trainer
        
        # Load query dataset
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
        
        # Load corpus dataset
        self.corpus = load_dataset(
            self.data_args.corpus_name,
            self.data_args.corpus_config,
            data_files=self.data_args.corpus_path,
            split=self.data_args.corpus_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        
        # Standardize column names
        self._standardize_columns()
        
        # Filter by ColPali source if specified
        if data_args.colpali_source != "all":
            self._filter_by_source()
        else:
            logger.info("Train with all data")
        
        # Build document ID to index mapping
        self._build_docid_mapping()
        
        # Shard dataset if needed
        if self.data_args.dataset_number_of_shards > 1:
            self.train_data = self.train_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    
    def _standardize_columns(self):
        """Rename columns to standard names."""
        try:
            self.train_data = self.train_data.rename_column("query", "query_text")
        except Exception:
            pass
        
        try:
            self.train_data = self.train_data.rename_column(
                "positive_document_ids", "positive_passages"
            )
        except Exception:
            pass
        
        try:
            self.train_data = self.train_data.rename_column(
                "negative_document_ids", "negative_passages"
            )
        except Exception:
            pass
    
    def _filter_by_source(self):
        """Filter dataset by ColPali source."""
        source = self.data_args.colpali_source
        logger.info(f"Train with samples from {source}")
        logger.info(f"Length of dataset before filtering: {len(self.train_data)}")
        
        self.train_data = self.train_data.filter(
            lambda x: source in x['source'],
            num_proc=4
        )
        
        logger.info(f"Length of dataset after filtering: {len(self.train_data)}")
        
        # Extract valid document IDs
        temp = self.train_data.remove_columns(["query_image"])
        
        def extract_docids(batch):
            docids = []
            for pos, src in zip(batch["positive_passages"], batch["source"]):
                if source in src:
                    docids.append(int(pos[0]))
            return {"_docid_candidate": docids}
        
        temp_with_docids = temp.map(extract_docids, batched=True, num_proc=4)
        docid_set = set(
            docid for docid in temp_with_docids["_docid_candidate"] 
            if docid is not None
        )
        
        del temp_with_docids
        
        # Filter corpus and negatives
        logger.info(f"Length of corpus before filtering: {len(self.corpus)}")
        self.corpus = self.corpus.select(list(docid_set))
        logger.info(f"Length of corpus after filtering: {len(self.corpus)}")
        
        def filter_negatives(batch):
            batch["negative_passages"] = [
                [i for i in negs if int(i) in docid_set] 
                for negs in batch["negative_passages"]
            ]
            return batch
        
        self.train_data = self.train_data.map(
            filter_negatives, batched=True, num_proc=4
        ).cast(self.train_data.features)
    
    def _build_docid_mapping(self):
        """Build mapping from document ID to corpus index."""
        self.docid2idx = {}
        
        if 'docid' in self.corpus.features:
            for idx, docid in enumerate(self.corpus['docid']):
                self.docid2idx[str(docid)] = idx
        else:
            # Handle DocMatrix-style datasets
            for idx in range(len(self.corpus)):
                self.docid2idx[str(idx)] = idx
    
    def set_trainer(self, trainer: Any):
        """Set trainer reference for epoch tracking."""
        self.trainer = trainer
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.train_data)
    
    def _get_image(self, docid: str) -> Any:
        """
        Retrieve image from corpus by document ID.
        
        Args:
            docid: Document identifier.
            
        Returns:
            Image data.
        """
        return self.corpus[self.docid2idx[docid]]['image']
    
    def __getitem__(self, item: int) -> Tuple[str, List[Any]]:
        """
        Get training sample by index.
        
        Args:
            item: Sample index.
            
        Returns:
            Tuple of (formatted_query, list_of_passage_images).
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)
        
        query = group.get("query_text", "")
        group_positives = group['positive_passages']
        
        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []
        
        # Select positive passage
        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(self._get_image(str(pos_psg)))
        
        # Sample negative passages
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
                negs = list(group_negatives)
                random.Random(_hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset: _offset + negative_size]
            
            for neg_psg in negs:
                formated_passages.append(self._get_image(str(neg_psg)))
        
        return formated_query, formated_passages


class EncodeDataset(Dataset):
    """
    Dataset for encoding queries or passages.
    
    This class loads data for inference/encoding tasks.
    """
    
    def __init__(self, data_args: DataArguments):
        """
        Initialize encoding dataset.
        
        Args:
            data_args: Data configuration arguments.
        """
        self.data_args = data_args
        
        self.data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, item: int) -> Tuple[str, str, Any]:
        """
        Get encoding sample by index.
        
        Args:
            item: Sample index.
            
        Returns:
            Tuple of (id, text, image/content).
        """
        example = self.data[item]
        
        if self.data_args.encode_is_query:
            # Query encoding
            content_id = example.get('id', item)
            text = example.get('query', example.get('text', ''))
            return content_id, text, None
        else:
            # Passage encoding
            content_id = example.get('id', example.get('docid', item))
            text = example.get('text', '')
            image = example.get('image', None)
            return content_id, text, [image] if image is not None else []
