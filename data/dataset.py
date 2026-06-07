from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from pydantic import BaseModel
from datasets import load_dataset


class BaseDatasetConfig(BaseModel):
    tokenizer_name: str = "DogaOytc/TRLM"
    max_seq_len: int = 1024
    seed: int = 42
    streaming: bool = True
    shuffle_buffer_size: int = 10000

class OpenWebTextConfig(BaseDatasetConfig):
    dataset_path: str = "dvruette/openwebtext"
    
class FineWebEduConfig(BaseDatasetConfig):
    dataset_path: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-10BT"  
    
class MiniPileConfig(BaseDatasetConfig):
    dataset_path: str = "JeanKaddour/minipile"

class FineMathConfig(BaseDatasetConfig):
    dataset_path: str = "HuggingFaceTB/FineMath"
    subset: str = "finemath-4plus" 


class BasePretrainDataset(IterableDataset):
    def __init__(self, tokenizer, config: BaseDatasetConfig, split: str = "train", is_eval: bool = False, eval_size: int = 5000):
        super().__init__()
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.is_eval = is_eval
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        subset = getattr(config, "subset", None)
        raw_dataset = load_dataset(
            config.dataset_path, 
            subset, 
            split=split, 
            streaming=config.streaming
        )

        if self.is_eval:
            # Grab only the first 'eval_size' examples for validation
            self.dataset = raw_dataset.take(eval_size)
        else:
            # Skip the first 'eval_size' examples for training to prevent data leakage
            self.dataset = raw_dataset.skip(eval_size)

        self.metadata = {
            "vocab_size": len(self.tokenizer),
            "max_seq_len": config.max_seq_len,
        }

    def __iter__(self):
        worker_info = get_worker_info()
        dataset = self.dataset
        
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        if not self.is_eval:
            dataset = dataset.shuffle(seed=self.config.seed, buffer_size=self.config.shuffle_buffer_size)

        token_buffer = []
        for example in dataset:
            text = example["text"]
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            token_buffer.extend(tokens)
            
            while len(token_buffer) >= self.config.max_seq_len:
                chunk = token_buffer[:self.config.max_seq_len]
                token_buffer = token_buffer[self.config.max_seq_len:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


class OpenWebTextDataset(BasePretrainDataset):
    def __init__(self, tokenizer, config: OpenWebTextConfig, split: str = "train", is_eval: bool = False, eval_size: int = 5000):
        super().__init__(tokenizer, config, split, is_eval, eval_size)

class FineWebEduDataset(BasePretrainDataset):
    def __init__(self, tokenizer,  config: FineWebEduConfig, split: str = "train", is_eval: bool = False, eval_size: int = 5000):
        super().__init__(tokenizer, config, split, is_eval, eval_size)

class MiniPileDataset(BasePretrainDataset):
    def __init__(self, tokenizer,  config: MiniPileConfig, split: str = "train", is_eval: bool = False, eval_size: int = 5000):
        super().__init__(tokenizer, config, split, is_eval, eval_size)

class FineMathDataset(BasePretrainDataset):
    def __init__(self, tokenizer,  config: FineMathConfig, split: str = "train", is_eval: bool = False, eval_size: int = 5000):
        super().__init__(tokenizer, config, split, is_eval, eval_size)


class MixedPretrainDataset(IterableDataset):
    def __init__(self, datasets: List[IterableDataset], probabilities: List[float]):
        
        super().__init__()
        assert len(datasets) == len(probabilities), "Number of datasets must match probabilities"
        assert abs(sum(probabilities) - 1.0) < 1e-5, "Probabilities must sum to 1.0"
        
        self.datasets = datasets
        self.probabilities = probabilities

    def __iter__(self):
        # Initialize iterators for all datasets
        iters = [iter(d) for d in self.datasets]
        
        while True:
            # Randomly select a dataset iterator based on probabilities
            idx = np.random.choice(len(iters), p=self.probabilities)
            
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.datasets[idx])
                yield next(iters[idx])