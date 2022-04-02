import os
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from filelock import FileLock
from transformers.utils import logging
from typing import Dict, List, Optional
import pickle
import time
import pandas as pd

logger = logging.get_logger(__name__)

class PoemDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    Parameters:
    ----------
    tokenizers : is pretrain tokenizer of PhoBERT
    file_path  : path to file train, test
    block_size : size of 1 block , optinal
    cache_dir  : just load 1 once and saved
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        lock_path = cached_features_file + ".lock"
        print('Loading file:', file_path)
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []

                train_df = pd.read_csv(file_path, sep="\t").astype(str)
                from tqdm import tqdm
                for i in tqdm(range(len(train_df))):
                    text = "<s>" + train_df['input_text'][i] + ' [SEP] ' + train_df['target_text'][i] + "</s>"
                    inds = tokenizer.encode(text)
                    self.examples.append(inds)
                    
                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)