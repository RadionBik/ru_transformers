import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
from yt_encoder import YTEncoder
import random


logger = logging.getLogger(__name__)


def batch_aligner(tok_lines, batch_size, shuffle=True):
    matched_batches = []
    batch = []
    prev_len = -1

    sorted_lines = sorted(tok_lines, key=len)

    for item_index, item in enumerate(sorted_lines):
        item_len = len(item)
        len_math = item_len == prev_len
        # if no match reset the batch
        if not len_math:
            batch = []

        batch.append(item)

        if len(batch) == batch_size:
            matched_batches.append(batch)
            batch = []

        prev_len = item_len

    if shuffle:
        matched_batches = random.sample(matched_batches, len(matched_batches))

    returned_lines = []
    for batch in matched_batches:
        returned_lines.extend(batch)

    logger.info(f'dropped {len(tok_lines) - len(returned_lines)} items to align batches')

    return returned_lines


class ConvDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', args=None, shuffle=True, batch=True):
        logger.info(f"Loading features from {file_path}")
        with open(file_path, 'r') as ft:
            text_lines = ft.readlines()

        tok_lines = tokenizer.encode(text_lines)
        if batch:
            self.examples = batch_aligner(tok_lines, args.train_batch_size)
        else:
            self.examples = tok_lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return torch.tensor(self.examples[index])


if __name__=='__main__':

    model_path = 'gpt/pelevin/m_checkpoint-3365357'

    tokenizer = YTEncoder.from_pretrained(model_path)

    with open('corpus/training_part.txt', 'r') as ft:
        text_lines = ft.readlines()

    tok_lines = tokenizer.encode(text_lines)

    BATCH_SIZE = 8
    old_counts = pd.Series(tok_lines).apply(lambda x: len(x)).value_counts()
    print(old_counts)
    print('total: ', old_counts.sum())
    matched_lines = batch_aligner(tok_lines, BATCH_SIZE)

    new_counts = pd.Series(matched_lines).apply(lambda x: len(x)).value_counts()
    print(new_counts)
    print('total: ', new_counts.sum())
    print(new_counts % BATCH_SIZE)