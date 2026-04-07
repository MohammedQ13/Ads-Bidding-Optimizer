import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# categorical features that have embedding layers in the model
CATEGORICAL_FEATURES = [
    "region",
    "city",
    "ad_exchange",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "advertiser_id",
]

# scaled continuous features
CONTINUOUS_FEATURES = [
    "slot_floor_price",
]

# sin/cos encoded time features
CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


class BidDataset(Dataset):
    def __init__(self, parquet_path):
        # load the processed parquet file
        self.df = pd.read_parquet(parquet_path)
        self.length = len(self.df)

        # pull categorical columns as a numpy int array so indexing is fast
        self.cat_data = self.df[CATEGORICAL_FEATURES].values.astype(np.int64)

        # pull continuous + cyclical into one float array
        cont_cols = CONTINUOUS_FEATURES + CYCLICAL_FEATURES
        self.cont_data = self.df[cont_cols].values.astype(np.float32)

        # tag indices stored as list of ints per row
        self.tag_data = self.df["tag_indices"].tolist()

        # log_payprice is what I'm training the model to predict (mu and sigma of this)
        self.log_targets = self.df["log_payprice"].values.astype(np.float32)

        # raw payprice kept around for bid regret calculation during evaluation
        self.raw_targets = self.df["payprice"].values.astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cat = torch.tensor(self.cat_data[idx], dtype=torch.long)
        cont = torch.tensor(self.cont_data[idx], dtype=torch.float32)

        # tag indices as a 1D tensor -- variable length, gets padded in collate_fn
        tags = self.tag_data[idx]
        if not isinstance(tags, list):
            tags = list(tags) # handle numpy array case from parquet
        tag_tensor = torch.tensor(tags, dtype=torch.long)

        log_target = torch.tensor(self.log_targets[idx], dtype=torch.float32)
        raw_target = torch.tensor(self.raw_targets[idx], dtype=torch.float32)

        return cat, cont, tag_tensor, log_target, raw_target


def collate_fn(batch):
    # batch is a list of tuples from __getitem__
    # unzip into separate lists
    cats, conts, tags, log_targets, raw_targets = [], [], [], [], []

    for cat, cont, tag, log_t, raw_t in batch:
        cats.append(cat)
        conts.append(cont)
        tags.append(tag)
        log_targets.append(log_t)
        raw_targets.append(raw_t)

    # stack fixed-size tensors normally
    cats = torch.stack(cats)
    conts = torch.stack(conts)
    log_targets = torch.stack(log_targets)
    raw_targets = torch.stack(raw_targets)

    # pad variable-length tag sequences to the longest one in this batch
    # padding value 0 is the unknown/empty tag index
    tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=0)

    return cats, conts, tags, log_targets, raw_targets


def get_dataloader(parquet_path, batch_size, shuffle, num_workers=0):
    dataset = BidDataset(parquet_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader
