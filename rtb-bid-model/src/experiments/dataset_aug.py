"""BidDatasetAug: adds bidding_price (raw and log) as input features.

Same parquet, different cont_arr layout. Model expects num_continuous = 11.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch


def load_artifacts(processed_dir):
    with open(os.path.join(processed_dir, 'artifacts.pkl'), 'rb') as f:
        return pickle.load(f)


class BidDatasetAug:

    def __init__(self, parquet_path, artifacts, has_extras=False, bp_mean=None, bp_std=None,
                 lp_bp_mean=None, lp_bp_std=None):
        df = pd.read_parquet(parquet_path)
        self.cat_features = artifacts['categorical_features']
        self.has_extras = has_extras
        self.cont_cols = ['log_floor_price', 'slot_area', 'tag_count']
        self.bin_cols = ['has_floor_price', 'is_weekend']
        self.cyc_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']

        n = len(df)
        cat_arr = np.zeros((n, len(self.cat_features)), dtype=np.int64)
        for i, feat in enumerate(self.cat_features):
            cat_arr[:, i] = df[feat + '_idx'].values.astype(np.int64)

        ncont = len(self.cont_cols) + len(self.bin_cols) + len(self.cyc_cols) + 2  # +2 for bp/log_bp
        cont_arr = np.zeros((n, ncont), dtype=np.float32)
        col = 0
        for c in self.cont_cols:
            cont_arr[:, col] = df[c].values.astype(np.float32); col += 1
        for c in self.bin_cols:
            cont_arr[:, col] = df[c].values.astype(np.float32); col += 1
        for c in self.cyc_cols:
            cont_arr[:, col] = df[c].values.astype(np.float32); col += 1
        bp = df['bidding_price'].values.astype(np.float32)
        lp_bp = np.log(np.clip(bp, 1.0, None)).astype(np.float32)
        if bp_mean is None:
            bp_mean = float(bp.mean()); bp_std = float(bp.std()) if bp.std() > 1e-6 else 1.0
            lp_bp_mean = float(lp_bp.mean()); lp_bp_std = float(lp_bp.std()) if lp_bp.std() > 1e-6 else 1.0
        cont_arr[:, col] = ((bp - bp_mean) / bp_std); col += 1
        cont_arr[:, col] = ((lp_bp - lp_bp_mean) / lp_bp_std); col += 1
        self.bp_mean = bp_mean; self.bp_std = bp_std
        self.lp_bp_mean = lp_bp_mean; self.lp_bp_std = lp_bp_std

        tags = df['tag_indices'].values
        tag_arr = np.zeros((n, 10), dtype=np.int64)
        try:
            stacked = np.array(list(tags), dtype=np.int64)
            if stacked.shape == (n, 10):
                tag_arr = stacked
        except Exception:
            for i in range(n):
                row = tags[i]
                for j in range(min(10, len(row))):
                    tag_arr[i, j] = int(row[j])

        self.cat = torch.from_numpy(cat_arr)
        self.cont = torch.from_numpy(cont_arr)
        self.tags = torch.from_numpy(tag_arr)
        self.log_pp = torch.from_numpy(df['log_payprice'].values.astype(np.float32))
        self.pp = torch.from_numpy(df['payprice'].values.astype(np.float32))
        self.bid = torch.from_numpy(df['bidding_price'].values.astype(np.float32))
        self.adv = df['advertiser_id'].values.astype(str)
        if has_extras:
            self.click = torch.from_numpy(df['click'].values.astype(np.int64))
            self.conv = torch.from_numpy(df['conversion'].values.astype(np.int64))
        self.n = n

    def num_continuous(self):
        return self.cont.shape[1]

    def __len__(self):
        return self.n

    def iter_batches(self, batch_size, shuffle=False, drop_last=False, generator=None):
        if shuffle:
            perm = torch.randperm(self.n, generator=generator) if generator is not None else torch.randperm(self.n)
        else:
            perm = torch.arange(self.n)
        cur = 0
        while cur < self.n:
            end = cur + batch_size
            if end > self.n:
                if drop_last:
                    break
                end = self.n
            idx = perm[cur:end]
            yield {
                'cat': self.cat[idx],
                'cont': self.cont[idx],
                'tags': self.tags[idx],
                'log_pp': self.log_pp[idx],
                'pp': self.pp[idx],
                'bid': self.bid[idx],
            }
            cur = end

    def num_batches(self, batch_size, drop_last=False):
        if drop_last:
            return self.n // batch_size
        return (self.n + batch_size - 1) // batch_size
