import os
import bz2
import math
import gc
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from config import load_config


# iPinYou Season 2 impression log columns (tab-separated, 24 fields)
COLNAMES = [
    'bid_id', 'timestamp', 'log_type', 'user_id', 'user_agent', 'ip',
    'region', 'city', 'ad_exchange', 'domain', 'url_hash', 'anonymous_url',
    'slot_id', 'slot_width', 'slot_height', 'slot_visibility', 'slot_format',
    'slot_floor_price', 'creative_id', 'bidding_price', 'payprice',
    'key_page_url', 'advertiser_id', 'user_tags',
]
# test file has 2 extra columns at the end
TEST_EXTRA = ['click', 'conversion']

# only keep columns we actually use for features, drop the rest early to save memory
KEEP_RAW = [
    'timestamp', 'region', 'city', 'ad_exchange', 'domain',
    'slot_width', 'slot_height', 'slot_visibility', 'slot_format',
    'slot_floor_price', 'bidding_price', 'payprice',
    'advertiser_id', 'user_tags',
]


def parse_chunked(path, is_test=False, chunk=200000):
    """Read a bz2 impression log in chunks to keep memory usage bounded.
    Skips malformed lines (wrong number of columns) silently.
    """
    n_cols = len(COLNAMES) + (len(TEST_EXTRA) if is_test else 0)
    cols = COLNAMES + (TEST_EXTRA if is_test else [])
    keep_cols = list(KEEP_RAW)
    if is_test:
        keep_cols = keep_cols + ['click', 'conversion']

    rows = []
    sub_dfs = []
    with bz2.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != n_cols:
                continue
            rows.append(parts)
            if len(rows) >= chunk:
                df = pd.DataFrame(rows, columns=cols)
                df = df[keep_cols]
                sub_dfs.append(df)
                rows = []
    if rows:
        df = pd.DataFrame(rows, columns=cols)
        df = df[keep_cols]
        sub_dfs.append(df)
    full = pd.concat(sub_dfs, ignore_index=True)
    return full


def basic_features_inplace(df, is_test=False):
    """Convert raw string columns to numeric types and compute engineered features.
    Adds: hour/weekday cyclical encoding, slot_area, log_floor_price, tag_count, etc.
    Drops raw columns that are no longer needed after feature extraction.
    """
    df['region'] = pd.to_numeric(df['region'], errors='coerce').fillna(0).astype('int32')
    df['city'] = pd.to_numeric(df['city'], errors='coerce').fillna(0).astype('int32')
    df['ad_exchange'] = pd.to_numeric(df['ad_exchange'], errors='coerce').fillna(0).astype('int32')
    df['slot_width'] = pd.to_numeric(df['slot_width'], errors='coerce').fillna(0).astype('int32')
    df['slot_height'] = pd.to_numeric(df['slot_height'], errors='coerce').fillna(0).astype('int32')
    df['slot_visibility'] = pd.to_numeric(df['slot_visibility'], errors='coerce').fillna(0).astype('int32')
    df['slot_floor_price'] = pd.to_numeric(df['slot_floor_price'], errors='coerce').fillna(0).astype('float32')
    df['bidding_price'] = pd.to_numeric(df['bidding_price'], errors='coerce').fillna(0).astype('float32')
    df['payprice'] = pd.to_numeric(df['payprice'], errors='coerce').fillna(0).astype('float32')

    sf = df['slot_format'].copy()
    sf = sf.replace({'Na': '0', 'na': '0', 'NA': '0', '': '0'})
    df['slot_format'] = pd.to_numeric(sf, errors='coerce').fillna(0).astype('int32')

    # advertiser stays string for now (later encoded)
    df['advertiser_id'] = df['advertiser_id'].astype(str)

    # parse timestamp (format: YYYYMMDDHHmmss)
    # encode hour and weekday as sin/cos so the model knows 23:00 is close to 00:00
    ts = df['timestamp'].astype(str)
    df['hour'] = ts.str.slice(8, 10).astype('int32')
    dt_str = ts.str.slice(0, 8)
    dts = pd.to_datetime(dt_str, format='%Y%m%d', errors='coerce')
    df['weekday'] = dts.dt.weekday.fillna(0).astype('int32').values
    h = df['hour'].astype('float32').values
    w = df['weekday'].astype('float32').values
    df['hour_sin'] = np.sin(2.0 * math.pi * h / 24.0).astype('float32')
    df['hour_cos'] = np.cos(2.0 * math.pi * h / 24.0).astype('float32')
    df['weekday_sin'] = np.sin(2.0 * math.pi * w / 7.0).astype('float32')
    df['weekday_cos'] = np.cos(2.0 * math.pi * w / 7.0).astype('float32')
    df['is_weekend'] = (df['weekday'] >= 5).astype('float32')

    df['slot_area'] = (df['slot_width'].astype('float32') * df['slot_height'].astype('float32')).astype('float32')
    df['has_floor_price'] = (df['slot_floor_price'] > 0).astype('float32')
    df['log_floor_price'] = np.log1p(df['slot_floor_price'].clip(lower=0)).astype('float32')

    # tag_count (count of comma-separated tokens in user_tags, "null"/empty -> 0)
    tags_raw = df['user_tags'].astype(str).fillna('').values
    tag_counts = np.zeros(len(df), dtype='int32')
    i = 0
    for s in tags_raw:
        if not s or s == 'null' or s == 'NaN':
            tag_counts[i] = 0
        else:
            c = 0
            for p in s.split(','):
                if p:
                    c += 1
            tag_counts[i] = c
        i += 1
    df['tag_count'] = tag_counts.astype('float32')

    df['log_payprice'] = np.log(df['payprice'].clip(lower=1.0)).astype('float32')

    df['domain'] = df['domain'].astype(str)

    if is_test:
        df['click'] = pd.to_numeric(df['click'], errors='coerce').fillna(0).astype('int32')
        df['conversion'] = pd.to_numeric(df['conversion'], errors='coerce').fillna(0).astype('int32')

    # drop columns no longer needed
    df.drop(columns=['timestamp', 'hour', 'weekday', 'slot_floor_price'], inplace=True)
    # NB: keep user_tags string for the second pass (will be encoded then dropped)
    return df


def fit_encoder(counter, min_count=0):
    """Build a value->index mapping from frequency counts.
    Index 0 is reserved for unknown/rare values. Values below min_count
    get mapped to 0 at encoding time (treated as unknown).
    """
    vocab = {'<UNK>': 0}
    idx = 1
    for v, c in counter.most_common():
        if c < min_count:
            continue
        if v in vocab:
            continue
        vocab[v] = idx
        idx += 1
    return vocab


def apply_encoder_series(values, vocab):
    """Map a list of raw values to integer indices using the vocab.
    Unknown values (not in vocab) map to 0.
    """
    out = np.zeros(len(values), dtype='int64')
    i = 0
    for v in values:
        out[i] = vocab.get(v, 0)
        i += 1
    return out


def encode_user_tags_to_arr(tags_series, vocab, max_len=10):
    """Convert comma-separated user tag strings to fixed-length int arrays.
    Each row gets up to max_len tag indices, padded with 0.
    Tags not in vocab are skipped (not counted toward max_len).
    """
    n = len(tags_series)
    arr = np.zeros((n, max_len), dtype='int32')
    i = 0
    tags_series = tags_series.astype(str).fillna('').values
    for s in tags_series:
        if not s or s == 'null' or s == 'NaN':
            i += 1
            continue
        j = 0
        for p in s.split(','):
            if j >= max_len:
                break
            if not p:
                continue
            tid = vocab.get(p, 0)
            if tid == 0:
                continue
            arr[i, j] = tid
            j += 1
        i += 1
    return arr


def main():
    cfg = load_config()
    out_dir = cfg['data']['processed_dir']
    os.makedirs(out_dir, exist_ok=True)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_files = cfg['data']['train_files']
    test_files = cfg['data']['test_files']
    train_frac = cfg['data']['train_frac']
    cat_features = cfg['features']['categorical']
    clip_min = cfg['features']['clip_min_count']

    # three-pass pipeline to handle large data without loading everything into RAM:
    # Pass A: parse raw bz2 files, compute basic features, save as intermediate parquets
    # Pass B: scan training portion to fit encoders and compute normalization stats
    # Pass C: apply encoders/normalization and write final train/val/test parquets

    inter_dir = os.path.join(out_dir, '_inter')
    os.makedirs(inter_dir, exist_ok=True)
    inter_paths = []
    file_row_counts = []
    t0 = time.time()
    for fp in train_files:
        full = os.path.join(here, fp)
        print('parsing', full, 'elapsed', round(time.time()-t0,1), flush=True)
        df = parse_chunked(full, is_test=False)
        df = basic_features_inplace(df, is_test=False)
        out_path = os.path.join(inter_dir, os.path.basename(fp).replace('.txt.bz2', '.parquet'))
        df.to_parquet(out_path, index=False)
        file_row_counts.append(len(df))
        inter_paths.append(out_path)
        print('rows:', len(df), 'elapsed:', round(time.time()-t0,1), 's', flush=True)
        del df
        gc.collect()

    # parse test
    test_inter_paths = []
    for fp in test_files:
        full = os.path.join(here, fp)
        print('parsing test', full, flush=True)
        df = parse_chunked(full, is_test=True)
        df = basic_features_inplace(df, is_test=True)
        out_path = os.path.join(inter_dir, 'test_' + os.path.basename(fp).replace('.txt.bz2', '.parquet'))
        df.to_parquet(out_path, index=False)
        test_inter_paths.append(out_path)
        print('rows:', len(df), flush=True)
        del df
        gc.collect()

    # Pass B: temporal split (first train_frac rows = train, rest = val)
    # only count frequencies from the training portion to avoid data leakage
    total_train = sum(file_row_counts)
    cut = int(total_train * train_frac)
    print('total train rows:', total_train, 'cut:', cut, flush=True)

    cat_counters = {}
    for feat in cat_features:
        cat_counters[feat] = Counter()
    tag_counter = Counter()
    sum_log = 0.0
    sum_log_sq = 0.0
    sum_lf = 0.0
    sum_lf_sq = 0.0
    sum_sa = 0.0
    sum_sa_sq = 0.0
    sum_tc = 0.0
    sum_tc_sq = 0.0

    rows_seen = 0
    for path in inter_paths:
        df = pd.read_parquet(path)
        # how many rows in this file go to training?
        if rows_seen >= cut:
            break
        end = min(rows_seen + len(df), cut)
        if end > rows_seen:
            sub = df.iloc[: end - rows_seen]
            for feat in cat_features:
                if feat == 'advertiser_id':
                    cat_counters[feat].update(sub[feat].astype(str).values.tolist())
                elif feat == 'domain':
                    cat_counters[feat].update(sub[feat].astype(str).values.tolist())
                else:
                    cat_counters[feat].update(sub[feat].astype('int64').values.tolist())
            tags = sub['user_tags'].astype(str).fillna('').values
            for s in tags:
                if not s or s == 'null' or s == 'NaN':
                    continue
                for p in s.split(','):
                    if p:
                        tag_counter[p] += 1
            v = sub['log_payprice'].values.astype('float64')
            sum_log += float(v.sum())
            sum_log_sq += float((v * v).sum())
            v2 = sub['log_floor_price'].values.astype('float64')
            sum_lf += float(v2.sum())
            sum_lf_sq += float((v2*v2).sum())
            v3 = sub['slot_area'].values.astype('float64')
            sum_sa += float(v3.sum())
            sum_sa_sq += float((v3*v3).sum())
            v4 = sub['tag_count'].values.astype('float64')
            sum_tc += float(v4.sum())
            sum_tc_sq += float((v4*v4).sum())
        rows_seen += len(df)
        del df
        gc.collect()

    n_train = cut
    train_mean_log = sum_log / n_train
    train_std_log = math.sqrt(max(0.0, sum_log_sq / n_train - train_mean_log ** 2))
    cont_mean = {
        'log_floor_price': sum_lf / n_train,
        'slot_area': sum_sa / n_train,
        'tag_count': sum_tc / n_train,
    }
    cont_var = {
        'log_floor_price': max(1e-9, sum_lf_sq / n_train - cont_mean['log_floor_price'] ** 2),
        'slot_area': max(1e-9, sum_sa_sq / n_train - cont_mean['slot_area'] ** 2),
        'tag_count': max(1e-9, sum_tc_sq / n_train - cont_mean['tag_count'] ** 2),
    }
    cont_std = {}
    for k, v in cont_var.items():
        cont_std[k] = math.sqrt(v)
    print('train mean log_payprice:', train_mean_log, 'std:', train_std_log, flush=True)
    print('cont means:', cont_mean, 'cont stds:', cont_std, flush=True)

    encoders = {}
    for feat in cat_features:
        mc = clip_min.get(feat, 0)
        encoders[feat] = fit_encoder(cat_counters[feat], min_count=mc)
        print('encoder', feat, 'size:', len(encoders[feat]), flush=True)

    tag_vocab = fit_encoder(tag_counter, min_count=200)
    # rename '<UNK>' to '<PAD>' for tag (semantically pad)
    new_tag_vocab = {}
    for k, v in tag_vocab.items():
        if k == '<UNK>':
            new_tag_vocab['<PAD>'] = v
        else:
            new_tag_vocab[k] = v
    tag_vocab = new_tag_vocab
    print('tag vocab size:', len(tag_vocab), flush=True)

    # Pass C: apply encoders and normalization, split into train/val, write final parquets
    cont_cols = ['log_floor_price', 'slot_area', 'tag_count']
    keep = []
    for feat in cat_features:
        keep.append(feat + '_idx')
    keep += ['has_floor_price', 'log_floor_price', 'slot_area', 'tag_count']
    keep += ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend']
    keep += ['tag_indices']
    keep += ['payprice', 'log_payprice', 'bidding_price', 'advertiser_id']

    train_chunks = []
    val_chunks = []
    rows_seen = 0
    for path in inter_paths:
        df = pd.read_parquet(path)
        # encode categoricals
        for feat in cat_features:
            if feat in ('advertiser_id', 'domain'):
                vals = df[feat].astype(str).values.tolist()
            else:
                vals = df[feat].astype('int64').values.tolist()
            df[feat + '_idx'] = apply_encoder_series(vals, encoders[feat])
        # standardize cont
        for c in cont_cols:
            df[c] = ((df[c].astype('float64').values - cont_mean[c]) / cont_std[c]).astype('float32')
        # encode tag indices (10 per row int32)
        tag_arr = encode_user_tags_to_arr(df['user_tags'], tag_vocab, max_len=10)
        # store as nested list for parquet
        ti = []
        for i in range(tag_arr.shape[0]):
            ti.append(tag_arr[i].tolist())
        df['tag_indices'] = ti
        # split rows
        n = len(df)
        if rows_seen >= cut:
            val_chunks.append(df[keep].copy())
        elif rows_seen + n <= cut:
            train_chunks.append(df[keep].copy())
        else:
            split = cut - rows_seen
            train_chunks.append(df.iloc[:split][keep].copy())
            val_chunks.append(df.iloc[split:][keep].copy())
        rows_seen += n
        del df, tag_arr
        gc.collect()

    train_df = pd.concat(train_chunks, ignore_index=True)
    val_df = pd.concat(val_chunks, ignore_index=True)
    train_df.to_parquet(os.path.join(out_dir, 'train.parquet'), index=False)
    val_df.to_parquet(os.path.join(out_dir, 'val.parquet'), index=False)
    print('wrote train', len(train_df), 'val', len(val_df), flush=True)
    del train_df, val_df, train_chunks, val_chunks
    gc.collect()

    # test
    keep_test = list(keep) + ['click', 'conversion']
    test_chunks = []
    for path in test_inter_paths:
        df = pd.read_parquet(path)
        for feat in cat_features:
            if feat in ('advertiser_id', 'domain'):
                vals = df[feat].astype(str).values.tolist()
            else:
                vals = df[feat].astype('int64').values.tolist()
            df[feat + '_idx'] = apply_encoder_series(vals, encoders[feat])
        for c in cont_cols:
            df[c] = ((df[c].astype('float64').values - cont_mean[c]) / cont_std[c]).astype('float32')
        tag_arr = encode_user_tags_to_arr(df['user_tags'], tag_vocab, max_len=10)
        ti = []
        for i in range(tag_arr.shape[0]):
            ti.append(tag_arr[i].tolist())
        df['tag_indices'] = ti
        test_chunks.append(df[keep_test].copy())
        del df, tag_arr
        gc.collect()
    test_df = pd.concat(test_chunks, ignore_index=True)
    test_df.to_parquet(os.path.join(out_dir, 'test.parquet'), index=False)
    print('wrote test', len(test_df), flush=True)

    embedding_dims = dict(cfg['features']['embedding_dims'])
    embedding_dims['_tag_vocab_size'] = len(tag_vocab)
    artifacts = {
        'encoders': encoders,
        'scalers': {'mean': cont_mean, 'std': cont_std, 'cols': cont_cols},
        'tag_vocab': tag_vocab,
        'categorical_features': cat_features,
        'embedding_dims': embedding_dims,
        'train_mean_log_payprice': float(train_mean_log),
        'train_std_log_payprice': float(train_std_log),
        'cont_cols': cont_cols,
        'keep_train_cols': keep,
        'keep_test_cols': keep_test,
    }
    with open(os.path.join(out_dir, 'artifacts.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)
    print('wrote artifacts.pkl', flush=True)

    # cleanup intermediates
    for p in inter_paths + test_inter_paths:
        try:
            os.remove(p)
        except Exception:
            pass
    try:
        os.rmdir(inter_dir)
    except Exception:
        pass
    print('done.', flush=True)


if __name__ == '__main__':
    main()
