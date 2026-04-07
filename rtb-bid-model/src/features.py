import os
import bz2
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# column definitions (same as eda.py, copied here so this file stands alone)
COLUMNS = [
    "bid_id", "timestamp", "log_type", "user_id", "user_agent", "ip",
    "region", "city", "ad_exchange", "domain", "url_hash", "anonymous_url",
    "slot_id", "slot_width", "slot_height", "slot_visibility", "slot_format",
    "slot_floor_price", "creative_id", "bidding_price", "payprice",
    "key_page_url", "advertiser_id", "user_tags",
]

KEEP_COLS = [
    "timestamp", "region", "city", "ad_exchange", "slot_width", "slot_height",
    "slot_visibility", "slot_format", "slot_floor_price", "payprice",
    "advertiser_id", "user_tags",
]

TRAIN_FILES = [
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130606.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130607.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130608.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130609.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130610.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130611.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130612.txt.bz2",
]

TEST_FILES = [
    "data/raw/archive/ipinyou.contest.dataset/testing2nd/leaderboard.test.data.20130613_15.txt.bz2",
]

PROCESSED_DIR = "data/processed"

# feature definitions

# these get label encoded and go into embedding layers
# slot_width and slot_height are discrete standard ad sizes (from EDA),
# so I'm treating them as categorical rather than continuous
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

# slot_floor_price is the only truly continuous feature I have
CONTINUOUS_FEATURES = [
    "slot_floor_price",
]

# derived from timestamp
CYCLICAL_FEATURES = ["hour", "weekday"]

TARGET = "payprice"


# loading
def detect_columns(filepath):
    """
    reads the first line of a bz2 file to count how many tab-separated
    columns it has. test files sometimes have a different column count
    than training files, so I check first.
    """
    with bz2.open(filepath, "rt", encoding="utf-8") as f:
        first_line = f.readline()
    return len(first_line.strip().split("\t"))


def load_file(filepath):
    """
    loads a single bz2 impression log into a dataframe.
    I detect the column count first to handle format differences between
    training files (24 cols) and test files (which may differ).
    only keeps columns that exist in both formats.
    """
    n_cols = detect_columns(filepath)
    print(f"{os.path.basename(filepath)} {n_cols} columns detected")

    # use only as many column names as the file actually has;
    # pad with generic names if the file has more columns than I defined
    col_names = COLUMNS[:n_cols]
    if len(col_names) < n_cols:
        col_names = col_names + [f"extra_{i}" for i in range(len(col_names), n_cols)]

    # only keep columns that exist in this file
    keep = [c for c in KEEP_COLS if c in col_names]

    with bz2.open(filepath, "rt", encoding="utf-8") as f:
        df = pd.read_csv(
            f,
            sep="\t",
            header=None,
            names=col_names,
            usecols=keep,
            dtype=str,
            low_memory=False,
        )
    return df


def load_files(filepaths, desc=""):
    """loads all the bz2 files and stacks them together."""
    dfs = []
    for fp in tqdm(filepaths, desc=desc):
        dfs.append(load_file(fp))
    return pd.concat(dfs, ignore_index=True)


# cleaning
def clean(df):
    """
    cleaning steps I do every time:
    - cast numeric columns to the right types
    - drop rows with zero or missing payprice (log is undefined at 0)
    - fill missing user_tags with empty string
    - normalize sentinel values (slot_visibility=255 stays as its own category)
    """
    # cast target and continuous features to float
    df["payprice"] = pd.to_numeric(df["payprice"], errors="coerce")
    df["slot_floor_price"] = pd.to_numeric(df["slot_floor_price"], errors="coerce").fillna(0.0)

    # drop rows where payprice is zero, negative, or couldn't be parsed
    before = len(df)
    df = df[df["payprice"] > 0].copy()
    after = len(df)
    print(f"dropped {before - after:,} rows with payprice <= 0")

    # fill missing user_tags -- no entry means no audience targeting data
    df["user_tags"] = df["user_tags"].fillna("").replace("null", "")

    # fill missing categoricals with "unknown"
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown")

    return df


# timestamp parsing
def parse_timestamp(df):
    """
    pulls hour and weekday out of the raw timestamp string.
    format is YYYYMMDDHHMMSSMS (e.g. 20130606143022192)
    """
    ts = df["timestamp"].astype(str)
    df["hour"] = ts.str[8:10].astype(int)
    df["weekday"] = pd.to_datetime(ts.str[:8], format="%Y%m%d").dt.dayofweek
    df = df.drop(columns=["timestamp"])
    return df


# user tag parsing

def build_tag_vocab(df):
    """
    builds a vocab of all unique tag IDs seen in training data.
    tags are comma-separated integers in the user_tags column.
    returns a dict of tag_id -> index, starting from 1 so I can use 0 as padding.
    """
    tag_set = set()
    for tag_str in df["user_tags"]:
        if tag_str:
            for tag in tag_str.split(","):
                tag = tag.strip()
                if tag:
                    tag_set.add(tag)

    # sort for determinism, reserve 0 for unknown/padding
    vocab = {tag: idx + 1 for idx, tag in enumerate(sorted(tag_set))}
    print(f"tag vocabulary size {len(vocab):,} unique tags")
    return vocab


def encode_tags(tag_str, vocab, max_tags=50):
    """
    converts a comma-separated tag string into a list of integer indices.
    truncates to max_tags. unknown tags get index 0.
    returns a list of ints (variable length, the dataset collate_fn handles padding).
    """
    if not tag_str:
        return [0]  # no tags -- single padding index
    tags = [t.strip() for t in tag_str.split(",") if t.strip()][:max_tags]
    return [vocab.get(tag, 0) for tag in tags]


# categorical encoding

def fit_label_encoders(df):
    """
    fits one LabelEncoder per categorical feature on training data.
    returns a dict of {feature_name: fitted LabelEncoder}.
    each encoder maps a string category to an integer index.
    """
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
        print(f"{col} {len(le.classes_)} unique categories")
    return encoders


def apply_label_encoders(df, encoders):
    """
    applies fitted label encoders to a dataframe.
    unseen categories (in val/test) get mapped to the 'unknown' class.
    """
    for col, le in encoders.items():
        known_classes = set(le.classes_)
        df[col] = df[col].astype(str).apply(
            lambda x: x if x in known_classes else "unknown"
        )
        # make sure 'unknown' is in the encoder -- add it if it's not there yet
        if "unknown" not in known_classes:
            le.classes_ = np.append(le.classes_, "unknown")
        df[col] = le.transform(df[col].astype(str))
    return df


# continuous feature scaling

def fit_scaler(df):
    """
    fits a StandardScaler on continuous features from training data.
    slot_floor_price has a heavy zero-spike (most slots have no floor price),
    so I scale it but don't drop it -- the model should figure out the spike.
    """
    scaler = StandardScaler()
    scaler.fit(df[CONTINUOUS_FEATURES])
    return scaler


def apply_scaler(df, scaler):
    """applies the fitted scaler to continuous features."""
    df[CONTINUOUS_FEATURES] = scaler.transform(df[CONTINUOUS_FEATURES])
    return df


# cyclical encoding
def add_cyclical_features(df):
    """
    encodes hour (0-23) and weekday (0-6) as sine/cosine pairs.
    this way the model understands that hour 23 and hour 0 are adjacent,
    and that Sunday and Monday wrap around correctly.
    I drop the raw hour/weekday columns after encoding them.
    """
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df = df.drop(columns=["hour", "weekday"])
    return df


# target transformation
def log_transform_target(df):
    """
    stores log(payprice) alongside raw payprice.
    the model predicts mu and sigma of log(payprice).
    I keep raw payprice around for bid regret calculation during evaluation.
    """
    df["log_payprice"] = np.log(df["payprice"])
    return df


# train/val split
def temporal_split(df, train_frac=0.85):
    """
    splits training data into train and val sets by time order.
    first 85% of rows (chronologically) go to train, rest to val.
    I don't shuffle -- temporal order has to be preserved to avoid leakage.
    the test set is a separate held-out file (testing2nd), not a random split.
    """
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    return train, val


# main pipeline
def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- load ---
    print("\nloading training files")
    train_df = load_files(TRAIN_FILES, desc="Training")

    print("\nloading test files")
    test_df = load_files(TEST_FILES, desc="Test")

    # --- clean ---
    print("\ncleaning training data")
    train_df = clean(train_df)

    print("\ncleaning test data")
    test_df = clean(test_df)

    # --- parse timestamps ---
    print("\nparsing timestamps")
    train_df = parse_timestamp(train_df)
    test_df = parse_timestamp(test_df)

    # --- build tag vocabulary from training data only ---
    print("\nbuilding tag vocabulary")
    tag_vocab = build_tag_vocab(train_df)

    # encode tags -- store as list of ints per row
    print("encoding user tags")
    train_df["tag_indices"] = train_df["user_tags"].apply(
        lambda x: encode_tags(x, tag_vocab)
    )
    test_df["tag_indices"] = test_df["user_tags"].apply(
        lambda x: encode_tags(x, tag_vocab)
    )
    train_df = train_df.drop(columns=["user_tags"])
    test_df = test_df.drop(columns=["user_tags"])

    # --- fit encoders and scaler on training data only ---
    print("\nfitting label encoders on training data")
    encoders = fit_label_encoders(train_df)

    print("\nfitting scaler on training data")
    scaler = fit_scaler(train_df)

    # --- apply encoders and scaler ---
    print("\napplying encoders")
    train_df = apply_label_encoders(train_df, encoders)
    test_df = apply_label_encoders(test_df, encoders)

    print("applying scaler")
    train_df = apply_scaler(train_df, scaler)
    test_df = apply_scaler(test_df, scaler)

    # --- cyclical features ---
    print("adding cyclical time features")
    train_df = add_cyclical_features(train_df)
    test_df = add_cyclical_features(test_df)

    # --- log transform target ---
    train_df = log_transform_target(train_df)
    test_df = log_transform_target(test_df)

    # --- temporal train/val split ---
    print("\nsplitting into train and val sets 85/15 temporal")
    train_split, val_split = temporal_split(train_df, train_frac=0.85)

    print(f"train rows {len(train_split):,}")
    print(f"val rows {len(val_split):,}")
    print(f"test rows {len(test_df):,}")

    # --- save splits to disk ---
    print("\nsaving processed splits")
    train_split.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"), index=False)
    val_split.to_parquet(os.path.join(PROCESSED_DIR, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"), index=False)

    # --- save encoders, scaler, and tag vocab ---
    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "tag_vocab": tag_vocab,
        "categorical_features": CATEGORICAL_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
    }
    with open(os.path.join(PROCESSED_DIR, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    print("\nfeature engineering done")
    print(f"splits saved to {PROCESSED_DIR}/")
    print(f"encoders and vocab saved to {PROCESSED_DIR}/artifacts.pkl")

    # print final feature summary
    print("\nfinal feature summary")
    sample = train_split.drop(columns=["payprice", "log_payprice", "tag_indices"])
    print(f"structured feature columns ({len(sample.columns)})")
    for col in sample.columns:
        print(f"  {col}")
    print("plus tag_indices variable-length list mean-pooled in model")
    print("target log_payprice float, raw payprice kept for bid regret")


if __name__ == "__main__":
    main()
