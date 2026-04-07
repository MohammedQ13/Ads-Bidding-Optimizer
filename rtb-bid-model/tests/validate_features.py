import os
import pickle
import pandas as pd
import numpy as np

PROCESSED_DIR = "data/processed"

def load_artifacts():
    with open(os.path.join(PROCESSED_DIR, "artifacts.pkl"), "rb") as f:
        return pickle.load(f)

def load_splits():
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val   = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    test  = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    return train, val, test


# individual checks -- each one prints pass or fail with some detail

def check_shapes(train, val, test):
    print("\nshape checks")
    print(f"Train: {len(train):>12,} rows, {train.shape[1]} columns")
    print(f"Val:   {len(val):>12,} rows, {val.shape[1]} columns")
    print(f"Test:  {len(test):>12,} rows, {test.shape[1]} columns")

    total = len(train) + len(val)
    actual_frac = len(train) / total
    expected_frac = 0.85
    if abs(actual_frac - expected_frac) < 0.01:
        print(f"pass Train/val split is {actual_frac:.2%} / {1-actual_frac:.2%} (expected ~85/15)")
    else:
        print(f"fail Train/val split is {actual_frac:.2%}, expected ~85%")


def check_no_leakage(train, val, test):
    """
    just checking bidding_price got dropped -- if it's here I messed up and left in a leakage column.
    """
    print("\nleakage check")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if "bidding_price" in df.columns:
            print(f"fail bidding_price found in {name} -- data leakage")
        else:
            print(f"pass bidding_price not present in {name}")


def check_target(train, val, test):
    """
    making sure payprice and log_payprice are there and look right.
    no zeros, no negatives, no NaN, and log values should all be finite.
    """
    print("\ntarget checks")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        issues = []
        if "payprice" not in df.columns:
            issues.append("missing payprice")
        if "log_payprice" not in df.columns:
            issues.append("missing log_payprice")
        if (df["payprice"] <= 0).any():
            issues.append(f"{(df['payprice'] <= 0).sum()} rows with payprice <= 0")
        if df["payprice"].isna().any():
            issues.append(f"{df['payprice'].isna().sum()} NaN in payprice")
        if not np.isfinite(df["log_payprice"]).all():
            issues.append(f"{(~np.isfinite(df['log_payprice'])).sum()} non-finite log_payprice")

        if issues:
            print(f"fail {name}: {', '.join(issues)}")
        else:
            print(f"pass {name}: payprice and log_payprice valid")
            print(f"payprice range [{df['payprice'].min():.0f}, {df['payprice'].max():.0f}]")
            print(f"log_payprice range [{df['log_payprice'].min():.3f}, {df['log_payprice'].max():.3f}]")


def check_categoricals(train, val, test, artifacts):
    """
    checking all categorical columns got integer encoded with no NaN.
    also making sure val/test don't have category indices outside the training vocab
    (unseen categories should have been mapped to the unknown index).
    """
    print("\ncategorical encoding checks")
    encoders = artifacts["encoders"]
    cat_cols = artifacts["categorical_features"]

    for col in cat_cols:
        if col not in train.columns:
            print(f"fail {col} missing from train")
            continue

        train_vals = set(train[col].unique())
        val_vals   = set(val[col].unique())
        test_vals  = set(test[col].unique())

        n_classes = len(encoders[col].classes_)
        max_idx = train[col].max()

        issues = []
        if train[col].isna().any():
            issues.append("NaN in train")
        if val[col].isna().any():
            issues.append("NaN in val")
        if test[col].isna().any():
            issues.append("NaN in test")

        # any index in val/test should be within the encoder's range
        oov_val  = val_vals - train_vals
        oov_test = test_vals - train_vals
        # OOV is fine only if they map to the unknown index
        unknown_idx = n_classes - 1  # unknown gets appended last
        real_oov_val  = {v for v in oov_val  if v != unknown_idx}
        real_oov_test = {v for v in oov_test if v != unknown_idx}

        if real_oov_val:
            issues.append(f"{len(real_oov_val)} unhandled OOV indices in val")
        if real_oov_test:
            issues.append(f"{len(real_oov_test)} unhandled OOV indices in test")

        if issues:
            print(f"fail {col}: {', '.join(issues)}")
        else:
            print(f"pass {col}: {n_classes} classes, max index {max_idx}, no NaN")


def check_continuous(train, val, test):
    """
    checking continuous features are scaled (mean ~0, std ~1 on train).
    if these are way off it means the scaler probably didn't get applied correctly.
    """
    print("\ncontinuous feature checks")
    cont_cols = ["slot_floor_price"]
    for col in cont_cols:
        mean = train[col].mean()
        std  = train[col].std()
        has_nan = train[col].isna().any()
        if has_nan:
            print(f"fail {col}: contains NaN")
        elif abs(mean) > 0.1 or abs(std - 1.0) > 0.2:
            print(f"fail {col}: mean={mean:.4f}, std={std:.4f} -- scaling looks wrong")
        else:
            print(f"pass {col}: mean={mean:.4f}, std={std:.4f} (expected ~0, ~1)")


def check_cyclical(train):
    """
    checking sin/cos features are in [-1, 1] and both are present.
    """
    print("\ncyclical feature checks")
    expected = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
    for col in expected:
        if col not in train.columns:
            print(f"fail {col} missing")
            continue
        mn, mx = train[col].min(), train[col].max()
        if mn < -1.01 or mx > 1.01:
            print(f"fail {col}: values outside [-1, 1] range: [{mn:.4f}, {mx:.4f}]")
        else:
            print(f"pass {col}: range [{mn:.4f}, {mx:.4f}]")


def check_tags(train, val, test, artifacts):
    """
    checking tag_indices column exists and has lists of integers.
    also making sure no tag index goes above the vocabulary size.
    """
    print("\nuser tag checks")
    tag_vocab = artifacts["tag_vocab"]
    vocab_size = len(tag_vocab) + 1  # +1 for the padding/unknown index 0

    for name, df in [("train", train), ("val", val), ("test", test)]:
        if "tag_indices" not in df.columns:
            print(f"fail {name} tag_indices column missing")
            continue

        sample = df["tag_indices"].iloc[:10000]
        all_flat = [idx for tags in sample for idx in tags]

        max_idx = max(all_flat) if all_flat else 0
        min_idx = min(all_flat) if all_flat else 0
        empty_rows = (df["tag_indices"].apply(len) == 0).sum()
        single_zero = (df["tag_indices"].apply(lambda x: list(x) == [0])).sum()

        issues = []
        if max_idx >= vocab_size:
            issues.append(f"tag index {max_idx} exceeds vocab size {vocab_size}")
        if min_idx < 0:
            issues.append(f"negative tag index {min_idx}")

        if issues:
            print(f"fail {name} {', '.join(issues)}")
        else:
            print(f"pass {name} vocab_size={vocab_size} max_tag_idx={max_idx}")
            print(f"rows with no tags (index=[0]): {single_zero:,}")


def check_no_raw_timestamp(train, val, test):
    """
    checking that the raw timestamp column got dropped.
    if it's still there it would be a string and break PyTorch tensor conversion.
    """
    print("\ntimestamp drop check")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if "timestamp" in df.columns:
            print(f"fail {name} raw timestamp column still present")
        elif "hour" in df.columns or "weekday" in df.columns:
            print(f"fail {name} raw hour/weekday not converted to sin/cos")
        else:
            print(f"pass {name} timestamp properly removed sin/cos features present")


def print_final_feature_list(train):
    print("\nfinal feature columns")
    skip = {"payprice", "log_payprice", "tag_indices"}
    structured = [c for c in train.columns if c not in skip]
    print(f"structured features ({len(structured)})")
    for col in structured:
        dtype = train[col].dtype
        print(f"  {col:<25} dtype={dtype}")
    print(f"tag indices variable-length list per row")
    print(f"target log_payprice (float)")
    print(f"kept raw payprice (for bid regret evaluation)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("loading processed data")
    artifacts = load_artifacts()
    train, val, test = load_splits()

    check_shapes(train, val, test)
    check_no_leakage(train, val, test)
    check_target(train, val, test)
    check_categoricals(train, val, test, artifacts)
    check_continuous(train, val, test)
    check_cyclical(train)
    check_tags(train, val, test, artifacts)
    check_no_raw_timestamp(train, val, test)
    print_final_feature_list(train)

    print("\nvalidation complete")


if __name__ == "__main__":
    main()
