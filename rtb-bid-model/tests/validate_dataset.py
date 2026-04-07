import sys
sys.path.append("src")

from dataset import get_dataloader, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, CYCLICAL_FEATURES

BATCH_SIZE = 512


def test_dataloader(name, path, shuffle):
    print(f"\n{name}")
    loader = get_dataloader(path, batch_size=BATCH_SIZE, shuffle=shuffle)

    # grab one batch to inspect
    batch = next(iter(loader))
    cats, conts, tags, log_targets, raw_targets = batch

    print(f"cats shape:        {cats.shape}        dtype={cats.dtype}")
    print(f"conts shape:       {conts.shape}       dtype={conts.dtype}")
    print(f"tags shape:        {tags.shape}         dtype={tags.dtype}")
    print(f"log_targets shape: {log_targets.shape}  dtype={log_targets.dtype}")
    print(f"raw_targets shape: {raw_targets.shape}  dtype={raw_targets.dtype}")

    # cats should be [batch, n_categorical]
    assert cats.shape == (BATCH_SIZE, len(CATEGORICAL_FEATURES)), "cats shape wrong"

    # conts should be [batch, n_continuous + n_cyclical]
    expected_cont = len(CONTINUOUS_FEATURES) + len(CYCLICAL_FEATURES)
    assert conts.shape == (BATCH_SIZE, expected_cont), "conts shape wrong"

    # tags should be [batch, max_seq_len_in_batch]
    assert tags.dim() == 2, "tags should be 2D after padding"
    assert tags.shape[0] == BATCH_SIZE, "tags batch size wrong"

    # targets should be [batch]
    assert log_targets.shape == (BATCH_SIZE,), "log_targets shape wrong"
    assert raw_targets.shape == (BATCH_SIZE,), "raw_targets shape wrong"

    # no NaN anywhere
    assert not conts.isnan().any(), "NaN in conts"
    assert not log_targets.isnan().any(), "NaN in log_targets"
    assert not raw_targets.isnan().any(), "NaN in raw_targets"

    # raw payprice should always be positive
    assert (raw_targets > 0).all(), "non-positive raw_targets"

    # categorical indices should be non-negative
    assert (cats >= 0).all(), "negative categorical index"

    # tag padding should only use 0
    assert (tags >= 0).all(), "negative tag index"

    print("pass everything looks good")
    print(f"tags max seq len in this batch: {tags.shape[1]}")
    print(f"log_target range: [{log_targets.min():.3f}, {log_targets.max():.3f}]")
    print(f"raw_target range: [{raw_targets.min():.1f}, {raw_targets.max():.1f}]")


def main():
    test_dataloader("Train loader", "data/processed/train.parquet", shuffle=True)
    test_dataloader("Val loader",   "data/processed/val.parquet",   shuffle=False)
    test_dataloader("Test loader",  "data/processed/test.parquet",  shuffle=False)
    print("\ndataset validation done")


if __name__ == "__main__":
    main()
