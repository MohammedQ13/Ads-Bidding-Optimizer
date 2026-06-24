# Dataset and Feature Engineering

## Source

iPinYou Global RTB Bidding Algorithm Competition, Season 2.
Public dataset released by iPinYou (Chinese DSP).
Data collected June 6-15, 2013.

## What RTB Data Looks Like

In real-time bidding, every time a user loads a webpage with ad slots,
the publisher's SSP (supply-side platform) sends a bid request to multiple
DSPs. Each DSP decides whether to bid and how much. The highest bidder wins
and their ad gets shown.

This data is from the second-price era (winner pays second-highest bid).
We apply a first-price framing: treat observed clearing prices as
competitor bid levels, and optimize bids assuming you pay what you bid.
This is more realistic to the modern industry (first-price became
dominant in 2019-2021).

## Files We Use

Training (7 days of impression logs):
- imp.20130606.txt.bz2 through imp.20130612.txt.bz2
- Each file is a bz2-compressed TSV

Testing (3 days):
- leaderboard.test.data.20130613_15.txt.bz2
- Same format but has 2 extra columns (clicks, conversions)

## Column Layout (24 columns, tab-separated)

| Col | Name             | What it is                          | Used? |
|-----|------------------|-------------------------------------|-------|
| 0   | bid_id           | unique auction ID                   | no    |
| 1   | timestamp        | YYYYMMDDHHMMSSMS                    | yes   |
| 2   | log_type         | always 1                            | no    |
| 3   | user_id          | anonymized session ID               | no    |
| 4   | user_agent       | raw UA string                       | no    |
| 5   | ip               | masked IP                           | no    |
| 6   | region           | integer region code                 | yes   |
| 7   | city             | integer city code                   | yes   |
| 8   | ad_exchange      | 1=Tanx, 2=Baidu, 3=Shenma          | yes   |
| 9   | domain           | anonymized publisher hash           | yes   |
| 10  | url_hash         | anonymized URL                      | no    |
| 11  | anonymous_url    | always null                         | no    |
| 12  | slot_id          | ad slot identifier                  | no    |
| 13  | slot_width       | pixels                              | yes   |
| 14  | slot_height      | pixels                              | yes   |
| 15  | slot_visibility  | 0=unknown, 1=above fold, 2=below   | yes   |
| 16  | slot_format      | 0=banner, 1=popup, 5=other          | yes   |
| 17  | slot_floor_price | publisher minimum price (CNY fen)   | yes   |
| 18  | creative_id      | which ad was shown                  | no    |
| 19  | bidding_price    | iPinYou's bid (not used as input feature; used only for V=bid evaluation) | no    |
| 20  | payprice         | clearing price (CNY fen) = TARGET   | yes   |
| 21  | key_page_url     | anonymized hash                     | no    |
| 22  | advertiser_id    | 5 advertisers in season 2           | yes   |
| 23  | user_tags        | comma-separated audience tag IDs    | yes   |

Test files have 2 extra columns (col 24 = clicks, col 25 = conversions).
We load these as labels but don't use them for training since the
training files don't have them.

---

## Feature Engineering Pipeline (features.py)

All feature engineering lives in `src/features.py`. It runs once to
convert raw bz2 impression logs into processed parquet files ready for
training. The pipeline uses a three-pass approach to keep memory usage
bounded on machines with limited RAM.

### Source File Constants

Three constants define the raw data schema:

- `COLNAMES`: list of 24 column names matching the iPinYou TSV format
- `TEST_EXTRA`: `['click', 'conversion']` - two extra columns in the test file
- `KEEP_RAW`: list of 14 columns we actually need (the rest are dropped early to save memory)

### Pass A: Parse and Basic Features

**parse_chunked(path, is_test=False, chunk=200000)**

Reads a bz2-compressed impression log in chunks of 200K rows. Opens
the file with `bz2.open`, reads line by line, splits on tabs. Skips
malformed lines (wrong number of columns). Each chunk is turned into
a pandas DataFrame, then only KEEP_RAW columns are kept. Chunks are
concatenated at the end.

The chunk-based reading is needed because some files have 2M+ rows
and loading all at once would spike memory.

**basic_features_inplace(df, is_test=False)**

Takes a raw DataFrame from parse_chunked and engineers features in-place.

Numeric conversions:
- region, city, ad_exchange, slot_width, slot_height, slot_visibility:
  string -> int32 (with coerce + fillna(0) for malformed values)
- slot_floor_price, bidding_price, payprice: string -> float32
- slot_format: special handling - replaces 'Na'/'na'/'NA'/'' with '0'
  before numeric conversion
- advertiser_id: stays as string (encoded later in Pass C)

Engineered features:
- **hour_sin, hour_cos**: `sin(2*pi*hour/24)`, `cos(2*pi*hour/24)`.
  Cyclical encoding so the model knows 23:00 is close to 00:00.
  Hour is extracted from the timestamp string (chars 8-10).
- **weekday_sin, weekday_cos**: `sin(2*pi*weekday/7)`, `cos(2*pi*weekday/7)`.
  Same idea. Weekday is extracted by parsing the date portion.
- **is_weekend**: binary, 1 if weekday >= 5 (Saturday or Sunday)
- **slot_area**: `slot_width * slot_height`. Computed before width/height
  become categorical (captures the continuous size signal).
- **has_floor_price**: binary, 1 if slot_floor_price > 0
- **log_floor_price**: `log1p(slot_floor_price)`. Log transform because
  floor prices are right-skewed.
- **tag_count**: number of comma-separated tokens in user_tags. Null/empty
  strings get count 0. Computed with a simple loop, not a comprehension.
- **log_payprice**: `log(payprice)` clipped to min 1 (the training target).

Columns dropped after feature extraction: timestamp, hour, weekday,
slot_floor_price (replaced by log_floor_price and has_floor_price).

user_tags string is kept here and gets encoded in Pass C.

Each file is saved as an intermediate parquet in `data/processed/_inter/`.
Memory is freed with `del df; gc.collect()` after each file.

### Pass B: Fit Encoders on Training Data

The temporal train/val split happens here. Total row count across all
7 training files is computed, and the first `train_frac` (85%) rows
form the training set. Only these rows are used to fit encoders and
compute normalization statistics, preventing data leakage from the
validation set.

**Categorical encoder fitting:**

For each categorical feature, a `Counter` counts value frequencies
across the training portion. Then `fit_encoder(counter, min_count)`
builds the vocabulary:

**fit_encoder(counter, min_count=0)**

Builds a value-to-index mapping. Index 0 is reserved for `<UNK>`
(unknown/rare values). Iterates `counter.most_common()` and assigns
sequential indices starting from 1. Values with count below `min_count`
are excluded - they will map to index 0 at encoding time.

Min count thresholds from config.yaml:
- domain: 100 (rare domains -> index 0)
- city: 500 (rare cities -> index 0)
- all others: 0 (no clipping)

**Tag vocabulary fitting:**

Same Counter + fit_encoder approach, with `min_count=200`. Tags
appearing fewer than 200 times in training are excluded. After
fitting, `<UNK>` is renamed to `<PAD>` since index 0 semantically
represents padding in the tag sequence. Final tag vocab ~14K entries.

**Normalization statistics:**

Running sums are computed in a single pass over the training portion:
- sum and sum-of-squares for log_payprice, log_floor_price, slot_area, tag_count
- Mean and std derived from these (one-pass formula, no explicit Welford:
  `mean = sum/n`, `std = sqrt(sum_sq/n - mean^2)`)

These are computed on the training set only and later applied to
train, val, and test.

### Pass C: Apply Encoders and Write Final Parquets

**apply_encoder_series(values, vocab)**

Maps a list of raw values to integer indices using the fitted vocab.
Unknown values (not in vocab) map to 0. Returns int64 numpy array.
Uses a simple loop, not a list comprehension.

**encode_user_tags_to_arr(tags_series, vocab, max_len=10)**

Converts comma-separated user tag strings to fixed-length int32 arrays.
Each row gets up to `max_len=10` tag indices, padded with 0.
Tags not in vocab are skipped (not counted toward max_len).
Uses nested loops: outer loop over rows, inner loop over
comma-split tokens.

For each intermediate parquet file:
1. Apply categorical encoders: create `{feat}_idx` columns
2. Standardize continuous features: `(x - mean) / std` using training stats
3. Encode user tags to fixed-length arrays
4. Split rows into train or val based on the temporal cutpoint
5. Keep only the final columns needed for training

Final output files:
- `data/processed/train.parquet` (~10.4M rows)
- `data/processed/val.parquet` (~1.8M rows)
- `data/processed/test.parquet` (~2.5M rows)

Intermediate parquet files in `_inter/` are deleted after processing.

### artifacts.pkl

Everything the downstream code needs to reconstruct the feature
encoding at inference time:

| Key | Type | Description |
|-----|------|-------------|
| `encoders` | dict of dicts | `{feat_name: {value: index}}` for each categorical |
| `scalers` | dict | `{'mean': {col: float}, 'std': {col: float}, 'cols': list}` |
| `tag_vocab` | dict | `{tag_string: index}` |
| `categorical_features` | list | ordered list of feature names |
| `embedding_dims` | dict | `{feat: dim}` from config + `_tag_vocab_size` |
| `train_mean_log_payprice` | float | mean of log(payprice) on training set |
| `train_std_log_payprice` | float | std of log(payprice) on training set |
| `cont_cols` | list | `['log_floor_price', 'slot_area', 'tag_count']` |
| `keep_train_cols` | list | column names kept in train/val parquets |
| `keep_test_cols` | list | column names kept in test parquet (includes click/conv) |

---

## BidDataset (dataset.py)

`src/dataset.py` provides the data loading layer for training and
evaluation. It reads processed parquets into CPU tensors held in RAM.

### load_artifacts(processed_dir)

Loads `artifacts.pkl` from the processed directory. Thin wrapper
around `pickle.load`.

### BidDataset class

**Constructor:** `BidDataset(parquet_path, artifacts, has_extras=False)`

Reads the parquet into pandas, then converts everything to PyTorch
tensors stored in RAM. The `has_extras` flag is True for the test
split, which has click and conversion columns.

**Tensor attributes after construction:**

| Attribute | Shape | Type | Description |
|-----------|-------|------|-------------|
| `self.cat` | (N, 9) | int64 | packed categorical feature indices |
| `self.cont` | (N, 9) | float32 | 3 standardized + 2 binary + 4 cyclical |
| `self.tags` | (N, 10) | int64 | user tag indices, padded with 0 |
| `self.log_pp` | (N,) | float32 | log(payprice) - training target |
| `self.pp` | (N,) | float32 | raw payprice (for regret computation) |
| `self.bid` | (N,) | float32 | iPinYou's bidding_price |
| `self.adv` | (N,) | str | advertiser ID strings |
| `self.click` | (N,) | int64 | click label (only if has_extras) |
| `self.conv` | (N,) | int64 | conversion label (only if has_extras) |
| `self.n` | int | - | number of samples |

The 9 continuous features are packed in this order:
1. log_floor_price (z-score standardized)
2. slot_area (z-score standardized)
3. tag_count (z-score standardized)
4. has_floor_price (binary)
5. is_weekend (binary)
6. hour_sin (cyclical)
7. hour_cos (cyclical)
8. weekday_sin (cyclical)
9. weekday_cos (cyclical)

**Methods:**

`num_continuous()`: Returns 9 (the number of continuous feature columns).

`__len__()`: Returns self.n.

`iter_batches(batch_size, shuffle=False, drop_last=False, generator=None)`:
Generator that yields batches as dicts with keys `cat`, `cont`, `tags`,
`log_pp`, `pp`, `bid`. If shuffle=True, generates a random permutation
of indices (using an optional torch.Generator for reproducibility).
If drop_last=True, drops the final incomplete batch. The shuffle
permutes indices, not the actual data arrays - this avoids copying
large tensors.

`num_batches(batch_size, drop_last=False)`:
Returns the number of batches per epoch. Floor division if drop_last,
ceil division otherwise. Used by train.py to compute total_steps for
the LR schedule.

---

## Data Splits

- Train: first 85% of training data (chronological, no shuffle) - ~10.4M impressions
- Val: last 15% of training data - ~1.8M impressions
- Test: separate held-out file (June 13-15) - ~2.5M impressions

Temporal ordering is preserved to avoid data leakage. The test set is
naturally from a later time period than training. The train-test
distribution shift is the main performance ceiling - the model cannot
predict market changes it has never seen.

## Selection Bias (Winner's Curse)

The impression log ONLY contains auctions that iPinYou won. We never see
auctions where they bid too low and lost. This means the observed clearing
prices are biased - we systematically miss expensive auctions.

iPinYou used a "fixed relatively high-price bidding strategy" for data
collection, so they won most auctions. Looking at the data, their bids
(col 19, ~227 fen) are well above typical clearing prices (20-200 fen).
The bias is mild but real.

For a production system you would want the bid logs (all auctions
including losses) and use censored regression or survival analysis
to correct for this. For this project it is a known limitation.

## Files Not Downloaded

The full dataset also includes bid logs, click logs, and conversion logs
for season 2. We only downloaded impression logs. The click/conversion
data would be needed to build a pCTR model (which would give us a proper
per-auction impression value instead of the hardcoded V=150).
