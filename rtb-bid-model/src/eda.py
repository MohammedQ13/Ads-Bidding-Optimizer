import os
import bz2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm

# column names for the season 2 impression log (24 tab-separated columns)
COLUMNS = [
    "bid_id",           # 1  - unique auction ID, dropping this
    "timestamp",        # 2  - YYYYMMDDHHMMSSMS, I'll pull hour and weekday out of this
    "log_type",         # 3  - always 1, useless, dropping
    "user_id",          # 4  - anonymized session ID, dropping
    "user_agent",       # 5  - raw UA string, dropping
    "ip",               # 6  - masked IP, dropping
    "region",           # 7  - integer region code
    "city",             # 8  - integer city code
    "ad_exchange",      # 9  - 1=Tanx, 2=Baidu, 3=Shenma
    "domain",           # 10 - anonymized publisher hash, dropping
    "url_hash",         # 11 - anonymized URL, dropping
    "anonymous_url",    # 12 - always null, dropping
    "slot_id",          # 13 - ad slot ID, dropping
    "slot_width",       # 14 - pixels
    "slot_height",      # 15 - pixels
    "slot_visibility",  # 16 - 0=unknown, 1=above fold, 2=below fold
    "slot_format",      # 17 - 0=banner, 1=popup, 5=other
    "slot_floor_price", # 18 - publisher minimum price in CNY fen
    "creative_id",      # 19 - which ad creative was shown, dropping
    "bidding_price",    # 20 - DSP bid price, dropping this (data leakage)
    "payprice",         # 21 - clearing price in CNY fen, this is my TARGET
    "key_page_url",     # 22 - anonymized hash, dropping
    "advertiser_id",    # 23 - 5 advertisers: 1458, 3358, 3386, 3427, 3476
    "user_tags",        # 24 - comma-separated audience segment tag IDs
]

# columns I'm keeping for modeling
KEEP_COLS = [
    "timestamp",
    "region",
    "city",
    "ad_exchange",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "slot_floor_price",
    "payprice",
    "advertiser_id",
    "user_tags",
]

# training files in chronological order
TRAIN_FILES = [
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130606.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130607.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130608.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130609.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130610.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130611.txt.bz2",
    "data/raw/archive/ipinyou.contest.dataset/training2nd/imp.20130612.txt.bz2",
]

PLOT_DIR = "results/plots"


# data loading

def load_impression_file(filepath):
    """
    loads a single bz2 impression log into a dataframe.
    only reads the columns I actually need for modeling.
    """
    with bz2.open(filepath, "rt", encoding="utf-8") as f:
        df = pd.read_csv(
            f,
            sep="\t",
            header=None,
            names=COLUMNS,
            usecols=KEEP_COLS,
            dtype={
                "region": str,
                "city": str,
                "ad_exchange": str,
                "slot_width": float,
                "slot_height": float,
                "slot_visibility": str,
                "slot_format": str,
                "slot_floor_price": float,
                "payprice": float,
                "advertiser_id": str,
                "user_tags": str,
            },
            low_memory=False,
        )
    return df


def load_all_training_data():
    """
    loads all 7 training days and stacks them into one big dataframe.
    keeps them in chronological order.
    """
    frames = []
    for filepath in tqdm(TRAIN_FILES, desc="Loading training files"):
        df = load_impression_file(filepath)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# timestamp parsing

def parse_timestamp(df):
    """
    pulls hour and weekday out of the raw timestamp string.
    format is YYYYMMDDHHMMSSMS (e.g. 20130606000104192)
    I'll use hour and weekday as cyclical features later.
    """
    ts = df["timestamp"].astype(str)
    df["hour"] = ts.str[8:10].astype(int)
    df["weekday"] = pd.to_datetime(ts.str[:8], format="%Y%m%d").dt.dayofweek
    df = df.drop(columns=["timestamp"])
    return df


# basic dataset stats

def print_basic_stats(df):
    print("\ndataset shape")
    print(f"rows {len(df):,}")
    print(f"columns {list(df.columns)}")

    print("\npayprice summary")
    print(df["payprice"].describe())

    print("\nmissing values")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "none")

    print("\nadvertiser distribution")
    print(df["advertiser_id"].value_counts())

    print("\nad exchange distribution")
    print(df["ad_exchange"].value_counts())

    print("\nslot visibility distribution")
    print(df["slot_visibility"].value_counts())

    print("\npayprice range")
    print(f"min {df['payprice'].min()}")
    print(f"max {df['payprice'].max()}")
    print(f"zero or negative values {(df['payprice'] <= 0).sum()}")

    print("\nuser tags coverage")
    has_tags = df["user_tags"].notna() & (df["user_tags"] != "null") & (df["user_tags"] != "")
    print(f"rows with tags {has_tags.sum():,} ({100 * has_tags.mean():.1f}%)")


# log-normal assumption check
def validate_lognormal(df):
    """
    the key question: is log(payprice) roughly normal?
    I check this visually with a histogram and QQ plot, and also run a KS test.
    if yes, it justifies using NLL loss with a log-normal distribution assumption.
    """
    # drop any zero or negative prices since log is undefined there
    prices = df["payprice"][df["payprice"] > 0].values
    log_prices = np.log(prices)

    print("\nlog-normal assumption check")
    print(f"samples used {len(prices):,}")
    print(f"log payprice mean {log_prices.mean():.4f}")
    print(f"log payprice std {log_prices.std():.4f}")

    # KS test: compare log(payprice) against a fitted normal
    mu, sigma = log_prices.mean(), log_prices.std()
    ks_stat, ks_pval = stats.kstest(log_prices, "norm", args=(mu, sigma))
    print(f"\nks test on log payprice vs normal mu={mu:.3f} sigma={sigma:.3f}")
    print(f"ks statistic {ks_stat:.4f}")
    print(f"p-value {ks_pval:.4f}")
    print(f"{'cannot reject normality' if ks_pval > 0.05 else 'reject normality distribution may not be log-normal'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Log-Normal Assumption Validation", fontsize=13)

    # raw payprice distribution
    axes[0].hist(prices, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_title("payprice (raw)")
    axes[0].set_xlabel("payprice (CNY fen)")
    axes[0].set_ylabel("Count")

    # log(payprice) with a fitted normal overlaid so I can see how close it is
    axes[1].hist(log_prices, bins=100, color="steelblue", edgecolor="none",
                 alpha=0.8, density=True, label="Empirical")
    x = np.linspace(log_prices.min(), log_prices.max(), 300)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Fitted Normal")
    axes[1].set_title("log(payprice) with Normal fit")
    axes[1].set_xlabel("log(payprice)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # QQ plot: if log(payprice) is actually normal the points should line up on the diagonal
    stats.probplot(log_prices, dist="norm", plot=axes[2])
    axes[2].set_title("QQ Plot: log(payprice) vs Normal")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lognormal_validation.png"), dpi=150)
    plt.close()
    print("\nsaved results/plots/lognormal_validation.png")

    return mu, sigma


# per-advertiser price distributions
def plot_advertiser_distributions(df):
    """
    each advertiser runs different campaigns so their auction dynamics are different.
    this shows how payprice distributions vary across advertisers.
    advertiser_id is probably going to be one of the strongest features.
    """
    advertiser_ids = sorted(df["advertiser_id"].unique())
    n = len(advertiser_ids)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    fig.suptitle("payprice Distribution by Advertiser", fontsize=13)

    for ax, adv_id in zip(axes, advertiser_ids):
        prices = df[df["advertiser_id"] == adv_id]["payprice"]
        prices = prices[prices > 0]
        log_prices = np.log(prices)
        ax.hist(log_prices, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(f"Advertiser {adv_id}\nn={len(prices):,}\nmedian={prices.median():.0f}")
        ax.set_xlabel("log(payprice)")
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "advertiser_distributions.png"), dpi=150)
    plt.close()
    print("saved results/plots/advertiser_distributions.png")

    print("\nper advertiser payprice stats")
    print(df.groupby("advertiser_id")["payprice"].describe().round(2))


# feature distribution plots

def plot_feature_distributions(df):
    """
    quick visual sanity check on the key categorical and continuous features.
    helps me spot encoding issues, unexpected values, or heavy class imbalance
    before I lock in a feature engineering strategy.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Feature Distributions", fontsize=13)

    # categorical counts
    cat_features = ["ad_exchange", "slot_visibility", "slot_format", "advertiser_id"]
    for ax, col in zip(axes[0], cat_features):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="steelblue", alpha=0.8)
        ax.set_title(col)
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    # continuous distributions
    cont_features = ["slot_width", "slot_height", "slot_floor_price"]
    for ax, col in zip(axes[1][:3], cont_features):
        data = df[col].dropna()
        ax.hist(data, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # hour of day distribution
    axes[1][3].hist(df["hour"], bins=24, range=(0, 24), color="steelblue",
                    edgecolor="none", alpha=0.8)
    axes[1][3].set_title("hour of day")
    axes[1][3].set_xlabel("Hour")
    axes[1][3].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_distributions.png"), dpi=150)
    plt.close()
    print("saved results/plots/feature_distributions.png")


# payprice vs features

def plot_payprice_vs_features(df):
    """
    median payprice broken down by key categorical features.
    helps me see which features actually have signal for price prediction.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Median payprice by Feature Value", fontsize=13)

    cat_features = ["ad_exchange", "slot_visibility", "slot_format", "advertiser_id"]
    for ax, col in zip(axes, cat_features):
        medians = df.groupby(col)["payprice"].median().sort_index()
        ax.bar(medians.index.astype(str), medians.values, color="steelblue", alpha=0.8)
        ax.set_title(f"Median payprice by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Median payprice (CNY fen)")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "payprice_vs_features.png"), dpi=150)
    plt.close()
    print("saved results/plots/payprice_vs_features.png")


# main
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("loading training data")
    df = load_all_training_data()

    print("parsing timestamps")
    df = parse_timestamp(df)

    print_basic_stats(df)

    mu, sigma = validate_lognormal(df)

    plot_advertiser_distributions(df)
    plot_feature_distributions(df)
    plot_payprice_vs_features(df)

    print("\neda done plots saved to results/plots/")
    print(f"global log-normal params mu={mu:.4f} sigma={sigma:.4f}")
    print("these will be used to check model calibration during evaluation")


if __name__ == "__main__":
    main()
