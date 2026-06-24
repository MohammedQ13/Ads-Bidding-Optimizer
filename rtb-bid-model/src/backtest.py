"""Backtest the bidding strategies on the real iPinYou test set.

For each real auction we run the model, compute the profit-maximizing bid, then
apply each company's strategy multiplier and check it against the REAL clearing
price (payprice). This answers: on real data, how much profit and what win rate
does each strategy get? It grounds the synthetic simulation against reality and
doubles as a sanity check (the profit-max strategy should reproduce the ~20-fen
regret the model was selected on).

Run:
    python src/backtest.py            # quick sample
    python src/backtest.py --limit 0  # full test set
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import onnxruntime as ort


# the 9 categorical columns in model order, and the 9 continuous columns in the
# order dataset.py packs them (3 standardized + 2 binary + 4 cyclical)
CAT_COLS = ["region_idx", "city_idx", "domain_idx", "ad_exchange_idx",
            "slot_width_idx", "slot_height_idx", "slot_visibility_idx",
            "slot_format_idx", "advertiser_id_idx"]
CONT_COLS = ["log_floor_price", "slot_area", "tag_count", "has_floor_price",
             "is_weekend", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]

# the companies: name -> bid multiplier (matches the C++ configs)
STRATEGIES = {
    "profit_max (A)": 1.0,
    "aggressive (B)": 1.2,
    "conservative (C)": 0.8,
}


def best_bids(probs, V, num_bins):
    """Profit-maximizing bid per row: argmax over bins of (V-b)*CDF(b)."""
    cdf = np.cumsum(probs, axis=1)
    bins = np.arange(num_bins)
    profit = (V - bins)[None, :] * cdf
    # bidding above V is never profitable
    mask = bins > V
    profit[:, mask] = 0.0
    return profit.argmax(axis=1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../cpp-bidder/models/bid_model.onnx")
    ap.add_argument("--parquet", default="data/processed/test.parquet")
    ap.add_argument("--V", type=float, default=150.0)
    ap.add_argument("--limit", type=int, default=300000,
                    help="rows to use, 0 = all")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--out", default="exports/backtest.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(here, model_path)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    num_bins = sess.get_outputs()[0].shape[1]

    df = pd.read_parquet(os.path.join(here, args.parquet))
    if args.limit and args.limit < len(df):
        df = df.iloc[: args.limit]
    n = len(df)
    print("backtesting on", n, "real auctions, V =", args.V, flush=True)

    payprice = df["payprice"].values.astype(np.float64)

    # accumulators per strategy
    acc = {}
    for name in STRATEGIES:
        acc[name] = {"profit": 0.0, "regret": 0.0, "wins": 0, "bid_sum": 0.0}
    # oracle: bid just above payprice when profitable
    oracle_profit = np.where(payprice < args.V, args.V - payprice, 0.0)

    V = args.V
    for start in range(0, n, args.batch):
        end = min(start + args.batch, n)
        sub = df.iloc[start:end]
        cat = np.stack([sub[c].values.astype(np.int64) for c in CAT_COLS], axis=1)
        cont = np.stack([sub[c].values.astype(np.float32) for c in CONT_COLS], axis=1)
        tags = np.stack(sub["tag_indices"].values).astype(np.int64)

        probs = sess.run(None, {"cat": cat, "cont": cont, "tags": tags})[0]
        bstar = best_bids(probs.astype(np.float64), V, num_bins)

        pp = payprice[start:end]
        for name, mult in STRATEGIES.items():
            bid = np.clip(bstar * mult, 0, num_bins - 1)
            won = bid >= pp
            profit = np.where(won, V - bid, 0.0)
            profit = np.where(profit < 0, 0.0, profit)
            realized = np.where(won, V - bid, 0.0)
            regret = oracle_profit[start:end] - np.where(won, realized, 0.0)
            a = acc[name]
            a["profit"] += float(profit.sum())
            a["regret"] += float(regret.sum())
            a["wins"] += int(won.sum())
            a["bid_sum"] += float(bid.sum())
        if (start // args.batch) % 5 == 0:
            print("  processed", end, "/", n, flush=True)

    print("\nstrategy             win_rate   avg_profit/imp   mean_regret   avg_bid")
    print("-" * 72)
    results = {}
    for name, mult in STRATEGIES.items():
        a = acc[name]
        win_rate = a["wins"] / n
        avg_profit = a["profit"] / n
        mean_regret = a["regret"] / n
        avg_bid = a["bid_sum"] / n
        results[name] = {
            "multiplier": mult,
            "win_rate": win_rate,
            "avg_profit_per_imp": avg_profit,
            "mean_regret": mean_regret,
            "avg_bid": avg_bid,
            "total_profit": a["profit"],
        }
        print(f"{name:20s} {win_rate:7.3f}   {avg_profit:13.3f}   {mean_regret:11.3f}   {avg_bid:7.2f}")
    print("\noracle avg profit/imp:", round(float(oracle_profit.mean()), 3))

    out_path = os.path.join(here, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n": n, "V": V, "strategies": results,
                   "oracle_avg_profit": float(oracle_profit.mean())}, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
