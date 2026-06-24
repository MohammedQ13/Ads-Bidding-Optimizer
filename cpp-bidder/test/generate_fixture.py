"""Generate a golden fixture for the C++ parity test.

This builds a set of raw bid requests, encodes them into model inputs using the
exact same rules as the training pipeline (features.py / dataset.py), runs the
exported ONNX model, and computes the optimal bid the same way evaluate.py does
(grid_optimize_bins). The C++ server must reproduce all of this. We dump
everything to golden_fixture.json so the C++ test can check it field by field.

Run from the cpp-bidder directory:
    python test/generate_fixture.py
"""

import os
import json
import math
import datetime
import numpy as np
import onnxruntime as ort


HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")


def load_config():
    with open(os.path.join(MODELS, "feature_config.json")) as f:
        return json.load(f)


def encode_categorical(cfg, req):
    """Map the 9 categorical fields to vocab indices, unknown -> 0."""
    out = []
    for feat in cfg["cat_order"]:
        vocab = cfg["cat_vocabs"][feat]
        if feat == "domain" or feat == "advertiser_id":
            key = str(req[feat])
        else:
            # numeric categoricals: the training vocab keys are the integer
            # value rendered as a string (json keys are always strings)
            key = str(int(req[feat]))
        out.append(int(vocab.get(key, 0)))
    return out


def count_tags(tags_str):
    """Raw token count, matching tag_count in basic_features_inplace."""
    if not tags_str or tags_str == "null" or tags_str == "NaN":
        return 0
    c = 0
    for p in tags_str.split(","):
        if p:
            c += 1
    return c


def encode_tags(cfg, tags_str):
    """Up to 10 known tag indices, unknown/pad skipped, rest padded with 0."""
    vocab = cfg["tag_vocab"]
    max_len = cfg["tag_max_len"]
    arr = [0] * max_len
    if not tags_str or tags_str == "null" or tags_str == "NaN":
        return arr
    j = 0
    for p in tags_str.split(","):
        if j >= max_len:
            break
        if not p:
            continue
        tid = int(vocab.get(p, 0))
        if tid == 0:
            continue
        arr[j] = tid
        j += 1
    return arr


def encode_continuous(cfg, req):
    """The 9 continuous slots: 3 standardized, 2 binary, 4 cyclical.
    Order must match dataset.py: cont_cols + bin_cols + cyc_cols.
    """
    means = cfg["cont_means"]
    stds = cfg["cont_stds"]

    floor = max(0.0, float(req["slot_floor_price"]))
    log_floor = math.log1p(floor)
    slot_area = float(req["slot_width"]) * float(req["slot_height"])
    tag_count = float(count_tags(req["user_tags"]))

    log_floor_z = (log_floor - means["log_floor_price"]) / stds["log_floor_price"]
    slot_area_z = (slot_area - means["slot_area"]) / stds["slot_area"]
    tag_count_z = (tag_count - means["tag_count"]) / stds["tag_count"]

    has_floor = 1.0 if float(req["slot_floor_price"]) > 0 else 0.0

    # timestamp is YYYYMMDDHHmmss
    ts = str(req["timestamp"])
    hour = int(ts[8:10])
    date = ts[0:8]
    dt = datetime.datetime.strptime(date, "%Y%m%d")
    weekday = dt.weekday()  # Monday=0 .. Sunday=6, matches pandas
    is_weekend = 1.0 if weekday >= 5 else 0.0
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
    weekday_sin = math.sin(2.0 * math.pi * weekday / 7.0)
    weekday_cos = math.cos(2.0 * math.pi * weekday / 7.0)

    return [
        log_floor_z, slot_area_z, tag_count_z,
        has_floor, is_weekend,
        hour_sin, hour_cos, weekday_sin, weekday_cos,
    ]


def optimal_bid(probs, V, num_bins):
    """grid_optimize_bins: bin index == price in fen. profit = (V-b)*CDF(b)."""
    cdf = np.cumsum(probs)
    bins = np.arange(num_bins, dtype=np.float64)
    profit = (V - bins) * cdf
    profit = np.where(bins > V, 0.0, profit)
    best_idx = int(np.argmax(profit))
    best_profit = float(profit[best_idx])
    if best_profit < 0:
        return 0.0, 0.0
    return float(best_idx), best_profit


def main():
    cfg = load_config()
    num_bins = cfg["num_bins"]
    sess = ort.InferenceSession(os.path.join(MODELS, "bid_model.onnx"),
                                providers=["CPUExecutionProvider"])

    # a spread of requests: known and unknown vocab entries, different
    # weekdays/hours, with and without floor prices and tags
    requests = [
        {"request_id": "r0", "region": 216, "city": 1, "domain": "trqRTuiEMaYZ",
         "ad_exchange": 1, "slot_width": 300, "slot_height": 250,
         "slot_visibility": 1, "slot_format": 0, "advertiser_id": "1458",
         "user_tags": "10063,10024,13403", "slot_floor_price": 0.0,
         "timestamp": 20130606120000},
        {"request_id": "r1", "region": 80, "city": 79, "domain": "unknown_dom_xyz",
         "ad_exchange": 2, "slot_width": 728, "slot_height": 90,
         "slot_visibility": 2, "slot_format": 1, "advertiser_id": "3358",
         "user_tags": "10006,99999", "slot_floor_price": 50.0,
         "timestamp": 20130608153000},
        {"region": 999, "city": 88888, "domain": "trqRTuiEMaYZ",
         "ad_exchange": 3, "slot_width": 160, "slot_height": 600,
         "slot_visibility": 0, "slot_format": 0, "advertiser_id": "9999999",
         "user_tags": "", "slot_floor_price": 5.0,
         "timestamp": 20130609000000, "request_id": "r2"},
        {"request_id": "r3", "region": 3, "city": 2, "domain": "abcd1234",
         "ad_exchange": 1, "slot_width": 468, "slot_height": 60,
         "slot_visibility": 1, "slot_format": 1, "advertiser_id": "2259",
         "user_tags": "null", "slot_floor_price": 100.0,
         "timestamp": 20130612235959},
        {"request_id": "r4", "region": 146, "city": 217, "domain": "trqRTuiEMaYZ",
         "ad_exchange": 1, "slot_width": 300, "slot_height": 600,
         "slot_visibility": 1, "slot_format": 0, "advertiser_id": "1458",
         "user_tags": "10063,10006,10110,10083,10024,10111,10059,10031,10075,10057,99999",
         "slot_floor_price": 20.0, "timestamp": 20130607093000},
    ]

    V_values = [50.0, 150.0, 300.0]

    cases = []
    for req in requests:
        cat = encode_categorical(cfg, req)
        cont = encode_continuous(cfg, req)
        tags = encode_tags(cfg, req["user_tags"])

        cat_in = np.array([cat], dtype=np.int64)
        cont_in = np.array([cont], dtype=np.float32)
        tags_in = np.array([tags], dtype=np.int64)
        probs = sess.run(None, {"cat": cat_in, "cont": cont_in, "tags": tags_in})[0][0]

        bids = {}
        for V in V_values:
            b, p = optimal_bid(probs.astype(np.float64), V, num_bins)
            bids[str(int(V))] = {"bid": b, "profit": p}

        cases.append({
            "request": req,
            "cat": cat,
            "cont": cont,
            "tags": tags,
            "probs": [float(x) for x in probs],
            "bids": bids,
        })

    fixture = {
        "num_bins": num_bins,
        "V_values": V_values,
        "cases": cases,
    }
    out_path = os.path.join(HERE, "golden_fixture.json")
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print("wrote", out_path, "with", len(cases), "cases")


if __name__ == "__main__":
    main()
