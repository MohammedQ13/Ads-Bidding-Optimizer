"""Unit tests for the retrainer.

The most important thing to check is that the retrainer encodes features the same
way the C++ server and the training pipeline do - otherwise it would fine-tune on
mis-encoded inputs. We check encode_record against the golden fixture (which the
C++ parity test already proved correct), plus a few standalone checks.

Run locally:
    python test_retrain.py
"""

import os
import json
import sys

import retrain


HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "cpp-bidder", "test", "golden_fixture.json")
CONFIG = os.path.join(HERE, "models", "feature_config.json")

failures = 0


def check(ok, msg):
    global failures
    if not ok:
        failures += 1
        print("  FAIL:", msg)


def split_tags(raw):
    """Match FeatureStore::split_tags: null/NaN/empty -> none, else comma tokens."""
    if not raw or raw == "null" or raw == "NaN":
        return []
    return [t for t in raw.split(",") if t]


def test_weekday():
    # 2013-06-06 was a Thursday -> Monday=0 convention gives 3
    check(retrain.weekday_monday0(20130606120000) == 3, "weekday Thu=3")
    # 2013-06-08 was a Saturday -> 5
    check(retrain.weekday_monday0(20130608000000) == 5, "weekday Sat=5")
    # 2013-06-09 was a Sunday -> 6
    check(retrain.weekday_monday0(20130609000000) == 6, "weekday Sun=6")


def test_encode_matches_fixture():
    with open(CONFIG) as f:
        cfg = json.load(f)
    with open(FIXTURE) as f:
        fx = json.load(f)

    for case in fx["cases"]:
        r = dict(case["request"])
        r["user_tags"] = split_tags(r["user_tags"])  # engine sends a list
        cat, cont, tags = retrain.encode_record(r, cfg)

        cid = r.get("request_id", "?")
        check(cat == case["cat"], cid + " cat mismatch: " + str(cat) + " vs " + str(case["cat"]))
        check(tags == case["tags"], cid + " tags mismatch: " + str(tags) + " vs " + str(case["tags"]))
        for i in range(9):
            diff = abs(cont[i] - case["cont"][i])
            check(diff < 1e-4, cid + " cont[" + str(i) + "] diff " + str(diff))


def test_build_tensors_clamps_label():
    with open(CONFIG) as f:
        cfg = json.load(f)
    num_bins = cfg["num_bins"]
    # a record whose clearing price is way above the top bin should clamp
    rec = {
        "region": 1, "city": 1, "domain": "x", "ad_exchange": 1,
        "slot_width": 300, "slot_height": 250, "slot_visibility": 1,
        "slot_format": 0, "advertiser_id": "1458", "user_tags": [],
        "slot_floor_price": 0.0, "timestamp": 20130606120000,
        "clearing_price": 99999.0,
    }
    cat_t, cont_t, tags_t, lab_t = retrain.build_tensors([rec], cfg, num_bins)
    check(int(cat_t.shape[1]) == 9, "cat tensor has 9 cols")
    check(int(cont_t.shape[1]) == 9, "cont tensor has 9 cols")
    check(int(tags_t.shape[1]) == 10, "tags tensor has 10 cols")
    check(int(lab_t[0]) == num_bins - 1, "huge clearing price clamps to last bin")

    # a normal clearing price rounds to its bin
    rec2 = dict(rec)
    rec2["clearing_price"] = 73.4
    _, _, _, lab2 = retrain.build_tensors([rec2], cfg, num_bins)
    check(int(lab2[0]) == 73, "clearing price 73.4 rounds to bin 73")


def main():
    test_weekday()
    test_encode_matches_fixture()
    test_build_tensors_clamps_label()
    if failures == 0:
        print("RETRAINER TESTS OK")
        return 0
    print("RETRAINER TESTS FAILED:", failures)
    return 1


if __name__ == "__main__":
    sys.exit(main())
