import sys
import json
import math
import torch
import numpy as np
import onnxruntime as ort

sys.path.append("src")
from model import BidTransformer, load_artifacts
from bid_optimizer import compute_optimal_bid, IMPRESSION_VALUE

# load onnx model and feature config
session = ort.InferenceSession("exports/bid_model.onnx")
with open("exports/feature_config.json") as f:
    config = json.load(f)

artifacts = load_artifacts()
encoders = artifacts["encoders"]
scaler = artifacts["scaler"]
tag_vocab = artifacts["tag_vocab"]


def encode_auction(region, city, ad_exchange, slot_width, slot_height,
                   slot_visibility, slot_format, advertiser_id,
                   slot_floor_price, hour, weekday, user_tags_str):
    # encode categorical features
    cat_feature_order = config["cat_feature_order"]
    cat_values = {
        "region": str(region),
        "city": str(city),
        "ad_exchange": str(ad_exchange),
        "slot_width": str(slot_width),
        "slot_height": str(slot_height),
        "slot_visibility": str(slot_visibility),
        "slot_format": str(slot_format),
        "advertiser_id": str(advertiser_id),
    }

    cat_indices = []
    for col in cat_feature_order:
        enc = encoders[col]
        val = cat_values[col]
        known = set(enc.classes_)
        if val not in known:
            val = "unknown"
        idx = enc.transform([val])[0]
        cat_indices.append(int(idx))

    cats = torch.tensor([cat_indices], dtype=torch.long)

    # encode continuous + cyclical
    scaled_floor = (slot_floor_price - scaler.mean_[0]) / scaler.scale_[0]
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    weekday_sin = math.sin(2 * math.pi * weekday / 7)
    weekday_cos = math.cos(2 * math.pi * weekday / 7)
    conts = torch.tensor([[scaled_floor, hour_sin, hour_cos, weekday_sin, weekday_cos]], dtype=torch.float32)

    # encode tags
    tag_indices = [0]
    if user_tags_str:
        for tag in user_tags_str.split(","):
            tag = tag.strip()
            if tag in tag_vocab:
                tag_indices.append(tag_vocab[tag])
    tags = torch.tensor([tag_indices], dtype=torch.long)

    return cats, conts, tags


def predict(cats, conts, tags):
    ort_inputs = {
        "cats":  cats.numpy(),
        "conts": conts.numpy(),
        "tags":  tags.numpy(),
    }
    mu_arr, sigma_arr = session.run(["mu", "sigma"], ort_inputs)
    mu = torch.tensor(mu_arr)
    sigma = torch.tensor(sigma_arr)
    return mu, sigma


def demo_auction(label, region, city, ad_exchange, slot_width, slot_height,
                 slot_visibility, slot_format, advertiser_id,
                 slot_floor_price, hour, weekday, user_tags_str=""):

    print(f"\n{'='*55}")
    print(f"Auction: {label}")
    print(f"  Exchange={ad_exchange}, Slot={slot_width}x{slot_height}, "
          f"Visibility={slot_visibility}, Format={slot_format}")
    print(f"  Advertiser={advertiser_id}, Floor={slot_floor_price} fen, Hour={hour}")

    cats, conts, tags = encode_auction(
        region, city, ad_exchange, slot_width, slot_height,
        slot_visibility, slot_format, advertiser_id,
        slot_floor_price, hour, weekday, user_tags_str
    )

    mu, sigma = predict(cats, conts, tags)

    mu_val = mu[0].item()
    sigma_val = sigma[0].item()
    median_price = math.exp(mu_val)
    optimal_bid = compute_optimal_bid(mu, sigma)[0].item()

    print(f"\n  Predicted distribution:")
    print(f"    mu    = {mu_val:.4f}  (log-space mean)")
    print(f"    sigma = {sigma_val:.4f}  (log-space std)")
    print(f"    Median clearing price = {median_price:.1f} fen")
    print(f"    Uncertainty (sigma):   {'low' if sigma_val < 0.4 else 'medium' if sigma_val < 0.8 else 'high'}")
    print(f"\n  Optimal bid = {optimal_bid:.1f} fen  (V={IMPRESSION_VALUE:.0f} fen)")
    print(f"  Bid shading = {100*(1 - optimal_bid/IMPRESSION_VALUE):.1f}% below impression value")


if __name__ == "__main__":
    print("RTB Distributional Bid Model demonstration")
    print("Impression value V = 150 CNY fen for all auctions")

    # auction 1 — premium above-fold slot on exchange 1
    demo_auction(
        label="Premium above-fold banner, peak hour",
        region=106, city=117, ad_exchange=1,
        slot_width=728, slot_height=90,
        slot_visibility=1, slot_format=0,
        advertiser_id=3358,
        slot_floor_price=50, hour=20, weekday=2
    )

    # auction 2 — low value below-fold slot
    demo_auction(
        label="Below-fold small banner, off-peak",
        region=106, city=117, ad_exchange=2,
        slot_width=300, slot_height=250,
        slot_visibility=2, slot_format=0,
        advertiser_id=1458,
        slot_floor_price=0, hour=3, weekday=0
    )

    # auction 3 — rare high-value format
    demo_auction(
        label="High-value format 5 slot",
        region=106, city=117, ad_exchange=1,
        slot_width=300, slot_height=250,
        slot_visibility=1, slot_format=5,
        advertiser_id=3427,
        slot_floor_price=100, hour=14, weekday=3
    )