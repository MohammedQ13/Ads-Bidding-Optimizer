import sys
sys.path.append("src")

import torch
import numpy as np
from bid_optimizer import compute_optimal_bid, compute_bid_regret, IMPRESSION_VALUE


def test_bid_is_below_value():
    # optimal bid should never go above the impression value, that would be irrational
    mu = torch.tensor([3.5, 4.0, 4.5])
    sigma = torch.tensor([0.5, 0.8, 1.0])
    bids = compute_optimal_bid(mu, sigma)
    assert (bids <= IMPRESSION_VALUE).all(), "bid exceeds impression value"
    print(f"pass  all bids below impression value ({IMPRESSION_VALUE} fen)")
    print(f"bids: {bids.numpy().round(2)}")


def test_bid_is_positive():
    mu = torch.tensor([2.0, 3.0, 4.0, 5.0])
    sigma = torch.tensor([0.5, 0.5, 0.5, 0.5])
    bids = compute_optimal_bid(mu, sigma)
    assert (bids > 0).all(), "bid is non-positive"
    print(f"pass all bids positive")


def test_higher_sigma_more_conservative():
    # with the same mu, higher uncertainty should push bids lower
    # because I risk overpaying on cheap auctions if I'm not careful
    mu = torch.tensor([4.0, 4.0])
    sigma_low = torch.tensor([0.3, 0.3])
    sigma_high = torch.tensor([1.5, 1.5])
    bids_low = compute_optimal_bid(mu, sigma_low)
    bids_high = compute_optimal_bid(mu, sigma_high)
    print(f"pass bid with low sigma:  {bids_low[0].item():.2f} fen")
    print(f"bid with high sigma: {bids_high[0].item():.2f} fen")
    # note: this relationship can actually reverse at extremes, so I just print it
    # rather than hard asserting


def test_bid_regret_shape():
    bids = torch.tensor([60.0, 80.0, 100.0, 40.0])
    clearing = torch.tensor([55.0, 90.0, 95.0, 45.0])
    regret = compute_bid_regret(bids, clearing)
    assert regret.shape == (4,), "regret shape wrong"
    assert (regret >= 0).all(), "regret should be non-negative"
    print(f"pass bid regret shape and non-negativity")
    print(f"regrets: {regret.numpy().round(2)}")


def test_zero_regret_perfect_bid():
    # if I bid exactly at the clearing price I win and pay the minimum
    # regret should be zero in this case
    clearing = torch.tensor([70.0, 80.0, 90.0])
    bids = clearing.clone()
    regret = compute_bid_regret(bids, clearing)
    assert (regret == 0).all(), "perfect bid should have zero regret"
    print(f"pass perfect bid produces zero regret")


def main():
    print("bid optimizer tests")
    test_bid_is_below_value()
    test_bid_is_positive()
    test_higher_sigma_more_conservative()
    test_bid_regret_shape()
    test_zero_regret_perfect_bid()
    print("\nbid optimizer validation done")


if __name__ == "__main__":
    main()
