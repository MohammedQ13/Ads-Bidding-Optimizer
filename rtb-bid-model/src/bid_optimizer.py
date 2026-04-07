import torch
import math
from scipy.stats import lognorm
import numpy as np


# impression value V in CNY fen -- what this impression is worth to the advertiser
# keeping it as a constant for now, in production this would come from a pCTR model
IMPRESSION_VALUE = 150.0

def compute_optimal_bid(mu, sigma, impression_value=IMPRESSION_VALUE, chunk_size=10000):
    mu_np = mu.detach().cpu().numpy()
    sigma_np = sigma.detach().cpu().numpy()
    candidates = np.linspace(1.0, impression_value, 500).astype(np.float32)
    log_candidates = np.log(candidates)
    all_bids = []

    for start in range(0, len(mu_np), chunk_size):
        end = min(start + chunk_size, len(mu_np))
        mu_chunk = mu_np[start:end].astype(np.float32)
        sigma_chunk = sigma_np[start:end].astype(np.float32)

        # log-normal CDF via the normal CDF: Phi((log(x) - mu) / sigma)
        # shape: [chunk, 500]
        z = (log_candidates[np.newaxis, :] - mu_chunk[:, np.newaxis]) / sigma_chunk[:, np.newaxis]
        win_probs = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))

        expected_profits = (impression_value - candidates[np.newaxis, :]) * win_probs
        best_indices = np.argmax(expected_profits, axis=1)
        all_bids.append(candidates[best_indices])

    return torch.tensor(np.concatenate(all_bids), dtype=torch.float32)

def _find_optimal_bid(mu, sigma, V):
    # keeping this for reference but I'm no longer calling it
    candidates = np.linspace(1.0, V, 500)
    win_probs = lognorm.cdf(candidates, s=sigma, scale=np.exp(mu))
    expected_profits = (V - candidates) * win_probs
    best_idx = np.argmax(expected_profits)
    return candidates[best_idx]


def compute_naive_bid(log_targets_train):
    """
    naive baseline: just always bid the historical average clearing price.
    this is the dumbest possible strategy, used as a comparison in bid regret evaluation.
    log_targets_train: log_payprice values from training set
    returns: a single constant bid value in CNY fen
    """
    mean_log_price = log_targets_train.mean().item()
    # convert from log space back to CNY fen
    naive_bid = math.exp(mean_log_price)
    return naive_bid


def compute_bid_regret(optimal_bids, actual_clearing_prices, impression_value=IMPRESSION_VALUE):
    """
    bid regret measures how much profit I leave on the table compared to a perfect bidder.

    with perfect information you'd know the clearing price exactly and bid just above it.
    perfect profit on a won auction = V - clearing_price.
    my actual profit = (V - my_bid) if I won, 0 if I lost.

    regret = perfect_profit - actual_profit, averaged over all auctions.

    optimal_bids:           [batch] tensor of model bids in CNY fen
    actual_clearing_prices: [batch] tensor of actual payprices in CNY fen
    impression_value:       advertiser value V in CNY fen
    """
    V = impression_value

    # did I win? I win if my bid >= clearing price
    won = (optimal_bids >= actual_clearing_prices).float()

    # actual profit: (V - bid) if won, 0 if lost
    actual_profit = won * (V - optimal_bids)

    # perfect profit: (V - clearing_price) if clearing_price < V, else 0
    # a perfect bidder wins whenever V > clearing_price
    perfect_won = (actual_clearing_prices < V).float()
    perfect_profit = perfect_won * (V - actual_clearing_prices)

    # regret per auction
    regret = perfect_profit - actual_profit

    return regret
