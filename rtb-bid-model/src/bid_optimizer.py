import math
import torch
import torch.nn.functional as F


def grid_optimize_mdn(pi_logits, mu, sigma, V, n_candidates=500, b_max=300.0):
    """Grid search bid optimizer for MDN.
    Tries n_candidates evenly spaced bids and picks the one with highest
    expected profit: (V - b) * P(payprice < b).
    Returns best_bid (B,), best_profit (B,), candidates, and profit grid.
    """
    device = mu.device
    B, K = mu.shape
    if not torch.is_tensor(V):
        V = torch.full((B,), float(V), device=device)
    elif V.dim() == 0:
        V = V.expand(B).to(device)
    else:
        V = V.to(device)

    # evenly spaced bids from 0.5 to b_max (skip 0 because log(0) is undefined)
    cands = torch.linspace(0.5, b_max, n_candidates, device=device)

    # compute CDF at each candidate bid for each sample
    # the MDN works in log-price space so we take log of the bid prices
    log_b = torch.log(cands)
    log_b_b = log_b.unsqueeze(0).expand(B, -1)
    pi = F.softmax(pi_logits, dim=-1)
    # standard normal CDF via erf for each mixture component
    z = (log_b_b.unsqueeze(2) - mu.unsqueeze(1)) / (sigma.unsqueeze(1) * math.sqrt(2.0))
    comp_cdf = 0.5 * (1.0 + torch.erf(z))  # (B, n_cand, K)
    cdf = (pi.unsqueeze(1) * comp_cdf).sum(dim=-1)  # (B, n_cand)

    # expected profit = (V - bid) * probability of winning at that bid
    profit = (V.unsqueeze(1) - cands.unsqueeze(0)) * cdf
    # bidding more than V guarantees negative profit, zero those out
    profit = torch.where(cands.unsqueeze(0) > V.unsqueeze(1), torch.zeros_like(profit), profit)

    best_profit, best_idx = profit.max(dim=1)
    best_bid = cands[best_idx]
    # if no bid is profitable, don't bid at all
    no_bid_mask = best_profit < 0
    best_bid = torch.where(no_bid_mask, torch.zeros_like(best_bid), best_bid)
    best_profit = torch.where(no_bid_mask, torch.zeros_like(best_profit), best_profit)
    return best_bid, best_profit, cands, profit


def grid_optimize_bins(probs, V, n_candidates=300, b_max=300.0):
    """Grid search bid optimizer for discrete bins.
    For bins, the CDF is just the cumulative sum of probabilities.
    Each bin index IS the bid price (integer fen), so we evaluate
    profit at every possible integer bid.
    probs: (B, num_bins). V: scalar or (B,).
    """
    device = probs.device
    B, num_bins = probs.shape
    if not torch.is_tensor(V):
        V = torch.full((B,), float(V), device=device)
    elif V.dim() == 0:
        V = V.expand(B).to(device)
    else:
        V = V.to(device)

    # CDF at bin k = P(payprice <= k) = cumulative sum up to k
    cdf = torch.cumsum(probs, dim=1)
    bins = torch.arange(num_bins, device=device, dtype=torch.float32)
    profit = (V.unsqueeze(1) - bins.unsqueeze(0)) * cdf
    # can't profitably bid more than V
    profit = torch.where(bins.unsqueeze(0) > V.unsqueeze(1), torch.zeros_like(profit), profit)

    best_profit, best_idx = profit.max(dim=1)
    best_bid = best_idx.float()
    no_bid_mask = best_profit < 0
    best_bid = torch.where(no_bid_mask, torch.zeros_like(best_bid), best_bid)
    best_profit = torch.where(no_bid_mask, torch.zeros_like(best_profit), best_profit)
    return best_bid, best_profit, bins, profit


def newton_optimize_mdn(pi_logits, mu, sigma, V, n_iters=25, n_starts=8, b_max=300.0):
    """Newton-Raphson bid optimizer for MDN. Multi-start, picks best root.
    Damped updates to prevent divergence. Grid search was more stable
    in practice so this is kept as an alternative.
    """
    device = mu.device
    B, K = mu.shape
    if not torch.is_tensor(V):
        V = torch.full((B,), float(V), device=device)
    elif V.dim() == 0:
        V = V.expand(B).to(device)
    V = V.to(device).clamp(min=1.0)

    pi = F.softmax(pi_logits, dim=-1)
    mu_l = mu
    sg_l = sigma

    def cdf_pdf(b):
        # compute both CDF and PDF at bid price b
        # need both for the Newton update: b_new = V - CDF(b) / pdf(b)
        log_b = torch.log(b.clamp(min=1e-6))
        z = (log_b.unsqueeze(1) - mu_l) / (sg_l * math.sqrt(2.0))
        comp_cdf = 0.5 * (1.0 + torch.erf(z))
        cdf_v = (pi * comp_cdf).sum(dim=-1)
        zz = (log_b.unsqueeze(1) - mu_l) / sg_l
        comp_pdf_log = torch.exp(-0.5 * zz * zz) / (sg_l * math.sqrt(2.0 * math.pi))
        pdf_log = (pi * comp_pdf_log).sum(dim=-1)
        # convert from log-space PDF to linear-space PDF
        pdf_lin = pdf_log / b.clamp(min=1e-6)
        return cdf_v, pdf_lin

    best_b = torch.zeros(B, device=device)
    best_p = torch.full((B,), -1e18, device=device)
    # try starting points spread evenly across [5% of V, 95% of V]
    starts_frac = torch.linspace(0.05, 0.95, n_starts, device=device)
    for s in range(n_starts):
        b = (V * starts_frac[s]).clamp(min=0.5)
        for _ in range(n_iters):
            b = b.clamp(min=1e-3, max=b_max)
            cdf_v, pdf_v = cdf_pdf(b)
            denom = pdf_v.clamp(min=1e-9)
            # first-order condition: optimal b satisfies V - b = CDF(b) / pdf(b)
            new_b = V - cdf_v / denom
            # damped update to avoid overshooting
            b = 0.5 * b + 0.5 * new_b
            b = torch.where(torch.isnan(b), torch.full_like(b, 0.5), b)
            b = b.clamp(min=0.5, max=b_max)
        # evaluate profit at the converged bid
        b = torch.where(b > V, V, b)
        cdf_v, _ = cdf_pdf(b.clamp(min=1e-3))
        profit = (V - b) * cdf_v
        # keep this start's result if it's better than previous starts
        better = profit > best_p
        best_b = torch.where(better, b, best_b)
        best_p = torch.where(better, profit, best_p)

    no_bid = best_p < 0
    best_b = torch.where(no_bid, torch.zeros_like(best_b), best_b)
    best_p = torch.where(no_bid, torch.zeros_like(best_p), best_p)
    return best_b, best_p


def newton_optimize_bins(probs, V, n_iters=15, n_starts=8, b_max=None):
    """Newton-Raphson bid optimizer for discrete bins. Multi-start.
    Interpolates CDF between integer bin edges.
    """
    device = probs.device
    B, num_bins = probs.shape
    if b_max is None:
        b_max = float(num_bins - 1)
    if not torch.is_tensor(V):
        V = torch.full((B,), float(V), device=device)
    elif V.dim() == 0:
        V = V.expand(B).to(device)
    V = V.to(device).clamp(min=1.0)

    cdf_full = torch.cumsum(probs, dim=1)

    def cdf_pdf(b):
        # interpolate CDF between integer bins for fractional bid values
        b_c = b.clamp(min=0.0, max=b_max)
        lo = b_c.floor().long().clamp(0, num_bins - 1)
        hi = (lo + 1).clamp(0, num_bins - 1)
        frac = (b_c - lo.float()).clamp(0.0, 1.0)
        cdf_lo = cdf_full.gather(1, lo.unsqueeze(1)).squeeze(1)
        cdf_hi = cdf_full.gather(1, hi.unsqueeze(1)).squeeze(1)
        cdf_v = cdf_lo + frac * (cdf_hi - cdf_lo)
        # approximate PDF as the slope of the CDF between adjacent bins
        pdf_v = (cdf_hi - cdf_lo).clamp(min=1e-9)
        return cdf_v, pdf_v

    best_b = torch.zeros(B, device=device)
    best_p = torch.full((B,), -1e18, device=device)
    starts_frac = torch.linspace(0.05, 0.95, n_starts, device=device)
    for s in range(n_starts):
        b = (V * starts_frac[s]).clamp(min=0.5)
        for _ in range(n_iters):
            b = b.clamp(min=0.0, max=b_max)
            cdf_v, pdf_v = cdf_pdf(b)
            new_b = V - cdf_v / pdf_v.clamp(min=1e-9)
            b = 0.5 * b + 0.5 * new_b
            b = torch.where(torch.isnan(b), torch.full_like(b, 0.5), b)
            b = b.clamp(min=0.0, max=b_max)
        b = torch.where(b > V, V, b)
        cdf_v, _ = cdf_pdf(b)
        profit = (V - b) * cdf_v
        better = profit > best_p
        best_b = torch.where(better, b, best_b)
        best_p = torch.where(better, profit, best_p)

    no_bid = best_p < 0
    best_b = torch.where(no_bid, torch.zeros_like(best_b), best_b)
    best_p = torch.where(no_bid, torch.zeros_like(best_p), best_p)
    return best_b, best_p


def perfect_profit(payprice, V):
    """Oracle profit: what you'd earn if you knew the clearing price exactly.
    Bid just above payprice when V > payprice (profitable), skip otherwise.
    """
    if not torch.is_tensor(V):
        V = torch.full_like(payprice, float(V))
    elif V.dim() == 0:
        V = V.expand_as(payprice)
    win = (V > payprice).float()
    return win * (V - payprice)


def regret(actual_bid, actual_profit, payprice, V):
    """Regret = oracle profit - realized profit per impression.
    You win if bid >= payprice, pay your bid. Regret = profit left on table.
    """
    if not torch.is_tensor(V):
        V = torch.full_like(payprice, float(V))
    elif V.dim() == 0:
        V = V.expand_as(payprice)
    win = (actual_bid >= payprice).float()
    realized = win * (V - actual_bid)
    perfect = perfect_profit(payprice, V)
    return perfect - realized
