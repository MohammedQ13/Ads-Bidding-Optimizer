"""Budget-constrained bidding simulation on the test set.

Process auctions in temporal order. For each auction, optimize the bid using
the model's CDF/probs and a per-row V (V=150 by default, or V=bidding_price).
Maintain cumulative spend; when budget exceeded, stop bidding.

Metrics: impressions won, clicks won (from test extras), total profit, mean cost.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch


def grid_optimize_from_probs(probs_chunk, V_chunk, num_bins):
    """Vectorized grid optimization on already-discrete bin probs."""
    cdf = np.cumsum(probs_chunk, axis=1)
    bins_arr = np.arange(num_bins, dtype='float32')
    profit = (V_chunk[:, None] - bins_arr[None, :]) * cdf
    profit = np.where(bins_arr[None, :] > V_chunk[:, None], 0.0, profit)
    best_idx = profit.argmax(axis=1)
    best_b = bins_arr[best_idx]
    best_p = profit[np.arange(profit.shape[0]), best_idx]
    no_bid = best_p < 0
    best_b = np.where(no_bid, 0.0, best_b)
    return best_b


def mdn_to_bins_probs_np(pi_logits, mu, sigma, num_bins=301, chunk=10000, device='cuda'):
    """Run on GPU because erf is the bottleneck. Returns numpy (N, num_bins)."""
    import torch.nn.functional as F
    import math
    pi_full = F.softmax(pi_logits, dim=-1)
    upper = torch.arange(num_bins, dtype=torch.float32) + 0.5
    lower = torch.cat([torch.zeros(1), torch.arange(num_bins - 1, dtype=torch.float32) + 0.5])
    log_upper = torch.log(upper.clamp(min=1e-3)).to(device)
    log_lower = torch.log(lower.clamp(min=1e-3)).to(device)
    log_lower[0] = -1e9
    N = pi_full.shape[0]
    out = np.zeros((N, num_bins), dtype='float32')
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        pi = pi_full[s:e].to(device)
        mu_b = mu[s:e].to(device)
        sg_b = sigma[s:e].to(device)
        z_up = (log_upper.view(1, -1, 1) - mu_b.unsqueeze(1)) / (sg_b.unsqueeze(1) * math.sqrt(2.0))
        z_lo = (log_lower.view(1, -1, 1) - mu_b.unsqueeze(1)) / (sg_b.unsqueeze(1) * math.sqrt(2.0))
        cu = (pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_up))).sum(dim=-1)
        cl = (pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_lo))).sum(dim=-1)
        p = (cu - cl).clamp(min=1e-10)
        p = p / p.sum(dim=1, keepdim=True)
        out[s:e] = p.cpu().numpy()
    return out


def quantiles_to_bins_np(quants, alphas, num_bins=301):
    N, A = quants.shape
    boundaries = np.arange(num_bins + 1, dtype='float32') - 0.5
    boundaries[0] = 0.0
    afull = np.concatenate([[0.0], alphas, [1.0]]).astype('float32')
    qfull = np.concatenate([np.full((N, 1), -1e9, dtype='float32'),
                            quants.astype('float32'),
                            np.full((N, 1), 1e9, dtype='float32')], axis=1)
    cdf = np.zeros((N, len(boundaries)), dtype='float32')
    chunk = 30000
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        qf = qfull[s:e]
        idx = (qf[:, :, None] <= boundaries[None, None, :]).sum(axis=1)
        idx = np.clip(idx, 1, A + 1)
        left_q = np.take_along_axis(qf, idx - 1, axis=1)
        right_q = np.take_along_axis(qf, idx, axis=1)
        left_a = afull[idx - 1]
        right_a = afull[idx]
        denom = right_q - left_q
        denom = np.where(denom <= 0, 1.0, denom)
        frac = (boundaries[None, :] - left_q) / denom
        frac = np.clip(frac, 0.0, 1.0)
        cdf[s:e] = left_a + frac * (right_a - left_a)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf, axis=1)
    pdf = np.diff(cdf, axis=1).astype('float32')
    pdf = np.clip(pdf, 1e-9, None)
    pdf = pdf / pdf.sum(axis=1, keepdims=True)
    return pdf


def simulate(bids, payprices, click, V_arr, total_budget):
    """Process auctions in order. Stop bidding when cumulative spend would exceed budget.

    bids: optimal bid per auction (already chosen by the model, not constrained).
    The simulator skips an auction if it cannot afford the bid in case of win.
    Returns metrics dict.
    """
    n = len(bids)
    spent = 0.0
    won = 0
    profit = 0.0
    clicks_won = 0
    for i in range(n):
        b = bids[i]
        if b <= 0:
            continue
        # would-cost-on-win = b (first-price), only commit if we still afford
        if spent + b > total_budget:
            continue
        if b >= payprices[i]:
            spent += b
            won += 1
            profit += V_arr[i] - b
            if click is not None and click[i] > 0:
                clicks_won += 1
    return {
        'won': int(won),
        'spent': float(spent),
        'profit': float(profit),
        'clicks_won': int(clicks_won),
        'mean_cost': float(spent / max(1, won)),
    }


def total_test_cost(payprices):
    """Reference: sum of payprices = cost to win every auction at oracle price."""
    return float(payprices.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True,
                        help='preds_test.pt with bins probs OR mdn pi/mu/sigma OR npz with quantiles')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='exports/budget')
    parser.add_argument('--V', type=float, default=150.0)
    parser.add_argument('--use_bidding_V', action='store_true')
    parser.add_argument('--num_bins', type=int, default=301)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print('loading preds', args.preds)

    if args.preds.endswith('.npz'):
        npz = np.load(args.preds, allow_pickle=True)
        probs = quantiles_to_bins_np(npz['test_q'], npz['alphas'], args.num_bins)
        pp = npz['y_test'].astype('float32')
        bid = npz['bid_test'].astype('float32')
        adv = npz['adv_test'].astype(str)
        click = None
        # try to load click from extras file
        ex = os.path.join(os.path.dirname(args.preds), 'preds_extras.npz')
        if os.path.exists(ex):
            ex_npz = np.load(ex, allow_pickle=True)
            click = ex_npz['click_test']
    else:
        d = torch.load(args.preds, map_location='cpu', weights_only=False)
        if 'probs' in d:
            probs = d['probs'].numpy()
            if probs.shape[1] != args.num_bins:
                if probs.shape[1] > args.num_bins:
                    probs = probs[:, :args.num_bins]
                else:
                    pad = np.zeros((probs.shape[0], args.num_bins - probs.shape[1]), dtype=probs.dtype)
                    probs = np.concatenate([probs, pad], axis=1)
                probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            probs = mdn_to_bins_probs_np(d['pi_logits'], d['mu'], d['sigma'], args.num_bins,
                                         device='cuda' if torch.cuda.is_available() else 'cpu')
        pp = d['pp'].numpy()
        bid = d['bid'].numpy()
        adv = d['adv'].astype(str) if 'adv' in d else None
        click = d.get('click', None)
        if torch.is_tensor(click):
            click = click.numpy()

    if args.use_bidding_V:
        V_arr = bid.astype('float32')
    else:
        V_arr = np.full(len(pp), args.V, dtype='float32')

    print('shape probs', probs.shape, 'pp', pp.shape)

    # compute optimal bid per auction (unconstrained), then filter by budget
    bids = np.zeros(len(pp), dtype='float32')
    chunk = 50000
    for s in range(0, len(pp), chunk):
        e = min(len(pp), s + chunk)
        bids[s:e] = grid_optimize_from_probs(probs[s:e], V_arr[s:e], args.num_bins)

    total_cost = total_test_cost(pp)
    print('total test oracle cost (sum payprice):', round(total_cost, 0))

    fractions = [1.0/32, 1.0/8, 1.0/2, 1.0]
    res = {'name': args.name, 'V_mode': 'bidding' if args.use_bidding_V else f'fixed{args.V}',
           'unconstrained_bid_count': int((bids > 0).sum()),
           'total_oracle_cost': total_cost, 'budgets': {}}
    for frac in fractions:
        budget = frac * total_cost
        m = simulate(bids, pp, click, V_arr, budget)
        m['budget_frac'] = frac
        m['budget'] = budget
        res['budgets'][f'{frac:.4f}'] = m
        print(f'frac={frac:.4f} budget={budget:.0f} won={m["won"]} spent={m["spent"]:.0f} profit={m["profit"]:.0f} clicks={m["clicks_won"]}')

    out_path = os.path.join(args.out_dir, f'{args.name}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(res, f)
    print('saved', out_path)


if __name__ == '__main__':
    main()
