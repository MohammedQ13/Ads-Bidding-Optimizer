"""Budget eval for an ENSEMBLE of model preds.

Streams chunks: for each chunk compute each member's bin probs (MDN integrate, bins direct,
LightGBM piecewise inverse CDF), weighted-average, run grid bid optimizer, simulate spend.
"""

import os
import math
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F


def mdn_chunk_probs(pi_logits, mu, sigma, log_upper, log_lower, device):
    pi = F.softmax(pi_logits.to(device), dim=-1)
    mu_d = mu.to(device)
    sg_d = sigma.to(device)
    z_up = (log_upper.view(1, -1, 1) - mu_d.unsqueeze(1)) / (sg_d.unsqueeze(1) * math.sqrt(2.0))
    z_lo = (log_lower.view(1, -1, 1) - mu_d.unsqueeze(1)) / (sg_d.unsqueeze(1) * math.sqrt(2.0))
    cu = (pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_up))).sum(dim=-1)
    cl = (pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_lo))).sum(dim=-1)
    p = (cu - cl).clamp(min=1e-10)
    p = p / p.sum(dim=1, keepdim=True)
    return p.cpu().numpy()


def quantile_chunk_probs(quants, alphas, num_bins):
    boundaries = np.arange(num_bins + 1, dtype='float32') - 0.5
    boundaries[0] = 0.0
    afull = np.concatenate([[0.0], alphas, [1.0]]).astype('float32')
    A = quants.shape[1]
    qfull = np.concatenate([np.full((quants.shape[0], 1), -1e9, dtype='float32'),
                            quants.astype('float32'),
                            np.full((quants.shape[0], 1), 1e9, dtype='float32')], axis=1)
    idx = (qfull[:, :, None] <= boundaries[None, None, :]).sum(axis=1)
    idx = np.clip(idx, 1, A + 1)
    left_q = np.take_along_axis(qfull, idx - 1, axis=1)
    right_q = np.take_along_axis(qfull, idx, axis=1)
    left_a = afull[idx - 1]
    right_a = afull[idx]
    denom = right_q - left_q
    denom = np.where(denom <= 0, 1.0, denom)
    frac = (boundaries[None, :] - left_q) / denom
    frac = np.clip(frac, 0.0, 1.0)
    cdf = left_a + frac * (right_a - left_a)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf, axis=1)
    pdf = np.diff(cdf, axis=1).astype('float32')
    pdf = np.clip(pdf, 1e-9, None)
    pdf = pdf / pdf.sum(axis=1, keepdims=True)
    return pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdn_preds', nargs='*', default=[])
    parser.add_argument('--bins_preds', nargs='*', default=[])
    parser.add_argument('--lgbm_npz', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='exports/budget')
    parser.add_argument('--num_bins', type=int, default=301)
    parser.add_argument('--V', type=float, default=150.0)
    parser.add_argument('--use_bidding_V', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    members = []
    pp = bid = adv = click = None
    for path in args.mdn_preds:
        d = torch.load(path, map_location='cpu', weights_only=False)
        members.append(('mdn', d))
        if pp is None:
            pp = d['pp'].numpy()
            bid = d['bid'].numpy()
            click = d.get('click')
            if torch.is_tensor(click):
                click = click.numpy()
    for path in args.bins_preds:
        d = torch.load(path, map_location='cpu', weights_only=False)
        members.append(('bins', d))
        if pp is None:
            pp = d['pp'].numpy()
            bid = d['bid'].numpy()
            click = d.get('click')
            if torch.is_tensor(click):
                click = click.numpy()
    if args.lgbm_npz is not None:
        npz = np.load(args.lgbm_npz, allow_pickle=True)
        members.append(('lgbm', {'quants': npz['test_q'], 'alphas': npz['alphas']}))
        if pp is None:
            pp = npz['y_test'].astype('float32')
            bid = npz['bid_test'].astype('float32')
        # for click
        ex = os.path.join(os.path.dirname(args.lgbm_npz), 'preds_extras.npz')
        if click is None and os.path.exists(ex):
            ex_npz = np.load(ex, allow_pickle=True)
            click = ex_npz['click_test']

    M = len(members)
    if args.weights is None:
        ws = [1.0 / M] * M
    else:
        parts = []
        for w in args.weights.split(','):
            parts.append(float(w))
        s = sum(parts)
        ws = []
        for w in parts:
            ws.append(w / s)
    member_names = []
    for k, _ in members:
        member_names.append(k)
    print('members:', member_names, 'weights:', ws)

    upper = torch.arange(args.num_bins, dtype=torch.float32) + 0.5
    lower = torch.cat([torch.zeros(1), torch.arange(args.num_bins - 1, dtype=torch.float32) + 0.5])
    log_upper = torch.log(upper.clamp(min=1e-3)).to(device)
    log_lower = torch.log(lower.clamp(min=1e-3)).to(device)
    log_lower[0] = -1e9

    n = len(pp)
    bins_arr = np.arange(args.num_bins, dtype='float32')
    if args.use_bidding_V:
        V_arr = bid.astype('float32')
    else:
        V_arr = np.full(n, args.V, dtype='float32')

    # compute bids in chunks
    bids = np.zeros(n, dtype='float32')
    chunk = 10000
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        avg = None
        for mi, (kind, d) in enumerate(members):
            if kind == 'mdn':
                p = mdn_chunk_probs(d['pi_logits'][s:e], d['mu'][s:e], d['sigma'][s:e],
                                    log_upper, log_lower, device)
            elif kind == 'bins':
                p_t = d['probs'][s:e]
                if p_t.shape[1] >= args.num_bins:
                    p = p_t[:, :args.num_bins].numpy()
                else:
                    p_full = np.zeros((p_t.shape[0], args.num_bins), dtype='float32')
                    p_full[:, :p_t.shape[1]] = p_t.numpy()
                    p = p_full
                p = p / p.sum(axis=1, keepdims=True)
            else:
                p = quantile_chunk_probs(d['quants'][s:e], d['alphas'], args.num_bins)
            if avg is None:
                avg = ws[mi] * p
            else:
                avg = avg + ws[mi] * p
        avg = avg / avg.sum(axis=1, keepdims=True)
        cdf = np.cumsum(avg, axis=1)
        Vc = V_arr[s:e]
        profit = (Vc[:, None] - bins_arr[None, :]) * cdf
        profit = np.where(bins_arr[None, :] > Vc[:, None], 0.0, profit)
        best_idx = profit.argmax(axis=1)
        best_b = bins_arr[best_idx]
        best_p = profit[np.arange(profit.shape[0]), best_idx]
        no_bid = best_p < 0
        best_b = np.where(no_bid, 0.0, best_b)
        bids[s:e] = best_b

    total_cost = float(pp.sum())
    print('oracle cost', round(total_cost, 0))
    res = {'name': args.name, 'V_mode': 'bidding' if args.use_bidding_V else f'fixed{args.V}',
           'unconstrained_bid_count': int((bids > 0).sum()),
           'total_oracle_cost': total_cost, 'budgets': {}}
    fractions = [1.0/32, 1.0/8, 1.0/2, 1.0]
    for frac in fractions:
        budget = frac * total_cost
        spent = 0.0
        won = 0
        profit = 0.0
        clicks_won = 0
        for i in range(n):
            b = bids[i]
            if b <= 0:
                continue
            if spent + b > budget:
                continue
            if b >= pp[i]:
                spent += b
                won += 1
                profit += V_arr[i] - b
                if click is not None and click[i] > 0:
                    clicks_won += 1
        m = {'won': int(won), 'spent': float(spent), 'profit': float(profit),
             'clicks_won': int(clicks_won), 'mean_cost': float(spent / max(1, won)),
             'budget_frac': frac, 'budget': budget}
        res['budgets'][f'{frac:.4f}'] = m
        print(f'frac={frac:.4f} won={m["won"]} prof={m["profit"]:.0f} clk={m["clicks_won"]}')

    out = os.path.join(args.out_dir, f'{args.name}.pkl')
    with open(out, 'wb') as f:
        pickle.dump(res, f)
    print('saved', out)


if __name__ == '__main__':
    main()
