import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd

from config import load_config


def quantile_to_cdf_pdf(quants, alphas, prices_grid):
    """Convert quantile predictions q[a] for each sample to CDF/PDF on a price grid.
    quants: (N, A) sorted ascending. alphas: (A,). prices_grid: (G,).
    For each sample: linear interpolation in (alpha, q) inversely yields CDF(p) = alpha s.t. q = p.
    Returns cdf: (N, G), pdf: (N, G) where pdf is forward-difference of CDF.
    """
    N, A = quants.shape
    G = len(prices_grid)
    cdf = np.zeros((N, G), dtype=np.float32)
    # for vectorized inverse interpolation, sort within row already, then for each grid point find left index
    # use searchsorted per-row via concat trick
    q_pad_left = np.zeros(N, dtype=np.float32) - 1e9
    q_pad_right = np.zeros(N, dtype=np.float32) + 1e9
    qfull = np.concatenate([q_pad_left[:, None], quants, q_pad_right[:, None]], axis=1)
    afull = np.concatenate([[0.0], alphas, [1.0]])
    for gi in range(G):
        p = prices_grid[gi]
        # find index where p falls in qfull row-wise
        # idx in [0, A+1]: q[idx-1] <= p < q[idx]
        # binary search per row
        # vectorized: use searchsorted along axis=1
        idx = np.empty(N, dtype=np.int64)
        for r in range(N):
            idx[r] = np.searchsorted(qfull[r], p, side='right')
        idx = np.clip(idx, 1, A + 1)
        left_q = qfull[np.arange(N), idx - 1]
        right_q = qfull[np.arange(N), idx]
        left_a = afull[idx - 1]
        right_a = afull[idx]
        denom = (right_q - left_q)
        denom[denom <= 0] = 1.0
        frac = (p - left_q) / denom
        frac = np.clip(frac, 0.0, 1.0)
        cdf[:, gi] = left_a + frac * (right_a - left_a)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf, axis=1)
    pdf = np.diff(cdf, prepend=0, axis=1)
    pdf = np.clip(pdf, 1e-9, None)
    return cdf, pdf


def quantile_to_cdf_pdf_vec(quants, alphas, prices_grid):
    """Fully-vectorized version: piecewise linear inverse CDF using cumulative interp."""
    N, A = quants.shape
    G = len(prices_grid)
    afull = np.concatenate([[0.0], alphas, [1.0]]).astype('float32')
    q_pad_left = np.full((N, 1), -1e9, dtype='float32')
    q_pad_right = np.full((N, 1), 1e9, dtype='float32')
    qfull = np.concatenate([q_pad_left, quants.astype('float32'), q_pad_right], axis=1)  # (N, A+2)
    # for each price, find index per row using searchsorted with axis=1 trick
    # convert price to array
    prices = np.asarray(prices_grid, dtype='float32')  # (G,)
    # compute idx (N, G): for each row, idx[g] = searchsorted(qfull[r], prices[g], side='right')
    # Use broadcasting: qfull[:, :, None] vs prices[None, None, :] -> (N, A+2, G); too big for full at once if N large
    # Do it in row chunks
    cdf = np.zeros((N, G), dtype='float32')
    chunk = 50000
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        qf = qfull[s:e]  # (n, A+2)
        # for each price find idx per row
        # we exploit qfull is monotone -> use ascending search with broadcasting
        # idx shape (n, G)
        # do: count of qf <= price - 1
        # actually, side='right' equivalent: idx = sum(qf <= p)
        idx = (qf[:, :, None] <= prices[None, None, :]).sum(axis=1)  # (n, G)
        # clamp
        idx = np.clip(idx, 1, A + 1)
        left_q = np.take_along_axis(qf, (idx - 1), axis=1)  # (n, G)
        right_q = np.take_along_axis(qf, idx, axis=1)
        left_a = afull[idx - 1]
        right_a = afull[idx]
        denom = (right_q - left_q)
        denom = np.where(denom <= 0, 1.0, denom)
        frac = (prices[None, :] - left_q) / denom
        frac = np.clip(frac, 0.0, 1.0)
        cdf[s:e] = left_a + frac * (right_a - left_a)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf, axis=1)
    pdf = np.diff(cdf, prepend=0, axis=1).astype('float32')
    pdf = np.clip(pdf, 1e-9, None)
    return cdf, pdf


def evaluate_lgbm(quants, alphas, y, bid, adv, name, out_dir, num_bins=300, V_fixed=150.0):
    print(f'evaluating LightGBM quantile model ({name})')
    prices_grid = np.arange(num_bins, dtype='float32')  # 0..num_bins-1
    t0 = time.time()
    cdf, pdf = quantile_to_cdf_pdf_vec(quants, alphas, prices_grid)
    print('cdf/pdf computed in', round(time.time()-t0,1), 's', cdf.shape)

    # NLL on integer prices: use pdf at price == int(round(payprice))
    yi = np.clip(np.round(y).astype(np.int64), 0, num_bins - 1)
    log_p = np.log(pdf[np.arange(len(y)), yi].clip(min=1e-30))
    anlp = float(-log_p.mean())
    print(f'ANLP (linear price): {anlp:.4f}')

    # PIT
    pit = cdf[np.arange(len(y)), yi]
    pcts = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    cov = {}
    for p in pcts:
        cov[p] = float((pit <= p/100.0).mean())
    n = len(pit)
    s = np.sort(pit)
    cdf_emp = np.arange(1, n + 1) / n
    d_plus = (cdf_emp - s).max()
    d_minus = (s - (np.arange(0, n) / n)).max()
    ks = float(max(d_plus, d_minus))
    print(f'KS: {ks:.4f}')
    print('coverage:')
    for p in sorted(cov.keys()):
        print(f'pct {p}: {cov[p]:.3f}')

    # bid optimization on grid
    bins_arr = prices_grid
    print('regret V=150 (grid)')
    profit150 = (V_fixed - bins_arr[None, :]) * cdf
    profit150 = np.where(bins_arr[None, :] > V_fixed, 0.0, profit150)
    best_idx_150 = profit150.argmax(axis=1)
    best_b_150 = bins_arr[best_idx_150]
    win = (best_b_150 >= y).astype('float32')
    realized = win * (V_fixed - best_b_150)
    perfect = ((V_fixed > y).astype('float32')) * (V_fixed - y)
    reg150 = float((perfect - realized).mean())
    print(f'mean regret: {reg150:.3f}')

    print('regret V=bidding (grid)')
    V_b = bid.astype('float32')
    profit_b = (V_b[:, None] - bins_arr[None, :]) * cdf
    profit_b = np.where(bins_arr[None, :] > V_b[:, None], 0.0, profit_b)
    best_idx_b = profit_b.argmax(axis=1)
    best_b_b = bins_arr[best_idx_b]
    winb = (best_b_b >= y).astype('float32')
    realb = winb * (V_b - best_b_b)
    perfb = ((V_b > y).astype('float32')) * (V_b - y)
    reg_b = float((perfb - realb).mean())
    print(f'mean regret: {reg_b:.3f}')

    # per-advertiser
    advs = sorted(set(adv.tolist()))
    per_adv = {}
    for a in advs:
        m = adv == a
        if m.sum() == 0:
            continue
        per_adv[a] = float((perfect[m] - realized[m]).mean())
    print('per-advertiser regret (V=150 grid):')
    for a in sorted(per_adv.keys()):
        print(f'{a}: n={int((adv==a).sum())} regret={per_adv[a]:.3f}')

    res = {
        'anlp': anlp,
        'ks': ks,
        'coverage': cov,
        'regret_v150_grid': reg150,
        'regret_vbid_grid': reg_b,
        'per_adv_regret_v150': per_adv,
    }
    with open(os.path.join(out_dir, 'eval_test.pkl'), 'wb') as f:
        pickle.dump(res, f)
    print('saved eval to', out_dir)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='exports/lgbm_quant')
    parser.add_argument('--name', type=str, default='lgbm_quant')
    args = parser.parse_args()
    npz = np.load(os.path.join(args.in_dir, 'preds.npz'), allow_pickle=True)
    quants = npz['test_q']
    alphas = npz['alphas']
    y = npz['y_test']
    bid = npz['bid_test']
    adv = npz['adv_test']
    evaluate_lgbm(quants, alphas, y, bid, adv, args.name, args.in_dir)


if __name__ == '__main__':
    main()
