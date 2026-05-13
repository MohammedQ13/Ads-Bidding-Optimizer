"""Eval LightGBM quantile preds without needing the full sweep saved.

Loads whatever q*.txt files exist, predicts on test set, computes the same metrics
as eval_preds.py (NLL via interp pdf, KS, regret grid, per-advertiser).
"""

import os
import re
import argparse
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import load_config


CAT_FEATS = [
    'region', 'city', 'domain', 'ad_exchange',
    'slot_width', 'slot_height', 'slot_visibility', 'slot_format', 'advertiser_id',
]
CONT_FEATS = ['has_floor_price', 'log_floor_price', 'slot_area', 'tag_count',
              'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='exports/lgbm_quant')
    parser.add_argument('--name', type=str, default='lgbm_quant')
    parser.add_argument('--num_bins', type=int, default=301)
    parser.add_argument('--out_dir', type=str, default='exports/preds_eval')
    parser.add_argument('--save_npz', type=str, default='exports/lgbm_quant/preds.npz')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config()
    pdir = cfg['data']['processed_dir']

    # collect saved alphas
    alphas = []
    paths = []
    for fn in sorted(os.listdir(args.model_dir)):
        m = re.match(r'q(\d+)\.txt$', fn)
        if not m:
            continue
        a = int(m.group(1)) / 100.0
        alphas.append(a)
        paths.append(os.path.join(args.model_dir, fn))
    if not alphas:
        print('no q*.txt models found in', args.model_dir)
        return
    print('using alphas:', alphas)

    # build feature df
    df = pd.read_parquet(os.path.join(pdir, 'test.parquet'))
    feats = []
    for c in CAT_FEATS:
        feats.append(c + '_idx')
    feats += CONT_FEATS
    X = df[feats].copy()
    for c in CAT_FEATS:
        X[c + '_idx'] = X[c + '_idx'].astype('int32')
    y = df['payprice'].values.astype('float32')
    bid = df['bidding_price'].values.astype('float32')
    adv = df['advertiser_id'].values.astype(str)
    click = df['click'].values.astype('int32') if 'click' in df.columns else None

    quants = np.zeros((len(y), len(alphas)), dtype='float32')
    import time
    for i, p in enumerate(paths):
        t = time.time()
        bst = lgb.Booster(model_file=p)
        quants[:, i] = bst.predict(X)
        print(f'predicted alpha={alphas[i]:.3f} in {time.time()-t:.1f}s', flush=True)
    quants = np.sort(quants, axis=1)

    # build piecewise-linear CDF using torch on GPU; broadcast over a larger tensor with searchsorted-like logic
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    boundaries = torch.arange(args.num_bins + 1, dtype=torch.float32, device=device) - 0.5
    boundaries[0] = 0.0
    afull = torch.tensor([0.0] + list(alphas) + [1.0], dtype=torch.float32, device=device)
    A = quants.shape[1]
    pad = torch.full((1,), -1e9, device=device)
    pad2 = torch.full((1,), 1e9, device=device)
    cdf_full = np.zeros((len(y), len(boundaries)), dtype='float32')
    chunk = 30000
    print('building CDF in chunks of', chunk, flush=True)
    t0 = time.time()
    for s in range(0, len(y), chunk):
        e = min(len(y), s + chunk)
        qf_t = torch.from_numpy(quants[s:e]).to(device)
        nrow = qf_t.size(0)
        l = pad.expand(nrow, 1)
        r = pad2.expand(nrow, 1)
        qfull = torch.cat([l, qf_t, r], dim=1)  # (n, A+2)
        # for each boundary, idx = number of qfull <= boundary (right-side count)
        idx = (qfull.unsqueeze(2) <= boundaries.view(1, 1, -1)).sum(dim=1)  # (n, B)
        idx = idx.clamp(min=1, max=A + 1)
        left_q = qfull.gather(1, idx - 1)
        right_q = qfull.gather(1, idx)
        left_a = afull[idx - 1]
        right_a = afull[idx]
        denom = right_q - left_q
        denom = torch.where(denom <= 0, torch.ones_like(denom), denom)
        frac = (boundaries.view(1, -1) - left_q) / denom
        frac = frac.clamp(0.0, 1.0)
        cdf_chunk = left_a + frac * (right_a - left_a)
        cdf_full[s:e] = cdf_chunk.cpu().numpy()
        if s % (chunk * 10) == 0:
            print(f'chunk {s}/{len(y)} elapsed {time.time()-t0:.1f}s', flush=True)
    cdf = np.clip(cdf_full, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf, axis=1)
    pdf = np.diff(cdf, axis=1).astype('float32')
    pdf = np.clip(pdf, 1e-9, None)
    pdf = pdf / pdf.sum(axis=1, keepdims=True)

    # cdf at int prices: cdf[:, k+1] is P(price <= k+0.5) ~= P(price <= k)
    cdf_int = cdf[:, 1:]  # (N, num_bins)
    pp_int = np.clip(np.round(y).astype(np.int64), 0, args.num_bins - 1)
    log_p = np.log(pdf[np.arange(len(y)), pp_int].clip(min=1e-30))
    anlp = float(-log_p.mean())
    pit = cdf_int[np.arange(len(y)), pp_int]
    n = len(pit)
    s_ = np.sort(pit)
    cdf_emp = np.arange(1, n + 1) / n
    d_plus = (cdf_emp - s_).max()
    d_minus = (s_ - (np.arange(0, n) / n)).max()
    ks = float(max(d_plus, d_minus))
    pcts = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    cov = {}
    for p in pcts:
        cov[p] = float((pit <= p / 100.0).mean())

    bins_arr = np.arange(args.num_bins, dtype='float32')
    # regret V=150
    V_fixed = 150.0
    profit150 = (V_fixed - bins_arr[None, :]) * cdf_int
    profit150 = np.where(bins_arr[None, :] > V_fixed, 0.0, profit150)
    best_idx = profit150.argmax(axis=1)
    best_b = bins_arr[best_idx]
    win = (best_b >= y).astype('float32')
    realized = win * (V_fixed - best_b)
    perfect = ((V_fixed > y).astype('float32')) * (V_fixed - y)
    reg150 = float((perfect - realized).mean())
    # regret V=bid
    Vb = bid
    profit_b = (Vb[:, None] - bins_arr[None, :]) * cdf_int
    profit_b = np.where(bins_arr[None, :] > Vb[:, None], 0.0, profit_b)
    best_idx_b = profit_b.argmax(axis=1)
    best_b_b = bins_arr[best_idx_b]
    winb = (best_b_b >= y).astype('float32')
    realb = winb * (Vb - best_b_b)
    perfb = ((Vb > y).astype('float32')) * (Vb - y)
    reg_vbid = float((perfb - realb).mean())

    print(f'{args.name}: ANLP={anlp:.4f} KS={ks:.4f} regret_v150={reg150:.3f} regret_vbid={reg_vbid:.3f}')

    per_adv = {}
    advs_set = sorted(set(adv.tolist()))
    for a in advs_set:
        mask = adv == a
        if mask.sum() == 0:
            continue
        per_adv[a] = float(((perfect - realized)[mask]).mean())
    for a in sorted(per_adv.keys()):
        print(f'{a}: n={int((adv == a).sum())} reg={per_adv[a]:.3f}')

    res = {
        'name': args.name, 'alphas': alphas, 'anlp': anlp, 'ks': ks, 'coverage': cov,
        'regret_v150_grid': reg150, 'regret_vbid_grid': reg_vbid,
        'per_adv_regret_v150': per_adv,
    }
    with open(os.path.join(args.out_dir, f'{args.name}.pkl'), 'wb') as f:
        pickle.dump(res, f)
    print('saved', os.path.join(args.out_dir, f'{args.name}.pkl'))

    if args.save_npz:
        np.savez(args.save_npz,
                 alphas=np.asarray(alphas, dtype='float32'),
                 test_q=quants,
                 y_test=y,
                 bid_test=bid,
                 adv_test=adv)
        print('saved', args.save_npz)


if __name__ == '__main__':
    main()
