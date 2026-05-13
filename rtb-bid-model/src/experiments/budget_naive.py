"""Naive baseline: always bid the train-mean payprice.

For comparison purposes, simulate the same way as budget_eval.py but with a
single fixed bid b = train_mean_payprice for every auction.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

from config import load_config
from dataset import load_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='budget_naive')
    parser.add_argument('--V', type=float, default=150.0)
    parser.add_argument('--out_dir', type=str, default='exports/budget')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config()
    art = load_artifacts(cfg['data']['processed_dir'])
    pdir = cfg['data']['processed_dir']
    df_tr = pd.read_parquet(os.path.join(pdir, 'train.parquet'))
    mean_pp = float(df_tr['payprice'].mean())
    print('train mean payprice:', round(mean_pp, 2))

    df_te = pd.read_parquet(os.path.join(pdir, 'test.parquet'))
    pp = df_te['payprice'].values.astype('float32')
    bid_col = df_te['bidding_price'].values.astype('float32')
    click = df_te['click'].values.astype('int32') if 'click' in df_te.columns else None

    n = len(pp)
    V_arr = np.full(n, args.V, dtype='float32')
    b = mean_pp
    if b > args.V:
        print('warn: mean exceeds V, naive cant bid')
        b = args.V

    total_cost = float(pp.sum())
    print('oracle cost', round(total_cost, 0))

    # also compute regret (V=150 grid, V=bid grid)
    win = (b >= pp).astype('float32')
    realized = win * (V_arr - b)
    perfect = ((V_arr > pp).astype('float32')) * (V_arr - pp)
    reg_v150 = float((perfect - realized).mean())
    Vb = bid_col
    win_b = (b >= pp).astype('float32')
    real_b = win_b * (Vb - b)
    perf_b = ((Vb > pp).astype('float32')) * (Vb - pp)
    reg_vbid = float((perf_b - real_b).mean())
    print('naive regret V=150:', round(reg_v150, 3), 'V=bid:', round(reg_vbid, 3))

    res = {'name': args.name, 'naive_bid': b, 'regret_v150': reg_v150, 'regret_vbid': reg_vbid,
           'total_oracle_cost': total_cost, 'budgets': {}}
    fractions = [1.0/32, 1.0/8, 1.0/2, 1.0]
    for frac in fractions:
        budget = frac * total_cost
        spent = 0.0
        won = 0
        profit = 0.0
        clicks_won = 0
        i = 0
        while i < n:
            if spent + b > budget:
                break
            if b >= pp[i]:
                spent += b
                won += 1
                profit += V_arr[i] - b
                if click is not None and click[i] > 0:
                    clicks_won += 1
            i += 1
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
