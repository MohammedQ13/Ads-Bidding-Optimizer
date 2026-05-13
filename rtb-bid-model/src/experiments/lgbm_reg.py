"""Fit a single LightGBM regression (MAE) and report MAE/RMSE on test."""
import os
import time
import argparse
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
    parser.add_argument('--out', type=str, default='exports/lgbm_quant/reg_l1.txt')
    parser.add_argument('--num_boost_round', type=int, default=200)
    parser.add_argument('--num_leaves', type=int, default=95)
    parser.add_argument('--lr', type=float, default=0.12)
    args = parser.parse_args()

    cfg = load_config()
    pdir = cfg['data']['processed_dir']
    print('loading data', flush=True)
    feats = []
    for c in CAT_FEATS:
        feats.append(c + '_idx')
    feats += CONT_FEATS
    df_tr = pd.read_parquet(os.path.join(pdir, 'train.parquet'))
    df_va = pd.read_parquet(os.path.join(pdir, 'val.parquet'))
    df_te = pd.read_parquet(os.path.join(pdir, 'test.parquet'))

    X_tr = df_tr[feats].copy()
    X_va = df_va[feats].copy()
    X_te = df_te[feats].copy()
    for c in CAT_FEATS:
        X_tr[c + '_idx'] = X_tr[c + '_idx'].astype('int32')
        X_va[c + '_idx'] = X_va[c + '_idx'].astype('int32')
        X_te[c + '_idx'] = X_te[c + '_idx'].astype('int32')
    y_tr = df_tr['payprice'].values.astype('float32')
    y_va = df_va['payprice'].values.astype('float32')
    y_te = df_te['payprice'].values.astype('float32')
    bid_te = df_te['bidding_price'].values.astype('float32')

    cats = []
    for c in CAT_FEATS:
        cats.append(c + '_idx')

    params = {
        'objective': 'regression_l1', 'metric': 'l1',
        'learning_rate': args.lr, 'num_leaves': args.num_leaves,
        'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1,
        'min_data_in_leaf': 200, 'max_bin': 255, 'verbose': -1, 'num_threads': 8,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cats, reference=dtr, free_raw_data=False)
    t = time.time()
    bst = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=args.num_boost_round,
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
    print('regression secs:', round(time.time() - t, 1))
    bst.save_model(args.out)

    pred_te = bst.predict(X_te)
    mae = float(np.mean(np.abs(pred_te - y_te)))
    rmse = float(np.sqrt(np.mean((pred_te - y_te) ** 2)))

    # treat point estimate as the bid (linear bid b=pred). compute regret V=150 and V=bid.
    V150 = 150.0
    bid150 = np.clip(pred_te, 0, V150)
    win = (bid150 >= y_te).astype('float32')
    realized = win * (V150 - bid150)
    perfect = ((V150 > y_te).astype('float32')) * (V150 - y_te)
    reg150 = float((perfect - realized).mean())

    bidvb = np.clip(pred_te, 0, bid_te)
    win_b = (bidvb >= y_te).astype('float32')
    real_b = win_b * (bid_te - bidvb)
    perf_b = ((bid_te > y_te).astype('float32')) * (bid_te - y_te)
    reg_vbid = float((perf_b - real_b).mean())

    print(f'point regression: test MAE={mae:.3f} RMSE={rmse:.3f}')
    print(f'using prediction directly as bid: regret V=150={reg150:.3f} V=bid={reg_vbid:.3f}')

    np.savez('exports/lgbm_quant/reg_preds.npz',
             pred_test=pred_te.astype('float32'), y_test=y_te,
             bid_test=bid_te,
             adv_test=df_te['advertiser_id'].values.astype(str),
             mae=mae, rmse=rmse, regret_v150=reg150, regret_vbid=reg_vbid)
    print('saved reg_preds.npz')


if __name__ == '__main__':
    main()
