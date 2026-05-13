import os
import math
import time
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


def load_features(processed_dir, split):
    df = pd.read_parquet(os.path.join(processed_dir, f'{split}.parquet'))
    feats = []
    for c in CAT_FEATS:
        feats.append(c + '_idx')
    feats += CONT_FEATS
    X = df[feats].copy()
    # cast cat cols to int32 for LightGBM categorical handling
    for c in CAT_FEATS:
        X[c + '_idx'] = X[c + '_idx'].astype('int32')
    y = df['payprice'].values.astype('float32')
    bid = df['bidding_price'].values.astype('float32')
    adv = df['advertiser_id'].values
    extras = {'bid': bid, 'adv': adv}
    if 'click' in df.columns:
        extras['click'] = df['click'].values.astype('int32')
        extras['conversion'] = df['conversion'].values.astype('int32')
    return X, y, extras, feats


def fit_quantile(X_tr, y_tr, X_val, y_val, alpha, params):
    p = dict(params)
    p['objective'] = 'quantile'
    p['alpha'] = alpha
    cats = []
    for c in CAT_FEATS:
        cats.append(c + '_idx')
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, reference=dtr, free_raw_data=False)
    bst = lgb.train(
        p, dtr, valid_sets=[dval], num_boost_round=p.get('num_boost_round', 200),
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )
    return bst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='exports/lgbm_quant')
    parser.add_argument('--alphas', type=str,
                        default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95')
    parser.add_argument('--num_leaves', type=int, default=127)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_boost_round', type=int, default=300)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_fraction', type=float, default=0.9)
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--also_regression', action='store_true',
                        help='Also fit one MAE regression model and save MAE/RMSE on test.')
    args = parser.parse_args()

    cfg = load_config()
    pdir = cfg['data']['processed_dir']
    os.makedirs(args.out, exist_ok=True)

    print('loading data')
    t0 = time.time()
    X_tr, y_tr, etr, feats = load_features(pdir, 'train')
    X_val, y_val, eva, _ = load_features(pdir, 'val')
    X_te, y_te, ete, _ = load_features(pdir, 'test')
    print('load secs:', round(time.time()-t0,1), 'train:', len(y_tr), 'val:', len(y_val), 'test:', len(y_te))

    base_params = {
        'metric': 'quantile',
        'verbose': -1,
        'learning_rate': args.lr,
        'num_leaves': args.num_leaves,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': 1,
        'min_data_in_leaf': 200,
        'max_bin': 255,
        'num_threads': args.num_threads,
        'num_boost_round': args.num_boost_round,
    }

    alphas = []
    for s in args.alphas.split(','):
        alphas.append(float(s.strip()))

    quants_val = np.zeros((len(y_val), len(alphas)), dtype='float32')
    quants_te = np.zeros((len(y_te), len(alphas)), dtype='float32')

    for ai, a in enumerate(alphas):
        save_path = os.path.join(args.out, f'q{int(round(a*100)):02d}.txt')
        if os.path.exists(save_path):
            print('reusing saved alpha=', a, save_path)
            bst = lgb.Booster(model_file=save_path)
        else:
            print('training quantile alpha=', a)
            t0 = time.time()
            bst = fit_quantile(X_tr, y_tr, X_val, y_val, a, base_params)
            print('alpha=', a, 'fit secs:', round(time.time() - t0, 1))
            bst.save_model(save_path)
        quants_val[:, ai] = bst.predict(X_val)
        quants_te[:, ai] = bst.predict(X_te)
        del bst

    # enforce monotone (sort each row's predictions ascending)
    quants_val = np.sort(quants_val, axis=1)
    quants_te = np.sort(quants_te, axis=1)

    np.savez(os.path.join(args.out, 'preds.npz'),
             alphas=np.asarray(alphas, dtype='float32'),
             val_q=quants_val, test_q=quants_te,
             y_val=y_val, y_test=y_te,
             bid_test=ete['bid'], bid_val=eva['bid'],
             adv_test=ete['adv'].astype(str),
             feats=np.asarray(feats))
    if 'click' in ete:
        np.savez(os.path.join(args.out, 'preds_extras.npz'),
                 click_test=ete['click'], conversion_test=ete['conversion'])

    if args.also_regression:
        print('training point-estimate regression (MAE)')
        cats = []
        for c in CAT_FEATS:
            cats.append(c + '_idx')
        rp = dict(base_params)
        rp['objective'] = 'regression_l1'
        rp['metric'] = 'l1'
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, reference=dtr, free_raw_data=False)
        t0 = time.time()
        bst = lgb.train(rp, dtr, valid_sets=[dval], num_boost_round=rp.get('num_boost_round', 300),
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(100)])
        print('regression fit secs:', round(time.time() - t0, 1))
        bst.save_model(os.path.join(args.out, 'reg_l1.txt'))
        pred_test = bst.predict(X_te)
        mae = float(np.mean(np.abs(pred_test - y_te)))
        rmse = float(np.sqrt(np.mean((pred_test - y_te) ** 2)))
        print('point regression: test MAE=', round(mae, 3), 'RMSE=', round(rmse, 3))
        np.savez(os.path.join(args.out, 'reg_preds.npz'),
                 pred_test=pred_test.astype('float32'), y_test=y_te,
                 bid_test=ete['bid'], adv_test=ete['adv'].astype(str), mae=mae, rmse=rmse)

    print('saved preds to', args.out)
    print('done.')


if __name__ == '__main__':
    main()
