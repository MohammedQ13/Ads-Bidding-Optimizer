"""Evaluate a bins-v2 ckpt that has custom edges. Reuses the integer-bid grid in
[0, 300] by remapping the model's variable-size bin probs onto a uniform 301-bin probs
array (probability mass per integer price), then reuses the standard regret formula.
"""
import os
import math
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from dataset import BidDataset, load_artifacts
from model import DiscreteBins
from bid_optimizer import grid_optimize_bins, regret


def remap_to_integer_probs(probs, edges, num_int_bins=301):
    """Map a (N, K) probability over variable-edge bins to a (N, num_int_bins) array
    over integer prices 0..num_int_bins-1, by linearly distributing each bin's mass
    proportionally to its overlap with each integer cell."""
    N, K = probs.shape
    # build a fixed transfer matrix W of shape (K, num_int_bins): W[k, j] = fraction of bin k
    # whose price interval [edges[k], edges[k+1]) overlaps integer cell [j-0.5, j+0.5).
    # so int_probs[:, j] = sum_k probs[:, k] * W[k, j].
    int_centers = np.arange(num_int_bins, dtype='float64')
    cell_lo = int_centers - 0.5
    cell_hi = int_centers + 0.5
    cell_lo[0] = 0.0  # first cell starts at 0
    edges = np.asarray(edges, dtype='float64')
    W = np.zeros((K, num_int_bins), dtype='float64')
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        bin_w = hi - lo
        # find overlapping cells
        j0 = int(max(0, np.floor(lo)))
        j1 = int(min(num_int_bins - 1, np.ceil(hi)))
        for j in range(j0, j1 + 1):
            ovl = max(0.0, min(hi, cell_hi[j]) - max(lo, cell_lo[j]))
            if ovl > 0:
                W[k, j] = ovl / bin_w
    # stamp out: each bin's mass distributed; rows sum to 1 already
    int_probs = probs.astype('float64') @ W
    int_probs = int_probs / int_probs.sum(axis=1, keepdims=True).clip(min=1e-30)
    return int_probs.astype('float32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--out_dir', type=str, default='exports/preds_eval')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config()
    art = load_artifacts(cfg['data']['processed_dir'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'test.parquet'), art, has_extras=True)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = ck['config']
    edges = np.asarray(mcfg['edges'], dtype='float32')
    num_bins = mcfg['num_bins']
    print('loaded ckpt', args.ckpt, 'num_bins', num_bins, 'edges range', edges.min(), edges.max())

    m = DiscreteBins(mcfg['vocab_sizes'], mcfg['emb_dims'], mcfg['tag_vocab_size'],
                     mcfg['tag_emb_dim'], mcfg['num_continuous'], mcfg['hidden'],
                     mcfg['dropout'], mcfg['num_bins']).to(device)
    m.load_state_dict(ck['full_state_dict'], strict=False)
    m.eval()

    # collect probs in chunks
    bs = 8192
    parts = []
    pp_parts = []
    bid_parts = []
    log_pp_parts = []
    with torch.no_grad():
        for batch in test_ds.iter_batches(bs, shuffle=False):
            cat = batch['cat'].to(device); cont = batch['cont'].to(device); tags = batch['tags'].to(device)
            logits = m(cat, cont, tags)
            probs = F.softmax(logits, dim=-1).cpu()
            parts.append(probs)
            pp_parts.append(batch['pp']); bid_parts.append(batch['bid']); log_pp_parts.append(batch['log_pp'])
    probs = torch.cat(parts, dim=0).numpy()
    pp = torch.cat(pp_parts, dim=0).numpy()
    bid = torch.cat(bid_parts, dim=0).numpy()
    log_pp = torch.cat(log_pp_parts, dim=0).numpy()
    adv = test_ds.adv.astype(str)

    print('computing remap')
    int_probs = remap_to_integer_probs(probs, edges, num_int_bins=301)

    # NLL: probability mass on integer bin = round(pp); use int_probs
    pp_int = np.clip(np.round(pp).astype(np.int64), 0, 300)
    log_p = np.log(int_probs[np.arange(len(pp)), pp_int].clip(min=1e-30))
    anlp = float(-log_p.mean())

    cdf = np.cumsum(int_probs, axis=1)
    pit = cdf[np.arange(len(pp)), pp_int]
    s_ = np.sort(pit)
    n = len(pit)
    cdf_emp = np.arange(1, n + 1) / n
    d_plus = (cdf_emp - s_).max()
    d_minus = (s_ - (np.arange(0, n) / n)).max()
    ks = float(max(d_plus, d_minus))

    bins_arr = np.arange(301, dtype='float32')
    V_fixed = 150.0
    profit150 = (V_fixed - bins_arr[None, :]) * cdf
    profit150 = np.where(bins_arr[None, :] > V_fixed, 0.0, profit150)
    best_idx = profit150.argmax(axis=1)
    best_b = bins_arr[best_idx]
    win = (best_b >= pp).astype('float32')
    realized = win * (V_fixed - best_b)
    perfect = ((V_fixed > pp).astype('float32')) * (V_fixed - pp)
    reg150 = float((perfect - realized).mean())

    Vb = bid.astype('float32')
    profit_b = (Vb[:, None] - bins_arr[None, :]) * cdf
    profit_b = np.where(bins_arr[None, :] > Vb[:, None], 0.0, profit_b)
    best_idx_b = profit_b.argmax(axis=1)
    best_b_b = bins_arr[best_idx_b]
    winb = (best_b_b >= pp).astype('float32')
    realb = winb * (Vb - best_b_b)
    perfb = ((Vb > pp).astype('float32')) * (Vb - pp)
    reg_vbid = float((perfb - realb).mean())

    print(f'{args.name}: ANLP={anlp:.4f} KS={ks:.4f} regret_v150={reg150:.3f} regret_vbid={reg_vbid:.3f}')

    per_adv = {}
    advs = sorted(set(adv.tolist()))
    for a in advs:
        m_ = adv == a
        if m_.sum() == 0:
            continue
        per_adv[a] = float((perfect - realized)[m_].mean())
    for a in sorted(per_adv.keys()):
        print(f'{a}: n={int((adv == a).sum())} reg={per_adv[a]:.3f}')

    res = {'name': args.name, 'anlp': anlp, 'ks': ks,
           'regret_v150_grid': reg150, 'regret_vbid_grid': reg_vbid,
           'per_adv_regret_v150': per_adv}
    with open(os.path.join(args.out_dir, f'{args.name}.pkl'), 'wb') as f:
        pickle.dump(res, f)

    if args.save_preds:
        save_path = os.path.join(os.path.dirname(args.ckpt), 'preds_test.pt')
        # save the remapped 301-bin probs so it can be ensembled with other bins models
        torch.save({
            'probs': torch.from_numpy(int_probs),
            'log_pp': torch.from_numpy(log_pp),
            'pp': torch.from_numpy(pp),
            'bid': torch.from_numpy(bid),
            'log_prob': torch.from_numpy(log_p.astype('float32')),
            'pit': torch.from_numpy(pit.astype('float32')),
            'adv': adv,
        }, save_path)
        print('saved', save_path)


if __name__ == '__main__':
    main()
