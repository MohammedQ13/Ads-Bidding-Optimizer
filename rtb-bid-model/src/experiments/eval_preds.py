"""Fast eval using saved preds_test.pt: NLL/ANLP, KS, calibration, regret with grid AND Newton.

Avoids re-running the model — just loads the prediction tensors from disk.
"""

import os
import math
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from bid_optimizer import (
    grid_optimize_mdn, grid_optimize_bins, newton_optimize_mdn, newton_optimize_bins,
    regret,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='exports/preds_eval')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = torch.load(args.preds, map_location='cpu', weights_only=False)
    pp = d['pp']
    bid = d['bid']
    adv = d.get('adv')
    if adv is not None:
        adv = np.asarray(adv).astype(str)

    is_mdn = 'pi_logits' in d
    print('model_type:', 'mdn' if is_mdn else 'bins', 'rows:', len(pp))

    # NLL / ANLP / PIT
    if is_mdn:
        anlp = float(-(d['log_prob'] - d['log_pp']).mean())
        # PIT
        log_t = d['log_pp'].unsqueeze(1)
        z = (log_t - d['mu']) / (d['sigma'] * math.sqrt(2.0))
        pit_t = (F.softmax(d['pi_logits'], dim=-1) * 0.5 * (1.0 + torch.erf(z))).sum(dim=-1)
        pit = pit_t.numpy()
    else:
        anlp = float(-d['log_prob'].mean())
        pp_int = pp.long().clamp(0, d['probs'].size(1) - 1)
        cdf = torch.cumsum(d['probs'], dim=1)
        pit = cdf.gather(1, pp_int.unsqueeze(1)).squeeze(1).numpy()

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

    print(f'ANLP={anlp:.4f} KS={ks:.4f}')

    # regret variants
    bs_ = 16384
    def bids_grid(V_arr):
        out = []
        for i in range(0, n, bs_):
            j = min(n, i + bs_)
            V_b = V_arr[i:j].to(device)
            if is_mdn:
                bb, _, _, _ = grid_optimize_mdn(d['pi_logits'][i:j].to(device),
                                                d['mu'][i:j].to(device),
                                                d['sigma'][i:j].to(device), V_b)
            else:
                bb, _, _, _ = grid_optimize_bins(d['probs'][i:j].to(device), V_b)
            out.append(bb.cpu())
        return torch.cat(out, dim=0)

    def bids_newton(V_arr):
        out = []
        for i in range(0, n, bs_):
            j = min(n, i + bs_)
            V_b = V_arr[i:j].to(device)
            if is_mdn:
                bb, _ = newton_optimize_mdn(d['pi_logits'][i:j].to(device),
                                            d['mu'][i:j].to(device),
                                            d['sigma'][i:j].to(device), V_b)
            else:
                bb, _ = newton_optimize_bins(d['probs'][i:j].to(device), V_b)
            out.append(bb.cpu())
        return torch.cat(out, dim=0)

    V150 = torch.full_like(pp, 150.0)
    Vbid = bid.clone()

    bg150 = bids_grid(V150)
    rg150 = regret(bg150, None, pp, V150).mean().item()
    bn150 = bids_newton(V150)
    rn150 = regret(bn150, None, pp, V150).mean().item()
    bg_b = bids_grid(Vbid)
    rg_b = regret(bg_b, None, pp, Vbid).mean().item()
    bn_b = bids_newton(Vbid)
    rn_b = regret(bn_b, None, pp, Vbid).mean().item()

    print(f'regret V=150  grid={rg150:.3f}  newton={rn150:.3f}')
    print(f'regret V=bid  grid={rg_b:.3f}  newton={rn_b:.3f}')

    per_adv = {}
    if adv is not None:
        reg150_arr = regret(bg150, None, pp, V150).numpy()
        for a in sorted(set(adv.tolist())):
            m = adv == a
            per_adv[a] = float(reg150_arr[m].mean())
        print('per-adv regret (V=150 grid):')
        for a in sorted(per_adv.keys()):
            print(f'{a}: n={int((adv==a).sum())} reg={per_adv[a]:.3f}')

    res = {
        'name': args.name, 'anlp': anlp, 'ks': ks, 'coverage': cov,
        'regret_v150_grid': rg150, 'regret_v150_newton': rn150,
        'regret_vbid_grid': rg_b, 'regret_vbid_newton': rn_b,
        'per_adv_regret_v150': per_adv,
    }
    with open(os.path.join(args.out_dir, f'{args.name}.pkl'), 'wb') as f:
        pickle.dump(res, f)
    print('saved', os.path.join(args.out_dir, f'{args.name}.pkl'))


if __name__ == '__main__':
    main()
