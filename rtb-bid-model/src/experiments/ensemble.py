"""Memory-efficient ensembling: process row chunks, never materialize full bins probs in RAM."""

import os
import math
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F


def mdn_chunk_probs(pi_logits_chunk, mu_chunk, sigma_chunk, log_upper, log_lower, device='cuda'):
    pi = F.softmax(pi_logits_chunk.to(device), dim=-1)
    mu = mu_chunk.to(device)
    sg = sigma_chunk.to(device)
    lu = log_upper.to(device)
    ll = log_lower.to(device)
    z_up = (lu.view(1, -1, 1) - mu.unsqueeze(1)) / (sg.unsqueeze(1) * math.sqrt(2.0))
    z_lo = (ll.view(1, -1, 1) - mu.unsqueeze(1)) / (sg.unsqueeze(1) * math.sqrt(2.0))
    cu = (pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_up))).sum(dim=-1)
    cl = (pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_lo))).sum(dim=-1)
    p = (cu - cl).clamp(min=1e-10)
    p = p / p.sum(dim=1, keepdim=True)
    return p.cpu().numpy()


def quantile_chunk_probs(quants_chunk, alphas, num_bins):
    boundaries = np.arange(num_bins + 1, dtype='float32') - 0.5
    boundaries[0] = 0.0
    afull = np.concatenate([[0.0], alphas, [1.0]]).astype('float32')
    A = quants_chunk.shape[1]
    qfull = np.concatenate([np.full((quants_chunk.shape[0], 1), -1e9, dtype='float32'),
                            quants_chunk.astype('float32'),
                            np.full((quants_chunk.shape[0], 1), 1e9, dtype='float32')], axis=1)
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


def evaluate_ensemble(mdn_preds_paths, bins_preds_paths, lgbm_npz_path, name, out_dir,
                      num_bins=301, V_fixed=150.0, weights=None, chunk=10000):
    """Stream chunk by chunk: compute each member's probs, average, then compute metrics."""
    # load metadata (pp, bid, adv) once, plus references to other arrays
    members = []
    pp_ref = None
    bid_ref = None
    adv_ref = None
    for path in (mdn_preds_paths or []):
        d = torch.load(path, map_location='cpu', weights_only=False)
        members.append(('mdn', d))
        if pp_ref is None:
            pp_ref = d['pp'].numpy().astype('float32')
            bid_ref = d['bid'].numpy().astype('float32')
            adv_ref = d['adv'].astype(str)
    for path in (bins_preds_paths or []):
        d = torch.load(path, map_location='cpu', weights_only=False)
        members.append(('bins', d))
        if pp_ref is None:
            pp_ref = d['pp'].numpy().astype('float32')
            bid_ref = d['bid'].numpy().astype('float32')
            adv_ref = d['adv'].astype(str)
    if lgbm_npz_path is not None:
        npz = np.load(lgbm_npz_path, allow_pickle=True)
        members.append(('lgbm', {'quants': npz['test_q'], 'alphas': npz['alphas']}))
        if pp_ref is None:
            pp_ref = npz['y_test'].astype('float32')
            bid_ref = npz['bid_test'].astype('float32')
            adv_ref = npz['adv_test'].astype(str)

    M = len(members)
    if weights is None:
        ws = [1.0/M] * M
    else:
        s = sum(weights)
        ws = []
        for w in weights:
            ws.append(w / s)

    upper = torch.arange(num_bins, dtype=torch.float32) + 0.5
    lower = torch.cat([torch.zeros(1), torch.arange(num_bins - 1, dtype=torch.float32) + 0.5])
    log_upper = torch.log(upper.clamp(min=1e-3))
    log_lower = torch.log(lower.clamp(min=1e-3))
    log_lower[0] = -1e9
    bins_arr = np.arange(num_bins, dtype='float32')

    N = len(pp_ref)
    total_log_p = 0.0
    pit_arr = np.zeros(N, dtype='float32')
    sum_reg150 = 0.0
    sum_reg_v = 0.0
    sum_perfect150 = 0.0
    sum_perfect_v = 0.0

    # per-advertiser accumulators
    advs_set = sorted(set(adv_ref.tolist()))
    adv_idx_map = {}
    for i, a in enumerate(advs_set):
        adv_idx_map[a] = i
    adv_reg150_sum = np.zeros(len(advs_set), dtype='float64')
    adv_reg_v_sum = np.zeros(len(advs_set), dtype='float64')
    adv_n = np.zeros(len(advs_set), dtype='int64')

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        # build avg probs over chunk
        avg = None
        for mi, (kind, d) in enumerate(members):
            if kind == 'mdn':
                pi = d['pi_logits'][s:e]
                mu = d['mu'][s:e]
                sg = d['sigma'][s:e]
                p = mdn_chunk_probs(pi, mu, sg, log_upper, log_lower)
            elif kind == 'bins':
                p_t = d['probs'][s:e]
                if p_t.shape[1] >= num_bins:
                    p = p_t[:, :num_bins].numpy()
                else:
                    p_full = np.zeros((p_t.shape[0], num_bins), dtype='float32')
                    p_full[:, :p_t.shape[1]] = p_t.numpy()
                    p = p_full
                p = p / p.sum(axis=1, keepdims=True)
            else:  # lgbm
                p = quantile_chunk_probs(d['quants'][s:e], d['alphas'], num_bins)
            if avg is None:
                avg = ws[mi] * p
            else:
                avg = avg + ws[mi] * p
        avg = avg / avg.sum(axis=1, keepdims=True)
        # cdf, pdf for chunk
        pp_int = np.clip(np.round(pp_ref[s:e]).astype(np.int64), 0, num_bins - 1)
        log_p = np.log(avg[np.arange(e - s), pp_int].clip(min=1e-30))
        total_log_p += float(log_p.sum())
        cdf = np.cumsum(avg, axis=1)
        pit_arr[s:e] = cdf[np.arange(e - s), pp_int]
        # regret V=fixed
        profit150 = (V_fixed - bins_arr[None, :]) * cdf
        profit150 = np.where(bins_arr[None, :] > V_fixed, 0.0, profit150)
        best_idx = profit150.argmax(axis=1)
        best_b = bins_arr[best_idx]
        win = (best_b >= pp_ref[s:e]).astype('float32')
        realized = win * (V_fixed - best_b)
        perfect = ((V_fixed > pp_ref[s:e]).astype('float32')) * (V_fixed - pp_ref[s:e])
        reg = perfect - realized
        sum_reg150 += float(reg.sum())
        sum_perfect150 += float(perfect.sum())
        # regret V=bid
        Vb = bid_ref[s:e]
        profit_b = (Vb[:, None] - bins_arr[None, :]) * cdf
        profit_b = np.where(bins_arr[None, :] > Vb[:, None], 0.0, profit_b)
        best_idx_b = profit_b.argmax(axis=1)
        best_b_b = bins_arr[best_idx_b]
        winb = (best_b_b >= pp_ref[s:e]).astype('float32')
        realb = winb * (Vb - best_b_b)
        perfb = ((Vb > pp_ref[s:e]).astype('float32')) * (Vb - pp_ref[s:e])
        reg_v = perfb - realb
        sum_reg_v += float(reg_v.sum())
        sum_perfect_v += float(perfb.sum())
        # per-adv accumulate
        adv_chunk = adv_ref[s:e]
        for r in range(e - s):
            ai = adv_idx_map[adv_chunk[r]]
            adv_reg150_sum[ai] += float(reg[r])
            adv_reg_v_sum[ai] += float(reg_v[r])
            adv_n[ai] += 1

    anlp = -total_log_p / N
    s_ = np.sort(pit_arr)
    cdf_emp = np.arange(1, N + 1) / N
    d_plus = (cdf_emp - s_).max()
    d_minus = (s_ - (np.arange(0, N) / N)).max()
    ks = float(max(d_plus, d_minus))
    pcts = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    cov = {}
    for p in pcts:
        cov[p] = float((pit_arr <= p/100.0).mean())
    reg150 = sum_reg150 / N
    reg_v = sum_reg_v / N
    per_adv = {}
    for a, i in adv_idx_map.items():
        per_adv[a] = float(adv_reg150_sum[i] / max(1, adv_n[i]))

    print(f'[{name}] ANLP={anlp:.4f} KS={ks:.4f} regret_v150={reg150:.3f} regret_vbid={reg_v:.3f}')
    print('per-adv regret v150:')
    for a in sorted(per_adv.keys()):
        print(f'{a}: n={int(adv_n[adv_idx_map[a]])} reg={per_adv[a]:.3f}')
    res = {
        'name': name, 'anlp': anlp, 'ks': ks, 'coverage': cov,
        'regret_v150_grid': reg150, 'regret_vbid_grid': reg_v, 'per_adv_regret_v150': per_adv,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{name}.pkl'), 'wb') as f:
        pickle.dump(res, f)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdn_preds', type=str, nargs='*', default=[])
    parser.add_argument('--bins_preds', type=str, nargs='*', default=[])
    parser.add_argument('--lgbm_npz', type=str, default=None)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='exports/ensembles')
    parser.add_argument('--num_bins', type=int, default=301)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    weights = None
    if args.weights is not None:
        weights = []
        for w in args.weights.split(','):
            weights.append(float(w))
    evaluate_ensemble(args.mdn_preds, args.bins_preds, args.lgbm_npz,
                      args.name, args.out_dir, args.num_bins, weights=weights)


if __name__ == '__main__':
    main()
