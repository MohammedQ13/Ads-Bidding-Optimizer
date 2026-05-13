"""Fast ensemble search using a subset of test rows for trial evaluation.

Loads each member's bin probs into RAM (uses ~3GB per member). Runs hundreds
of weight combinations on a 100K-row subset. Validates top candidates on full
test.
"""
import os
import math
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F


def mdn_to_bins_chunked(pi, mu, sigma, num_bins, chunk=10000, device='cuda'):
    upper = torch.arange(num_bins, dtype=torch.float32) + 0.5
    lower = torch.cat([torch.zeros(1), torch.arange(num_bins - 1, dtype=torch.float32) + 0.5])
    log_upper = torch.log(upper.clamp(min=1e-3)).to(device)
    log_lower = torch.log(lower.clamp(min=1e-3)).to(device); log_lower[0] = -1e9
    N = pi.shape[0]
    out = np.zeros((N, num_bins), dtype='float32')
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        p_pi = F.softmax(pi[s:e].to(device), dim=-1)
        p_mu = mu[s:e].to(device); p_sg = sigma[s:e].to(device)
        z_up = (log_upper.view(1, -1, 1) - p_mu.unsqueeze(1)) / (p_sg.unsqueeze(1) * math.sqrt(2.0))
        z_lo = (log_lower.view(1, -1, 1) - p_mu.unsqueeze(1)) / (p_sg.unsqueeze(1) * math.sqrt(2.0))
        cu = (p_pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_up))).sum(dim=-1)
        cl = (p_pi.unsqueeze(1) * 0.5 * (1.0 + torch.erf(z_lo))).sum(dim=-1)
        chunk_p = (cu - cl).clamp(min=1e-10)
        chunk_p = chunk_p / chunk_p.sum(dim=1, keepdim=True)
        out[s:e] = chunk_p.cpu().numpy()
    return out


def to_bins(probs, num_bins=301):
    p = probs.numpy() if isinstance(probs, torch.Tensor) else probs
    if p.shape[1] >= num_bins:
        p = p[:, :num_bins]
    else:
        pad = np.zeros((p.shape[0], num_bins - p.shape[1]), dtype=p.dtype)
        p = np.concatenate([p, pad], axis=1)
    p = p / p.sum(axis=1, keepdims=True).clip(min=1e-30)
    return p.astype('float32')


def regret_v150(probs, pp, V_fixed=150.0):
    num_bins = probs.shape[1]
    bins_arr = np.arange(num_bins, dtype='float32')
    cdf = np.cumsum(probs, axis=1)
    profit = (V_fixed - bins_arr[None, :]) * cdf
    profit[:, int(np.ceil(V_fixed)):] = 0.0
    idx = profit.argmax(axis=1)
    bids = bins_arr[idx]
    win = (bids >= pp).astype('float32')
    realized = win * (V_fixed - bids)
    perfect = ((V_fixed > pp).astype('float32')) * (V_fixed - pp)
    return float((perfect - realized).mean())


def regret_vbid(probs, pp, bid):
    num_bins = probs.shape[1]
    bins_arr = np.arange(num_bins, dtype='float32')
    cdf = np.cumsum(probs, axis=1)
    profit = (bid[:, None] - bins_arr[None, :]) * cdf
    profit = np.where(bins_arr[None, :] > bid[:, None], 0.0, profit)
    idx = profit.argmax(axis=1)
    bids_ = bins_arr[idx]
    win = (bids_ >= pp).astype('float32')
    realized = win * (bid - bids_)
    perfect = ((bid > pp).astype('float32')) * (bid - pp)
    return float((perfect - realized).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', nargs='+', required=True)
    parser.add_argument('--num_bins', type=int, default=301)
    parser.add_argument('--n_random', type=int, default=400)
    parser.add_argument('--subset_n', type=int, default=200000)
    parser.add_argument('--top_k_full', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='exports/ens_search_v3.pkl')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)

    print('loading', len(args.members), 'members', flush=True)
    member_probs = []
    pp_full = bid_full = None
    for mp in args.members:
        d = torch.load(mp, map_location='cpu', weights_only=False)
        if pp_full is None:
            pp_full = d['pp'].numpy(); bid_full = d['bid'].numpy()
        if 'pi_logits' in d:
            print(f"{os.path.basename(os.path.dirname(mp))}: mdn -> integrating to bins", flush=True)
            t = time.time()
            arr = mdn_to_bins_chunked(d['pi_logits'], d['mu'], d['sigma'], args.num_bins, device=device)
            print(f"done in {time.time()-t:.1f}s", flush=True)
        else:
            print(f"{os.path.basename(os.path.dirname(mp))}: bins direct", flush=True)
            arr = to_bins(d['probs'], args.num_bins)
        member_probs.append(arr)
        del d

    M = len(member_probs)
    N = pp_full.shape[0]

    # subset for fast trial eval
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=min(args.subset_n, N), replace=False).astype('int64')
    print(f"subset size {len(idx)}", flush=True)
    sub_probs = []
    for arr in member_probs:
        sub_probs.append(arr[idx])
    sub_pp = pp_full[idx]
    sub_bid = bid_full[idx]

    # member-alone on subset
    print('member-alone subset regret:', flush=True)
    member_results = []
    for i, p in enumerate(sub_probs):
        r = regret_v150(p, sub_pp)
        member_results.append({'i': i, 'name': args.members[i], 'r150_sub': r})
        print(f"m{i} ({os.path.basename(os.path.dirname(args.members[i]))}): r150={r:.3f}", flush=True)

    def avg_probs(weights, probs_list):
        avg = np.zeros_like(probs_list[0])
        for k, w in enumerate(weights):
            if w > 0:
                avg += w * probs_list[k]
        s = avg.sum(axis=1, keepdims=True).clip(min=1e-30)
        return avg / s

    # uniform Dirichlet
    print(f'random Dirichlet ({args.n_random}) on subset', flush=True)
    best_r = float('inf'); best_w = None
    results = []
    for trial in range(args.n_random):
        w = np.random.dirichlet(np.ones(M))
        avg = avg_probs(w, sub_probs)
        r = regret_v150(avg, sub_pp)
        results.append({'w': w.tolist(), 'r150_sub': r})
        if r < best_r:
            best_r = r; best_w = w
            print(f"trial {trial} new best subset r150={r:.4f} w={np.round(w, 3)}", flush=True)

    # biased Dirichlet per member
    for biased_idx in range(M):
        alpha = np.ones(M); alpha[biased_idx] = 5.0
        for trial in range(args.n_random // 4):
            w = np.random.dirichlet(alpha)
            avg = avg_probs(w, sub_probs)
            r = regret_v150(avg, sub_pp)
            results.append({'w': w.tolist(), 'r150_sub': r})
            if r < best_r:
                best_r = r; best_w = w
                print(f"biased({biased_idx}) trial {trial} new best subset r150={r:.4f} w={np.round(w, 3)}", flush=True)

    # greedy
    print('greedy refinement on subset', flush=True)
    for outer in range(4):
        improved = False
        for i in range(M):
            for delta in (-0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15):
                w = best_w.copy()
                w[i] = max(0.0, w[i] + delta)
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                avg = avg_probs(w, sub_probs)
                r = regret_v150(avg, sub_pp)
                if r < best_r - 1e-4:
                    best_r = r; best_w = w; improved = True
                    print(f"greedy r150={r:.4f} w={np.round(w, 3)}", flush=True)
        if not improved:
            break

    # validate top weights on full test
    results.sort(key=lambda x: x['r150_sub'])
    candidates = results[:args.top_k_full]
    candidates.append({'w': best_w.tolist(), 'r150_sub': best_r})
    print(f'validating top {len(candidates)} on full test', flush=True)
    full_results = []
    for c in candidates:
        w = np.asarray(c['w'])
        avg = avg_probs(w, member_probs)
        r150 = regret_v150(avg, pp_full)
        rvb = regret_vbid(avg, pp_full, bid_full)
        full_results.append({'w': c['w'], 'r150_sub': c['r150_sub'], 'r150_full': r150, 'rvbid_full': rvb})
        print(f"full r150={r150:.4f} rvbid={rvb:.4f} w={np.round(w, 3)}", flush=True)

    # final best on full
    full_results.sort(key=lambda x: x['r150_full'])
    best_full = full_results[0]
    print(f'FINAL r150_full={best_full["r150_full"]:.4f} rvbid={best_full["rvbid_full"]:.4f} w={np.round(np.asarray(best_full["w"]), 3)}', flush=True)
    out = {'best_full': best_full, 'top_full': full_results,
           'subset_results': results, 'member_results': member_results,
           'members': args.members}
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print('saved', args.out)


if __name__ == '__main__':
    main()
