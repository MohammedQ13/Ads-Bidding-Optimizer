"""Memory-efficient ensemble weight search.

Strategy: load member preds on disk as compressed memory-mapped arrays. For each
weight combination, stream chunks from disk, compute weighted avg, compute bid,
accumulate regret. Doesn't materialize any (N, K) array except the small bids.

Pre-step: convert each member's preds_test.pt to a memory-mapped (N, num_bins) bins
array on disk. Reuses the work across weight trials.
"""
import os
import math
import argparse
import pickle
import json
import time
import numpy as np
import torch
import torch.nn.functional as F


def mdn_to_bins_chunked_save(pi, mu, sigma, num_bins, out_path, chunk=10000, device='cuda'):
    upper = torch.arange(num_bins, dtype=torch.float32) + 0.5
    lower = torch.cat([torch.zeros(1), torch.arange(num_bins - 1, dtype=torch.float32) + 0.5])
    log_upper = torch.log(upper.clamp(min=1e-3)).to(device)
    log_lower = torch.log(lower.clamp(min=1e-3)).to(device); log_lower[0] = -1e9
    N = pi.shape[0]
    arr = np.memmap(out_path, dtype='float32', mode='w+', shape=(N, num_bins))
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
        arr[s:e] = chunk_p.cpu().numpy()
    arr.flush()


def bins_to_mmap_save(probs_t, num_bins, out_path):
    arr = np.memmap(out_path, dtype='float32', mode='w+', shape=(probs_t.shape[0], num_bins))
    p = probs_t.numpy()
    if p.shape[1] >= num_bins:
        p = p[:, :num_bins]
    else:
        pad = np.zeros((p.shape[0], num_bins - p.shape[1]), dtype=p.dtype)
        p = np.concatenate([p, pad], axis=1)
    p = p / p.sum(axis=1, keepdims=True).clip(min=1e-30)
    arr[:] = p
    arr.flush()


def regret_v150_streaming(member_paths, weights, pp, num_bins, V_fixed=150.0, chunk=200000):
    """Stream chunks from each member's mmap, compute weighted avg, compute bid, accumulate."""
    N = len(pp)
    bids = np.zeros(N, dtype='float32')
    bins_arr = np.arange(num_bins, dtype='float32')
    M = len(member_paths)
    mmaps = []
    for p in member_paths:
        a = np.memmap(p, dtype='float32', mode='r', shape=(N, num_bins))
        mmaps.append(a)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        avg = np.zeros((e - s, num_bins), dtype='float32')
        for mi, mm in enumerate(mmaps):
            avg += weights[mi] * np.asarray(mm[s:e])
        s_avg = avg.sum(axis=1, keepdims=True).clip(min=1e-30)
        avg /= s_avg
        cdf = np.cumsum(avg, axis=1)
        profit = (V_fixed - bins_arr[None, :]) * cdf
        profit[:, int(np.ceil(V_fixed)):] = 0.0
        idx = profit.argmax(axis=1)
        bids[s:e] = bins_arr[idx]
    win = (bids >= pp).astype('float32')
    realized = win * (V_fixed - bids)
    perfect = ((V_fixed > pp).astype('float32')) * (V_fixed - pp)
    return float((perfect - realized).mean())


def regret_vbid_streaming(member_paths, weights, pp, bid, num_bins, chunk=200000):
    N = len(pp)
    bids_out = np.zeros(N, dtype='float32')
    bins_arr = np.arange(num_bins, dtype='float32')
    M = len(member_paths)
    mmaps = []
    for p in member_paths:
        mmaps.append(np.memmap(p, dtype='float32', mode='r', shape=(N, num_bins)))
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        avg = np.zeros((e - s, num_bins), dtype='float32')
        for mi, mm in enumerate(mmaps):
            avg += weights[mi] * np.asarray(mm[s:e])
        avg /= avg.sum(axis=1, keepdims=True).clip(min=1e-30)
        cdf = np.cumsum(avg, axis=1)
        Vc = bid[s:e]
        profit = (Vc[:, None] - bins_arr[None, :]) * cdf
        profit = np.where(bins_arr[None, :] > Vc[:, None], 0.0, profit)
        idx = profit.argmax(axis=1)
        bids_out[s:e] = bins_arr[idx]
    win = (bids_out >= pp).astype('float32')
    realized = win * (bid - bids_out)
    perfect = ((bid > pp).astype('float32')) * (bid - pp)
    return float((perfect - realized).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', nargs='+', required=True)
    parser.add_argument('--cache_dir', type=str, default='exports/_ens_cache')
    parser.add_argument('--num_bins', type=int, default=301)
    parser.add_argument('--n_random', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='exports/ens_search_results.pkl')
    parser.add_argument('--rebuild_cache', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)

    # build cache mmaps
    member_paths = []
    pp = bid = None
    for mpath in args.members:
        d = torch.load(mpath, map_location='cpu', weights_only=False)
        if pp is None:
            pp = d['pp'].numpy()
            bid = d['bid'].numpy()
        cache_name = os.path.basename(os.path.dirname(mpath)) + '.f32'
        cache_path = os.path.join(args.cache_dir, cache_name)
        member_paths.append(cache_path)
        if (not os.path.exists(cache_path)) or args.rebuild_cache:
            print('building cache for', mpath, '->', cache_path, flush=True)
            t = time.time()
            if 'pi_logits' in d:
                mdn_to_bins_chunked_save(d['pi_logits'], d['mu'], d['sigma'],
                                         args.num_bins, cache_path, device=device)
            else:
                bins_to_mmap_save(d['probs'], args.num_bins, cache_path)
            print('cache built in', round(time.time() - t, 1), 's', flush=True)
        else:
            print('reusing cache', cache_path, flush=True)
        # free the loaded tensors
        del d

    M = len(member_paths)
    # baseline: each member alone
    print('member-alone regret:')
    member_results = []
    for i, p in enumerate(member_paths):
        w = np.zeros(M); w[i] = 1.0
        r = regret_v150_streaming(member_paths, w, pp, args.num_bins)
        rb = regret_vbid_streaming(member_paths, w, pp, bid, args.num_bins)
        print(f"m{i} ({os.path.basename(p)}): r150={r:.3f} rvbid={rb:.3f}", flush=True)
        member_results.append({'i': i, 'name': p, 'r150': r, 'rvbid': rb})

    # random Dirichlet
    best_r = float('inf'); best_w = None
    results = []
    print('random Dirichlet search', flush=True)
    for trial in range(args.n_random):
        w = np.random.dirichlet(np.ones(M))
        r = regret_v150_streaming(member_paths, w, pp, args.num_bins)
        results.append({'w': w.tolist(), 'r150': r})
        if r < best_r:
            best_r = r; best_w = w
            print(f"trial {trial} new best r150={r:.4f} w={np.round(w, 3)}", flush=True)

    # Dirichlet skewed toward each member
    for biased_idx in range(M):
        alpha = np.ones(M); alpha[biased_idx] = 5.0
        for trial in range(args.n_random // 2):
            w = np.random.dirichlet(alpha)
            r = regret_v150_streaming(member_paths, w, pp, args.num_bins)
            results.append({'w': w.tolist(), 'r150': r})
            if r < best_r:
                best_r = r; best_w = w
                print(f"biased({biased_idx}) trial {trial} new best r150={r:.4f} w={np.round(w, 3)}", flush=True)

    # greedy
    print('greedy refinement', flush=True)
    for outer in range(3):
        improved = False
        for i in range(M):
            for delta in (-0.1, -0.05, -0.02, 0.02, 0.05, 0.1):
                w = best_w.copy()
                w[i] = max(0.0, w[i] + delta)
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                r = regret_v150_streaming(member_paths, w, pp, args.num_bins)
                if r < best_r - 1e-4:
                    best_r = r; best_w = w; improved = True
                    print(f"greedy r150={r:.4f} w={np.round(w, 3)}", flush=True)
        if not improved:
            break

    rvb = regret_vbid_streaming(member_paths, best_w, pp, bid, args.num_bins)
    print(f"FINAL best r150={best_r:.4f} rvbid={rvb:.4f} w={np.round(best_w, 3)}", flush=True)

    out = {'best_r150': best_r, 'best_rvbid': rvb,
           'best_w': best_w.tolist(), 'members': args.members,
           'member_paths_cache': member_paths,
           'member_alone': member_results,
           'all_results': results}
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print('saved', args.out)


if __name__ == '__main__':
    main()
