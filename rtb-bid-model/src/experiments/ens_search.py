"""Random + greedy weight search for ensembling.

Loads a list of preds_test.pt members, converts each to discrete bin probs (MDN→integrate,
bins→identity), then explores weight vectors to minimize V=150 grid regret.
"""
import os
import math
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F


def to_bins(probs, num_bins=301):
    """Ensure (N, num_bins) probs. Pads/truncates if needed and renormalizes."""
    if isinstance(probs, torch.Tensor):
        p = probs.numpy()
    else:
        p = probs
    if p.shape[1] >= num_bins:
        p = p[:, :num_bins]
    else:
        pad = np.zeros((p.shape[0], num_bins - p.shape[1]), dtype=p.dtype)
        p = np.concatenate([p, pad], axis=1)
    p = p / p.sum(axis=1, keepdims=True).clip(min=1e-30)
    return p.astype('float32')


def mdn_to_bins_chunked(pi, mu, sigma, num_bins=301, chunk=10000, device='cuda'):
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


def regret_v150_grid(probs, pp, V_fixed=150.0):
    """Memory-light regret. Process in chunks, keep only the (N,) bid indices."""
    N, num_bins = probs.shape
    bins_arr = np.arange(num_bins, dtype='float32')
    chunk = 200000
    bids = np.zeros(N, dtype='float32')
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        cdf = np.cumsum(probs[s:e], axis=1)
        profit = (V_fixed - bins_arr[None, :]) * cdf
        profit[:, int(np.ceil(V_fixed)):] = 0.0
        idx = profit.argmax(axis=1)
        bids[s:e] = bins_arr[idx]
    win = (bids >= pp).astype('float32')
    realized = win * (V_fixed - bids)
    perfect = ((V_fixed > pp).astype('float32')) * (V_fixed - pp)
    return float((perfect - realized).mean())


def regret_vbid_grid(probs, pp, bid):
    num_bins = probs.shape[1]
    bins_arr = np.arange(num_bins, dtype='float32')
    cdf = np.cumsum(probs, axis=1)
    Vb = bid.astype('float32')
    profit = (Vb[:, None] - bins_arr[None, :]) * cdf
    profit = np.where(bins_arr[None, :] > Vb[:, None], 0.0, profit)
    best_idx = profit.argmax(axis=1)
    best_b = bins_arr[best_idx]
    win = (best_b >= pp).astype('float32')
    realized = win * (Vb - best_b)
    perfect = ((Vb > pp).astype('float32')) * (Vb - pp)
    return float((perfect - realized).mean())


def load_member(path, num_bins, device):
    d = torch.load(path, map_location='cpu', weights_only=False)
    pp = d['pp'].numpy(); bid = d['bid'].numpy()
    if 'pi_logits' in d:
        kind = 'mdn'
        probs = mdn_to_bins_chunked(d['pi_logits'], d['mu'], d['sigma'], num_bins, device=device)
    else:
        kind = 'bins'
        probs = to_bins(d['probs'], num_bins)
    return probs, pp, bid, kind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', nargs='+', required=True)
    parser.add_argument('--num_bins', type=int, default=301)
    parser.add_argument('--n_random', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='exports/ens_search_results.pkl')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)

    print('loading', len(args.members), 'members', flush=True)
    member_probs = []
    pp = bid = None
    for p in args.members:
        probs, pp_p, bid_p, kind = load_member(p, args.num_bins, device)
        if pp is None:
            pp = pp_p; bid = bid_p
        member_probs.append(probs)
        print('', os.path.basename(os.path.dirname(p)), 'kind=', kind, 'shape', probs.shape, flush=True)

    M = len(member_probs)

    # baseline: each member alone
    print('member-alone regret:')
    for i, p in enumerate(member_probs):
        r = regret_v150_grid(p, pp)
        rb = regret_vbid_grid(p, pp, bid)
        print(f"m{i} (member {i}): r150={r:.3f} rvbid={rb:.3f}", flush=True)

    # random Dirichlet weight search
    best_r = float('inf')
    best_w = None
    results = []
    for trial in range(args.n_random):
        w = np.random.dirichlet(np.ones(M))
        avg = np.zeros_like(member_probs[0])
        for i in range(M):
            avg += w[i] * member_probs[i]
        avg = avg / avg.sum(axis=1, keepdims=True)
        r = regret_v150_grid(avg, pp)
        results.append({'w': w.tolist(), 'r150': r})
        if r < best_r:
            best_r = r
            best_w = w
            print(f"trial {trial} new best r150={r:.4f} w={np.round(w, 3)}", flush=True)

    # also sample some "MDN-heavy" Dirichlets (higher alpha for MDN positions)
    # caller is expected to put MDN(s) first
    print('MDN-heavy random search (alpha=[5, 1, ]):')
    alpha = np.ones(M); alpha[0] = 5.0  # bias toward member 0 (MDN)
    for trial in range(args.n_random):
        w = np.random.dirichlet(alpha)
        avg = np.zeros_like(member_probs[0])
        for i in range(M):
            avg += w[i] * member_probs[i]
        avg = avg / avg.sum(axis=1, keepdims=True)
        r = regret_v150_grid(avg, pp)
        results.append({'w': w.tolist(), 'r150': r})
        if r < best_r:
            best_r = r
            best_w = w
            print(f"trial {trial} new best r150={r:.4f} w={np.round(w, 3)}", flush=True)

    # greedy local search around best
    print('greedy refinement around best')
    for _ in range(3):
        improved = False
        for i in range(M):
            for delta in (-0.1, -0.05, -0.02, 0.02, 0.05, 0.1):
                w = best_w.copy()
                w[i] = max(0.0, w[i] + delta)
                # renormalize
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                avg = np.zeros_like(member_probs[0])
                for k in range(M):
                    avg += w[k] * member_probs[k]
                avg = avg / avg.sum(axis=1, keepdims=True)
                r = regret_v150_grid(avg, pp)
                if r < best_r - 1e-4:
                    best_r = r
                    best_w = w
                    improved = True
                    print(f"greedy r150={r:.4f} w={np.round(w, 3)}", flush=True)
        if not improved:
            break

    # final stats with V=bid for the best w
    avg = np.zeros_like(member_probs[0])
    for k in range(M):
        avg += best_w[k] * member_probs[k]
    avg = avg / avg.sum(axis=1, keepdims=True)
    rvb = regret_vbid_grid(avg, pp, bid)
    print(f"FINAL best r150={best_r:.4f} rvbid={rvb:.4f} w={np.round(best_w, 3)}", flush=True)

    out = {'best_r150': best_r, 'best_rvbid': rvb,
           'best_w': best_w.tolist(), 'members': args.members,
           'all_results': results}
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print('saved', args.out)


if __name__ == '__main__':
    main()
