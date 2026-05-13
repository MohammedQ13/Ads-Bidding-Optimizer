import os
import math
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from dataset import BidDataset, load_artifacts
from model import MDN, DiscreteBins, build_vocab_sizes
from loss import mdn_log_prob
from bid_optimizer import (
    grid_optimize_mdn, grid_optimize_bins, newton_optimize_mdn, newton_optimize_bins,
    perfect_profit, regret,
)


def to_dev(d, device):
    """Move all tensors in a dict to the target device."""
    out = {}
    for k, v in d.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def load_model(ckpt_path, device):
    """Load checkpoint and rebuild the model from its saved config."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck['config']
    if cfg['model_type'] == 'mdn':
        model = MDN(
            vocab_sizes=cfg['vocab_sizes'],
            emb_dims=cfg['emb_dims'],
            tag_vocab_size=cfg['tag_vocab_size'],
            tag_emb_dim=cfg['tag_emb_dim'],
            num_continuous=cfg['num_continuous'],
            hidden=cfg['hidden'],
            dropout=cfg['dropout'],
            K=cfg['K'],
            sigma_floor=cfg['sigma_floor'],
        )
    else:
        model = DiscreteBins(
            vocab_sizes=cfg['vocab_sizes'],
            emb_dims=cfg['emb_dims'],
            tag_vocab_size=cfg['tag_vocab_size'],
            tag_emb_dim=cfg['tag_emb_dim'],
            num_continuous=cfg['num_continuous'],
            hidden=cfg['hidden'],
            dropout=cfg['dropout'],
            num_bins=cfg['num_bins'],
        )
    model.load_state_dict(ck['full_state_dict'], strict=False)
    model = model.to(device).eval()
    return model, cfg


@torch.no_grad()
def collect_predictions(model, ds, device, model_type, batch_size, num_bins=None):
    """Run the model on every sample and collect outputs.
    For MDN: stores pi_logits, mu, sigma, and per-sample log probability.
    For bins: stores the full probability vector and per-sample log probability.
    Also stores ground truth (payprice, log_payprice, bidding_price) for metrics.
    """
    out = {'log_pp': [], 'pp': [], 'bid': []}
    if model_type == 'mdn':
        out['pi_logits'] = []
        out['mu'] = []
        out['sigma'] = []
        out['log_prob'] = []
    else:
        out['probs'] = []
        out['log_prob'] = []

    for batch in ds.iter_batches(batch_size, shuffle=False):
        b = to_dev(batch, device)
        if model_type == 'mdn':
            pi, mu, sigma = model(b['cat'], b['cont'], b['tags'])
            lp = mdn_log_prob(pi, mu, sigma, b['log_pp'])
            out['pi_logits'].append(pi.cpu())
            out['mu'].append(mu.cpu())
            out['sigma'].append(sigma.cpu())
            out['log_prob'].append(lp.cpu())
        else:
            logits = model(b['cat'], b['cont'], b['tags'])
            probs = F.softmax(logits, dim=-1)
            pp_int = b['pp'].long().clamp(0, num_bins - 1)
            # log prob of the true bin: grab the predicted probability at the true price
            lp = torch.log(probs.gather(1, pp_int.unsqueeze(1)).squeeze(1).clamp(min=1e-30))
            out['probs'].append(probs.cpu())
            out['log_prob'].append(lp.cpu())
        out['log_pp'].append(b['log_pp'].cpu())
        out['pp'].append(b['pp'].cpu())
        out['bid'].append(b['bid'].cpu())

    # concatenate all batches into single tensors
    cat_outs = {}
    for k, v in out.items():
        cat_outs[k] = torch.cat(v, dim=0)
    return cat_outs


def compute_pit_mdn(preds):
    """PIT for MDN: CDF evaluated at the true value.
    Should be uniform on [0,1] if model is well calibrated.
    """
    pi_logits = preds['pi_logits']
    mu = preds['mu']
    sigma = preds['sigma']
    t = preds['log_pp']
    pi = F.softmax(pi_logits, dim=-1)
    z = (t.unsqueeze(1) - mu) / (sigma * math.sqrt(2.0))
    comp = 0.5 * (1.0 + torch.erf(z))
    cdf = (pi * comp).sum(dim=-1)
    return cdf.numpy()


def compute_pit_bins(preds):
    """PIT for discrete bins: CDF at the true payprice bin."""
    probs = preds['probs']
    pp = preds['pp'].long()
    cdf = torch.cumsum(probs, dim=1)
    pit = cdf.gather(1, pp.unsqueeze(1).clamp(0, probs.size(1)-1)).squeeze(1)
    return pit.numpy()


def ks_statistic(pit):
    """Kolmogorov-Smirnov statistic: max gap between empirical CDF of PIT
    and the uniform CDF. Lower = better calibrated. Perfect model gives 0.
    """
    n = len(pit)
    s = np.sort(pit)
    cdf_emp = np.arange(1, n + 1) / n
    d_plus = (cdf_emp - s).max()
    d_minus = (s - (np.arange(0, n) / n)).max()
    return float(max(d_plus, d_minus))


def coverage_table(pit, pcts):
    """Fraction of PIT values below each percentile threshold."""
    out = {}
    for p in pcts:
        out[p] = float((pit <= (p / 100.0)).mean())
    return out


def regret_metrics_mdn_grid(preds, V_value, n_candidates=500, b_max=300.0, batch_size=8192, device='cuda'):
    """Grid search regret for MDN. V_value is a number or 'bidding'."""
    pi = preds['pi_logits']
    mu = preds['mu']
    sigma = preds['sigma']
    pp = preds['pp']
    bid = preds['bid']

    if isinstance(V_value, str) and V_value == 'bidding':
        V = bid.clone()
    else:
        V = torch.full_like(pp, float(V_value))

    bids = []
    n = pi.shape[0]
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        pi_b = pi[i:j].to(device)
        mu_b = mu[i:j].to(device)
        sigma_b = sigma[i:j].to(device)
        V_b = V[i:j].to(device)
        bb, _, _, _ = grid_optimize_mdn(pi_b, mu_b, sigma_b, V_b, n_candidates=n_candidates, b_max=b_max)
        bids.append(bb.cpu())
    bids = torch.cat(bids, dim=0)
    reg = regret(bids, None, pp, V)
    return reg, bids


def regret_metrics_bins_grid(preds, V_value, batch_size=8192, device='cuda'):
    """Same as regret_metrics_mdn_grid but for discrete bins model."""
    probs = preds['probs']
    pp = preds['pp']
    bid = preds['bid']
    if isinstance(V_value, str) and V_value == 'bidding':
        V = bid.clone()
    else:
        V = torch.full_like(pp, float(V_value))
    bids = []
    n = probs.shape[0]
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        p_b = probs[i:j].to(device)
        V_b = V[i:j].to(device)
        bb, _, _, _ = grid_optimize_bins(p_b, V_b)
        bids.append(bb.cpu())
    bids = torch.cat(bids, dim=0)
    reg = regret(bids, None, pp, V)
    return reg, bids


def regret_bins_newton(preds, V_value, batch_size=8192, device='cuda'):
    """Regret using Newton optimizer for bins. Alternative to grid search."""
    probs = preds['probs']
    pp = preds['pp']
    bid = preds['bid']
    if isinstance(V_value, str) and V_value == 'bidding':
        V = bid.clone()
    else:
        V = torch.full_like(pp, float(V_value))
    bids = []
    n = probs.shape[0]
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        p_b = probs[i:j].to(device)
        V_b = V[i:j].to(device)
        bb, _ = newton_optimize_bins(p_b, V_b)
        bids.append(bb.cpu())
    bids = torch.cat(bids, dim=0)
    return regret(bids, None, pp, V)


def regret_metrics_mdn_newton(preds, V_value, batch_size=8192, device='cuda', b_max=300.0):
    """Regret using Newton optimizer for MDN. Alternative to grid search."""
    pi = preds['pi_logits']
    mu = preds['mu']
    sigma = preds['sigma']
    pp = preds['pp']
    bid = preds['bid']
    if isinstance(V_value, str) and V_value == 'bidding':
        V = bid.clone()
    else:
        V = torch.full_like(pp, float(V_value))
    bids = []
    n = pi.shape[0]
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        pi_b = pi[i:j].to(device)
        mu_b = mu[i:j].to(device)
        sigma_b = sigma[i:j].to(device)
        V_b = V[i:j].to(device)
        bb, _ = newton_optimize_mdn(pi_b, mu_b, sigma_b, V_b, b_max=b_max)
        bids.append(bb.cpu())
    bids = torch.cat(bids, dim=0)
    reg = regret(bids, None, pp, V)
    return reg, bids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--save_preds', action='store_true')
    args = parser.parse_args()

    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    art = load_artifacts(cfg['data']['processed_dir'])
    parquet = os.path.join(cfg['data']['processed_dir'], f'{args.split}.parquet')
    has_extras = (args.split == 'test')
    ds = BidDataset(parquet, art, has_extras=has_extras)
    print(f'{args.split} rows:', len(ds), flush=True)

    model, mcfg = load_model(args.ckpt, device)
    model_type = mcfg['model_type']
    num_bins = mcfg.get('num_bins')
    print('model_type:', model_type, 'num_bins:', num_bins, flush=True)

    print('collecting predictions', flush=True)
    t0 = time.time()
    preds = collect_predictions(model, ds, device, model_type, batch_size=8192, num_bins=num_bins)
    print('collected in', round(time.time()-t0,1), 's', flush=True)

    # density metrics: NLL and ANLP (average negative log probability)
    # ANLP is the key density metric for comparing to published baselines
    nll = float(-preds['log_prob'].mean())
    if model_type == 'mdn':
        # for MDN, ANLP subtracts log_payprice (Jacobian correction for log-space)
        anlp = float(-(preds['log_prob'] - preds['log_pp']).mean())
        pit = compute_pit_mdn(preds)
    else:
        anlp = float(-preds['log_prob'].mean())
        pit = compute_pit_bins(preds)

    ks = ks_statistic(pit)
    cov = coverage_table(pit, cfg['evaluation']['percentiles'])
    print(f'NLL: {nll:.4f}')
    print(f'ANLP_linear: {anlp:.4f}')
    print(f'KS: {ks:.4f}')
    print('coverage:')
    for p in sorted(cov.keys()):
        print(f'pct {p}: {cov[p]:.3f}')

    print('regret V=150 (grid)')
    if model_type == 'mdn':
        reg150, bids150 = regret_metrics_mdn_grid(preds, 150.0, device=device)
    else:
        reg150, bids150 = regret_metrics_bins_grid(preds, 150.0, device=device)
    print(f'mean regret: {reg150.mean().item():.3f} fen')

    print('regret V=bidding_price (grid)')
    if model_type == 'mdn':
        reg_bid, bids_bid = regret_metrics_mdn_grid(preds, 'bidding', device=device)
    else:
        reg_bid, bids_bid = regret_metrics_bins_grid(preds, 'bidding', device=device)
    print(f'mean regret: {reg_bid.mean().item():.3f} fen')

    if model_type == 'mdn':
        print('regret V=150 (newton)')
        reg_n150, _ = regret_metrics_mdn_newton(preds, 150.0, device=device)
        print(f'mean regret: {reg_n150.mean().item():.3f}')
        print('regret V=bidding (newton)')
        reg_nbid, _ = regret_metrics_mdn_newton(preds, 'bidding', device=device)
        print(f'mean regret: {reg_nbid.mean().item():.3f}')
    else:
        print('regret V=150 (newton-bins)')
        reg_n150 = regret_bins_newton(preds, 150.0, device=device)
        print(f'mean regret: {reg_n150.mean().item():.3f}')
        print('regret V=bidding (newton-bins)')
        reg_nbid = regret_bins_newton(preds, 'bidding', device=device)
        print(f'mean regret: {reg_nbid.mean().item():.3f}')

    adv_arr = np.asarray(ds.adv)
    advs = sorted(set(adv_arr.tolist()))
    per_adv = {}
    print('per-advertiser regret (V=150 grid):')
    for a in advs:
        m = adv_arr == a
        if m.sum() == 0:
            continue
        r = reg150[m].mean().item()
        per_adv[a] = r
        print(f'adv {a}: n={int(m.sum())}, regret={r:.3f}')

    out_dir = os.path.join('exports', args.name)
    os.makedirs(out_dir, exist_ok=True)
    res = {
        'nll': nll,
        'anlp': anlp,
        'ks': ks,
        'coverage': cov,
        'regret_v150_grid': float(reg150.mean()),
        'regret_vbid_grid': float(reg_bid.mean()),
        'per_adv_regret_v150': per_adv,
    }
    res['regret_v150_newton'] = float(reg_n150.mean())
    res['regret_vbid_newton'] = float(reg_nbid.mean())
    with open(os.path.join(out_dir, f'eval_{args.split}.pkl'), 'wb') as f:
        pickle.dump(res, f)

    if args.save_preds:
        save_path = os.path.join(out_dir, f'preds_{args.split}.pt')
        light = {
            'log_pp': preds['log_pp'],
            'pp': preds['pp'],
            'bid': preds['bid'],
            'log_prob': preds['log_prob'],
            'pit': torch.from_numpy(pit),
            'adv': adv_arr,
        }
        if model_type == 'mdn':
            light['pi_logits'] = preds['pi_logits']
            light['mu'] = preds['mu']
            light['sigma'] = preds['sigma']
        else:
            light['probs'] = preds['probs']
        if has_extras:
            light['click'] = ds.click
            light['conv'] = ds.conv
        torch.save(light, save_path)
        print('saved preds to', save_path)


if __name__ == '__main__':
    main()
