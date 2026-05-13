"""Eval many MDN/bins ckpts in one process. Loads test_ds once, iterates ckpts,
computes test ANLP/KS/regret_v150_grid/regret_vbid_grid, prints CSV-style.
"""
import os
import math
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from dataset import BidDataset, load_artifacts
from model import MDN, DiscreteBins
from loss import mdn_log_prob
from bid_optimizer import grid_optimize_mdn, grid_optimize_bins, regret


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck['config']
    if cfg['model_type'] == 'mdn':
        m = MDN(cfg['vocab_sizes'], cfg['emb_dims'], cfg['tag_vocab_size'],
                cfg['tag_emb_dim'], cfg['num_continuous'], cfg['hidden'], cfg['dropout'],
                cfg['K'], cfg['sigma_floor'])
    else:
        m = DiscreteBins(cfg['vocab_sizes'], cfg['emb_dims'], cfg['tag_vocab_size'],
                         cfg['tag_emb_dim'], cfg['num_continuous'], cfg['hidden'], cfg['dropout'],
                         cfg['num_bins'])
    m.load_state_dict(ck['full_state_dict'], strict=False)
    return m.to(device).eval(), cfg


@torch.no_grad()
def collect_mdn(m, ds, device, bs=8192):
    pi_list, mu_list, sg_list = [], [], []
    log_pp = ds.log_pp; pp = ds.pp; bid = ds.bid
    log_prob_list = []
    for batch in ds.iter_batches(bs, shuffle=False):
        cat = batch['cat'].to(device); cont = batch['cont'].to(device); tags = batch['tags'].to(device)
        pi, mu, sg = m(cat, cont, tags)
        pi_list.append(pi.cpu()); mu_list.append(mu.cpu()); sg_list.append(sg.cpu())
        lp = mdn_log_prob(pi, mu, sg, batch['log_pp'].to(device))
        log_prob_list.append(lp.cpu())
    return (torch.cat(pi_list), torch.cat(mu_list), torch.cat(sg_list),
            torch.cat(log_prob_list))


@torch.no_grad()
def collect_bins(m, ds, device, num_bins, bs=8192):
    probs_list = []
    log_prob_list = []
    for batch in ds.iter_batches(bs, shuffle=False):
        cat = batch['cat'].to(device); cont = batch['cont'].to(device); tags = batch['tags'].to(device)
        logits = m(cat, cont, tags)
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs.cpu())
        pp_int = batch['pp'].long().clamp(0, num_bins - 1).to(device)
        lp = torch.log(probs.gather(1, pp_int.unsqueeze(1)).squeeze(1).clamp(min=1e-30))
        log_prob_list.append(lp.cpu())
    return torch.cat(probs_list, dim=0), torch.cat(log_prob_list, dim=0)


def compute_metrics(name, model_kind, preds, ds, device, save_preds_path=None):
    pp = ds.pp; bid = ds.bid; log_pp = ds.log_pp
    n = len(pp)

    if model_kind == 'mdn':
        pi, mu, sg, log_prob = preds
        anlp = float(-(log_prob - log_pp).mean())
        # PIT
        z = (log_pp.unsqueeze(1) - mu) / (sg * math.sqrt(2.0))
        pit = (F.softmax(pi, dim=-1) * 0.5 * (1.0 + torch.erf(z))).sum(dim=-1).numpy()
    else:
        probs, log_prob = preds
        anlp = float(-log_prob.mean())
        pp_int = pp.long().clamp(0, probs.size(1) - 1)
        cdf = torch.cumsum(probs, dim=1)
        pit = cdf.gather(1, pp_int.unsqueeze(1)).squeeze(1).numpy()
    s_ = np.sort(pit)
    cdf_emp = np.arange(1, n + 1) / n
    d_plus = (cdf_emp - s_).max()
    d_minus = (s_ - (np.arange(0, n) / n)).max()
    ks = float(max(d_plus, d_minus))

    # regret V=150
    bs_grid = 16384
    bids150 = []
    bidsvb = []
    for i in range(0, n, bs_grid):
        j = min(n, i + bs_grid)
        if model_kind == 'mdn':
            pi_b = pi[i:j].to(device); mu_b = mu[i:j].to(device); sg_b = sg[i:j].to(device)
            V_b = torch.full((j - i,), 150.0, device=device)
            bb, _, _, _ = grid_optimize_mdn(pi_b, mu_b, sg_b, V_b)
            bids150.append(bb.cpu())
            V_b2 = bid[i:j].to(device)
            bb2, _, _, _ = grid_optimize_mdn(pi_b, mu_b, sg_b, V_b2)
            bidsvb.append(bb2.cpu())
        else:
            p_b = probs[i:j].to(device)
            V_b = torch.full((j - i,), 150.0, device=device)
            bb, _, _, _ = grid_optimize_bins(p_b, V_b)
            bids150.append(bb.cpu())
            V_b2 = bid[i:j].to(device)
            bb2, _, _, _ = grid_optimize_bins(p_b, V_b2)
            bidsvb.append(bb2.cpu())
    bids150 = torch.cat(bids150)
    bidsvb = torch.cat(bidsvb)
    V150 = torch.full_like(pp, 150.0)
    Vbid = bid
    r150 = regret(bids150, None, pp, V150).mean().item()
    rvb = regret(bidsvb, None, pp, Vbid).mean().item()

    if save_preds_path is not None:
        out = {'pp': pp, 'bid': bid, 'log_pp': log_pp, 'log_prob': log_prob, 'pit': torch.from_numpy(pit),
               'adv': ds.adv}
        if model_kind == 'mdn':
            out['pi_logits'] = pi
            out['mu'] = mu
            out['sigma'] = sg
        else:
            out['probs'] = probs
        torch.save(out, save_preds_path)

    return {'name': name, 'anlp': anlp, 'ks': ks, 'r150': r150, 'rvb': rvb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True)
    parser.add_argument('--save_preds', action='store_true')
    args = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg['data']['processed_dir'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading test', flush=True)
    t = time.time()
    ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'test.parquet'), art, has_extras=True)
    print('loaded in', round(time.time() - t, 1), 's. n=', len(ds), flush=True)

    print('NAME,ANLP,KS,REG_V150,REG_VBID,DT', flush=True)
    for ck_path in args.ckpts:
        if not os.path.exists(ck_path):
            print(f'missing,{ck_path}', flush=True)
            continue
        name = os.path.basename(os.path.dirname(ck_path))
        t0 = time.time()
        m, mcfg = load_model(ck_path, device)
        kind = mcfg['model_type']
        if kind == 'mdn':
            preds = collect_mdn(m, ds, device)
        else:
            preds = collect_bins(m, ds, device, mcfg['num_bins'])
        save_path = None
        if args.save_preds:
            save_path = os.path.join(os.path.dirname(ck_path), 'preds_test.pt')
        res = compute_metrics(name, kind, preds, ds, device, save_path)
        dt = time.time() - t0
        print(f"{name},{res['anlp']:.4f},{res['ks']:.4f},{res['r150']:.3f},{res['rvb']:.3f},{dt:.1f}", flush=True)
        # free
        del m, preds
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
