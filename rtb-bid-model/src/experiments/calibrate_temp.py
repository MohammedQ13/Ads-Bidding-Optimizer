"""Post-hoc temperature scaling on a saved bins ckpt.

Loads val preds from preds_test.pt (using val.parquet via dataset, NOT test).
Actually — we use val data to fit T, then evaluate on test.

For bins: probs = softmax(logits / T). T tuned to minimize val NLL.
For MDN: temperature on log_pi; sigma scale.
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
from loss import discrete_bins_nll, mdn_log_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='exports/calibrated')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config()
    art = load_artifacts(cfg['data']['processed_dir'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'val.parquet'), art)
    test_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'test.parquet'), art, has_extras=True)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = ck['config']
    if mcfg['model_type'] == 'mdn':
        m = MDN(mcfg['vocab_sizes'], mcfg['emb_dims'], mcfg['tag_vocab_size'],
                mcfg['tag_emb_dim'], mcfg['num_continuous'], mcfg['hidden'],
                mcfg['dropout'], mcfg['K'], mcfg['sigma_floor'])
    else:
        m = DiscreteBins(mcfg['vocab_sizes'], mcfg['emb_dims'], mcfg['tag_vocab_size'],
                         mcfg['tag_emb_dim'], mcfg['num_continuous'], mcfg['hidden'],
                         mcfg['dropout'], mcfg['num_bins'])
    m.load_state_dict(ck['full_state_dict'], strict=False)
    m = m.to(device).eval()

    # collect val logits/sigma
    bs = 8192
    is_mdn = (mcfg['model_type'] == 'mdn')
    val_logits = []
    val_pi = []; val_mu = []; val_sg = []
    val_log_pp = []; val_pp_int = []
    with torch.no_grad():
        for batch in val_ds.iter_batches(bs, shuffle=False):
            cat = batch['cat'].to(device); cont = batch['cont'].to(device); tags = batch['tags'].to(device)
            if is_mdn:
                pi, mu, sg = m(cat, cont, tags)
                val_pi.append(pi.cpu()); val_mu.append(mu.cpu()); val_sg.append(sg.cpu())
                val_log_pp.append(batch['log_pp'])
            else:
                logits = m(cat, cont, tags)
                val_logits.append(logits.cpu())
                val_pp_int.append(batch['pp'].long().clamp(0, mcfg['num_bins'] - 1))
    if is_mdn:
        pi = torch.cat(val_pi); mu = torch.cat(val_mu); sg = torch.cat(val_sg); log_pp = torch.cat(val_log_pp)
    else:
        logits = torch.cat(val_logits); pp_int = torch.cat(val_pp_int)

    # find T
    print('searching T', flush=True)
    Ts = np.linspace(0.5, 3.0, 26)
    best_t = 1.0; best_nll = float('inf')
    for T in Ts:
        if is_mdn:
            # apply T to log_pi (rescale mixture sharpness) AND scale sigma by sqrt(T)
            sg_t = sg * float(np.sqrt(T))
            log_pi = F.log_softmax(pi / float(T), dim=-1)
            log_pi_e = log_pi
            t = log_pp.unsqueeze(1)
            z = (t - mu) / sg_t
            log_comp = -0.5 * (z * z + math.log(2 * math.pi)) - torch.log(sg_t)
            log_p = torch.logsumexp(log_pi_e + log_comp, dim=-1)
            nll = float(-log_p.mean())
        else:
            log_p = F.log_softmax(logits / float(T), dim=-1)
            lp = log_p.gather(1, pp_int.unsqueeze(1)).squeeze(1)
            nll = float(-lp.mean())
        if nll < best_nll:
            best_nll = nll; best_t = T
            print(f'T={T:.3f} val_nll={nll:.4f} (new best)', flush=True)
    print(f'best T={best_t:.3f} val_nll={best_nll:.4f}', flush=True)

    # apply T to test predictions and re-collect
    test_logits = []
    test_pi = []; test_mu = []; test_sg = []
    test_log_pp = []; test_pp = []; test_bid = []
    test_pp_int = []
    with torch.no_grad():
        for batch in test_ds.iter_batches(bs, shuffle=False):
            cat = batch['cat'].to(device); cont = batch['cont'].to(device); tags = batch['tags'].to(device)
            if is_mdn:
                pi_, mu_, sg_ = m(cat, cont, tags)
                test_pi.append(pi_.cpu()); test_mu.append(mu_.cpu()); test_sg.append(sg_.cpu())
                test_log_pp.append(batch['log_pp'])
            else:
                logits_ = m(cat, cont, tags)
                test_logits.append(logits_.cpu())
                test_pp_int.append(batch['pp'].long().clamp(0, mcfg['num_bins'] - 1))
            test_pp.append(batch['pp']); test_bid.append(batch['bid'])
    pp = torch.cat(test_pp); bid = torch.cat(test_bid)

    # save calibrated preds
    save_path = os.path.join(args.out_dir, f'{args.name}.pt')
    if is_mdn:
        pi_t = torch.cat(test_pi); mu_t = torch.cat(test_mu); sg_t = torch.cat(test_sg)
        # apply temperature transform
        sg_T = sg_t * float(np.sqrt(best_t))
        pi_T = F.log_softmax(pi_t / float(best_t), dim=-1)  # log probs
        # NOTE: ensemble.py expects pi_logits not log_pi. Convert by storing logits = pi/T.
        out = {'pi_logits': pi_t / float(best_t), 'mu': mu_t, 'sigma': sg_T,
               'log_pp': torch.cat(test_log_pp), 'pp': pp, 'bid': bid,
               'best_T': best_t,
               'adv': test_ds.adv}
    else:
        log_t = torch.cat(test_logits)
        probs = F.softmax(log_t / float(best_t), dim=-1)
        out = {'probs': probs, 'log_pp': None,
               'pp': pp, 'bid': bid, 'best_T': best_t, 'adv': test_ds.adv}
        out['log_pp'] = pp.float().clamp(min=1.0).log()
    torch.save(out, save_path)
    print(f'saved {save_path} (best_T={best_t:.3f})')


if __name__ == '__main__':
    main()
