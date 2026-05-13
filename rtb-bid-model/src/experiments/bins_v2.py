"""DLF discrete-bins variants with configurable edges and EMD loss option.

Trains a DiscreteBins model whose K outputs map to non-uniform price bins. Edges
are stored alongside the checkpoint so eval can use them.
"""
import os
import math
import time
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import load_config
from dataset import BidDataset, load_artifacts
from model import DiscreteBins, build_vocab_sizes
from loss import discrete_bins_nll, discrete_bins_smoothed_nll


def make_edges(kind, num_bins, train_pp, max_price=300.0):
    if kind == 'uniform':
        # bins 0..num_bins-1 cover prices [0, max_price]
        e = np.linspace(0.0, max_price, num_bins + 1)
    elif kind == 'log':
        # log-spaced from 1 to max_price; first bin starts at 0
        log_edges = np.linspace(np.log(1.0), np.log(max_price + 1.0), num_bins + 1)
        e = np.exp(log_edges) - 1.0
        e[0] = 0.0
    elif kind == 'sqrt':
        u = np.linspace(0.0, np.sqrt(max_price), num_bins + 1)
        e = u ** 2
    elif kind == 'quantile':
        # use empirical quantiles of training payprice clipped to [0, max_price]
        v = np.clip(train_pp, 0.0, max_price)
        qs = np.linspace(0.0, 1.0, num_bins + 1)
        e = np.quantile(v, qs)
        e[0] = 0.0
        e[-1] = max_price
        # ensure monotone increasing (quantiles can have ties)
        for i in range(1, len(e)):
            if e[i] <= e[i - 1]:
                e[i] = e[i - 1] + 1e-3
    else:
        raise ValueError(kind)
    return e.astype(np.float32)


def assign_bins(pp, edges):
    # pp can be float numpy. digitize returns 1..len(edges)-1, then -1 → 0..num_bins-1
    idx = np.digitize(pp, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2).astype(np.int64)
    return idx


def emd_loss(logits, target_bin, num_bins):
    """1D EMD: |CDF_pred - CDF_true| summed over bins (one-hot true)."""
    probs = F.softmax(logits, dim=-1)
    cdf_pred = torch.cumsum(probs, dim=1)
    # true cdf: 0 below true bin, 1 from true bin onward
    arr = torch.arange(num_bins, device=logits.device).unsqueeze(0)
    cdf_true = (arr >= target_bin.unsqueeze(1)).float()
    return torch.abs(cdf_pred - cdf_true).sum(dim=1).mean()


def train_one(args, train_ds, val_ds, art, device, edges, log_path):
    seed = args.get('seed', 42)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    K = args.get('num_bins', 200)
    hidden = args.get('hidden', [512, 256, 128, 64])
    bs = args['batch_size']
    lr = args['lr']
    dp = args['dropout']
    wd = args['weight_decay']
    epochs = args.get('epochs', 6)
    ema_decay = args.get('ema_decay', 0.99)
    warmup = args.get('warmup', 500)
    smooth_sigma = args.get('smooth_sigma', 0.0)
    emd_lambda = args.get('emd_lambda', 0.0)
    label_smooth = args.get('label_smooth', 0.0)

    vs = build_vocab_sizes(art)
    emb_dims = art['embedding_dims']
    tag_dim = emb_dims['user_tags']
    tag_vocab_size = emb_dims['_tag_vocab_size']
    num_continuous = train_ds.num_continuous()
    num_bins = K

    model = DiscreteBins(vs, emb_dims, tag_vocab_size, tag_dim, num_continuous, hidden, dp, num_bins).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = train_ds.num_batches(bs, drop_last=True)
    total_steps = steps_per_epoch * epochs

    edges_t = torch.tensor(edges, device=device)
    # cache bin ids for train and val targets once, by computing on dataset .pp
    def bin_ids_of(pp_tensor):
        return torch.tensor(assign_bins(pp_tensor.numpy(), edges)).to(device)

    train_bin = bin_ids_of(train_ds.pp)
    val_bin = bin_ids_of(val_ds.pp)

    gen = torch.Generator(); gen.manual_seed(seed)

    def cosine_lr_step(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return lr * (0.05 + 0.95 * cos)

    best_val = float('inf')
    best_state = None
    step = 0
    t0 = time.time()

    perm_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_ds), generator=gen)
        cur = 0
        while cur < len(train_ds) - bs:
            idx = perm[cur:cur + bs]
            cat = train_ds.cat[idx].to(device, non_blocking=True)
            cont = train_ds.cont[idx].to(device, non_blocking=True)
            tags = train_ds.tags[idx].to(device, non_blocking=True)
            tgt = train_bin[idx]
            cur_lr = cosine_lr_step(step)
            for pg in opt.param_groups:
                pg['lr'] = cur_lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(cat, cont, tags)
                if smooth_sigma > 0:
                    loss = discrete_bins_smoothed_nll(logits, tgt, sigma=smooth_sigma)
                elif label_smooth > 0:
                    n = num_bins
                    loss = F.cross_entropy(logits, tgt, label_smoothing=label_smooth)
                else:
                    loss = discrete_bins_nll(logits, tgt)
                if emd_lambda > 0:
                    loss = loss + emd_lambda * emd_loss(logits, tgt, num_bins)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            cur += bs
            step += 1
        # val
        model.eval()
        total = 0.0; cnt = 0
        with torch.no_grad():
            for batch in val_ds.iter_batches(bs * 2, shuffle=False):
                cat = batch['cat'].to(device); cont = batch['cont'].to(device); tags = batch['tags'].to(device)
                logits = model(cat, cont, tags)
                pp_int = assign_bins(batch['pp'].numpy(), edges)
                pp_t = torch.from_numpy(pp_int).to(device)
                loss = discrete_bins_nll(logits, pp_t)
                total += loss.item() * cat.size(0); cnt += cat.size(0)
        val_loss = total / cnt
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {}
            for k, v in model.state_dict().items():
                best_state[k] = v.cpu().clone()
        with open(log_path, 'a') as f:
            f.write(f"{args['name']} epoch {epoch} val={val_loss:.4f} (best={best_val:.4f})\n")

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0

    out_dir = os.path.join('exports', args['name'])
    os.makedirs(out_dir, exist_ok=True)
    cfg_save = {
        'model_type': 'bins', 'K': None, 'num_bins': num_bins,
        'hidden': hidden, 'dropout': dp, 'vocab_sizes': vs,
        'emb_dims': emb_dims, 'tag_vocab_size': tag_vocab_size,
        'num_continuous': num_continuous, 'sigma_floor': None, 'tag_emb_dim': tag_dim,
        'edges': edges.tolist(),
    }
    ckpt = {'config': cfg_save, 'full_state_dict': model.state_dict(),
            'val_loss': best_val, 'use_ema': False,
            'edges': edges.tolist()}
    torch.save(ckpt, os.path.join(out_dir, 'best.pt'))

    line = (f"name={args['name']} bin_kind={args.get('bin_kind')} num_bins={num_bins} bs={bs} lr={lr} "
            f"dp={dp} wd={wd} epochs={epochs} smooth={smooth_sigma} emd={emd_lambda} ls={label_smooth} "
            f"hidden={hidden} val={best_val:.4f} time={elapsed:.0f}s")
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')
    return {'name': args['name'], 'val': best_val, 'time_s': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, required=True)
    parser.add_argument('--log', type=str, default='logs/bins_v2_sweep.log')
    args_main = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg['data']['processed_dir'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading data', flush=True)
    t = time.time()
    train_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'train.parquet'), art)
    val_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'val.parquet'), art)
    print('loaded in', round(time.time() - t, 1), 's', flush=True)
    train_pp_arr = train_ds.pp.numpy()

    with open(args_main.config_json, 'r') as f:
        runs = json.load(f)
    open(args_main.log, 'w').close()
    results = []
    for r in runs:
        edges = make_edges(r.get('bin_kind', 'uniform'), r.get('num_bins', 200),
                           train_pp_arr, max_price=r.get('max_price', 300.0))
        res = train_one(r, train_ds, val_ds, art, device, edges, args_main.log)
        results.append(res)
    with open('exports/bins_v2_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('done.', flush=True)


if __name__ == '__main__':
    main()
