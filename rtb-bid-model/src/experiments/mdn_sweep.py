"""MDN hyperparameter sweep — train multiple MDNs with different configs in one process.

Re-uses BidDataset loaded once across runs.
Logs each run's best val NLL + a quick test regret estimate.
"""
import os
import math
import time
import json
import argparse
import pickle
import numpy as np
import torch

from config import load_config
from dataset import BidDataset, load_artifacts
from model import MDN, build_vocab_sizes
from loss import mdn_nll, mdn_log_prob
from bid_optimizer import grid_optimize_mdn, regret


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])


def cosine_lr(step, warmup, total, base_lr, min_lr_ratio=0.05):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def to_dev(d, device):
    out = {}
    for k, v in d.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def evaluate_loss(model, ds, device, batch_size):
    model.eval()
    total = 0.0
    cnt = 0
    with torch.no_grad():
        for batch in ds.iter_batches(batch_size, shuffle=False):
            b = to_dev(batch, device)
            pi, mu, sigma = model(b['cat'], b['cont'], b['tags'])
            loss = mdn_nll(pi, mu, sigma, b['log_pp'])
            total += loss.item() * b['cat'].size(0)
            cnt += b['cat'].size(0)
    return total / cnt


def quick_test_regret(model, test_ds, device, V_const=150.0, batch_size=8192, sample_n=200000):
    """Estimate test regret on a random subsample for fast HP screening."""
    model.eval()
    n = len(test_ds)
    idx = torch.randperm(n)[:sample_n]
    pp_arr = test_ds.pp[idx]
    bid_arr = test_ds.bid[idx]
    cat_arr = test_ds.cat[idx]
    cont_arr = test_ds.cont[idx]
    tag_arr = test_ds.tags[idx]
    out_b = []
    out_b_v = []
    with torch.no_grad():
        for i in range(0, sample_n, batch_size):
            j = min(sample_n, i + batch_size)
            cat = cat_arr[i:j].to(device)
            cont = cont_arr[i:j].to(device)
            tags = tag_arr[i:j].to(device)
            pi, mu, sigma = model(cat, cont, tags)
            V_b = torch.full((j - i,), V_const, device=device)
            bb, _, _, _ = grid_optimize_mdn(pi, mu, sigma, V_b)
            out_b.append(bb.cpu())
            V_b2 = bid_arr[i:j].to(device)
            bb2, _, _, _ = grid_optimize_mdn(pi, mu, sigma, V_b2)
            out_b_v.append(bb2.cpu())
    bids = torch.cat(out_b)
    bids_v = torch.cat(out_b_v)
    V150 = torch.full_like(pp_arr, V_const)
    Vbid = bid_arr
    r150 = regret(bids, None, pp_arr, V150).mean().item()
    rvb = regret(bids_v, None, pp_arr, Vbid).mean().item()
    return r150, rvb


def train_one(args, train_ds, val_ds, test_ds, art, device, log_path):
    set_seed(args['seed'])
    K = args.get('K', 12)
    hidden = args.get('hidden', [512, 256, 128, 64])
    bs = args['batch_size']
    lr = args['lr']
    dp = args['dropout']
    wd = args['weight_decay']
    sf = args['sigma_floor']
    epochs = args.get('epochs', 5)
    ema_decay = args.get('ema_decay', 0.999)
    warmup = args.get('warmup', 500)
    ent_bonus = args.get('ent_bonus', 0.0)

    vs = build_vocab_sizes(art)
    emb_dims = art['embedding_dims']
    tag_dim = emb_dims['user_tags']
    tag_vocab_size = emb_dims['_tag_vocab_size']
    num_continuous = train_ds.num_continuous()
    model = MDN(vs, emb_dims, tag_vocab_size, tag_dim, num_continuous, hidden, dp, K, sf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ema = EMA(model, decay=ema_decay)
    scaler = torch.amp.GradScaler('cuda')
    steps_per_epoch = train_ds.num_batches(bs, drop_last=True)
    total_steps = steps_per_epoch * epochs

    gen = torch.Generator()
    gen.manual_seed(args['seed'])

    best_val_raw = float('inf')
    best_val_ema = float('inf')
    best_state_raw = None
    best_state_ema = None
    step = 0
    t0 = time.time()
    print(f"[{args['name']}] starting train, steps/epoch={steps_per_epoch}", flush=True)
    for epoch in range(epochs):
        ep_t0 = time.time()
        model.train()
        for batch in train_ds.iter_batches(bs, shuffle=True, drop_last=True, generator=gen):
            cur_lr = cosine_lr(step, warmup, total_steps, lr, min_lr_ratio=0.05)
            for pg in opt.param_groups:
                pg['lr'] = cur_lr
            b = to_dev(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pi, mu, sigma = model(b['cat'], b['cont'], b['tags'])
                loss = mdn_nll(pi, mu, sigma, b['log_pp'], ent_bonus=ent_bonus)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
        train_dur = time.time() - ep_t0
        torch.cuda.synchronize()
        print(f"[{args['name']}] epoch {epoch} train done in {train_dur:.1f}s, evaluating", flush=True)
        val_t0 = time.time()
        val_raw = evaluate_loss(model, val_ds, device, bs * 2)
        print(f"[{args['name']}] epoch {epoch} val_raw {val_raw:.4f} in {time.time()-val_t0:.1f}s", flush=True)
        # eval ema (snapshot, restore)
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                backup[n] = p.detach().clone()
        ema.copy_to(model)
        ve_t0 = time.time()
        val_ema = evaluate_loss(model, val_ds, device, bs * 2)
        print(f"[{args['name']}] epoch {epoch} val_ema {val_ema:.4f} in {time.time()-ve_t0:.1f}s", flush=True)
        if val_ema < best_val_ema - 1e-5:
            best_val_ema = val_ema
            best_state_ema = {}
            for k, v in model.state_dict().items():
                best_state_ema[k] = v.cpu().clone()
        # restore raw
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(backup[n])
        if val_raw < best_val_raw - 1e-5:
            best_val_raw = val_raw
            best_state_raw = {}
            for k, v in model.state_dict().items():
                best_state_raw[k] = v.cpu().clone()
        print(f"[{args['name']}] epoch {epoch} train_dur {train_dur:.1f}s val_raw {val_raw:.4f} val_ema {val_ema:.4f}", flush=True)

    # pick the better of raw vs ema
    if best_val_ema < best_val_raw:
        model.load_state_dict(best_state_ema)
        chosen = 'ema'
        best_val = best_val_ema
    else:
        if best_state_raw is not None:
            model.load_state_dict(best_state_raw)
        chosen = 'raw'
        best_val = best_val_raw

    elapsed = time.time() - t0
    log_line = (f"name={args['name']} bs={bs} lr={lr} dp={dp} wd={wd} sf={sf} K={K} "
                f"hidden={hidden} seed={args['seed']} epochs={epochs} ent_bonus={ent_bonus} "
                f"val={best_val:.4f}({chosen}) time={elapsed:.0f}s")
    with open(log_path, 'a') as f:
        f.write(log_line + '\n')
    print(log_line, flush=True)

    # Save the best ckpt
    out_dir = os.path.join('exports', args['name'])
    os.makedirs(out_dir, exist_ok=True)
    cfg_save = {
        'model_type': 'mdn', 'K': K, 'num_bins': None,
        'hidden': hidden, 'dropout': dp, 'vocab_sizes': vs,
        'emb_dims': emb_dims, 'tag_vocab_size': tag_vocab_size,
        'num_continuous': num_continuous, 'sigma_floor': sf, 'tag_emb_dim': tag_dim,
    }
    ckpt = {'config': cfg_save, 'full_state_dict': model.state_dict(),
            'val_loss': best_val, 'use_ema': (chosen == 'ema'),
            'ema_state': {}}
    for k, v in ema.shadow.items():
        ckpt['ema_state'][k] = v.detach().cpu()
    torch.save(ckpt, os.path.join(out_dir, 'best.pt'))

    # free gpu
    del model, opt, ema, scaler
    if best_state_raw is not None:
        del best_state_raw
    if best_state_ema is not None:
        del best_state_ema
    torch.cuda.empty_cache()
    return {'name': args['name'], 'val': best_val,
            'chosen': chosen, 'time_s': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, required=True,
                        help='JSON list of run configs')
    parser.add_argument('--log', type=str, default='logs/mdn_sweep.log')
    args_main = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg['data']['processed_dir'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading data', flush=True)
    t = time.time()
    train_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'train.parquet'), art)
    val_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'val.parquet'), art)
    test_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'test.parquet'), art, has_extras=True)
    print('loaded in', round(time.time() - t, 1), 's. train/val/test:', len(train_ds), len(val_ds), len(test_ds), flush=True)

    with open(args_main.config_json, 'r') as f:
        runs = json.load(f)

    open(args_main.log, 'w').close()
    results = []
    for r in runs:
        try:
            res = train_one(r, train_ds, val_ds, test_ds, art, device, args_main.log)
            results.append(res)
        except Exception as e:
            print('failed run', r.get('name'), e, flush=True)
    with open('exports/mdn_sweep_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('sweep done. saved exports/mdn_sweep_results.pkl', flush=True)


if __name__ == '__main__':
    main()
