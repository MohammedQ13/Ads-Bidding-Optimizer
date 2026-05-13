import os
import math
import time
import argparse
import numpy as np
import torch

from config import load_config
from dataset import BidDataset, load_artifacts
from model import MDN, DiscreteBins, build_vocab_sizes
from loss import mdn_nll, discrete_bins_nll, discrete_bins_smoothed_nll


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class EMA:
    """Exponential moving average of model weights.
    shadow = decay * shadow + (1 - decay) * current weights each step.
    Smooths out noisy updates, often generalizes better than raw weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        # snapshot of each parameter at init time
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    def update(self, model):
        # blend current weights into the shadow: shadow = decay*shadow + (1-decay)*param
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        # overwrite model weights with the smoothed shadow weights
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def state_dict(self):
        out = {}
        for k, v in self.shadow.items():
            out[k] = v.clone()
        return out


def cosine_lr(step, warmup, total, base_lr, min_lr_ratio=0.05):
    """Cosine annealing learning rate with linear warmup.
    First `warmup` steps: linearly ramp from 0 to base_lr.
    After that: cosine decay from base_lr down to base_lr * min_lr_ratio.
    """
    if step < warmup:
        # linear warmup: avoids large updates with random weights
        return base_lr * (step + 1) / warmup
    # cosine decay phase
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def to_dev(d, device):
    """Move all tensors in a dict to the target device."""
    out = {}
    for k, v in d.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def evaluate_loss(model, ds, device, model_type, batch_size, num_bins=None):
    """Compute average loss over a full dataset (used for validation).
    Returns the mean NLL (for MDN) or cross-entropy (for bins).
    """
    model.eval()
    total = 0.0
    cnt = 0
    with torch.no_grad():
        for batch in ds.iter_batches(batch_size, shuffle=False):
            b = to_dev(batch, device)
            if model_type == 'mdn':
                pi, mu, sigma = model(b['cat'], b['cont'], b['tags'])
                loss = mdn_nll(pi, mu, sigma, b['log_pp'])
            else:
                # bins target is the integer payprice clamped to valid range
                pp_int = b['pp'].long().clamp(0, num_bins - 1)
                logits = model(b['cat'], b['cont'], b['tags'])
                loss = discrete_bins_nll(logits, pp_int)
            total += loss.item() * b['cat'].size(0)
            cnt += b['cat'].size(0)
    return total / cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mdn', choices=['mdn', 'bins'])
    parser.add_argument('--name', type=str, default='mdn_v1')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--num_bins', type=int, default=None)
    parser.add_argument('--smooth_sigma', type=float, default=0.0)
    parser.add_argument('--hidden', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--ema_decay', type=float, default=None)
    parser.add_argument('--ent_bonus', type=float, default=0.0)
    parser.add_argument('--target_jitter', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--sigma_floor', type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg['training']['seed']
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    art = load_artifacts(cfg['data']['processed_dir'])
    vocab_sizes = build_vocab_sizes(art)
    emb_dims = art['embedding_dims']
    tag_dim = emb_dims['user_tags']
    tag_vocab_size = emb_dims['_tag_vocab_size']

    bs = args.batch_size if args.batch_size is not None else cfg['training']['batch_size']
    epochs = args.epochs if args.epochs is not None else cfg['training']['max_epochs']
    base_lr = args.lr if args.lr is not None else cfg['training']['learning_rate']
    ema_decay = args.ema_decay if args.ema_decay is not None else cfg['training']['ema_decay']
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg['training']['weight_decay']
    sigma_floor_override = args.sigma_floor

    if args.hidden is not None:
        hidden = []
        for x in args.hidden.split(','):
            hidden.append(int(x))
    else:
        if args.model == 'mdn':
            hidden = cfg['mdn']['hidden_layers']
        else:
            hidden = cfg['bins']['hidden_layers']

    if args.dropout is not None:
        dropout = args.dropout
    else:
        if args.model == 'mdn':
            dropout = cfg['mdn']['dropout']
        else:
            dropout = cfg['bins']['dropout']

    print('loading data', flush=True)
    t0 = time.time()
    train_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'train.parquet'), art)
    val_ds = BidDataset(os.path.join(cfg['data']['processed_dir'], 'val.parquet'), art)
    print('train:', len(train_ds), 'val:', len(val_ds), 'load secs:', round(time.time()-t0,1), flush=True)
    num_continuous = train_ds.num_continuous()

    if args.model == 'mdn':
        K = args.K if args.K is not None else cfg['mdn']['n_components']
        sf = sigma_floor_override if sigma_floor_override is not None else cfg['mdn']['sigma_floor']
        model = MDN(
            vocab_sizes=vocab_sizes, emb_dims=emb_dims,
            tag_vocab_size=tag_vocab_size, tag_emb_dim=tag_dim,
            num_continuous=num_continuous, hidden=hidden,
            dropout=dropout, K=K, sigma_floor=sf,
        )
        num_bins = None
    else:
        num_bins = args.num_bins if args.num_bins is not None else cfg['bins']['num_bins']
        model = DiscreteBins(
            vocab_sizes=vocab_sizes, emb_dims=emb_dims,
            tag_vocab_size=tag_vocab_size, tag_emb_dim=tag_dim,
            num_continuous=num_continuous, hidden=hidden,
            dropout=dropout, num_bins=num_bins,
        )
        K = None

    model = model.to(device)
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    ema = EMA(model, decay=ema_decay)
    # mixed precision: fp16 forward/backward, fp32 optimizer step
    # GradScaler prevents underflow in fp16 gradients
    scaler = torch.amp.GradScaler('cuda')

    steps_per_epoch = train_ds.num_batches(bs, drop_last=True)
    total_steps = steps_per_epoch * epochs
    warmup = cfg['training']['warmup_steps']
    grad_clip = cfg['training']['grad_clip']

    out_dir = os.path.join('exports', args.name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f'train_{args.name}.log')
    logf = open(log_path, 'w')
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log(f'name={args.name} model={args.model} seed={seed} params={n_params}')
    log(f'epochs={epochs} bs={bs} lr={base_lr} hidden={hidden} dropout={dropout} ema={ema_decay}')
    if args.model == 'mdn':
        log(f'K={K} sigma_floor={cfg["mdn"]["sigma_floor"]}')
    else:
        log(f'num_bins={num_bins} smooth_sigma={args.smooth_sigma}')
    log(f'steps/epoch={steps_per_epoch} total_steps={total_steps}')

    best_val = float('inf')
    bad = 0
    step = 0
    gen = torch.Generator()
    gen.manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        running_n = 0
        et0 = time.time()
        batch_idx = 0
        for batch in train_ds.iter_batches(bs, shuffle=True, drop_last=True, generator=gen):
            # update learning rate every step (cosine schedule)
            lr = cosine_lr(step, warmup, total_steps, base_lr, min_lr_ratio=0.05)
            for pg in opt.param_groups:
                pg['lr'] = lr

            b = to_dev(batch, device)
            opt.zero_grad(set_to_none=True)
            # autocast: run forward pass in fp16 for speed
            with torch.amp.autocast('cuda', dtype=torch.float16):
                if args.model == 'mdn':
                    pi, mu, sigma = model(b['cat'], b['cont'], b['tags'])
                    loss = mdn_nll(pi, mu, sigma, b['log_pp'], ent_bonus=args.ent_bonus, target_jitter=args.target_jitter)
                else:
                    pp_int = b['pp'].long().clamp(0, num_bins - 1)
                    logits = model(b['cat'], b['cont'], b['tags'])
                    if args.smooth_sigma > 0:
                        loss = discrete_bins_smoothed_nll(logits, pp_int, sigma=args.smooth_sigma)
                    else:
                        loss = discrete_bins_nll(logits, pp_int)

            # backward pass: scale loss to prevent fp16 underflow,
            # unscale before clipping, then step the optimizer
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            # update EMA shadow weights after each optimizer step
            ema.update(model)

            running += loss.item() * b['cat'].size(0)
            running_n += b['cat'].size(0)
            step += 1
            batch_idx += 1

            if batch_idx % 100 == 0:
                log(f'epoch {epoch} step {batch_idx}/{steps_per_epoch} lr {lr:.5f} loss {loss.item():.4f}')

        train_loss = running / running_n

        # evaluate on both raw and EMA weights to see which is better
        # raw weights: the actual model parameters after gradient updates
        val_loss_raw = evaluate_loss(model, val_ds, device, args.model, batch_size=bs * 2, num_bins=num_bins)

        # EMA weights: smoothed version, often better late in training
        # temporarily swap EMA weights in, evaluate, then restore raw weights
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                backup[n] = p.detach().clone()
        ema.copy_to(model)
        val_loss_ema = evaluate_loss(model, val_ds, device, args.model, batch_size=bs * 2, num_bins=num_bins)
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(backup[n])

        # pick whichever val loss is lower for checkpoint selection
        val_loss = min(val_loss_raw, val_loss_ema)
        use_ema = val_loss_ema <= val_loss_raw

        epoch_time = time.time() - et0
        log(f'epoch {epoch} train_loss {train_loss:.4f} val_raw {val_loss_raw:.4f} val_ema {val_loss_ema:.4f} time {epoch_time:.1f}s')

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad = 0
            # new best: save checkpoint with whichever weights are better
            if use_ema:
                # temporarily swap in EMA weights for saving
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(ema.shadow[n])
            # save everything needed to reconstruct and evaluate the model
            ckpt = {
                'ema_state': ema.state_dict(),
                'val_loss': val_loss,
                'use_ema': use_ema,
                'config': {
                    'model_type': args.model,
                    'K': K,
                    'num_bins': num_bins,
                    'hidden': hidden,
                    'dropout': dropout,
                    'vocab_sizes': vocab_sizes,
                    'emb_dims': emb_dims,
                    'tag_vocab_size': tag_vocab_size,
                    'num_continuous': num_continuous,
                    'sigma_floor': cfg['mdn']['sigma_floor'] if args.model == 'mdn' else None,
                    'tag_emb_dim': tag_dim,
                },
                'full_state_dict': model.state_dict(),
            }
            torch.save(ckpt, os.path.join(out_dir, 'best.pt'))
            log(f'saved best ckpt val={val_loss:.4f} (use_ema={use_ema})')
            # restore raw weights for next epoch's training
            if use_ema:
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(backup[n])
        else:
            bad += 1
            log(f'no improve ({bad}/{cfg["training"]["early_stopping_patience"]})')
            if bad >= cfg['training']['early_stopping_patience']:
                log('early stop')
                break

    logf.close()
    print('done. best val:', best_val, flush=True)


if __name__ == '__main__':
    main()
