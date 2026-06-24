"""Sliding-window retrainer for the feedback loop.

Reads the auction outcomes the Go engine logs, keeps the most recent W of them,
fine-tunes the bins model on them (warm-started from the iPinYou model), and
exports a new ONNX file. The C++ servers watch that file and hot-swap it in.

The model classes here are copied from rtb-bid-model/src/model.py on purpose:
they must match exactly so the warm-start checkpoint loads cleanly.

Run a single round (for testing):
    python retrain.py --once
Run the loop (in the compose demo):
    python retrain.py
"""

import os
import json
import math
import time
import argparse
import datetime

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- model (copied from rtb-bid-model/src/model.py so weights load) -----

class CategoricalEmbeddings(nn.Module):
    """Embedding tables for the categorical features and the user tags."""

    def __init__(self, vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim):
        super().__init__()
        self.embs = nn.ModuleList()
        self.feat_names = list(vocab_sizes.keys())
        out_dim = 0
        for name in self.feat_names:
            self.embs.append(nn.Embedding(vocab_sizes[name], emb_dims[name]))
            out_dim += emb_dims[name]
        self.tag_emb = nn.Embedding(tag_vocab_size, tag_emb_dim, padding_idx=0)
        self.tag_dim = tag_emb_dim
        self.out_dim = out_dim + tag_emb_dim

    def forward(self, cat, tags):
        parts = []
        for i, e in enumerate(self.embs):
            parts.append(e(cat[:, i]))
        tag_e = self.tag_emb(tags)
        mask = (tags != 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        bag = (tag_e * mask).sum(dim=1) / denom
        parts.append(bag)
        return torch.cat(parts, dim=1)


class MLPBackbone(nn.Module):
    """Shared MLP: Linear -> LayerNorm -> GELU -> Dropout per layer."""

    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class DiscreteBins(nn.Module):
    """Softmax over price bins. Predicts P(payprice == k) for each bin k."""

    def __init__(self, vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim,
                 num_continuous, hidden, dropout, num_bins):
        super().__init__()
        self.emb = CategoricalEmbeddings(vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim)
        self.cont_norm = nn.Identity()
        in_dim = self.emb.out_dim + num_continuous
        self.backbone = MLPBackbone(in_dim, hidden, dropout)
        self.num_bins = num_bins
        self.head = nn.Linear(self.backbone.out_dim, num_bins)

    def forward(self, cat, cont, tags):
        e = self.emb(cat, tags)
        c = self.cont_norm(cont)
        x = torch.cat([e, c], dim=1)
        h = self.backbone(x)
        return self.head(h)


class BinsForExport(nn.Module):
    """Wraps the model so the ONNX graph outputs probabilities (softmax baked in),
    exactly like the original export so the C++ server sees the same thing.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, cat, cont, tags):
        return F.softmax(self.m(cat, cont, tags), dim=-1)


# ----- feature encoding (matches the C++ feature store / training) -----

def weekday_monday0(ts):
    """Monday=0..Sunday=6 from a YYYYMMDDHHmmss timestamp."""
    date = ts // 1000000
    y = int(date // 10000)
    mo = int((date // 100) % 100)
    d = int(date % 100)
    try:
        return datetime.datetime(y, mo, d).weekday()
    except ValueError:
        return 0


def encode_record(rec, cfg):
    """Turn one outcome record into (cat[9], cont[9], tags[10], payprice_bin)."""
    cat = []
    for feat in cfg["cat_order"]:
        vocab = cfg["cat_vocabs"][feat]
        if feat == "domain" or feat == "advertiser_id":
            key = str(rec[feat])
        else:
            key = str(int(rec[feat]))
        cat.append(int(vocab.get(key, 0)))

    means = cfg["cont_means"]
    stds = cfg["cont_stds"]
    floor = max(0.0, float(rec["slot_floor_price"]))
    log_floor = math.log1p(floor)
    slot_area = float(rec["slot_width"]) * float(rec["slot_height"])
    tags_list = rec.get("user_tags") or []
    tag_count = float(len(tags_list))

    cont = [
        (log_floor - means["log_floor_price"]) / stds["log_floor_price"],
        (slot_area - means["slot_area"]) / stds["slot_area"],
        (tag_count - means["tag_count"]) / stds["tag_count"],
        1.0 if float(rec["slot_floor_price"]) > 0 else 0.0,
    ]
    ts = int(rec["timestamp"])
    hour = int((ts % 1000000) // 10000)
    wd = weekday_monday0(ts)
    cont.append(1.0 if wd >= 5 else 0.0)
    cont.append(math.sin(2 * math.pi * hour / 24.0))
    cont.append(math.cos(2 * math.pi * hour / 24.0))
    cont.append(math.sin(2 * math.pi * wd / 7.0))
    cont.append(math.cos(2 * math.pi * wd / 7.0))

    vocab = cfg["tag_vocab"]
    tag_ids = [0] * cfg["tag_max_len"]
    j = 0
    for t in tags_list:
        if j >= cfg["tag_max_len"]:
            break
        tid = int(vocab.get(str(t), 0))
        if tid == 0:
            continue
        tag_ids[j] = tid
        j += 1

    return cat, cont, tag_ids


def read_window(path, window):
    """Read the last `window` lines of the outcome log."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if len(rows) > window:
        rows = rows[-window:]
    return rows


def build_tensors(rows, cfg, num_bins):
    """Encode a window of records into model input tensors + the price bin label."""
    cats = []
    conts = []
    tagss = []
    labels = []
    for rec in rows:
        cat, cont, tags = encode_record(rec, cfg)
        cats.append(cat)
        conts.append(cont)
        tagss.append(tags)
        # the label is the clearing price, rounded to a bin (0..num_bins-1)
        cp = int(round(float(rec["clearing_price"])))
        if cp < 0:
            cp = 0
        if cp > num_bins - 1:
            cp = num_bins - 1
        labels.append(cp)
    cat_t = torch.tensor(cats, dtype=torch.int64)
    cont_t = torch.tensor(conts, dtype=torch.float32)
    tags_t = torch.tensor(tagss, dtype=torch.int64)
    lab_t = torch.tensor(labels, dtype=torch.int64)
    return cat_t, cont_t, tags_t, lab_t


def load_model(ckpt_path):
    """Build the model from the warm-start checkpoint and load its weights."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = DiscreteBins(
        vocab_sizes=c["vocab_sizes"],
        emb_dims=c["emb_dims"],
        tag_vocab_size=c["tag_vocab_size"],
        tag_emb_dim=c["tag_emb_dim"],
        num_continuous=c["num_continuous"],
        hidden=c["hidden"],
        dropout=0.1,
        num_bins=c["num_bins"],
    )
    m.load_state_dict(ck["full_state_dict"], strict=False)
    return m, c["num_bins"]


def export_onnx(model, out_path):
    """Export the fine-tuned model to ONNX as a single self-contained file,
    written atomically so the C++ watcher only ever sees a complete model.

    Newer torch exporters can split the weights into a separate .data file. The
    C++ server loads one file, so we load the exported model back and re-save it
    with the weights embedded, then move it into place in one atomic step.
    """
    model.eval()
    wrap = BinsForExport(model)
    cat = torch.zeros(2, 9, dtype=torch.int64)
    cont = torch.zeros(2, 9, dtype=torch.float32)
    tags = torch.zeros(2, 10, dtype=torch.int64)

    raw = out_path + ".raw.onnx"
    torch.onnx.export(
        wrap, (cat, cont, tags), raw,
        input_names=["cat", "cont", "tags"],
        output_names=["probs"],
        dynamic_axes={"cat": {0: "B"}, "cont": {0: "B"}, "tags": {0: "B"}, "probs": {0: "B"}},
        opset_version=17,
    )
    # load (pulls in any external weights) and re-save embedded into one file
    m = onnx.load(raw)
    embed = out_path + ".embed.tmp"
    onnx.save(m, embed)
    os.replace(embed, out_path)
    # clean up the raw export and its possible external data file
    for p in (raw, raw + ".data"):
        if os.path.exists(p):
            os.remove(p)


def write_status(args, state, status, rows, last_loss):
    """Write a small JSON status file the Go engine reads and surfaces in the UI.
    Written atomically so the reader never sees a half-written file.
    """
    if not args.status:
        return
    payload = {
        "round": state["round"],
        "exports": state["exports"],
        "rows": rows,
        "last_loss": last_loss,
        "exported_at": state["exported_at"],
        "model_version": float(state["exports"]),
        "window": args.window,
        "status": status,
    }
    try:
        tmp = args.status + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, args.status)
    except Exception as e:
        print("status write failed:", e, flush=True)


def train_round(cfg, args, state):
    """One retraining round: read window, fine-tune, export. Returns rows used."""
    state["round"] += 1
    rows = read_window(args.outcome_log, args.window)
    if len(rows) < args.min_rows:
        print("only", len(rows), "rows, need", args.min_rows, "-- skipping", flush=True)
        write_status(args, state, "waiting", len(rows), state["last_loss"])
        return 0

    write_status(args, state, "training", len(rows), state["last_loss"])
    model, num_bins = load_model(args.warm_start)
    cat_t, cont_t, tags_t, lab_t = build_tensors(rows, cfg, num_bins)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    n = cat_t.shape[0]
    bs = args.batch_size
    last_loss = state["last_loss"]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0
        steps = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(cat_t[idx], cont_t[idx], tags_t[idx])
            loss = F.cross_entropy(logits, lab_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        last_loss = round(total / max(1, steps), 4)
        print("epoch", epoch, "loss", last_loss, flush=True)

    export_onnx(model, args.out)
    state["exports"] += 1
    state["last_loss"] = last_loss
    state["exported_at"] = int(time.time())
    write_status(args, state, "exported", n, last_loss)
    print("exported new model to", args.out, "from", n, "outcomes", flush=True)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outcome_log", default="/data/outcomes.jsonl")
    p.add_argument("--feature_config", default="models/feature_config.json")
    p.add_argument("--warm_start", default="models/warm_start.pt")
    p.add_argument("--out", default="/models/bid_model.onnx")
    p.add_argument("--window", type=int, default=200000)
    p.add_argument("--min_rows", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--interval", type=int, default=60, help="seconds between rounds")
    p.add_argument("--status", default="/models/retrainer_status.json",
                   help="JSON status file the engine reads for the UI")
    p.add_argument("--once", action="store_true", help="run one round and exit")
    args = p.parse_args()

    with open(args.feature_config) as f:
        cfg = json.load(f)

    # state carried across rounds so the status file shows totals, not just the
    # current round
    state = {"round": 0, "exports": 0, "last_loss": 0.0, "exported_at": 0}

    if args.once:
        train_round(cfg, args, state)
        return

    print("retrainer loop: window", args.window, "every", args.interval, "s", flush=True)
    while True:
        try:
            train_round(cfg, args, state)
        except Exception as e:
            print("retrain round failed:", e, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
