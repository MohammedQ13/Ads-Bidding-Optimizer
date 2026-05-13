"""Export discrete-bins model to ONNX for the C++ inference server.
Also writes feature_config.json with everything the server needs to
encode raw features (vocab mappings, normalization stats, etc).
"""

import os
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from dataset import load_artifacts
from model import DiscreteBins, build_vocab_sizes


class BinsForExport(torch.nn.Module):
    """Wraps DiscreteBins so the ONNX graph outputs probabilities, not raw logits.
    The C++ server can use these probabilities directly for bid optimization.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, cat, cont, tags):
        logits = self.m(cat, cont, tags)
        return F.softmax(logits, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='exports/bins_300/best.pt')
    parser.add_argument('--out', type=str, default='exports/best_model.onnx')
    parser.add_argument('--feature_config', type=str, default='exports/feature_config.json')
    parser.add_argument('--n_test', type=int, default=8)
    args = parser.parse_args()

    cfg = load_config()
    art = load_artifacts(cfg['data']['processed_dir'])
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    mcfg = ck['config']

    # rebuild model from checkpoint config, dropout=0 for inference
    m = DiscreteBins(
        vocab_sizes=mcfg['vocab_sizes'],
        emb_dims=mcfg['emb_dims'],
        tag_vocab_size=mcfg['tag_vocab_size'],
        tag_emb_dim=mcfg['tag_emb_dim'],
        num_continuous=mcfg['num_continuous'],
        hidden=mcfg['hidden'],
        dropout=0.0,
        num_bins=mcfg['num_bins'],
    )
    m.load_state_dict(ck['full_state_dict'], strict=False)
    m.eval()
    wrap = BinsForExport(m)

    # build dummy inputs matching dataset shape conventions
    B = args.n_test
    cat = torch.zeros(B, len(mcfg['vocab_sizes']), dtype=torch.int64)
    cont = torch.zeros(B, mcfg['num_continuous'], dtype=torch.float32)
    tags = torch.zeros(B, 10, dtype=torch.int64)

    # forward pass to get reference probs
    with torch.no_grad():
        probs_ref = wrap(cat, cont, tags).numpy()

    print('exporting', args.out)
    torch.onnx.export(
        wrap, (cat, cont, tags), args.out,
        input_names=['cat', 'cont', 'tags'],
        output_names=['probs'],
        dynamic_axes={'cat': {0: 'B'}, 'cont': {0: 'B'}, 'tags': {0: 'B'}, 'probs': {0: 'B'}},
        opset_version=17,
    )

    # verify with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        out = sess.run(None, {
            'cat': cat.numpy(),
            'cont': cont.numpy(),
            'tags': tags.numpy(),
        })[0]
        max_diff = float(np.abs(out - probs_ref).max())
        print('onnxruntime verify: max abs diff vs torch =', max_diff)
    except ImportError:
        print('onnxruntime not installed, skipping numerical verify')

    # write feature_config.json: the C++ server reads this to know how to
    # encode raw ad request fields into the same format the model expects
    feat_cfg = {
        'cat_order': list(art['categorical_features']),
        'cat_vocabs': {},
    }
    for feat in art['categorical_features']:
        feat_cfg['cat_vocabs'][feat] = dict(art['encoders'][feat])
    feat_cfg.update({
        'tag_vocab': dict(art['tag_vocab']),
        'cont_cols_order': ['log_floor_price', 'slot_area', 'tag_count',
                            'has_floor_price', 'is_weekend',
                            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos'],
        'cont_means': art['scalers']['mean'],
        'cont_stds': art['scalers']['std'],
        'tag_max_len': 10,
        'tag_pad_idx': 0,
        'num_bins': mcfg['num_bins'],
        'embedding_dims': dict(mcfg['emb_dims']),
        'tag_emb_dim': mcfg['tag_emb_dim'],
    })
    with open(args.feature_config, 'w') as f:
        json.dump(feat_cfg, f, indent=2)
    print('wrote feature config:', args.feature_config)


if __name__ == '__main__':
    main()
