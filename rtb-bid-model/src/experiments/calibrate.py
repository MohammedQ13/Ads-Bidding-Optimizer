"""Isotonic-style PIT calibration. Fits a monotone mapping from observed PIT (val)
to uniform[0,1], then applies to test predictions.

For MDN: shift CDF -> g(CDF) where g is monotone fitted to make val PIT uniform.
For bins: same idea on the cumulative.
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression


def fit_pit_calibrator(val_pit):
    """Returns IsotonicRegression mapping val_pit (CDF at true) -> uniform CDF on [0,1].
    Sort val_pit, target = (rank+1)/N (empirical CDF of PIT).
    """
    n = len(val_pit)
    s = np.sort(val_pit)
    target = (np.arange(1, n + 1) / n).astype('float32')
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(s, target)
    return iso


def apply_iso_to_cdf(iso, cdf_arr):
    """Apply the isotonic on a (N, B) cdf array. cdf_arr in [0,1]."""
    flat = cdf_arr.reshape(-1)
    out = iso.transform(flat).reshape(cdf_arr.shape)
    out = np.clip(out, 0.0, 1.0)
    out = np.maximum.accumulate(out, axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_val', type=str, required=True,
                        help='Path to val preds (with pi/mu/sigma or probs)')
    parser.add_argument('--preds_test', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    val = torch.load(args.preds_val, map_location='cpu', weights_only=False)
    test = torch.load(args.preds_test, map_location='cpu', weights_only=False)

    pit_val = val['pit'].numpy() if torch.is_tensor(val['pit']) else val['pit']
    iso = fit_pit_calibrator(pit_val)
    print('fitted isotonic. checking val PIT remapped')
    val_pit_remap = iso.transform(pit_val)
    n = len(val_pit_remap)
    s = np.sort(val_pit_remap)
    cdf_emp = np.arange(1, n + 1) / n
    ks_after = max(np.abs(cdf_emp - s).max(), np.abs(s - np.arange(0, n)/n).max())
    print('val KS after iso:', ks_after)

    # apply on test - need to remap CDF arrays
    if 'probs' in test:
        cdf = torch.cumsum(test['probs'], dim=1).numpy()
        cdf_cal = apply_iso_to_cdf(iso, cdf)
        probs_cal = np.diff(cdf_cal, prepend=0, axis=1)
        probs_cal = np.clip(probs_cal, 1e-10, None)
        probs_cal = probs_cal / probs_cal.sum(axis=1, keepdims=True)
        out = {
            'probs': torch.from_numpy(probs_cal.astype('float32')),
            'pp': test['pp'],
            'bid': test['bid'],
            'log_pp': test['log_pp'],
            'adv': test.get('adv'),
        }
    else:
        # MDN: convert to bins probs first, then calibrate, then save as bins
        from ensemble import mdn_to_bins_probs
        probs = mdn_to_bins_probs(test['pi_logits'], test['mu'], test['sigma'], num_bins=300).numpy()
        cdf = np.cumsum(probs, axis=1)
        cdf_cal = apply_iso_to_cdf(iso, cdf)
        probs_cal = np.diff(cdf_cal, prepend=0, axis=1)
        probs_cal = np.clip(probs_cal, 1e-10, None)
        probs_cal = probs_cal / probs_cal.sum(axis=1, keepdims=True)
        out = {
            'probs': torch.from_numpy(probs_cal.astype('float32')),
            'pp': test['pp'],
            'bid': test['bid'],
            'log_pp': test['log_pp'],
            'adv': test.get('adv'),
        }

    torch.save(out, args.out)
    print('saved calibrated preds to', args.out)


if __name__ == '__main__':
    main()
