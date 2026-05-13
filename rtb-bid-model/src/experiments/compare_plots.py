"""Small comparison plot showing PIT for each model."""
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = 'results/plots/comparison_pit.png'
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    paths = [
        ('exports/mdn_s42/preds_test.pt', 'MDN K=12'),
        ('exports/bins_300/preds_test.pt', 'DLF bins seed=42'),
        ('exports/bins_300_smooth/preds_test.pt', 'DLF bins smoothed'),
        ('exports/bins_300_s1/preds_test.pt', 'DLF bins seed=1'),
    ]
    for ax, (path, name) in zip(axes.flat, paths):
        if not os.path.exists(path):
            continue
        d = torch.load(path, map_location='cpu', weights_only=False)
        pit = d['pit'].numpy() if torch.is_tensor(d['pit']) else d['pit']
        ax.hist(pit, bins=40, range=(0, 1), edgecolor='black')
        ax.set_title(name)
        ax.set_xlabel('PIT')
        ax.axhline(len(pit) / 40, color='red', ls='--', label='uniform')
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print('saved', out)

    # also calibration overlay
    plt.figure(figsize=(7, 7))
    for path, name in paths:
        if not os.path.exists(path):
            continue
        d = torch.load(path, map_location='cpu', weights_only=False)
        pit = d['pit'].numpy() if torch.is_tensor(d['pit']) else d['pit']
        s = np.sort(pit)
        emp = np.arange(1, len(s) + 1) / len(s)
        plt.plot(s, emp, label=name)
    plt.plot([0, 1], [0, 1], 'k--', label='ideal')
    plt.xlabel('predicted CDF')
    plt.ylabel('empirical CDF')
    plt.title('Calibration comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/plots/comparison_calibration.png', dpi=110)
    plt.close()
    print('saved results/plots/comparison_calibration.png')


if __name__ == '__main__':
    main()
