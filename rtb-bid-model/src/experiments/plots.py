import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_pit(pit, out_path, title='PIT histogram'):
    plt.figure(figsize=(6, 4))
    plt.hist(pit, bins=50, range=(0, 1), edgecolor='black')
    plt.xlabel('PIT')
    plt.ylabel('count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_pp_hist(pp, out_path, title='Payprice histogram'):
    plt.figure(figsize=(6, 4))
    plt.hist(pp, bins=80, range=(0, 300), edgecolor='black')
    plt.xlabel('payprice (fen)')
    plt.ylabel('count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_calibration(pit, out_path, title='Calibration'):
    n = len(pit)
    s = np.sort(pit)
    cdf_emp = np.arange(1, n + 1) / n
    plt.figure(figsize=(5, 5))
    plt.plot(s, cdf_emp, label='empirical')
    plt.plot([0,1],[0,1],'k--', label='ideal')
    plt.xlabel('predicted CDF')
    plt.ylabel('empirical CDF')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    d = torch.load(args.preds, map_location='cpu', weights_only=False)
    pit = d['pit'].numpy() if torch.is_tensor(d['pit']) else d['pit']
    pp = d['pp'].numpy() if torch.is_tensor(d['pp']) else d['pp']
    out_dir = 'results/plots'
    os.makedirs(out_dir, exist_ok=True)
    plot_pit(pit, os.path.join(out_dir, f'pit_{args.name}.png'), f'PIT {args.name}')
    plot_calibration(pit, os.path.join(out_dir, f'cal_{args.name}.png'), f'calibration {args.name}')
    plot_pp_hist(pp, os.path.join(out_dir, f'pp_{args.name}.png'), f'payprice {args.name}')
    print('saved plots for', args.name)


if __name__ == '__main__':
    main()
