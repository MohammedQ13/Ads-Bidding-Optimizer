"""Build the final comparison table from all saved eval pkls + ensembles + budget pkls."""
import os
import pickle
import argparse


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='exports/final_table.txt')
    args = parser.parse_args()

    rows = []
    rows.append('Model | ANLP | KS | Regret V=150 grid | Regret V=150 Newton | Regret V=bid grid | Regret V=bid Newton')
    rows.append('---|---|---|---|---|---|---')
    # individual models
    for n in sorted(os.listdir('exports/preds_eval')) if os.path.isdir('exports/preds_eval') else []:
        if not n.endswith('.pkl'):
            continue
        d = load_pkl(os.path.join('exports/preds_eval', n))
        n150 = d.get('regret_v150_newton', None)
        nb = d.get('regret_vbid_newton', None)
        rows.append(f"{d['name']} | {d['anlp']:.3f} | {d['ks']:.3f} | {d['regret_v150_grid']:.3f} | {('-' if n150 is None else f'{n150:.3f}')} | {d['regret_vbid_grid']:.3f} | {('-' if nb is None else f'{nb:.3f}')}")
    # ensembles
    if os.path.isdir('exports/ensembles'):
        for n in sorted(os.listdir('exports/ensembles')):
            if not n.endswith('.pkl'):
                continue
            d = load_pkl(os.path.join('exports/ensembles', n))
            rows.append(f"ENS {d['name']} | {d['anlp']:.3f} | {d['ks']:.3f} | {d['regret_v150_grid']:.3f} | - | {d['regret_vbid_grid']:.3f} | -")
    # budget
    rows.append('')
    rows.append('Budget runs:')
    if os.path.isdir('exports/budget'):
        rows.append('Model | budget 1/32 (won/profit/clicks) | 1/8 | 1/2 | unconstrained')
        rows.append('---|---|---|---|---')
        for n in sorted(os.listdir('exports/budget')):
            if not n.endswith('.pkl'):
                continue
            d = load_pkl(os.path.join('exports/budget', n))
            cells = [d['name']]
            for f in ['0.0312', '0.1250', '0.5000', '1.0000']:
                m = d['budgets'].get(f)
                if m is None:
                    cells.append('-')
                else:
                    cells.append(f"won={m['won']} prof={m['profit']:.0f} clk={m['clicks_won']}")
            rows.append(' | '.join(cells))

    text = '\n'.join(rows)
    print(text)
    with open(args.out, 'w') as f:
        f.write(text + '\n')
    print('saved', args.out)


if __name__ == '__main__':
    main()
