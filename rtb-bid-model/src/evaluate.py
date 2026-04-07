import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append("src")
from dataset import get_dataloader, BidDataset
from model import BidTransformer, MLPBaseline, load_artifacts
from loss import NLLLoss
from bid_optimizer import compute_optimal_bid, compute_naive_bid, compute_bid_regret, IMPRESSION_VALUE
from torch.utils.data import DataLoader

CHECKPOINT_DIR = "exports"
PLOT_DIR = "results/plots"
PROCESSED_DIR = "data/processed"
BATCH_SIZE = 1024
PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def load_model(model_class, checkpoint_path, artifacts, device):
    model = model_class(artifacts)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"loaded {checkpoint_path} epoch {checkpoint['epoch']} val_loss={checkpoint['val_loss']:.4f}")
    return model


def run_transformer_inference(model, loader, device):
    # runs the full test set through the transformer, collecting mu, sigma, and raw targets
    all_mu = []
    all_sigma = []
    all_log_targets = []
    all_raw_targets = []

    with torch.no_grad():
        for cats, conts, tags, log_targets, raw_targets in loader:
            cats = cats.to(device)
            conts = conts.to(device)
            tags = tags.to(device)
            mu, sigma = model(cats, conts, tags)
            all_mu.append(mu.cpu())
            all_sigma.append(sigma.cpu())
            all_log_targets.append(log_targets)
            all_raw_targets.append(raw_targets)

    all_mu = torch.cat(all_mu)
    all_sigma = torch.cat(all_sigma)
    all_log_targets = torch.cat(all_log_targets)
    all_raw_targets = torch.cat(all_raw_targets)
    return all_mu, all_sigma, all_log_targets, all_raw_targets


def run_mlp_inference(model, loader, device):
    # runs the full test set through the MLP, collecting point predictions and targets
    all_preds = []
    all_log_targets = []
    all_raw_targets = []

    with torch.no_grad():
        for cats, conts, tags, log_targets, raw_targets in loader:
            cats = cats.to(device)
            conts = conts.to(device)
            tags = tags.to(device)
            pred = model(cats, conts, tags)
            all_preds.append(pred.cpu())
            all_log_targets.append(log_targets)
            all_raw_targets.append(raw_targets)

    all_preds = torch.cat(all_preds)
    all_log_targets = torch.cat(all_log_targets)
    all_raw_targets = torch.cat(all_raw_targets)
    return all_preds, all_log_targets, all_raw_targets


def compute_held_out_nll(mu, sigma, log_targets):
    # NLL on the test set -- this is my main metric for distribution quality
    sigma_clamped = sigma.clamp(min=1e-3, max=10.0)
    nll = torch.log(sigma_clamped) + (log_targets - mu) ** 2 / (2 * sigma_clamped ** 2)
    return nll.mean().item()


def plot_calibration_curve(mu, sigma, log_targets, model_name):
    # calibration check: does the predicted p-th percentile actually contain p% of real prices?
    # a well-calibrated model's 70th percentile prediction should contain roughly 70% of observed prices
    from scipy.stats import norm

    mu_np = mu.numpy()
    sigma_np = sigma.numpy()
    log_targets_np = log_targets.numpy()

    empirical_coverages = []
    for p in PERCENTILES:
        # compute the p-th percentile of the predicted log-normal for each sample
        predicted_pth = norm.ppf(p / 100, loc=mu_np, scale=sigma_np)
        # what fraction of actual log prices fall below this predicted percentile
        empirical_coverage = (log_targets_np <= predicted_pth).mean()
        empirical_coverages.append(empirical_coverage * 100)

    plt.figure(figsize=(6, 6))
    plt.plot(PERCENTILES, empirical_coverages, "o-", label="Model", color="steelblue")
    plt.plot([0, 100], [0, 100], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted percentile")
    plt.ylabel("Empirical coverage (%)")
    plt.title(f"Calibration Curve -- {model_name}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")

    print(f"\ncalibration {model_name}")
    for p, ec in zip(PERCENTILES, empirical_coverages):
        diff = ec - p
        print(f"predicted {p:2d}th percentile empirical {ec:.1f}% error={diff:+.1f}%")

    return empirical_coverages


def plot_bid_regret_comparison(transformer_regret, mlp_regret, naive_regret):
    # the headline result: bid regret distribution for all three approaches
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Bid Regret Distribution (lower is better)", fontsize=13)

    for ax, regret, name, color in zip(
        axes,
        [transformer_regret, mlp_regret, naive_regret],
        ["Transformer + NLL", "MLP + MSE", "Naive Average"],
        ["steelblue", "darkorange", "gray"]
    ):
        regret_np = regret.numpy()
        ax.hist(regret_np, bins=80, color=color, edgecolor="none", alpha=0.8)
        ax.axvline(regret_np.mean(), color="red", linewidth=2, label=f"mean={regret_np.mean():.2f}")
        ax.set_title(name)
        ax.set_xlabel("Regret (CNY fen)")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "bid_regret_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def plot_bid_landscape(mu, sigma, log_targets, raw_targets, n_samples=3):
    # visualize the predicted distribution vs actual price for a few test samples
    # lets me see whether the uncertainty estimates look sensible
    from scipy.stats import lognorm

    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4))
    fig.suptitle("Predicted vs Empirical Bid Landscape (Sample Auctions)", fontsize=12)

    indices = np.random.choice(len(mu), n_samples, replace=False)

    for ax, idx in zip(axes, indices):
        mu_val = mu[idx].item()
        sigma_val = sigma[idx].item()
        actual_price = raw_targets[idx].item()

        # plot the predicted log-normal distribution
        x = np.linspace(1, 300, 500)
        pdf = lognorm.pdf(x, s=sigma_val, scale=np.exp(mu_val))
        ax.plot(x, pdf, "b-", linewidth=2, label="Predicted dist")

        # mark the actual clearing price
        ax.axvline(actual_price, color="red", linewidth=2, linestyle="--", label=f"Actual={actual_price:.0f}")

        # compute and mark the optimal bid
        mu_t = torch.tensor([mu_val])
        sigma_t = torch.tensor([sigma_val])
        optimal_bid = compute_optimal_bid(mu_t, sigma_t)[0].item()
        ax.axvline(optimal_bid, color="green", linewidth=2, linestyle="--", label=f"Optimal bid={optimal_bid:.0f}")

        ax.set_xlabel("payprice (CNY fen)")
        ax.set_ylabel("Density")
        ax.set_title(f"Sample auction {idx}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "bid_landscape_samples.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def per_advertiser_ks_test(mu, sigma, log_targets, test_df):
    # check the log-normal assumption per advertiser
    # I'm curious if it holds better per-advertiser since each has different dynamics
    print("\nper advertiser ks test")
    advertisers = test_df["advertiser_id"].unique()

    for adv in sorted(advertisers):
        mask = torch.tensor(test_df["advertiser_id"].values == adv)
        adv_log_targets = log_targets[mask].numpy()
        adv_mu = mu[mask].numpy()
        adv_sigma = sigma[mask].numpy()

        # compare empirical log prices to a fitted normal
        fitted_mu = adv_mu.mean()
        fitted_sigma = adv_sigma.mean()
        ks_stat, ks_pval = stats.kstest(adv_log_targets, "norm", args=(fitted_mu, fitted_sigma))
        print(f"advertiser {adv} n={mask.sum():,} ks={ks_stat:.4f} p={ks_pval:.4f} {'reject' if ks_pval < 0.05 else 'cannot reject'}")

def per_advertiser_bid_regret(transformer_bids, mlp_bids, naive_bids, raw_targets, test_df):
    # break down bid regret per advertiser to see where each model wins and loses
    print("\nper advertiser bid regret")
    print(f"{'advertiser':<15} {'n':>8} {'transformer':>14} {'mlp':>10} {'naive':>10}")
    print("-" * 60)

    advertiser_col = test_df["advertiser_id"].values

    for adv in sorted(test_df["advertiser_id"].unique()):
        mask = torch.tensor(advertiser_col == adv)
        t_regret = compute_bid_regret(transformer_bids[mask], raw_targets[mask]).mean().item()
        m_regret = compute_bid_regret(mlp_bids[mask], raw_targets[mask]).mean().item()
        n_regret = compute_bid_regret(naive_bids[mask], raw_targets[mask]).mean().item()
        n = mask.sum().item()
        print(f"{adv:<15} {n:>8,} {t_regret:>14.2f} {m_regret:>10.2f} {n_regret:>10.2f}")


def report_sigma_stats(sigma, test_df):
    # sigma distribution on the test set -- useful for setting C++ circuit breaker thresholds
    # if sigma is way outside these ranges at inference time something is probably broken
    print("\nsigma distribution on test set")
    sigma_np = sigma.numpy()
    print(f"mean {sigma_np.mean():.4f}")
    print(f"std {sigma_np.std():.4f}")
    print(f"min {sigma_np.min():.6f}")
    print(f"max {sigma_np.max():.4f}")
    print(f"p5 {np.percentile(sigma_np, 5):.4f}")
    print(f"p25 {np.percentile(sigma_np, 25):.4f}")
    print(f"p50 {np.percentile(sigma_np, 50):.4f}")
    print(f"p75 {np.percentile(sigma_np, 75):.4f}")
    print(f"p95 {np.percentile(sigma_np, 95):.4f}")
    print(f"fraction below 0.1 {(sigma_np < 0.1).mean()*100:.2f}%")
    print(f"fraction above 2.0 {(sigma_np > 2.0).mean()*100:.2f}%")
    print("\nc++ circuit breaker should flag sigma outside [0.05, 3.0]")


def print_summary(transformer_regret, mlp_regret, naive_regret, nll):
    print("\nevaluation summary")
    print(f"held-out nll transformer {nll:.4f}")
    print(f"mean bid regret transformer {transformer_regret.mean().item():.4f} fen")
    print(f"mean bid regret mlp baseline {mlp_regret.mean().item():.4f} fen")
    print(f"mean bid regret naive {naive_regret.mean().item():.4f} fen")

    transformer_win_rate = (transformer_regret < mlp_regret).float().mean().item()
    print(f"\ntransformer beats mlp on {transformer_win_rate*100:.1f}% of auctions")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    print("loading artifacts")
    artifacts = load_artifacts()

    print("loading test data")
    test_loader = get_dataloader("data/processed/test.parquet", batch_size=BATCH_SIZE, shuffle=False)
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

    # naive baseline uses the global log-normal mu from EDA -- avoids loading the 10M row train file
    # from eda.py: global log-normal mu=4.0523 on training data
    train_log_targets = torch.tensor([4.0523], dtype=torch.float32)

    # load both models
    transformer = load_model(
        BidTransformer,
        os.path.join(CHECKPOINT_DIR, "BidTransformer_best.pt"),
        artifacts,
        device
    )
    mlp = load_model(
        MLPBaseline,
        os.path.join(CHECKPOINT_DIR, "MLPBaseline_best.pt"),
        artifacts,
        device
    )

    # run inference
    print("\nrunning transformer inference on test set")
    mu, sigma, log_targets, raw_targets = run_transformer_inference(transformer, test_loader, device)

    print("running mlp inference on test set")
    mlp_preds, _, _ = run_mlp_inference(mlp, test_loader, device)

    # held-out NLL
    nll = compute_held_out_nll(mu, sigma, log_targets)
    print(f"\nheld-out nll {nll:.4f}")

    # calibration curve
    print("\ncomputing calibration curve")
    plot_calibration_curve(mu, sigma, log_targets, "BidTransformer")

    # compute optimal bids for all three approaches
    print("\ncomputing optimal bids")
    transformer_bids = compute_optimal_bid(mu, sigma)

    # MLP bids -- point estimate only, so I use predicted log price as mu with a fixed sigma
    # since MLP has no sigma, I use the global sigma from training as a stand-in
    global_sigma = sigma.mean()
    mlp_bids = compute_optimal_bid(mlp_preds, torch.full_like(mlp_preds, global_sigma.item()))

    # naive bid -- historical average clearing price
    naive_bid_value = compute_naive_bid(train_log_targets)
    naive_bids = torch.full((len(raw_targets),), naive_bid_value)
    print(f"naive bid value {naive_bid_value:.2f} fen")

    # compute bid regret for all three
    transformer_regret = compute_bid_regret(transformer_bids, raw_targets)
    mlp_regret = compute_bid_regret(mlp_bids, raw_targets)
    naive_regret = compute_bid_regret(naive_bids, raw_targets)

    # plots
    print("\ngenerating plots")
    plot_bid_regret_comparison(transformer_regret, mlp_regret, naive_regret)
    plot_bid_landscape(mu, sigma, log_targets, raw_targets)

    # per-advertiser KS test
    per_advertiser_ks_test(mu, sigma, log_targets, test_df)

    # summary
    print_summary(transformer_regret, mlp_regret, naive_regret, nll)

    per_advertiser_bid_regret(transformer_bids, mlp_bids, naive_bids, raw_targets, test_df)
    report_sigma_stats(sigma, test_df)

    print("\nevaluation done plots saved to results/plots/")


if __name__ == "__main__":
    main()
