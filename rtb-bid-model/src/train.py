import os
import sys
import time
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append("src")
from dataset import get_dataloader
from model import BidTransformer, MLPBaseline, load_artifacts, count_parameters
from loss import NLLLoss, MSELoss

SEED = 42
BATCH_SIZE = 1024
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
GRAD_CLIP = 1.0
CHECKPOINT_DIR = "exports"
PLOT_DIR = "results/plots"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device, is_transformer):
    model.train()
    total_loss = 0.0
    total_batches = 0
    epoch_start = time.time()

    for cats, conts, tags, log_targets, raw_targets in loader:
        cats = cats.to(device)
        conts = conts.to(device)
        tags = tags.to(device)
        log_targets = log_targets.to(device)

        optimizer.zero_grad()

        if is_transformer:
            mu, sigma = model(cats, conts, tags)
            loss = loss_fn(mu, sigma, log_targets)
        else:
            pred = model(cats, conts, tags)
            loss = loss_fn(pred, log_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        # print progress every 500 batches so I know it's not frozen
        if total_batches % 500 == 0:
            elapsed = time.time() - epoch_start
            batches_per_sec = total_batches / elapsed
            remaining = (len(loader) - total_batches) / batches_per_sec
            print(f"batch {total_batches}/{len(loader)} loss={loss.item():.4f} {batches_per_sec:.1f} batches/sec ~{remaining/60:.1f} min remaining")

    epoch_time = time.time() - epoch_start
    print(f"epoch time {epoch_time/60:.1f} min")
    return total_loss / total_batches


def evaluate_one_epoch(model, loader, loss_fn, device, is_transformer):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    sigma_vals = []

    with torch.no_grad():
        for cats, conts, tags, log_targets, raw_targets in loader:
            cats = cats.to(device)
            conts = conts.to(device)
            tags = tags.to(device)
            log_targets = log_targets.to(device)

            if is_transformer:
                mu, sigma = model(cats, conts, tags)
                sigma_vals.append(sigma.mean().item())
                loss = loss_fn(mu, sigma, log_targets)
            else:
                pred = model(cats, conts, tags)
                loss = loss_fn(pred, log_targets)

            total_loss += loss.item()
            total_batches += 1

    if sigma_vals:
        print(f"sigma mean={np.mean(sigma_vals):.4f} min={np.min(sigma_vals):.6f} max={np.max(sigma_vals):.4f}")

    return total_loss / total_batches


def save_learning_curves(train_losses, val_losses, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} learning curves")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def train_model(model, model_name, is_transformer, train_loader, val_loader, device):
    print(f"\n{'='*50}")
    print(f"training {model_name}")
    print(f"{'='*50}")

    total, trainable = count_parameters(model)
    print(f"parameters {total:,} total {trainable:,} trainable")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    if is_transformer:
        loss_fn = NLLLoss()
    else:
        loss_fn = MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, is_transformer)
        val_loss = evaluate_one_epoch(model, val_loader, loss_fn, device, is_transformer)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}/{MAX_EPOCHS} train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "model_name": model_name,
            }, best_checkpoint_path)
            print(f"checkpoint saved val_loss={best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"no improvement {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"early stopping at epoch {epoch}")
            break

    save_learning_curves(train_losses, val_losses, model_name)
    print(f"best val loss {best_val_loss:.4f}")
    return best_val_loss


def main():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    print("loading artifacts")
    artifacts = load_artifacts()

    print("loading data")
    train_loader = get_dataloader("data/processed/train.parquet", batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader("data/processed/val.parquet", batch_size=BATCH_SIZE, shuffle=False)

    # train transformer with NLL loss
    transformer = BidTransformer(artifacts)
    train_model(transformer, "BidTransformer", True, train_loader, val_loader, device)

    # train MLP baseline with MSE loss
    mlp = MLPBaseline(artifacts)
    train_model(mlp, "MLPBaseline", False, train_loader, val_loader, device)

    print("\ntraining done")
    print(f"checkpoints saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
