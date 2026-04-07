import sys
import os
import torch
import pandas as pd
import pickle

sys.path.append("src")
from dataset import BidDataset, collate_fn, get_dataloader
from model import BidTransformer, MLPBaseline, load_artifacts
from loss import NLLLoss, MSELoss
from torch.utils.data import DataLoader, Subset

SMOKE_ROWS = 1000
SMOKE_EPOCHS = 2
BATCH_SIZE = 64


def get_smoke_loader(parquet_path, n_rows, batch_size):
    # only load the first n_rows for a fast smoke test -- no need to run the whole dataset
    dataset = BidDataset(parquet_path)
    subset = Subset(dataset, list(range(min(n_rows, len(dataset)))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader


def smoke_train(model, loader, loss_fn, optimizer, is_transformer, device):
    model.train()
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
        optimizer.step()

    return loss.item()


def test_transformer_trains(artifacts, device):
    print("\ntransformer smoke train")
    model = BidTransformer(artifacts).to(device)
    loader = get_smoke_loader("data/processed/train.parquet", SMOKE_ROWS, BATCH_SIZE)
    loss_fn = NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    for epoch in range(SMOKE_EPOCHS):
        loss = smoke_train(model, loader, loss_fn, optimizer, True, device)
        losses.append(loss)
        print(f"epoch {epoch+1} loss={loss:.4f}")

    assert torch.isfinite(torch.tensor(losses)).all(), "non-finite loss during transformer training"
    print("pass transformer trains without nan or inf loss")


def test_mlp_trains(artifacts, device):
    print("\nmlp smoke train")
    model = MLPBaseline(artifacts).to(device)
    loader = get_smoke_loader("data/processed/train.parquet", SMOKE_ROWS, BATCH_SIZE)
    loss_fn = MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    for epoch in range(SMOKE_EPOCHS):
        loss = smoke_train(model, loader, loss_fn, optimizer, False, device)
        losses.append(loss)
        print(f"epoch {epoch+1} loss={loss:.4f}")

    assert torch.isfinite(torch.tensor(losses)).all(), "non-finite loss during MLP training"
    print("pass mlp trains without nan or inf loss")


def test_checkpoint_save_load(artifacts, device):
    print("\ncheckpoint save and load")
    model = BidTransformer(artifacts).to(device)
    loader = get_smoke_loader("data/processed/train.parquet", SMOKE_ROWS, BATCH_SIZE)
    loss_fn = NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    smoke_train(model, loader, loss_fn, optimizer, True, device)

    # save checkpoint
    path = "exports/smoke_test_checkpoint.pt"
    os.makedirs("exports", exist_ok=True)
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "val_loss": 0.0,
    }, path)

    # reload and verify weights match
    model2 = BidTransformer(artifacts).to(device)
    checkpoint = torch.load(path, map_location=device)
    model2.load_state_dict(checkpoint["model_state_dict"])

    # run the same batch through both models and check outputs match
    batch = next(iter(loader))
    cats, conts, tags = batch[0].to(device), batch[1].to(device), batch[2].to(device)

    model.eval()
    model2.eval()

    with torch.no_grad():
        mu1, sigma1 = model(cats, conts, tags)
        mu2, sigma2 = model2(cats, conts, tags)

    assert torch.allclose(mu1, mu2), "mu mismatch after checkpoint reload"
    assert torch.allclose(sigma1, sigma2), "sigma mismatch after checkpoint reload"

    os.remove(path)
    print("pass checkpoint saves and reloads correctly")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    artifacts = load_artifacts()

    test_transformer_trains(artifacts, device)
    test_mlp_trains(artifacts, device)
    test_checkpoint_save_load(artifacts, device)

    print("\ntraining validation done")


if __name__ == "__main__":
    main()
