import sys
sys.path.append("src")

import torch
import pickle
from model import BidTransformer, MLPBaseline, load_artifacts, count_parameters

BATCH_SIZE = 32


def make_fake_batch(artifacts):
    # build a fake batch with the exact shapes from validate_dataset.py
    encoders = artifacts["encoders"]
    tag_vocab = artifacts["tag_vocab"]

    n_cat = len(encoders)
    n_cont = 5  # 1 continuous + 4 cyclical

    # random categorical indices within valid range for each feature
    cats = torch.zeros(BATCH_SIZE, n_cat, dtype=torch.long)
    for i, (col, enc) in enumerate(encoders.items()):
        n_classes = len(enc.classes_)
        cats[:, i] = torch.randint(0, n_classes, (BATCH_SIZE,))

    conts = torch.randn(BATCH_SIZE, n_cont, dtype=torch.float32)

    # fake tag sequences of length 10, with some padding zeros at the end
    tag_vocab_size = len(tag_vocab) + 1
    tags = torch.randint(0, tag_vocab_size, (BATCH_SIZE, 10), dtype=torch.long)
    tags[:, 7:] = 0  # last 3 positions are padding

    return cats, conts, tags


def test_transformer(artifacts):
    print("\nbid transformer")
    model = BidTransformer(artifacts)
    model.eval()

    cats, conts, tags = make_fake_batch(artifacts)

    with torch.no_grad():
        mu, sigma = model(cats, conts, tags)

    print(f"mu shape {mu.shape} dtype={mu.dtype}")
    print(f"sigma shape {sigma.shape} dtype={sigma.dtype}")

    assert mu.shape == (BATCH_SIZE,), f"mu shape wrong: {mu.shape}"
    assert sigma.shape == (BATCH_SIZE,), f"sigma shape wrong: {sigma.shape}"
    assert mu.dtype == torch.float32, "mu dtype wrong"
    assert sigma.dtype == torch.float32, "sigma dtype wrong"

    # sigma must always be positive -- Softplus enforces this
    assert (sigma > 0).all(), "sigma contains non-positive values"

    # mu and sigma should both be finite
    assert torch.isfinite(mu).all(), "mu contains non-finite values"
    assert torch.isfinite(sigma).all(), "sigma contains non-finite values"

    # mu should be in a reasonable range for log(payprice)
    # from EDA: log_payprice range is roughly [1.4, 5.7]
    assert mu.mean() > -10 and mu.mean() < 10, "mu mean looks unreasonable"

    total, trainable = count_parameters(model)
    print(f"total parameters {total:,}")
    print(f"trainable parameters {trainable:,}")
    print("pass bidtransformer forward pass")


def test_mlp(artifacts):
    print("\nmlp baseline")
    model = MLPBaseline(artifacts)
    model.eval()

    cats, conts, tags = make_fake_batch(artifacts)

    with torch.no_grad():
        pred = model(cats, conts, tags)

    print(f"pred shape {pred.shape} dtype={pred.dtype}")

    assert pred.shape == (BATCH_SIZE,), f"pred shape wrong: {pred.shape}"
    assert pred.dtype == torch.float32, "pred dtype wrong"
    assert torch.isfinite(pred).all(), "pred contains non-finite values"

    total, trainable = count_parameters(model)
    print(f"total parameters {total:,}")
    print(f"trainable parameters {trainable:,}")
    print("pass mlp baseline forward pass")


def test_gradient_flow(artifacts):
    print("\ngradient flow")
    model = BidTransformer(artifacts)
    model.train()

    cats, conts, tags = make_fake_batch(artifacts)
    mu, sigma = model(cats, conts, tags)

    # simple loss -- just sum of outputs, enough to test that backprop works
    loss = mu.sum() + sigma.sum()
    loss.backward()

    # check that gradients exist and are finite for all parameters
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"fail no gradient for {name}")
            return
        if not torch.isfinite(param.grad).all():
            print(f"fail non-finite gradient for {name}")
            return

    print("pass all gradients are finite and present")


def main():
    print("loading artifacts")
    artifacts = load_artifacts()

    test_transformer(artifacts)
    test_mlp(artifacts)
    test_gradient_flow(artifacts)

    print("\nmodel validation done")


if __name__ == "__main__":
    main()
