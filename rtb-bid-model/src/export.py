import os
import sys
import json
import pickle
import torch
import numpy as np
import onnxruntime as ort

sys.path.append("src")
from model import BidTransformer, load_artifacts

CHECKPOINT_PATH = "exports/BidTransformer_best.pt"
ONNX_PATH = "exports/bid_model.onnx"
FEATURE_CONFIG_PATH = "exports/feature_config.json"
PROCESSED_DIR = "data/processed"
OPSET_VERSION = 18
EQUIVALENCE_SAMPLES = 1000
TOLERANCE = 1e-4


def load_model(artifacts, device):
    model = BidTransformer(artifacts)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint epoch {checkpoint['epoch']} val_loss={checkpoint['val_loss']:.4f}")
    return model


def make_sample_inputs(artifacts, batch_size=8, tag_seq_len=20):
    # create sample inputs for ONNX export tracing
    encoders = artifacts["encoders"]
    tag_vocab = artifacts["tag_vocab"]

    n_cat = len(encoders)
    n_cont = 5  # 1 continuous + 4 cyclical

    # sample categorical indices within valid range
    cats = torch.zeros(batch_size, n_cat, dtype=torch.long)
    for i, (col, enc) in enumerate(encoders.items()):
        n_classes = len(enc.classes_)
        cats[:, i] = torch.randint(0, n_classes, (batch_size,))

    conts = torch.randn(batch_size, n_cont, dtype=torch.float32)

    # sample tag indices
    tag_vocab_size = len(tag_vocab) + 1
    tags = torch.randint(0, tag_vocab_size, (batch_size, tag_seq_len), dtype=torch.long)

    return cats, conts, tags

def export_onnx(model, artifacts):
    print("\nexporting model to onnx")
    cats, conts, tags = make_sample_inputs(artifacts)

    dynamic_axes = {
        "cats":  {0: "batch_size"},
        "conts": {0: "batch_size"},
        "tags":  {0: "batch_size", 1: "tag_seq_len"},
        "mu":    {0: "batch_size"},
        "sigma": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (cats, conts, tags),
        ONNX_PATH,
        opset_version=18,
        input_names=["cats", "conts", "tags"],
        output_names=["mu", "sigma"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,  # use legacy exporter, avoids onnxscript version conflicts
    )
    print(f"saved {ONNX_PATH}")


def verify_onnx_outputs(model, artifacts):
    # run the same inputs through both PyTorch and ONNX Runtime
    # check that outputs match within tolerance
    # if this passes then the C++ server should produce identical predictions
    print(f"\nrunning equivalence check {EQUIVALENCE_SAMPLES} samples")

    cats, conts, tags = make_sample_inputs(artifacts, batch_size=EQUIVALENCE_SAMPLES)

    # pytorch outputs
    with torch.no_grad():
        pt_mu, pt_sigma = model(cats, conts, tags)
    pt_mu = pt_mu.numpy()
    pt_sigma = pt_sigma.numpy()

    # onnx runtime outputs
    session = ort.InferenceSession(ONNX_PATH)
    ort_inputs = {
        "cats":  cats.numpy(),
        "conts": conts.numpy(),
        "tags":  tags.numpy(),
    }
    ort_mu, ort_sigma = session.run(["mu", "sigma"], ort_inputs)

    # check mu
    mu_max_diff = np.abs(pt_mu - ort_mu).max()
    mu_mean_diff = np.abs(pt_mu - ort_mu).mean()
    print(f"mu max diff {mu_max_diff:.8f} mean diff {mu_mean_diff:.8f}")

    # check sigma
    sigma_max_diff = np.abs(pt_sigma - ort_sigma).max()
    sigma_mean_diff = np.abs(pt_sigma - ort_sigma).mean()
    print(f"sigma max diff {sigma_max_diff:.8f} mean diff {sigma_mean_diff:.8f}")

    if mu_max_diff < TOLERANCE and sigma_max_diff < TOLERANCE:
        print(f"pass pytorch and onnx runtime outputs match within {TOLERANCE}")
    else:
        print(f"fail output mismatch exceeds tolerance {TOLERANCE}")
        print("do not use this onnx file for inference")
        sys.exit(1)

    return session


def verify_onnx_inputs(session):
    # print input/output tensor names, shapes, and dtypes
    # the C++ server needs to use these exact names and types
    print("\nonnx model input spec for c++ server")
    for inp in session.get_inputs():
        print(f"input name={inp.name} shape={inp.shape} dtype={inp.type}")

    print("\nonnx model output spec for c++ server")
    for out in session.get_outputs():
        print(f"output name={out.name} shape={out.shape} dtype={out.type}")


def verify_dynamic_axes(session, artifacts):
    # make sure different batch sizes and tag lengths actually work
    print("\ndynamic axes verification")
    test_cases = [
        (1, 5),    # single request, few tags
        (32, 15),  # typical batch, medium tags
        (64, 50),  # large batch, many tags
        (1, 1),    # edge case: single request, single tag
    ]

    for batch_size, tag_len in test_cases:
        cats, conts, tags = make_sample_inputs(artifacts, batch_size=batch_size, tag_seq_len=tag_len)
        ort_inputs = {
            "cats":  cats.numpy(),
            "conts": conts.numpy(),
            "tags":  tags.numpy(),
        }
        try:
            ort_mu, ort_sigma = session.run(["mu", "sigma"], ort_inputs)
            assert ort_mu.shape == (batch_size,), f"mu shape wrong: {ort_mu.shape}"
            assert ort_sigma.shape == (batch_size,), f"sigma shape wrong: {ort_sigma.shape}"
            assert (ort_sigma > 0).all(), "sigma contains non-positive values"
            print(f"pass batch={batch_size} tag_len={tag_len} mu shape {ort_mu.shape} sigma range [{ort_sigma.min():.3f}, {ort_sigma.max():.3f}]")
        except Exception as e:
            print(f"fail batch={batch_size} tag_len={tag_len} {e}")
            sys.exit(1)


def export_feature_config(artifacts):
    # write out all the feature encoding info needed by the C++ server
    # the C++ feature store loads this file at startup to encode incoming auction requests
    print("\nexporting feature config")

    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    tag_vocab = artifacts["tag_vocab"]

    # build encoder mappings: category string -> integer index
    categorical_encoders = {}
    for col, enc in encoders.items():
        mapping = {}
        for idx, cls in enumerate(enc.classes_):
            mapping[str(cls)] = int(idx)
        categorical_encoders[col] = mapping

    # scaler params for slot_floor_price
    scaler_params = {
        "slot_floor_price_mean": float(scaler.mean_[0]),
        "slot_floor_price_std":  float(scaler.scale_[0]),
    }

    # tag vocab: tag_id string -> integer index
    tag_vocab_serializable = {str(k): int(v) for k, v in tag_vocab.items()}

    # categorical feature order has to match exactly what the model expects
    # the C++ server must pass features in this exact order
    cat_feature_order = list(encoders.keys())

    config = {
        "categorical_encoders": categorical_encoders,
        "scaler": scaler_params,
        "tag_vocab": tag_vocab_serializable,
        "cat_feature_order": cat_feature_order,
        "n_continuous": 1,
        "n_cyclical": 4,
        "continuous_features": ["slot_floor_price"],
        "cyclical_features": ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"],
        "embedding_dim": 16,
        "tag_vocab_size": len(tag_vocab) + 1,
        "impression_value_fen": 150.0,
        "sigma_circuit_breaker_min": 0.05,
        "sigma_circuit_breaker_max": 3.0,
    }

    with open(FEATURE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved {FEATURE_CONFIG_PATH}")
    print(f"categorical features {len(categorical_encoders)}")
    print(f"tag vocab size {len(tag_vocab_serializable):,}")
    print(f"scaler mean={scaler_params['slot_floor_price_mean']:.4f} std={scaler_params['slot_floor_price_std']:.4f}")


def main():
    os.makedirs("exports", exist_ok=True)

    device = torch.device("cpu")  # export on CPU for maximum compatibility
    print(f"device {device}")

    print("loading artifacts")
    artifacts = load_artifacts()

    print("loading model")
    model = load_model(artifacts, device)

    # export ONNX
    export_onnx(model, artifacts)

    # verify ONNX outputs match PyTorch
    session = verify_onnx_outputs(model, artifacts)

    # print input/output spec for C++ reference
    verify_onnx_inputs(session)

    # verify dynamic axes work with different shapes
    verify_dynamic_axes(session, artifacts)

    # export feature config for C++ feature store
    export_feature_config(artifacts)

    print("\nexport done")
    print(f"onnx model {ONNX_PATH}")
    print(f"feature config {FEATURE_CONFIG_PATH}")
    print("\nthese two files are everything the c++ inference server needs")


if __name__ == "__main__":
    main()
