# Retraining Worker (Phase 3)

Closes the feedback loop. It reads the auction outcomes the Go engine logs,
keeps the most recent window of them, fine-tunes the bins model on that window
(warm-started from the iPinYou model), and exports a new ONNX file. The C++
servers watch that file and hot-swap it in with zero downtime.

```
/data/outcomes.jsonl   ->  sliding window (last W)
                       ->  fine-tune the bins model (warm start = iPinYou model)
                       ->  export /models/bid_model.onnx  (single self-contained file)
                       ->  C++ servers detect the change and atomically swap it in
                       ->  write /models/retrainer_status.json (round, rows, loss,
                           exports, model_version) for the live dashboard
```

## Why a sliding window

The window keeps only the most recent ~W outcomes, so the model tracks the
*current* competitive environment instead of accumulating stale synthetic data.
The iPinYou-trained checkpoint is the warm start: it grounds the model in real
auction behavior before the simulation takes over.

## Files

```
retrain.py            the loop: read window -> fine-tune -> export ONNX
models/warm_start.pt  the iPinYou-trained bins model (the warm start)
models/feature_config.json  the same encoding the C++ server / training uses
```

The model classes in `retrain.py` are copied from
`rtb-bid-model/src/model.py` on purpose, so the warm-start checkpoint loads with
matching layer names.

## Run

```
# one round, against a local outcome log (for testing)
python retrain.py --once --outcome_log ../go-engine/_sample/outcomes.jsonl

# the loop (in the compose demo): reads /data, writes /models every 60s
docker build -t rtb-retrainer:latest .
```

## Important details

- The export is written as a **single self-contained ONNX** (weights embedded)
  and moved into place atomically, so the C++ watcher never reads a half-written
  or split (external-data) model.
- The exported model has the exact same inputs/outputs as the original
  (`cat[B,9]`, `cont[B,9]`, `tags[B,10]` -> `probs[B,301]`), so it is a drop-in
  swap with no C++ changes.
- Each round writes `/models/retrainer_status.json` atomically (`--status` to
  change the path). The Go engine reads it and surfaces the training loop
  (round, window rows, last cross-entropy loss, export count, model version) in
  the live dashboard's "Retraining loop" panel.
- File-based for the MVP. Kafka is the documented production version of the
  outcome stream.
