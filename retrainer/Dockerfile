# The retraining worker. CPU PyTorch is enough since the model is tiny and we
# only fine-tune on a sliding window.
FROM python:3.11-slim

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy onnx onnxscript

WORKDIR /app
COPY . .

# /data has the engine's outcome log, /models is where the new ONNX goes (both
# shared volumes in docker-compose)
CMD ["python", "retrain.py"]
