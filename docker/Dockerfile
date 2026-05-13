# CPU inference image for AikiSol.
#
# Build:
#   docker build -f docker/Dockerfile -t aikisol:cpu .
#
# Run (mount a checkpoint dir; the released curve-head .pt files live there):
#   docker run --rm -it \
#     -v $HOME/.cache/aikisol/checkpoints:/checkpoints \
#     -v $HOME/.cache/huggingface:/hf \
#     -e AIKISOL_CKPT_DIR=/checkpoints \
#     -e HF_HOME=/hf \
#     aikisol:cpu --sequence "MAEILVTQNMK..."

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small.
RUN pip install --no-cache-dir \
        torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        numpy>=1.24 \
        pandas>=2.0 \
        scipy>=1.11 \
        tqdm \
        transformers>=4.40 \
        sentencepiece \
        protobuf \
        accelerate \
        huggingface-hub

ENV HF_HOME=/hf
ENV AIKISOL_CKPT_DIR=/checkpoints
VOLUME ["/hf", "/checkpoints"]

COPY aikisol/ aikisol/
COPY scripts/predict.py scripts/predict.py
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install the package itself so `aikisol-predict` and `import aikisol` both work.
RUN pip install --no-cache-dir .

ENTRYPOINT ["aikisol-predict"]
