# ── Stage 1 — Python build environment ───────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Alias python3.11 → python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3     1

WORKDIR /build

# Copy dependency manifests first for layer caching
COPY pyproject.toml ./

# Install Python dependencies (CPU-only torch for non-GPU targets; override in
# the compose file with TORCH_INDEX_URL env var for GPU wheels).
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
 && pip install "jax[cuda12]" jaxlib \
 && pip install fastapi "uvicorn[standard]" python-multipart psycopg2-binary asyncpg \
 && pip install grpcio grpcio-tools grpcio-reflection protobuf \
 && pip install ollama chromadb sentence-transformers rank-bm25 pymupdf networkx \
 && pip install numpy scipy scikit-learn xgboost \
 && pip install resfo lasio pydantic "pydantic-settings" loguru tqdm rich click \
 && pip install h5py pyarrow matplotlib sqlalchemy alembic cryptography


# ── Stage 2 — Runtime image ───────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3     1

# Copy installed packages from builder
COPY --from=builder /usr/lib/python3/dist-packages        /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/lib/python3.11/dist-packages \
                    /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy PhysicsFlow engine source
COPY . .

# Install the engine package (no deps — already installed above)
RUN pip install --no-deps -e .

# Create data directories
RUN mkdir -p /data/projects /data/uploads /data/models /data/logs /data/db

EXPOSE 50051 8000

# Health-check: ping the REST API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/health || exit 1

CMD ["physicsflow-engine", "--port", "50051", "--log-level", "INFO"]
