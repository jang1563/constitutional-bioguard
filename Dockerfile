FROM python:3.11-slim

# System dependencies
# libgomp1: required by DeBERTa / PyTorch OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Copy source package (no extras — skip dev/llamaguard)
COPY constitutional_bioguard/ ./constitutional_bioguard/

# Install the package and its runtime dependencies (no dev extras)
RUN pip install --no-cache-dir .

# Copy scripts
COPY scripts/ ./scripts/

EXPOSE 8000

# Model is mounted at runtime — not baked into the image
ENV BIOGUARD_MODEL_DIR=/models/deberta_bioguard_v1

CMD ["python", "scripts/serve.py"]
