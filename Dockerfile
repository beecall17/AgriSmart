# syntax=docker/dockerfile:1
#
# AgriSmart — AI Supply-Chain & Inventory Coordinator
# Production-ready image running the Streamlit UI on port 8501.
#
# Build:    docker build -t agrismart:latest .
# Run:      docker run --rm -p 8501:8501 --env-file .env agrismart:latest

FROM python:3.11-slim

# --------------------------------------------------------------------------- #
# Runtime environment
# --------------------------------------------------------------------------- #
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Containers run headless and telemetry is disabled for production.
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# --------------------------------------------------------------------------- #
# System dependencies
# --------------------------------------------------------------------------- #
# curl is required at runtime for the container HEALTHCHECK.
# build-essential guarantees any Python package without a prebuilt wheel still
# compiles, and is purged afterwards to keep the final image lightweight.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Python dependencies (this layer is cached until requirements.txt changes)
# --------------------------------------------------------------------------- #
WORKDIR /app

COPY requirements-docker.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-docker.txt

# Drop the build toolchain now that every wheel/sdist is installed.
RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Application code & enterprise data
# --------------------------------------------------------------------------- #
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

CMD ["python","-m","streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]