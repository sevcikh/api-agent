# Python with DuckDB for data processing
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install CA certificates for HTTPS, remove unused pip/wheel/setuptools from base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip wheel \
    && pip uninstall -y pip wheel setuptools 2>/dev/null; true

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY api_agent ./api_agent
COPY start.sh ./

# Copy bundled OpenAPI specs (if any)
COPY gitlab-openapi.yaml ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

EXPOSE 3000

RUN sed -i 's/\r$//' ./start.sh && chmod +x ./start.sh

ENTRYPOINT ["/app/start.sh"]
