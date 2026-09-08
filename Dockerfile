FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.10.9
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY assets ./assets
COPY infra/postgres ./infra/postgres
COPY configs ./configs
COPY workflows ./workflows
COPY infra/runpod/models.lock.json ./infra/runpod/models.lock.json
RUN uv sync --frozen --no-dev && useradd --create-home app \
    && mkdir -p /app/artifacts && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "talotrace.main:app", "--host", "0.0.0.0", "--port", "8000"]
