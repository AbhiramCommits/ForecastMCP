FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORECASTMCP_HOME=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# CPU-only torch keeps the image small and is sufficient for the classical baselines.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY forecastmcp ./forecastmcp
COPY configs ./configs
RUN pip install .

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/data /app/data/processed /app/artifacts /app/results \
    && chown -R app:app /app

USER app

EXPOSE 8000
ENTRYPOINT ["forecastmcp"]
CMD ["serve"]
