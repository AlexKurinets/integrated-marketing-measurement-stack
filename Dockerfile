FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAX_PLATFORMS=cpu \
    PIP_NO_CACHE_DIR=1 \
    IMMS_CONFIG_DIR=/app/config \
    IMMS_SQL_DIR=/app/sql

WORKDIR /app

# Dependencies first so code changes do not reinstall JAX.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p measurement seed && pip install --upgrade pip && pip install .

COPY measurement ./measurement
COPY seed ./seed
COPY config ./config
COPY sql ./sql
RUN pip install --no-deps .

EXPOSE 8000 8001
CMD ["imms-api"]
