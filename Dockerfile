FROM ghcr.io/astral-sh/uv:0.11.21 AS uv

FROM docker.io/library/python:3.14-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM docker.io/library/python:3.14-slim
# Runs with a read-only root filesystem: bytecode is compiled at build time
# and nothing is written at runtime.
ENV PATH=/app/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=build /app/.venv /app/.venv
USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["semantic-search"]
CMD ["serve"]
