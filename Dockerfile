FROM ghcr.io/astral-sh/uv:0.13.10 AS uv

FROM docker.io/library/python:3.15-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
# uv's cache is a BuildKit cache mount, so a rebuild reuses earlier downloads.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM docker.io/library/python:3.15-slim
# Runs with a read-only root filesystem: bytecode is compiled at build time
# and nothing is written at runtime.
ENV PATH=/app/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=build /app/.venv /app/.venv
USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["semantic-search"]
CMD ["serve"]

# Smoke test, run by every CI build (job pods have no Docker daemon, so the
# image can't be started as a separate step): the entry point runs, every
# module imports on this Python, the migrations shipped in the wheel and the
# app builds. Nothing here touches the network or the database.
RUN semantic-search --help >/dev/null && API_TOKEN=smoke python - <<'EOF'
import importlib
import importlib.resources
import pkgutil

import semantic_search as pkg
from semantic_search.api import create_app

for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    importlib.import_module(m.name)
migrations = importlib.resources.files(pkg).joinpath("migrations")
assert any(f.name.endswith(".sql") for f in migrations.iterdir()), "no migrations in the image"
create_app()
EOF
