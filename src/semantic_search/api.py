"""The long-running process: REST, MCP, the paperless webhook and metrics.

  GET  /healthz                     liveness/readiness, unauthenticated
  GET  /metrics                     Prometheus, unauthenticated
  GET  /search?q=&k=&sort=&source=...     bearer API_TOKEN
  GET  /documents/{source}/{id}     bearer API_TOKEN
  GET  /status                      bearer API_TOKEN
  POST /mcp                         bearer API_TOKEN (MCP streamable HTTP)
  POST /hooks/paperless             bearer WEBHOOK_TOKEN
"""

import contextlib
import hmac
import logging
import os
import threading
import time

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from . import db, sources
from . import search as search_mod
from .mcp_server import mcp
from .reconcile import reconcile

log = logging.getLogger(__name__)


def _token(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} must be set for serve")
    return value


class BearerAuth:
    """Require a bearer token on everything except probes, metrics and the
    webhook (which checks its own, different token)."""

    OPEN = ("/healthz", "/metrics", "/hooks/")

    def __init__(self, app: ASGIApp, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http" and not scope["path"].startswith(self.OPEN):
            supplied = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self.expected):
                await JSONResponse({"error": "unauthorized"}, 401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class Debouncer:
    """Collapse a burst of webhook events into one reconcile.

    Consuming 20 documents fires 20 webhooks. Each event only sets a flag;
    the worker waits for `quiet` seconds without events (capped at
    `max_delay` from the first) and then runs one full reconcile. An event
    arriving during a run earns exactly one more run afterwards, so a change
    landing mid-reconcile is never missed. Correctness against the CronJob
    comes from the per-source advisory lock, not from this.
    """

    def __init__(self, source: str, quiet: float = 10, max_delay: float = 120):
        self.source, self.quiet, self.max_delay = source, quiet, max_delay
        self._wake = threading.Event()
        threading.Thread(target=self._run, daemon=True, name=f"webhook-{source}").start()

    def poke(self):
        self._wake.set()

    def _run(self):
        while True:
            self._wake.wait()
            deadline = time.monotonic() + self.max_delay
            while True:
                self._wake.clear()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._wake.wait(min(self.quiet, remaining)):
                    break
            self._wake.clear()
            try:
                log.info("webhook: reconciling %s", self.source)
                stats = reconcile(sources.get(self.source))
                log.info("webhook: %s reconciled: %s", self.source, stats)
            except Exception:
                # Not retried here: the scheduled CronJob is the retry.
                log.exception("webhook: %s reconcile failed; leaving it to the CronJob", self.source)


def create_app() -> Starlette:
    api_token = _token("API_TOKEN")
    webhook_token = os.environ.get("WEBHOOK_TOKEN", "")
    paperless_hook = Debouncer("paperless") if webhook_token and os.environ.get("PAPERLESS_URL") else None

    async def healthz(_: Request):
        return PlainTextResponse("ok")

    async def metrics(_: Request):
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    async def search(request: Request):
        q = request.query_params.get("q", "").strip()
        if not q:
            return JSONResponse({"error": "q is required"}, 400)
        p = request.query_params
        filters = search_mod.Filters(
            sources=p.getlist("source") or None,
            tags=p.getlist("tag") or None,
            correspondent=p.get("correspondent"),
            document_type=p.get("document_type"),
            created_after=p.get("created_after"),
            created_before=p.get("created_before"),
            path_prefix=p.get("path_prefix"),
        )
        k = min(max(int(p.get("k", "5")), 1), 50)
        sort = p.get("sort", "relevance")
        if sort not in ("relevance", "newest"):
            return JSONResponse({"error": "sort must be relevance or newest"}, 400)
        result = await run_in_threadpool(search_mod.search, q, k=k, filters=filters, sort=sort)
        return JSONResponse({"results": result.results, "degraded": result.degraded})

    async def document(request: Request):
        p = request.query_params
        doc = await run_in_threadpool(
            search_mod.get_document,
            request.path_params["source"], request.path_params["source_id"],
            int(p.get("offset", "0")), int(p.get("limit", "50000")),
        )
        if doc is None:
            return JSONResponse({"error": "not found"}, 404)
        return JSONResponse(doc)

    async def status(_: Request):
        return JSONResponse({"sources": await run_in_threadpool(search_mod.status)})

    async def paperless_webhook(request: Request):
        await request.body()
        supplied = request.headers.get("authorization", "")
        if not webhook_token or not hmac.compare_digest(supplied, f"Bearer {webhook_token}"):
            log.warning("rejected webhook from %s", request.client.host if request.client else "?")
            return JSONResponse({"error": "unauthorized"}, 401)
        if paperless_hook is None:
            return JSONResponse({"error": "paperless source not configured"}, 503)
        # Ack before any work: paperless gives the webhook 5s and does not
        # retry a timeout, so a slow response loses the event.
        paperless_hook.poke()
        return PlainTextResponse("queued", 202)

    # Stateless + JSON responses: every MCP call is a self-contained POST, so
    # the pod can restart (or scale) without stranding agent sessions.
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True, host="0.0.0.0",
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        db.migrate()
        db.pool()
        # A mounted app's own lifespan never runs, so the MCP session manager
        # has to be started from ours.
        async with mcp.session_manager.run():
            yield

    return Starlette(
        routes=[
            Route("/healthz", healthz),
            Route("/metrics", metrics),
            Route("/search", search),
            Route("/documents/{source}/{source_id:path}", document),
            Route("/status", status),
            Route("/hooks/paperless", paperless_webhook, methods=["POST"]),
            Mount("/", app=mcp_app),
        ],
        middleware=[Middleware(BearerAuth, token=api_token)],
        lifespan=lifespan,
    )
