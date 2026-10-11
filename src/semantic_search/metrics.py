"""Prometheus metrics.

Sync health is read from the database at scrape time rather than counted in
process: the syncs run as short-lived CronJob pods that are never scraped,
and `sync_runs` is the one place every run -- CronJob or webhook -- records
its outcome.
"""

import logging

from prometheus_client import REGISTRY, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily

from . import db

log = logging.getLogger(__name__)

SEARCH_SECONDS = Histogram(
    "semantic_search_search_seconds",
    "End-to-end search latency",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
SEARCH_DEGRADED = Counter(
    "semantic_search_degraded_searches_total",
    "Searches answered without part of the pipeline",
    ["reason"],
)


class _IndexCollector:
    def collect(self):
        docs = GaugeMetricFamily("semantic_search_documents", "Indexed documents", labels=["source"])
        chunks = GaugeMetricFamily("semantic_search_chunks", "Indexed chunks", labels=["source"])
        last_ok = GaugeMetricFamily(
            "semantic_search_last_sync_success_timestamp_seconds",
            "Finish time of the last successful sync", labels=["source"])
        last_failed = GaugeMetricFamily(
            "semantic_search_last_sync_failed",
            "1 if the most recent finished sync failed", labels=["source"])
        try:
            with db.pool().connection() as conn:
                for r in conn.execute(
                    "SELECT d.source, count(DISTINCT d.id) AS docs, count(c.id) AS chunks"
                    " FROM documents d LEFT JOIN chunks c ON c.document_id = d.id"
                    " GROUP BY d.source"
                ):
                    docs.add_metric([r["source"]], r["docs"])
                    chunks.add_metric([r["source"]], r["chunks"])
                for r in conn.execute(
                    "SELECT source, max(extract(epoch FROM finished_at)) AS ts"
                    " FROM sync_runs WHERE ok GROUP BY source"
                ):
                    last_ok.add_metric([r["source"]], float(r["ts"]))
                for r in conn.execute(
                    "SELECT DISTINCT ON (source) source, ok FROM sync_runs"
                    " WHERE finished_at IS NOT NULL ORDER BY source, finished_at DESC"
                ):
                    last_failed.add_metric([r["source"]], 0 if r["ok"] else 1)
        except Exception:
            # A scrape must not fail because the database is briefly away;
            # the missing series are themselves the signal.
            log.exception("index metrics unavailable")
        yield from (docs, chunks, last_ok, last_failed)


REGISTRY.register(_IndexCollector())
