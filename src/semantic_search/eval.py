"""Retrieval evaluation against a golden set.

eval/queries.yaml:

    - query: escritura do apartamento
      expect: [paperless:123]            # any of these counts as a hit
      tags: [pt-pt, paperless]           # free-form, reported per tag

Reports recall@1/@5/@10 and MRR per tag and overall. `--baseline openwebui`
runs the same queries against the Open WebUI PoC collections, so the cutover
gate compares like with like.
"""

import json
import os
import statistics
from collections import defaultdict

import httpx
import yaml

from . import search as search_mod

_OPENWEBUI_COLLECTIONS = {
    "paperless": os.environ.get("OPENWEBUI_PAPERLESS_KB", ""),
    "obsidian": os.environ.get("OPENWEBUI_OBSIDIAN_KB", ""),
}


def _ours(query: str, k: int) -> list[str]:
    result = search_mod.search(query, k=k)
    return [f"{r['source']}:{r['source_id']}" for r in result.results]


def _openwebui_name_to_key(name: str) -> str | None:
    # The PoC's filename conventions: paperless-<id>.txt and %2F-escaped paths.
    if name.startswith("paperless-") and name.endswith(".txt"):
        return f"paperless:{name.removeprefix('paperless-').removesuffix('.txt')}"
    if name.endswith(".md"):
        return "obsidian:" + name.replace("%2F", "/").replace("%25", "%")
    return None


def _openwebui(query: str, k: int) -> list[str]:
    url = os.environ["OPENWEBUI_URL"].rstrip("/")
    token = os.environ["OPENWEBUI_TOKEN"]
    collections = [c for c in _OPENWEBUI_COLLECTIONS.values() if c]
    # Ask for more chunks than documents: several chunks of one document
    # collapse into a single entry, as they do in our own results.
    r = httpx.post(
        f"{url}/api/v1/retrieval/query/collection",
        headers={"Authorization": f"Bearer {token}"},
        json={"collection_names": collections, "query": query, "k": k * 4},
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    # Already merged across collections and sorted best-first by Open WebUI;
    # its `distances` mean different things in hybrid and vector-only mode,
    # so the order is trusted rather than re-sorted.
    names = [m.get("name") or m.get("source") for m in (body.get("metadatas") or [[]])[0]]
    out: list[str] = []
    for name in names:
        key = _openwebui_name_to_key(name or "")
        if key and key not in out:
            out.append(key)
    return out[:k]


def run(path: str, baseline: str | None = None, k: int = 10, verbose: bool = False) -> dict:
    cases = yaml.safe_load(open(path))
    backend = _openwebui if baseline == "openwebui" else _ours
    per_tag: dict[str, list[tuple[int | None]]] = defaultdict(list)
    misses = []
    for case in cases:
        expect = set(case["expect"])
        got = backend(case["query"], k)
        rank = next((i for i, key in enumerate(got, 1) if key in expect), None)
        for tag in ["all", *case.get("tags", [])]:
            per_tag[tag].append(rank)
        if rank is None or rank > 5:
            misses.append({"query": case["query"], "expect": sorted(expect), "rank": rank, "got": got[:5]})
        if verbose:
            print(f"{'MISS' if rank is None else f'@{rank:<3}'} {case['query']}")

    def summary(ranks):
        return {
            "n": len(ranks),
            "recall@1": sum(1 for r in ranks if r and r <= 1) / len(ranks),
            "recall@5": sum(1 for r in ranks if r and r <= 5) / len(ranks),
            "recall@10": sum(1 for r in ranks if r and r <= 10) / len(ranks),
            "mrr": statistics.fmean(1 / r if r else 0 for r in ranks),
        }

    return {
        "backend": baseline or "semantic-search",
        "by_tag": {tag: summary(r) for tag, r in sorted(per_tag.items())},
        "misses": misses,
    }


def print_report(report: dict) -> None:
    print(f"\n== {report['backend']}")
    print(f"{'tag':<20} {'n':>4} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    for tag, s in report["by_tag"].items():
        print(f"{tag:<20} {s['n']:>4} {s['recall@1']:>6.2f} {s['recall@5']:>6.2f}"
              f" {s['recall@10']:>6.2f} {s['mrr']:>6.3f}")
    if report["misses"]:
        print(f"\nmisses (not in top 5): {len(report['misses'])}")
        for m in report["misses"]:
            print(json.dumps(m, ensure_ascii=False))
