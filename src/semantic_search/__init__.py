"""semantic-search: hybrid search over paperless and Obsidian.

  semantic-search serve                   API, MCP and webhook on :8080
  semantic-search sync <source>           reconcile one source and exit
  semantic-search migrate                 apply schema migrations and exit
  semantic-search search "<query>"        one-off search, JSON to stdout
  semantic-search eval eval/queries.yaml  retrieval metrics on a golden set
"""

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    # httpx logs every request at INFO; that is a line per embedding batch.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # One "Terminating session: None" per stateless MCP request.
    logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="semantic-search")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)

    p = sub.add_parser("sync")
    p.add_argument("source", choices=["paperless", "obsidian"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-delete-ratio", type=float)

    sub.add_parser("migrate")

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--source", action="append")

    p = sub.add_parser("eval")
    p.add_argument("queries")
    p.add_argument("--baseline", choices=["openwebui"])
    p.add_argument("--both", action="store_true", help="run ours and the openwebui baseline")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from .api import create_app
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info",
                    access_log=False, proxy_headers=True)
        return 0

    from . import db
    db.migrate()
    try:
        return _run(args)
    finally:
        # The pool's worker threads cannot be joined during interpreter
        # shutdown, so close it while the interpreter is still whole.
        db.close()


def _run(args) -> int:
    if args.command == "migrate":
        return 0

    if args.command == "sync":
        from . import sources
        from .reconcile import UnsafeSync, reconcile
        try:
            stats = reconcile(sources.get(args.source), dry_run=args.dry_run,
                              max_delete_ratio=args.max_delete_ratio)
        except UnsafeSync as e:
            logging.error("REFUSING TO SYNC: %s", e)
            return 1
        print(stats)
        return 0

    if args.command == "search":
        from .search import Filters, search
        result = search(args.query, k=args.k, filters=Filters(sources=args.source))
        print(json.dumps({"results": result.results, "degraded": result.degraded},
                         ensure_ascii=False, indent=2, default=str))
        return 0

    if args.command == "eval":
        from . import eval as ev
        backends = [None, "openwebui"] if args.both else [args.baseline]
        reports = [ev.run(args.queries, baseline=b, verbose=args.verbose) for b in backends]
        if args.json:
            print(json.dumps(reports, ensure_ascii=False, indent=2))
        else:
            for r in reports:
                ev.print_report(r)
        return 0

    return 2
