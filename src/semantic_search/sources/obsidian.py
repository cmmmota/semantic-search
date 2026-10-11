"""The Obsidian vault, read from a local directory.

In the cluster an rclone init container copies the vault's markdown out of
OpenCloud into an emptyDir before the sync runs, so this source only ever
reads files. The vault-relative path is the source_id: a moved or renamed
note is a delete plus an add, which is right, because its path is part of
what gets embedded.
"""

import os
import pathlib
import re
import urllib.parse
from collections.abc import Iterator

from ..chunking import Chunk, parse_frontmatter, split_markdown
from ..config import settings
from . import Document

# Obsidian's own state and its trash are not notes.
_SKIP = (".obsidian/", ".trash/")
_WIKILINK = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_INLINE_TAG = re.compile(r"(?:^|\s)#([\w/-]*[^\W\d][\w/-]*)", re.U)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
_EMBED = re.compile(r"!\[\[[^\]]*\]\]")
_WIKI_ALIAS = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")


def _for_embedding(body: str) -> str:
    """Markdown as a reader sees it: link and image text without URLs, no
    attachment embeds. A note that is one pasted image is otherwise a few
    hundred characters of URL to the embedder, and lands nowhere near the
    question it answers."""
    body = _MD_IMAGE.sub(lambda m: m.group(1), body)
    body = _MD_LINK.sub(lambda m: m.group(1), body)
    body = _EMBED.sub("", body)
    return _WIKI_ALIAS.sub(lambda m: m.group(1).split("#")[0], body)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(",", " ").split() if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value)]


class Obsidian:
    name = "obsidian"

    def __init__(self):
        self.root = pathlib.Path(os.environ.get("VAULT_DIR", "/vault"))
        self.vault_name = os.environ.get("OBSIDIAN_VAULT_NAME", "Obsidian")
        if not self.root.is_dir():
            raise RuntimeError(f"vault directory {self.root} does not exist")

    def _notes(self) -> Iterator[tuple[str, pathlib.Path]]:
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root).as_posix()
            if not rel.startswith(_SKIP):
                yield rel, path

    def listing(self) -> dict[str, str]:
        # rclone carries the remote mtime across, so this is stable between
        # runs even though every run copies into a fresh emptyDir.
        return {rel: str(path.stat().st_mtime_ns) for rel, path in self._notes()}

    def fetch(self, ids: list[str]) -> Iterator[Document]:
        s = settings()
        for rel in ids:
            path = self.root / rel
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue
            front, body = parse_frontmatter(raw)
            title = pathlib.PurePosixPath(rel).stem
            folder = str(pathlib.PurePosixPath(rel).parent)
            tags = sorted(
                {t.lstrip("#") for t in _as_list(front.get("tags") or front.get("tag"))}
                | set(_INLINE_TAG.findall(body))
            )
            aliases = _as_list(front.get("aliases") or front.get("alias"))
            links = sorted({m.strip() for m in _WIKILINK.findall(body)})
            meta = {"path": rel, "folder": "" if folder == "." else folder}
            if tags:
                meta["tags"] = tags
            if aliases:
                meta["aliases"] = aliases
            if links:
                meta["links"] = links

            header = [f"Note: {title}", f"Path: {rel}"]
            if aliases:
                header.append(f"Aliases: {', '.join(aliases)}")
            if tags:
                header.append(f"Tags: {', '.join(tags)}")
            chunks = split_markdown(
                _for_embedding(body), s.chunk_size, s.chunk_overlap, s.chunk_min
            ) or [Chunk(title)]
            yield Document(
                source_id=rel,
                title=title,
                content=body,
                chunks=chunks,
                header=header,
                uri="obsidian://open?" + urllib.parse.urlencode(
                    {"vault": self.vault_name, "file": rel.removesuffix(".md")},
                    quote_via=urllib.parse.quote,
                ),
                metadata=meta,
            )
