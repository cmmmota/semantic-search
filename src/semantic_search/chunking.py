"""Text and markdown chunking.

Sizes are in characters, like the Open WebUI PoC these settings were tuned
in (1000 / 100 overlap / 200 minimum). bge-m3 accepts far longer inputs; the
size is about retrieval granularity, not the model's limit.
"""

import re
from dataclasses import dataclass

import yaml

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: str | None = None


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than `size`, preferring coarse
    separators (paragraphs) over fine ones (words)."""
    if len(text) <= size:
        return [text]
    sep, *rest = separators
    if sep == "":
        return [text[i:i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    out: list[str] = []
    for i, part in enumerate(parts):
        # Keep the separator attached so rejoining loses nothing.
        piece = part + sep if i < len(parts) - 1 else part
        if len(piece) <= size:
            out.append(piece)
        else:
            out.extend(_split(piece, size, rest))
    return out


def split_text(text: str, size: int, overlap: int, minimum: int) -> list[str]:
    """Pack pieces into chunks of at most `size`, each starting with up to
    `overlap` characters of the previous one. A short tail is folded into the
    chunk before it rather than embedded on its own."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    current, carried = "", 0
    for piece in _split(text, size, _SEPARATORS):
        if len(current) > carried and len(current) + len(piece) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # Start the overlap on a word boundary so no chunk opens mid-word.
            if " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current, carried = tail, len(tail)
        current += piece
    fresh = current[carried:]
    if fresh.strip():
        if chunks and len(fresh) < minimum:
            chunks[-1] += fresh
        else:
            chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from a note. Malformed YAML is left in the body
    rather than dropped -- losing text is worse than indexing some syntax."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text[m.end():]


def split_markdown(text: str, size: int, overlap: int, minimum: int) -> list[Chunk]:
    """Chunk a markdown body section by section.

    Each chunk carries its heading trail ("Laundry > Dryer > Filter"), which
    is embedded with it: a section's text alone often never names what it is
    about. Headings inside fenced code blocks are not headings.
    """
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    trail: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level = len(m.group(1))
            trail = [(lvl, title) for lvl, title in trail if lvl < level]
            trail.append((level, m.group(2)))
            sections.append(([t for _, t in trail], []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for headings, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        path = " > ".join(headings) or None
        for piece in split_text(body, size, overlap, minimum):
            chunks.append(Chunk(piece, path))
    return chunks
