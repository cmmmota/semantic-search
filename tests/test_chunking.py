from semantic_search.chunking import parse_frontmatter, split_markdown, split_text


def test_short_text_is_one_chunk():
    assert split_text("hello world", 1000, 100, 200) == ["hello world"]


def test_empty_text_has_no_chunks():
    assert split_text("   \n ", 1000, 100, 200) == []


def test_chunks_respect_size_and_cover_text():
    words = [f"word{i}" for i in range(2000)]
    text = " ".join(words)
    chunks = split_text(text, 500, 50, 100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 + 100 for c in chunks)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)


def test_overlap_starts_on_word_boundary():
    text = " ".join(f"w{i:04d}" for i in range(500))
    for c in split_text(text, 300, 60, 50)[1:]:
        assert c.startswith("w") and len(c.split()[0]) == 5


def test_short_tail_is_folded_into_previous_chunk():
    text = ("a" * 90 + " ") * 10 + "tail"
    chunks = split_text(text, 400, 0, 50)
    assert chunks[-1].endswith("tail")
    assert not any(c == "tail" for c in chunks)


def test_paragraphs_preferred_over_words():
    text = "First paragraph here.\n\nSecond paragraph here."
    assert split_text(text, 30, 0, 0) == ["First paragraph here.", "Second paragraph here."]


def test_frontmatter_parsed_and_removed():
    meta, body = parse_frontmatter("---\ntags: [a, b]\naliases: x\n---\n# Title\nBody")
    assert meta == {"tags": ["a", "b"], "aliases": "x"}
    assert body == "# Title\nBody"


def test_malformed_frontmatter_kept_in_body():
    text = "---\n: : bad: [\n---\nBody"
    assert parse_frontmatter(text) == ({}, text)


def test_markdown_heading_paths():
    md = "intro\n# A\ntext a\n## B\ntext b\n# C\ntext c\n### D\ntext d"
    chunks = split_markdown(md, 1000, 0, 0)
    assert [(c.heading_path, c.text) for c in chunks] == [
        (None, "intro"),
        ("A", "text a"),
        ("A > B", "text b"),
        ("C", "text c"),
        ("C > D", "text d"),
    ]


def test_heading_only_sections_produce_no_chunks():
    assert split_markdown("# A\n# B\nbody", 1000, 0, 0)[0].heading_path == "B"


def test_hashes_in_code_fences_are_not_headings():
    md = "# Real\n```bash\n# a comment\necho hi\n```"
    chunks = split_markdown(md, 1000, 0, 0)
    assert len(chunks) == 1 and chunks[0].heading_path == "Real"
    assert "# a comment" in chunks[0].text


def test_obsidian_markdown_rendered_for_embedding():
    from semantic_search.sources.obsidian import _for_embedding
    body = (
        "![Pinout | T568B](https://x/y.png?a=1) see [docs](https://d)"
        " and [[Note#Sec|alias]] [[Other]] ![[img.jpg]]"
    )
    assert _for_embedding(body) == "Pinout | T568B see docs and alias Other "
