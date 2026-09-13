from scraper.ingest.chunk import chunk_text
from scraper.ingest.extract import extract_text, is_divider, plain, table_cells, title_of

HTML = b"""
<html><head><title>Hayden Library Hours</title></head>
<body><nav>Home | About</nav>
<main>
  <h1>Hours</h1>
  <p>Monday through Thursday: 7am to 2am.</p>
  <p>Friday: 7am to 6pm.</p>
  <script>track()</script>
</main>
<footer>Copyright</footer></body></html>
"""


def test_extract_keeps_main_and_drops_boilerplate() -> None:
    text = extract_text(HTML)
    assert "7am to 2am" in text
    assert "Home | About" not in text
    assert "Copyright" not in text
    assert "track()" not in text


def test_title_is_read() -> None:
    assert title_of(HTML) == "Hayden Library Hours"


def test_chunks_respect_max_and_overlap() -> None:
    paragraphs = [f"Paragraph {i}. " + ("word " * 40) for i in range(12)]
    chunks = chunk_text("\n".join(paragraphs), max_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[0][-50:].strip() in chunks[1]


def test_long_paragraph_is_split() -> None:
    text = "sentence one. " * 200
    chunks = chunk_text(text, max_chars=300, overlap_chars=0)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) >= len(text) * 0.95


def test_a_divider_row_with_a_caption_after_it_is_still_a_divider() -> None:
    # lib.asu.edu/hours puts the table caption after the closing pipe of the divider row.
    cells = table_cells("| --- | --- | --- |Display of Opening hours")
    assert cells == ["---", "---", "---"]
    assert is_divider(cells)


def test_a_plain_table_row_keeps_every_cell() -> None:
    assert table_cells("| Hayden | 7am | Closed |") == ["Hayden", "7am", "Closed"]
    assert table_cells("not a table") is None


def test_a_line_break_tag_becomes_a_space() -> None:
    # lib.asu.edu/hours puts the weekday under the date with a break tag inside the cell.
    assert plain("Sep 11  <br>Friday") == "Sep 11 Friday"
    assert plain("a<br/>b") == "a b"


def test_overlap_starts_on_a_line_boundary() -> None:
    row = "Library {}: Monday 7am - 2am; Tuesday 7am - 2am; Sunday 10am - 6pm"
    rows = [row.format(i) for i in range(20)]
    chunks = chunk_text("\n".join(rows), max_chars=400, overlap_chars=120)
    assert len(chunks) > 1
    for c in chunks[1:]:
        assert c.startswith("Library "), f"chunk opens mid-line: {c[:60]!r}"
