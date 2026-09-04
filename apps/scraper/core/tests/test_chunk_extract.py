from scraper.chunk import chunk_text
from scraper.extract import extract_text, title_of

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
