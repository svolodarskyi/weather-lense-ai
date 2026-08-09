"""
WeatherLens AI — text chunking and sentence embedding.

chunk_text()  splits a narrative into overlapping windows snapped to
              natural boundaries (paragraph > sentence > word).
Encoder       wraps SentenceTransformer; loaded once, reused per worker.
"""

MODEL_NAME   = "all-MiniLM-L6-v2"
DIMS         = {MODEL_NAME: 384}   # model_name → embedding width
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 100
BATCH_SIZE   = 64


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows snapped to natural boundaries.

    Boundary preference: paragraph (\\n\\n) > sentence ([.!?] + space) > word.
    A boundary is accepted only when the window is at least 50% full — this
    prevents a period 30 chars into an 800-char window from emitting a tiny
    first chunk.

    Args:
        text:       raw narrative text (stripped on entry)
        chunk_size: target window size in characters
        overlap:    how many characters the next chunk re-reads from the current one
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    half = chunk_size // 2
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Last piece — take the remainder as-is.
        if end >= len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # Only consider break points in the second half of the window.
        min_pos = start + half

        # 1. Paragraph break (\n\n) — highest priority and a clean semantic
        #    boundary; no overlap is needed across it.
        p = text.rfind("\n\n", min_pos, end)
        if p != -1:
            chunk = text[start:p].strip()
            if chunk:
                chunks.append(chunk)
            next_start = p + 2   # skip past the two newlines
            while next_start < len(text) and text[next_start] in " \t":
                next_start += 1
            start = next_start
            continue

        # 2. Sentence break — [.!?] followed by whitespace.
        brk: int | None = None
        for i in range(end - 1, min_pos - 1, -1):
            if text[i] in ".!?" and i + 1 < len(text) and text[i + 1] in " \n\r\t":
                brk = i + 1  # include the punctuation in this chunk
                break

        # 3. Word break — any whitespace character.
        if brk is None:
            for i in range(end - 1, min_pos - 1, -1):
                if text[i] in " \n\r\t":
                    brk = i
                    break

        # 4. Forced break — no whitespace at all in the window (rare in NWS text).
        if brk is None:
            brk = end

        chunk = text[start:brk].strip()
        if chunk:
            chunks.append(chunk)

        # Overlap: go back `overlap` chars from the break, then snap FORWARD
        # to the next word boundary so every chunk begins at a complete word.
        # The snap is capped at `brk` to prevent overshooting into the next
        # window when the text has no whitespace (force-break case).
        next_start = max(start + 1, brk - overlap)
        # If we landed mid-word (prev char and current char are both non-space),
        # advance to the end of the partial word — but no further than brk.
        if (next_start > 0
                and text[next_start - 1] not in " \t\n\r"
                and next_start < len(text)
                and text[next_start] not in " \t\n\r"):
            while next_start < brk and text[next_start] not in " \t\n\r":
                next_start += 1
        # Skip any whitespace to land at the first character of the next word.
        while next_start < len(text) and text[next_start] in " \t\n\r":
            next_start += 1
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder:
    """Thin wrapper around SentenceTransformer that validates model width.

    Raises ValueError for an unknown model so a wrong-width embedding cannot
    reach a VECTOR(384) column silently — the assignment would succeed but
    the pgvector operator would reject it with an unhelpful dimension mismatch
    error at query time rather than at write time.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        if model_name not in DIMS:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Known models and dims: {DIMS}. "
                "Add an entry to DIMS in embeddings.py to use a new model."
            )
        # Deferred import so the module can be imported in tests without
        # pulling in sentence_transformers / torch.
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device="cpu")
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dims(self) -> int:
        return DIMS[self._model_name]

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return a list of float vectors, one per input text."""
        return self._model.encode(texts, normalize_embeddings=True).tolist()
