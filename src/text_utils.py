from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[-+]|\()?\s*[$€£₹]?\s*"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?:trillion|billion|million|thousand|bn|[kmbt]|%|percent|percentage|×|x)?\s*\)?",
    re.IGNORECASE,
)

SCALE = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "m": Decimal("1000000"),
    "million": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
    "t": Decimal("1000000000000"),
    "trillion": Decimal("1000000000000"),
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "which", "with", "year", "page",
    "value", "amount", "number", "figure", "table", "chart", "e", "percent",
}


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def chunk_words(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def parse_decimal_token(raw: str) -> Decimal | None:
    token = raw.strip().lower().replace("−", "-").replace("–", "-")
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("() ")
    token = token.replace("$", "").replace("€", "").replace("£", "").replace("₹", "")
    token = token.replace(",", "").strip()
    match = re.fullmatch(
        r"(?P<sign>[-+]?)\s*(?P<number>\d+(?:\.\d+)?)\s*"
        r"(?P<suffix>trillion|billion|million|thousand|bn|[kmbt]|%|percent|percentage|×|x)?",
        token,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        value = Decimal(match.group("number"))
    except InvalidOperation:
        return None
    if match.group("sign") == "-" or negative:
        value = -value
    suffix = (match.group("suffix") or "").lower()
    if suffix in SCALE:
        value *= SCALE[suffix]
    return value.normalize()


def decimal_key(value: Decimal | None) -> str | None:
    if value is None:
        return None
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def extract_numeric_spans(text: str) -> list[tuple[str, Decimal | None, int, int]]:
    clean = re.sub(r"\[E\d+\]", "", text)
    spans: list[tuple[str, Decimal | None, int, int]] = []
    for match in NUMBER_RE.finditer(clean):
        raw = match.group(0).strip()
        value = parse_decimal_token(raw)
        if value is not None:
            spans.append((raw, value, match.start(), match.end()))
    return spans


def context_tokens(text: str, start: int | None = None, end: int | None = None, window: int = 70) -> set[str]:
    if start is not None and end is not None:
        text = text[max(0, start - window) : min(len(text), end + window)]
    return {token for token in tokenize(text) if token not in STOPWORDS and not token.isdigit()}


def normalize_for_search(text: str) -> str:
    lowered = text.lower().replace("−", "-").replace("–", "-")
    lowered = re.sub(r"\(([$€£₹]?\s*\d[\d,]*(?:\.\d+)?)\)", r"-\1", lowered)
    canonical: list[str] = []
    for raw, value, _, _ in extract_numeric_spans(lowered):
        key = decimal_key(value)
        if key is not None:
            canonical.append(key)
        if "%" in raw or "percent" in raw.lower():
            canonical.append("percent")
        if "$" in raw:
            canonical.extend(["usd", "dollar"])
        elif "€" in raw:
            canonical.append("euro")
        elif "£" in raw:
            canonical.append("pound")
        elif "₹" in raw:
            canonical.extend(["inr", "rupee"])
    searchable = " ".join(tokenize(lowered) + canonical)
    return re.sub(r"\s+", " ", searchable).strip()


def is_numeric_query(question: str) -> bool:
    lowered = question.lower()
    signals = (
        "how much", "how many", "percent", "percentage", "revenue", "income",
        "cost", "margin", "total", "difference", "increase", "decrease", "rate",
        "parameters", "tokens", "flops", "accuracy",
    )
    return bool(extract_numeric_spans(question)) or any(signal in lowered for signal in signals)


def is_visual_query(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("chart", "figure", "plot", "graph", "visual", "trend", "curve"))


def is_method_query(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered
        for term in ("method", "methodology", "approach", "how does", "how do", "how it works")
    )


def expand_retrieval_query(question: str, document_names: list[str] | None = None) -> str:
    """Add intent terms for underspecified research questions without changing the user prompt."""
    lowered = question.lower()
    additions: list[str] = []
    if any(term in lowered for term in ("methodology", "method used", "core method", "approach used")):
        additions.extend(["proposed method", "main contribution", "objective", "key idea", "approach", "how it works", "ours"])
    if any(term in lowered for term in ("main idea", "core idea", "summarize", "summary")):
        additions.extend(["abstract", "main contribution", "conclusion", "proposed approach"])
    if document_names:
        additions.extend(
            re.sub(r"[_-]+", " ", name.rsplit(".", 1)[0]) for name in document_names
        )
    return " ".join([question, *additions]).strip()


def is_structural_number(text: str, raw: str, start: int, end: int) -> bool:
    """Return true for list/section identifiers that are not factual numeric claims."""
    prefix = text[max(0, start - 28) : start].lower()
    suffix = text[end : min(len(text), end + 12)].lower()
    structural_prefixes = (
        "approach", "appendix", "chapter", "equation", "eq.", "figure", "fig.",
        "item", "method", "part", "phase", "section", "step", "table",
    )
    if any(re.search(rf"\b{re.escape(label)}\s*$", prefix) for label in structural_prefixes):
        return True
    compact = raw.strip()
    if compact.endswith(")") and re.fullmatch(r"\d+\)", compact):
        return True
    if re.fullmatch(r"\d+", compact) and (suffix.lstrip().startswith(("-", "–", "—"))):
        return True
    if re.fullmatch(r"\d+", compact) and prefix.rstrip().endswith(("-", "–", "—")):
        return True
    return False
