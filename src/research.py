"""Bounded multi-query research and parent-page context assembly."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

from .models import DocumentChunk, RetrievedEvidence
from .text_utils import is_method_query, is_visual_query, tokenize


def query_facets(question: str, answer_mode: str) -> list[str]:
    queries = [question.strip()]
    q = question.lower()
    if answer_mode == "Quick":
        return queries
    if is_method_query(q) or any(x in q for x in ("summar", "overview", "explain")):
        queries.extend([
            f"{question} proposed method mechanism algorithm design main contribution",
            f"{question} experimental results evaluation baseline limitations",
        ])
    elif any(x in q for x in ("evaluat", "comparison", "compare", "window", "cache size")):
        queries.extend([
            f"{question} experimental setup settings context window cache size benchmark",
            f"{question} baseline comparison results dataset limitations",
        ])
    elif is_visual_query(q):
        queries.append(f"{question} caption axes legend trend interpretation")
    else:
        queries.append(f"{question} explanation supporting evidence context limitations")
    return queries[:2] if answer_mode == "Detailed" else queries[:3]


def normalize_queries(values: object, question: str) -> list[str]:
    if not isinstance(values, list):
        return [question]
    return list(dict.fromkeys([question, *[
        value.strip()[:400] for value in values if isinstance(value, str) and value.strip()
    ]]))[:5]


def _unique_text(parts: Sequence[str], word_budget: int) -> str:
    """Remove the exact word overlap introduced by sliding-window chunking."""
    words: list[str] = []
    for part in parts:
        incoming = part.split()
        overlap = 0
        for n in range(min(len(words), len(incoming), 80), 4, -1):
            if words[-n:] == incoming[:n]:
                overlap = n
                break
        words.extend(incoming[overlap:])
        if len(words) >= word_budget:
            break
    return " ".join(words[:word_budget])


def assemble_evidence(
    result_lists: list[list[RetrievedEvidence]],
    chunks: Sequence[DocumentChunk],
    *,
    max_items: int = 10,
    word_budget: int = 5200,
) -> list[RetrievedEvidence]:
    """Fuse query results, cover query winners, diversify pages, expand parent context.

    Context stays on the cited page. It is separately recorded and is exactly the
    text used for generation and verification, never an uncited adjacent page.
    """
    scores: dict[tuple[str, int], float] = {}
    representatives: dict[tuple[str, int], RetrievedEvidence] = {}
    winners: list[tuple[str, int]] = []
    for query_index, results in enumerate(result_lists):
        seen = set()
        if results:
            winners.append(results[0].page_key)
        for rank, item in enumerate(results, 1):
            key = item.page_key
            if key in seen:
                continue
            seen.add(key)
            weight = 2.0 if query_index == 0 else 1.0
            scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
            representatives.setdefault(key, item)
    if not scores:
        return []
    ordered = sorted(scores, key=scores.get, reverse=True)
    # Preserve each subquestion/document's strongest page before diversity filling.
    chosen = list(dict.fromkeys(winners))[:max_items]
    token_sets = {key: set(tokenize(representatives[key].chunk.text)) for key in ordered}
    while len(chosen) < max_items:
        remaining = [key for key in ordered if key not in chosen]
        if not remaining:
            break
        def relevance_diversity(key: tuple[str, int]) -> float:
            similarity = max((
                len(token_sets[key] & token_sets[other]) / max(1, len(token_sets[key] | token_sets[other]))
                for other in chosen
            ), default=0.0)
            return 0.8 * scores[key] / max(scores.values()) - 0.2 * similarity
        chosen.append(max(remaining, key=relevance_diversity))

    by_page: dict[tuple[str, int], list[DocumentChunk]] = {}
    for chunk in chunks:
        by_page.setdefault((chunk.document_id, chunk.page_number), []).append(chunk)
    selected = []
    used_words = 0
    for key in chosen:
        item = representatives[key]
        page = by_page[key]
        # Put the hit first; retain tables as Markdown and append surrounding prose.
        hit_text = item.chunk.text
        surrounding = [c.text for c in page if c.modality == "prose" and c.chunk_id != item.chunk.chunk_id]
        remaining = word_budget - used_words
        if remaining < 80:
            break
        if item.chunk.modality == "table":
            context = hit_text + "\n\nPage context: " + _unique_text(surrounding, min(260, remaining))
        else:
            tables = [c.text for c in page if c.modality == "table"]
            context = _unique_text([hit_text, *surrounding], min(500 if tables else 700, remaining))
            if tables and remaining - len(context.split()) >= 100:
                context += "\n\n" + tables[0]
        # Global budget also applies to unusually large tables.
        if len(context.split()) > remaining:
            lines, count = [], 0
            for line in context.splitlines():
                if count + len(line.split()) > remaining:
                    break
                lines.append(line)
                count += len(line.split())
            context = "\n".join(lines)
        if not context:
            continue
        used_words += len(context.split())
        selected.append(replace(item, evidence_id=f"E{len(selected)+1}", context_text=context))
    return selected


def citation_audit(answer: str, evidence: Sequence[RetrievedEvidence]) -> tuple[float, list[str]]:
    valid_ids = {item.evidence_id for item in evidence}
    claimed = set(re.findall(r"\[(E\d+)\]", answer))
    warnings = []
    if claimed - valid_ids:
        warnings.append("The answer contains unknown citation labels: " + ", ".join(sorted(claimed - valid_ids)))
    units = [unit.strip() for unit in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", answer) if unit.strip()]
    units = [unit for unit in units if len(unit.split()) >= 8 and not unit.startswith(("#", "Sources:"))]
    covered = sum(bool(set(re.findall(r"\[(E\d+)\]", unit)) & valid_ids) for unit in units)
    coverage = covered / len(units) if units else 0.0
    if units and coverage < 1:
        warnings.append("Some answer passages lack an inline source citation. Citation coverage does not prove the claims are correct.")
    return coverage, warnings


def focus_passages(question: str, evidence: Sequence[RetrievedEvidence]) -> str:
    """Extract a short, verbatim relevance guide ahead of the broader context."""
    ignored = {"what", "which", "was", "were", "the", "and", "for", "this", "that", "with", "used", "explain", "document", "paper", "models"}
    terms = set(tokenize(question)) - ignored
    if "window" in terms:
        terms.update(("cache", "size"))
    candidates = []
    for rank, item in enumerate(evidence[:4]):
        for sentence in re.split(r"(?<=[.!?])\s+|\n", item.context_text or item.chunk.text):
            words = set(tokenize(sentence))
            if len(sentence.split()) < 8 or len(sentence.split()) > 100:
                continue
            score = len(words & terms) / max(1, len(terms)) + 0.15 / (1 + rank)
            if "window" in terms and "cache size" in sentence.lower():
                score += 0.5
            candidates.append((score, sentence, item.evidence_id))
    winners = sorted(candidates, reverse=True)[:4]
    return "\n".join(f"[{eid}] {sentence}" for _, sentence, eid in winners)


def export_markdown(result) -> str:
    sections = [f"# {result.question}\n", result.answer, "\n## Sources\n"]
    for item in result.evidence:
        if item.evidence_id in result.used_evidence_ids:
            sections.append(f"- [{item.evidence_id}] {item.chunk.document_name}, PDF page {item.chunk.page_number}")
    if result.warnings or result.parse_warning:
        sections.append("\n## Review notes\n")
        sections.extend(f"- {warning}" for warning in [*result.warnings, *([result.parse_warning] if result.parse_warning else [])])
    sections.append(f"\nProvider: {result.provider or 'Evidence search'} · Model: {result.model or 'none'}")
    return "\n".join(sections)
