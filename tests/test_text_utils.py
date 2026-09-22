from decimal import Decimal

from src.text_utils import (
    decimal_key,
    extract_numeric_spans,
    is_numeric_query,
    normalize_for_search,
    parse_decimal_token,
)


def test_scaled_and_parenthesized_numbers_are_normalized() -> None:
    assert parse_decimal_token("$4.2B") == Decimal("4.2") * Decimal("1000000000")
    assert parse_decimal_token("(14,264)") == Decimal("-14264")
    assert decimal_key(parse_decimal_token("1.40 trillion")) == "1400000000000"


def test_search_normalization_adds_numeric_and_currency_variants() -> None:
    normalized = normalize_for_search("Revenue was $4.2bn, or 67.5% of sales.")
    assert "4200000000" in normalized
    assert "usd" in normalized
    assert "percent" in normalized


def test_citation_numbers_are_not_extracted_as_claims() -> None:
    spans = extract_numeric_spans("Net income was $112,010 million [E1].")
    assert [decimal_key(value) for _, value, _, _ in spans] == ["112010000000"]


def test_numeric_query_detection() -> None:
    assert is_numeric_query("How much revenue was reported?")
    assert not is_numeric_query("Why was the method selected?")
